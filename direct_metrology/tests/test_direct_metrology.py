"""Fast unit tests for the direct-metrology prototype. These are the correctness invariants that must
hold independent of the full gate run. Run: pytest direct_metrology/tests/ -q

They use a SMALL config (fewer cameras) so the suite is quick; the full statistical gates live in
run.py. Nothing here tunes parameters -- each test asserts a structural property."""
from __future__ import annotations
import numpy as np
import pytest

from direct_metrology import phantom as PH
from direct_metrology import detect as DT
from direct_metrology import estimator as ES


def _cfg(n_cam=10):
    return dict(
        phantom=dict(R_ref_mm=6.0, r_throat_mm=3.0, length_mm=40.0, stenosis_z_mm=25.0, sigma_mm=4.0,
                     n_cameras=n_cam, cam_depth_mm=12.0, cam_cone_deg=8.0, albedo=1.0, seed=0,
                     intrinsics=dict(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)),
        optimizer=dict(lambda_contour=1.0, lambda_photo=0.02, max_nfev=4000, wall_sample_offset=1.15))


@pytest.fixture(scope="module")
def scene():
    cfg = _cfg()
    P = PH.render(cfg)
    cons = DT.detect_all(P.images)
    return cfg, P, cons


def test_gt_is_self_consistent(scene):
    _, P, _ = scene
    g = P.gt
    assert abs(g["csa_mm2"] - np.pi * g["throat_radius_mm"] ** 2) < 1e-9
    assert abs(g["dce_mm"] - 2 * g["throat_radius_mm"]) < 1e-9


def test_render_is_deterministic():
    P1 = PH.render(_cfg()); P2 = PH.render(_cfg())
    assert all(np.array_equal(a, b) for a, b in zip(P1.images, P2.images))


def test_detection_finds_apertures(scene):
    _, P, cons = scene
    ndet = sum(c is not None for c in cons)
    assert ndet >= 0.6 * len(P.images), f"only {ndet}/{len(P.images)} detected"


def test_exact_contour_recovers_gt_to_machine_precision(scene):
    """Estimator on NOISE-FREE projected-GT contours must return GT radius to <0.5% (isolates estimator)."""
    cfg, P, _ = scene
    g = P.gt
    th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    circ = np.array(g["throat_center_mm"])[None, :] + g["throat_radius_mm"] * np.c_[np.cos(th), np.sin(th), 0 * th]
    exact = []
    for cam in P.cameras:
        uv, z = ES._project(circ, cam.R, cam.C, P.K)
        exact.append(uv[z > 0])
    th0 = ES.theta_from(g["throat_center_mm"], g["throat_axis"], g["throat_radius_mm"])
    fr = ES.fit(exact, P.cameras, P.K, P.images, th0, lam_c=1.0, lam_p=0.0,
                wall_offset=cfg["optimizer"]["wall_sample_offset"], use_photo=False)
    assert 100 * abs(fr.r - g["throat_radius_mm"]) / g["throat_radius_mm"] < 0.5


def test_pipeline_gt_init_within_5pct(scene):
    cfg, P, cons = scene
    g = P.gt
    th = ES.theta_from(g["throat_center_mm"], g["throat_axis"], g["throat_radius_mm"])
    o = cfg["optimizer"]
    fr = ES.fit(cons, P.cameras, P.K, P.images, th, lam_c=o["lambda_contour"], lam_p=o["lambda_photo"],
                wall_offset=o["wall_sample_offset"])
    assert 100 * abs(fr.dce - g["dce_mm"]) / g["dce_mm"] < 5.0
    assert 100 * abs(fr.csa - g["csa_mm2"]) / g["csa_mm2"] < 5.0


def test_sim3_equivariance(scene):
    """Scaling the whole scene by s must scale recovered DCE by exactly s (scale-free ratio invariant)."""
    cfg, P, cons = scene
    g = P.gt; o = cfg["optimizer"]
    th = ES.theta_from(g["throat_center_mm"], g["throat_axis"], g["throat_radius_mm"])
    fr0 = ES.fit(cons, P.cameras, P.K, P.images, th, lam_c=o["lambda_contour"], lam_p=o["lambda_photo"],
                 wall_offset=o["wall_sample_offset"])
    s = 2.5; Rg = np.eye(3); tg = np.array([10.0, -5.0, 3.0])
    cams2 = [PH.Camera(R=cam.R @ Rg.T, C=s * (Rg @ cam.C) + tg) for cam in P.cameras]
    th2 = ES.theta_from(s * np.array(g["throat_center_mm"]) + tg, g["throat_axis"], s * g["throat_radius_mm"])
    fr1 = ES.fit(cons, cams2, P.K, P.images, th2, lam_c=o["lambda_contour"], lam_p=o["lambda_photo"],
                 wall_offset=o["wall_sample_offset"])
    assert 100 * abs(fr1.dce - s * fr0.dce) / (s * fr0.dce) < 1.0


def test_photometry_only_cannot_recover_radius(scene):
    """The WEAK photometric term alone is underdetermined -> must NOT accidentally match GT."""
    cfg, P, cons = scene
    g = P.gt; o = cfg["optimizer"]
    th_bad = ES.theta_from(g["throat_center_mm"], g["throat_axis"], g["throat_radius_mm"] * 1.6)
    fr = ES.fit(cons, P.cameras, P.K, P.images, th_bad, lam_c=o["lambda_contour"], lam_p=o["lambda_photo"],
                wall_offset=o["wall_sample_offset"], use_contour=False, use_photo=True)
    # photo-only from a biased init stays biased (weak term does not pull to GT radius)
    assert 100 * abs(fr.dce - g["dce_mm"]) / g["dce_mm"] > 5.0


def test_covariance_is_finite_and_positive(scene):
    cfg, P, cons = scene
    g = P.gt; o = cfg["optimizer"]
    th = ES.theta_from(g["throat_center_mm"], g["throat_axis"], g["throat_radius_mm"])
    fr = ES.fit(cons, P.cameras, P.K, P.images, th, lam_c=o["lambda_contour"], lam_p=o["lambda_photo"],
                wall_offset=o["wall_sample_offset"])
    assert np.isfinite(fr.r_var) and fr.r_var > 0
    assert np.isfinite(fr.csa_var) and np.isfinite(fr.dce_var)


def test_eval_adapter_roundtrips_through_frozen_framework(scene):
    """The fit must evaluate through evaluation/ unchanged, returning finite CSA/DCE % error."""
    from direct_metrology import eval_adapter as EA
    cfg, P, cons = scene
    g = P.gt; o = cfg["optimizer"]
    th = ES.theta_from(g["throat_center_mm"], g["throat_axis"], g["throat_radius_mm"])
    fr = ES.fit(cons, P.cameras, P.K, P.images, th, lam_c=o["lambda_contour"], lam_p=o["lambda_photo"],
                wall_offset=o["wall_sample_offset"])
    ev, res, gt = EA.evaluate_fit(fr, g)
    gt.assert_immutable()
    assert np.isfinite(ev.CSA_error["CSA_error_pct"]["rmse"])
    assert np.isfinite(ev.DCE_error["DCE_error_pct"]["rmse"])
    assert res.units == "mm"
