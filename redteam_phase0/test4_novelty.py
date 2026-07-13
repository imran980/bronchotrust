"""TEST 4 -- novelty audit. Assume Reviewer #2 is right. Compare the PROPOSED estimator
mathematically (optimization variables / objective / unknowns / output) against prior work. Do NOT
argue. If ours is only a combination of previous work, state it. References characterized by their
known/likely class; uncertainty is flagged explicitly (no fabricated equations)."""
import json

OURS = dict(name="PROPOSED",
            variables="SoR radius profile r(s) (low-dim), + albedo A, + light intensity I",
            objective="photometric residual ||B_obs - A*(n.l)/d^2||^2 under a near-light Lambertian model",
            unknowns="r(s), albedo, light",
            fixed="camera POSES (from COLMAP/Barbour) + intrinsics",
            output="airway cross-section CSA / DCE profile (scene units)")

PRIOR = [
 dict(name="NFL-BA (near-field-lighting bundle adjustment)", confidence="medium",
      variables="camera poses + per-vertex/dense geometry + near-field light params",
      objective="photometric BA (near-field light) reprojection consistency",
      unknowns="poses, geometry, lighting", output="refined poses + dense surface",
      relation="Ours = NFL-BA with POSES FIXED and geometry restricted to a SoR prior. A constrained special case; same near-light photometric objective. NOT fundamentally new."),
 dict(name="Sengupta et al. (endoscopic photometric / shape-from-shading)", confidence="low (exact formulation uncertain)",
      variables="per-pixel depth/normals (+ light/albedo)", objective="shape-from-shading / photometric depth",
      unknowns="depth, normals, albedo, light", output="depth / surface",
      relation="Shares the shading cue. Ours adds a SoR parametrization + fixed COLMAP poses. If Sengupta is near-light SfS, ours is a tubular-constrained variant. Uncertain reference -> characterized by class."),
 dict(name="LightNeuS (near-light neural implicit surface, Zaragoza/MICCAI'23)", confidence="medium-high",
      variables="neural SDF weights + light params", objective="near-light (inverse-square) photometric loss + SDF/eikonal reg",
      unknowns="implicit surface (general topology), light", output="implicit 3D surface of the lumen",
      relation="SAME near-light photometric loss. LightNeuS uses a general neural SDF; ours replaces it with an explicit low-dim SoR. Ours is a STRICT SPECIALIZATION (less general). NOT fundamentally new; arguably a downgrade in surface model."),
 dict(name="Barbour (clinical 30-frame COLMAP SfM+MVS recipe)", confidence="high (our own baseline)",
      variables="(feature-based SfM/MVS; not an explicit photometric optimizer)", objective="geometric reprojection / patch photo-consistency (parallax)",
      unknowns="poses + dense point cloud", output="airway point cloud -> cross-section",
      relation="Different CUE: parallax/feature geometry, not shading. Ours REUSES Barbour/COLMAP poses as fixed input and adds a photometric layer. Complementary, not a replacement; ours depends on it."),
 dict(name="Zaragoza group (near-light photometric endoscopy: EndoMapper / NR-SLAM / LightNeuS line)", confidence="medium",
      variables="poses and/or surface + near-light photometric model", objective="near-light photometric consistency (+ SLAM/BA)",
      unknowns="poses, surface, light", output="endoscopic surface / map",
      relation="Ours sits INSIDE this line (near-light photometric + endoscopy). The SoR + fixed-pose + clinical-CSA choices are a narrowing, not a new estimator. Heavy overlap."),
 dict(name="BREA-Depth (learned bronchoscopy depth)", confidence="low (exact method uncertain)",
      variables="network weights (trained offline)", objective="supervised/self-supervised monocular depth",
      unknowns="network params", output="per-frame depth maps",
      relation="Different paradigm: LEARNED depth vs our per-scene physics optimization. Ours needs no training data but assumes the shading model; BREA needs training data. Orthogonal; not the same estimator. Uncertain reference."),
]


def main():
    verdict = dict(
        is_fundamentally_new=False,
        summary=("The proposed estimator is a COMBINATION of existing components: the near-light Lambertian "
                 "photometric objective (NFL-BA / LightNeuS / Zaragoza / Sengupta) + a surface-of-revolution "
                 "low-dim geometry prior (a strict specialization of LightNeuS's implicit surface; tubular) + "
                 "FIXED COLMAP/Barbour poses (parallax init, not optimized) + a clinical CSA/DCE readout. "
                 "The mathematics (variables, objective, unknowns) are not distinct from the near-light "
                 "photometric endoscopy line; the only new element is the SPECIFIC combination and the clinical "
                 "cross-section output, not the estimator itself. Relative to LightNeuS it is LESS general."),
        distinct_element_if_any=("Only: explicit SoR parametrization + fixed-pose photometric fit yielding a "
                                 "direct CSA/DCE clinical profile. This is an engineering combination, not a "
                                 "new inverse problem."))
    open("test4_novelty.json", "w").write(json.dumps({"ours": OURS, "prior": PRIOR, "verdict": verdict}, indent=2))
    print("=== TEST 4 novelty audit ===")
    print("OURS:", OURS["variables"], "|", OURS["objective"], "| fixed:", OURS["fixed"])
    for p in PRIOR:
        print(f"\n[{p['name']}]  (confidence: {p['confidence']})")
        print(f"  vars: {p['variables']}\n  obj:  {p['objective']}\n  vs ours: {p['relation']}")
    print("\nVERDICT: fundamentally new =", verdict["is_fundamentally_new"])
    print(verdict["summary"])


if __name__ == "__main__":
    main()
