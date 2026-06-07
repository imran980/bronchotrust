Project goal in one paragraph. Calibration-free, CT-free bronchoscopy reconstruction with uncertainty. Validated against 4 paired pediatric cases (Barbour cohort).
Hard rules learned the hard way:

Never add uninvited preprocessing (ROI crops, normalizations) — match training-time exactly.
When restoring a previously-working path, change only what's demonstrably broken.
Verify train-vs-inference preprocessing parity before shipping any model on real video (means, stds, channel order, resize method, ROI bounds).
Depth scale conventions: document at every interface. Past trap: C3VD raw=655.35 vs preprocessed=2.55.
Endo-2DTAM is shelved permanently. Do not revisit.


Phase-gate discipline:

Each phase has a written gate. Don't skip gates.
One change at a time. Don't simultaneously tune multiple things.
Time-box experiments. If 2× over estimate, stop and escalate.
If reaching for justifications, wait — pause and check with user.


Reusable assets and their conventions:

Textured synthetic renderer location + depth scale (2.55).
PCA-SSM location.
Real video corpus location + 4 paired cases location.
5-class scheme: vocal_cord, trachea, main_carina, rmb, lmb.

SSM-fit outputs (under runs/<session>/recon_dust3r_foe/ssm_fit/):
  mesh_ssm.ply               147996-vert MAP airway mesh, all verts always
                             included. Vertex colors encode posterior STD.
  mesh_ssm_residual_color.ply Same geometry, colored by nearest-obs distance.
  figure_mask.npy            Per-vertex bool (Fix 1, 2026-05-26). True = OK
                             for paper figures; False = spike-adjacent
                             (incident edge > 3x median edge length). NEVER
                             use to delete vertices from the mesh data; only
                             apply when rendering. See diag_fix1_spikes.py.
  is_supported.npy           Per-vertex bool (Fix 2 redo, 2026-05-27).
                             ANATOMICAL definition: True iff
                                (vertex_segment[v] in visited_segments) AND
                                (dist(v, visited_centerline) < tol_mm).
                             Quantitative claims (diameter, CI coverage,
                             sizing) computed only over True. Outside it,
                             mesh is SSM-prior-only extrapolation - use 20%
                             gray opacity in paper figures (diag_fix2_extent).
  is_supported_geometric.npy Earlier geometric definition kept as a
                             fallback: per_vert_dist_mm < support_d_mm.
                             Anatomical version is the default.
  visited_segments.npy       (K,) int8, K <= 5. Subset of {0:TRACHEA,
                             1:RMB, 2:LMB, 3:BI, 4:DISTAL} visited by the
                             camera trajectory.
  vert_dist_to_visited_centerline_mm.npy  (V,) float, vertex distance to
                             nearest centerline node that received an
                             observation projection. Used in is_supported.
  diameter_profile.csv       Fix 4b (2026-05-28): validated cross-
                             sectional diameter along the airway, ONLY
                             at centerline nodes that pass the hard
                             validity gate. Columns: arclength_mm,
                             diameter_mm, diameter_std_mm,
                             diameter_obs_mm, diameter_advance_mm,
                             diameter_withdrawal_mm, n_obs,
                             n_obs_advance, n_obs_withdrawal, segment,
                             n_wall_verts, centerline_node_idx,
                             validated. Units in template-mm.
  diameter_profile_unvalidated.csv  Preserved Fix 4 output (had RMB/BI
                             diameter values that were SSM-prior
                             hallucinations - never use for clinical
                             claims). Kept for ablation comparison only.
  diameter_profile.png       Plot of diameter (mean +/- 1 sigma posterior
                             CI) vs arclength. Each segment in its own
                             colour, segment transitions marked. Raw
                             per-node line shown faintly underneath the
                             Gaussian-smoothed (sigma=4 nodes) curve.

Diameter-profile conventions:
  - HARD RULE (Fix 4b, 2026-05-28): NEVER report diameter for any
    segment in scope_extent.json's segments_never_reached. Physical
    reality (scope_extent) overrides detector / HMM output. Peeked and
    reached_not_entered segments are also excluded (peek tolerance = 0
    in the dev set). Measurable = segments_fully_traversed only.
  - Per-node validity gate (both must pass):
      (a) node's segment in scope_extent.segments_fully_traversed, AND
      (b) obs_count_per_node >= --min_obs_per_node (default 20).
    If either fails -> node is dropped; no prior-fill.
  - Sanity assertion fires hard if any never_reached segment has a
    validated node (-> validity gate broken).
  - Per-node radius = median Euclidean distance from centerline node to
    same-segment wall vertices within a (slab_mm)=4 mm perpendicular
    slab and within search_radius_mm=20 mm sphere, after figure_mask.
    Diameter = 2 * radius.
  - Posterior CI from N=100 alpha samples: vertex positions per sample,
    median radius per sample, std across samples.
  - Direct-obs diameter (diameter_obs_mm column): 2 * median perpendicular
    distance from centerline node to dust3r-foe observation points in
    the same slab.
  - Advance/withdrawal consistency: deepest_frame_idx = argmax of
    valid-camera-position projection onto PCA-1 of trajectory; obs
    classified by source frame index <= or > deepest. Per-node
    diameter computed from each subset independently. Large median
    |advance - withdrawal| flags a tracking-quality issue.

scope_extent.json (Fix 4b, 2026-05-28). One per video, at
runs/<vid>/scope_extent.json. Schema:
  deepest_segment_reached: "trachea" | "main_carina" | "rmb_peek" |
                            "lmb_peek" | ... (informational)
  segments_fully_traversed: [...]           # measurable
  segments_reached_not_entered: [...]       # excluded
  segments_peeked: [...]                    # excluded (peek tol 0)
  segments_never_reached: [...]             # hard-excluded; assertion
                                              fires if any have nodes
Segment vocabulary (lowercase strings): trachea | main_carina | rmb |
lmb | bi | distal. main_carina is a junction landmark, not a segment.

Scale-recovery known issue (Fix 4b finding, 2026-05-28):
  - Trachea median diameter on 3 dev videos: 5V1 37.6 mm, 13V2 19.6 mm,
    16V1 25.8 mm.  13V2 is plausible adult (15-25 mm range), 5V1 and
    16V1 are OUTSIDE plausible range.  This confirms similarity-scale
    recovery from dust3r-foe + SSM-fit is unreliable on its own.
  - Until metric calibration is added (anatomical anchor: trachea
    diameter from CT segmentation; or AnyCalib fine-tune from Phase 3),
    absolute diameter values are NOT reliable.  Profile SHAPE
    (narrowings/widenings, relative changes) IS reliable within a
    single fit's scale.
  ssm_fit.npz                alpha (z-score clipped to +/-3sigma), R, t, s,
                             per_vert_std (Fix 3 2026-05-27: now from
                             Sigma_alpha = (A^T A / sigma^2 + lambda*I)^-1,
                             prior precision = lambda*I, not (lambda/sigma^2)*I
                             as before), per_vert_dist, per_vert_dist_mm,
                             figure_mask, is_supported, support_d_mm,
                             posterior_samples, Sigma_alpha, sigma2_hat.
  ssm_fit.json               summary + history.

ssm_fit conventions:
  Modes pre-normalised so alpha is in z-score units.
  alpha clipped to +/-3 inside the optimisation loop (Fix 1, 2026-05-26).
  lambda_alpha default 10000 in z-score parameterisation.
  support_d_mm default 5 mm (geometric fallback); sensitivity reported 3-10 mm.
  support_anatomical_tol_mm default 10 mm (anatomical primary definition).
  All quantitative metrics (diameter, CI coverage, Hausdorff vs CT) are
  computed only over is_supported=True vertices. Mesh-extent visualisation
  outside this mask must be visually de-emphasised in paper figures
  (20% gray opacity) to avoid overselling coverage to reviewers.

Init ensemble (Fix 2b, 2026-05-27): per video, ssm_fit runs 9 init
candidates and picks the winner by (TRACHEA in visited, named-only
supported fraction):
  - 8 landmark-anchored rotations at rolls 0,45,...,315 around the
    trajectory PCA-1 (= trachea axis). Trajectory-PCA init uses
    template_landmarks.json's vocal_cord and main_carina to anchor
    the trachea axis from observed trajectory direction (PCA-1, signed
    by first-to-last). Scale = traj_PCA1_span / (2.2 * trachea_len_mm)
    so the initial trajectory extends past the carina into bronchi.
  - bbox-scale baseline (no anatomical anchor).
The right init varies per video (verified empirically on 5V1, 13V2,
16V1: winners were roll90, roll180, bbox respectively). Without the
ensemble, single-init runs leave 5V1 anatomically broken (no TRACHEA
in visited). The bbox baseline is retained because it occasionally
beats all landmark candidates (16V1 case).

SSM-template anatomical segmentation (ssm_template/):
  vertex_segment.npy         (V,) int8, labels 0..4 (TRACHEA, RMB, LMB,
                             BI, DISTAL). Inherited from nearest centerline
                             node's segment.
  centerline_segment.npy     (M,) int8, same labels on centerline nodes.
  centerline_nodes.npy       (M,3) template centerline positions (mm).
  Computed once via compute_vertex_segment.py from the template subject's
  named-landmark graph (TRACHEA, CARINA, RMB_BIF, LMB_BIF, BI_BIF).


What NOT to do:

Don't pivot domains (no colonoscopy).
Don't add motion blur / bubbles / extra augmentations without explicit ask.
Don't rebuild the SSM correspondence pipeline (known TPS issue, accept as-is).
Don't propose Endo-2DTAM as a solution.


Default tone: ask before destructive changes, honest negative results acceptable, no overclaiming, no "I think it's fine" without evidence.