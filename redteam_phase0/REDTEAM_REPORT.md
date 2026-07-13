# Phase-0 Red-Team Validation — proposed near-light photometric SoR airway estimator

**Estimator under test:** fit a surface-of-revolution airway cross-section `r(s)` (+ albedo, light) to
image intensities under a near-light Lambertian model `B = A·(n·l)/d²`, using **fixed COLMAP/Barbour
poses**. Output: scale-free CSA/DCE. Goal of this phase: **falsify it** before implementation (assume
Reviewer #2 is right). Simulation code + figures + JSON in this folder. Sanity: on the ideal circular
phantom the estimator recovers r to 0% bias, full rank (2/2), condition 44 (2.5° parallax).

## Deliverable — PASS / FAIL

| test | result | key number |
|---|---|---|
| **1 — model mismatch** | **PASS** | worst CSA bias **7.6%** (2:1 ellipse) < 10% |
| **2 — spatially-varying albedo** | **FAIL** (at spec) | **44.6%** CSA bias (vascular stripes) at the specified 5–10 albedo DOF |
| **3 — pose uncertainty** | **FAIL** | CSA **std 14.7% + bias 21%** at realistic COLMAP noise |
| **4 — novelty** | **FAIL (not novel)** | a combination of existing near-light photometric + SoR + fixed poses |

**Overall: DOES NOT CLEAR PHASE-0.** Two quantitative failures (albedo, pose) and no novelty.

---

### Test 1 — model mismatch — **PASS**
Circular-SoR estimator vs elliptical / lobed / asymmetric-scar / off-center / non-circular
cross-sections (constant albedo, clean poses), 2.5° parallax:

| phantom | CSA bias | DCE bias | J rank | cond |
|---|---|---|---|---|
| ellipse b/a=0.7 | −2.5% | −1.3% | 2/2 | 40 |
| ellipse b/a=0.5 | **−7.6%** | −3.9% | 2/2 | 36 |
| lobed 3×25% | −3.9% | −2.0% | 2/2 | 45 |
| asym scar 35% | −3.4% | −1.7% | 2/2 | 42 |
| non-circular | −3.4% | −1.7% | 2/2 | 44 |
| off-center 0.4R | −3.4% | −1.7% | 2/2 | 48 |

CSA (area) is robust to *geometry* mismatch (worst 7.6% < 10%); full rank, well-conditioned. **PASS.**

### Test 2 — spatially-varying albedo — **FAIL (at the specified DOF)**
Circular geometry, non-constant albedo, albedo free (Fourier, **9 DOF = the specified 5–10**), 2.5°:

| albedo | CSA bias | J rank / params | geom↔albedo null-mix | coupled |
|---|---|---|---|---|
| gradient | 0.0% | 18/18 | 0.0 | no |
| **stripes (vascular)** | **44.6%** | 18/18 | 0.0 | no |
| low-freq texture | 0.1% | 18/18 | 0.0 | no |
| high-freq texture | 0.0% | 18/18 | 0.0 | no |

**Two findings:** (i) there is **no fundamental geometry–albedo coupling** — the Jacobian stays
full-rank even at 34 params (near-light + known poses breaks the classic shape-from-shading
ambiguity; null-space mixing 0.0). (ii) **But** at the specified 5–10 albedo DOF, realistic
**vascular stripes** cause a **44.6% CSA bias**: the smooth albedo basis cannot represent the sharp
texture, and the residual **aliases into geometry**. A DOF sweep confirms the dilemma is a *modeling*
one — 13+ albedo DOF drives the bias to ~0% and stays full-rank — but the correct DOF is **unknown a
priori on real vascular texture**, so the bias is unverifiable/uncontrolled on real data. **FAIL at
spec.**

### Test 3 — pose uncertainty — **FAIL**
Ray-based model (translation, rotation, focal all act via the pixel→wall correspondence), 100
Monte-Carlo runs, realistic COLMAP-ish noise (translation ~1% of depth, rotation 0.3°, focal 1%):

| source | CSA std | CSA bias | p5 / p95 |
|---|---|---|---|
| **combined** | **14.7%** | **+21.0%** | +0.9% / +46.8% |
| translation only | 10.3% | +15.4% | +0.3% / +30.5% |
| rotation only | 9.1% | −4.1% | −21.3% / +8.7% |
| focal only | 0.0% | 0.0% | — |

**Sensitivity ranking: translation > rotation ≫ focal.** Pose noise not only adds ~15% variance but
**systematically biases CSA +21%** (the `1/d²` nonlinearity rectifies symmetric pose noise into a
positive area bias — Jensen). At our data's low parallax (2.5° cone), COLMAP poses are *more*
uncertain than tested, so this is optimistic. **FAIL.**

### Test 4 — novelty — **NOT fundamentally new**
| method | optimization variables | objective | vs ours |
|---|---|---|---|
| NFL-BA | poses + geometry + near-field light | near-light photometric BA | ours = NFL-BA with **poses fixed** + SoR prior (special case) |
| LightNeuS (Zaragoza) | neural SDF + light | near-light photometric + SDF reg | ours replaces the SDF with an explicit SoR — a **strict specialization, less general** |
| Zaragoza near-light line | poses/surface + light | near-light photometric consistency | ours sits **inside** this line |
| Sengupta (SfS) *(ref uncertain)* | depth/normals + albedo/light | shape-from-shading | ours = tubular-constrained variant of the same cue |
| Barbour (our COLMAP recipe) | (feature SfM/MVS) | geometric parallax | different cue; ours **reuses its poses** and adds shading |
| BREA-Depth *(ref uncertain)* | network weights | learned depth | different paradigm (learned vs per-scene optimization) |

The **variables, objective and unknowns are not distinct** from the near-light photometric endoscopy
line. The only new element is the **specific combination** (SoR + fixed poses + clinical CSA readout),
which is engineering, not a new inverse problem. Relative to LightNeuS it is **less general**.
_(Sengupta / BREA-Depth exact formulations are uncertain — characterized by class, flagged.)_

---

## What must change before implementation
1. **Pose (Test 3) — blocking.** The estimator **must** propagate pose uncertainty or **jointly
   refine poses** (which makes it NFL-BA, not a fixed-pose fit), and must debias the `1/d²`
   nonlinearity. Fundamentally, on **low-parallax** footage (our data) COLMAP poses are too uncertain
   for a ±10% CSA — the same parallax limit that constrains the geometric baseline. No photometric
   trick removes it.
2. **Albedo (Test 2) — blocking on real tissue.** The albedo basis must be validated against the
   *actual* vascular texture; too few DOF biases CSA by tens of %, and the right DOF is not knowable a
   priori on real data. Requires a data-driven / adaptive albedo model with a bias-verification
   procedure — non-trivial.
3. **Novelty (Test 4).** Reframe honestly: this is an **application/specialization** of existing
   near-light photometric reconstruction (NFL-BA / LightNeuS / Zaragoza), not a new estimator. Do not
   claim a new method; if pursued, build **on** LightNeuS/NFL-BA rather than reinventing a weaker
   fixed-pose SoR variant.

**Consistency with prior work in this project:** the empirical photometric pivot already showed the
same-scale narrowing is not recoverable from photometry on real 2_V2 (a scale/parallax limit); this
Phase-0 simulation independently reaches the same wall via pose sensitivity (Test 3) plus an albedo
hazard (Test 2), and finds no novelty (Test 4). **Recommendation: do not implement as proposed.**
