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


What NOT to do:

Don't pivot domains (no colonoscopy).
Don't add motion blur / bubbles / extra augmentations without explicit ask.
Don't rebuild the SSM correspondence pipeline (known TPS issue, accept as-is).
Don't propose Endo-2DTAM as a solution.


Default tone: ask before destructive changes, honest negative results acceptable, no overclaiming, no "I think it's fine" without evidence.