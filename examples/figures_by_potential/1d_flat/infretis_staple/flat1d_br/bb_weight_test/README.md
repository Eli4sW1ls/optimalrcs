# flat1d_br: why both weightings were biased (notes section 15.1)

| file | what |
|---|---|
| `fit_runs.py`, `fit_runs.log`, `older_runs_profiles.png` | the notebook's fit (A/B traps, stopped ends, mirrored pieces, path_f) on the six older 20 000-cycle runs of `_shared/param_lambda/op_loop/staple_weights`: the noise level (all within 0.018 of exact) |
| `older_runs_forward_profiles.png` | the same, forward pieces only (`fit_runs3.py ...:fwd`) |
| `fit_runs3.py`, `fit_halves.log`, `halves_profiles.png` | this run split in rows 30-65 % / 65-100 %, with and without mirrored pieces: both halves biased upward |
| `fit_runs4.py`, `fit_bb.log`, `bb_reweight_profiles.png` | the B-to-B pairs of the last ensemble reweighted x2 / x0.5: x0.5 removes the bias |
| `drift_test.py`, `drift_test.log` | a fit-free check E[q_exact(piece end) - q_exact(piece start)] = 0 (too little power for path_f; shows the per-interface bias of path_f / path_w) |

Spec format of the fit scripts: `RUN:WEIGHTING[:fwd][:bi=F][:df=FILE][:ppe=N][:bb=FACTOR]`. They were run from the
session scratchpad; the output folder is their first argument. The loader now applies the x0.5 itself
(`--bb-weight-fix auto`).
