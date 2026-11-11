# Parametric order parameter λ(x) for StapleTIS

A parametric λ(x) was fitted to the nonparametric committor q_np of staple runs, to replace the order
parameter of the sampler. Priorities: (1) cost per frame, (2) smoothness along trajectories, (3) λ isosurfaces
close to isocommittors, (4) pointwise accuracy.

## Findings

**Recommendation.**
- For 2D systems: the MLP (M3) on the raw CVs, with the smallest network whose validation SD(Z_q) is within 10%
  of the best.
  - Z: [`exported/z_br_m3-fast.npz`](../examples/param_lambda/exported/z_br_m3-fast.npz) (64×64, β = 10).
    10 µs per frame, against 8.6 µs per toytis MD step.
  - cc: [`exported/cc_br_m3-fast.npz`](../examples/param_lambda/exported/cc_br_m3-fast.npz) (32 units, β = 1).
    6.7 µs per frame.
- M1 is not within tolerance of M3 on either 2D system, so it is not recommended there.
- In 1D every monotone λ ties, so M1 is recommended by the rule. 1D does not need a new λ.

**Z potential** (test paths; the OOD block agrees). Spread is against the exact committor, whose floor is 0.012.

| λ | spread vs exact | SD(Z_q) | turns / exact committor | Δr² / q_np | µs/frame |
|---|---|---|---|---|---|
| y (current) | 1.26 | 93 | 4.1 | 11.6 | 2.4 |
| M1 L1-logistic / spline GAM | 1.14 / 0.93 | 8.9 / 12.7 | 0.95 / 0.90 | 2.2 | 3.4 / 5.8 |
| M3, lowest validation SD(Z_q) (128×128×128) | 0.055 | 2.29 | 1.06 | 1.01 | 64 |
| **M3, smallest within 10% (64×64, β = 10)** | **0.043** | **2.17** | **1.03** | **1.02** | **10** |
| q_np (not exportable) | 0.049 | 2.26 | 1.05 | 1.00 | – |
| exact committor | 0.012 | 2.19 | 1 | 0.98 | – |

- **y is the problem.** Its interfaces cut across the 45° ridge (interface map).
  - It turns 4× as often as the committor used as order parameter.
  - Its stopped Z_q fails at every lag.
  - This is the order-parameter failure that also breaks the turn-to-turn MSM on this system
    (notes §6.5, errors up to 0.43).
- **M1 cannot represent the step across the ridge**: linear in (y, x), or additive in the screened features.
  The M2 screening does find the ridge normal u = (y + x)/√2 as by far the most important feature.
- **The MLP matches q_np.** Its isosurfaces are as close to the exact ones as q_np's, slightly closer for the
  smaller network, which smooths q_np's noise. Dynamics and turn rates are those of the committor.
- **The increment loss helps small networks.**
  - Validation SD(Z_q), β = 0 → 10: 32 units 26 → 9.8; 64×64 5.1 → 3.5.
  - At 128×128×128 it makes no difference.

**Curved channel** (test paths; floor 0.013). The same ordering with smaller gaps:

| λ | spread vs exact | SD(Z_q) |
|---|---|---|
| y | 0.50 | 8.0 |
| M1 | 0.08–0.09 | 3.8–3.9 |
| M3 (selected, and the smallest within 10%) | 0.047 | 3.3–3.5 |
| q_np | 0.019 | 3.5 |
| exact | 0.013 | 3.5 |

**Smoothness.**
- The step-size ratio of the fitted λ is 0.5–0.8; y's is 1.1–2.3. Sign-change ratios are ≈ 1 for all of them:
  lag-1 increments are thermal noise.
- The turn replay separates the λ's instead: in its own interface system, y turns far more often than the
  committor.

**Model selection: the specified rule is the weak point.**
- The validation SD(Z_q) separates gross failures (y, M1 on Z) but cannot rank good models by isosurface
  alignment: their values lie within its noise. On cc it picks an M3 checkpoint at epoch 3.
- Selecting by validation isosurface spread instead, computed from q_np alone (selection-sensitivity tables):
  - Z: test spread against the exact committor 0.037 instead of 0.055, with equal test SD(Z_q).
  - cc: not reliably better. In an earlier sweep it gave 0.014 instead of 0.047. After a refit that differs
    only by GPU rounding it gives 0.061: the validation spread of the cc runs is itself noisy at this level.
- Suggestion: SD(Z_q) within a tolerance → validation spread → cost; on cc no rule separates the M3 runs.

**What q_np can and cannot tell.**
- Below q ≈ 1e-3 the q_np labels are noise: logit q_np varies by 10–100× the slab width on exact isocommittor
  slabs. Alignment there can only be checked against an exact committor.
- On Z, the planned levels 0.015–0.41 lie on the thin ridge (77–463 of 2.1M test frames). Interfaces equally
  spaced in ln q bunch there.
- The NPNEq fits work in linear q. A q_np that resolves small q better (more paths, or a fit in log q) would
  help interface placement more than a better λ model.

**Stopped Z_q at long lags.** On the Z test and OOD blocks, every λ including the exact committor has
zqs_z ≈ 10.5 (test) and 7.7 (OOD), from lags ≥ 256. Paths in a block of MC cycles are correlated, so the standard
error is underestimated there. Compare zqs_z with the exact committor's on the same split; SD(Z_q) ranks fine.

**Cost.**
- Exported λ on one frame: 2–10 µs, except the 128×128×128 network (60 µs).
- TorchScript is slower per frame: 9.6 µs for the forward pass alone, against 8.9 µs for the whole numpy λ.
- In the staple extension the adapter's cache removes the second λ call per MD step.

**Not done / next.**
- M0 (no replayable nonparametric model exists).
- A q_np refit without the OOD block.
- A production staple run with the new λ: compare the crossing probability and rate with the exact rate, and
  the turn-to-turn MSM against the exact committor. Only a 4-cycle toytis smoke test was run.


## What was compared

**Data.** Staple runs of four toy systems: forward and mirrored pieces, rejection weights, and the frames that
`tis_data.prepare_data` gives each run's q_np fit. The loader checks the frames, pieces, order parameter and
A/B are identical to those of the fit.

| system | run | CVs | frames | q_np | q_np − exact, RMS (mid q) |
|---|---|---|---|---|---|
| Z (2D Brownian) | `sim_istarz_br_g10`, 1500 paths/ens. | y, x | 20.9M | existing 25000-iteration fit | 0.033 |
| curved channel (2D Brownian) | `test_cc_br` (2000 cycles) | y, x | 2.7M | new fit, 4000 iterations | 0.019 |
| 1D flat | `sim_istar0br10intf`, 1000 paths/ens. | x | 1.3M | new fit, 3000 iterations | 0.013 |
| 1D two bumps | `sim_istar2cosbr_2108`, 2000 paths/ens. | x | 3.3M | new fit, 3000 iterations | 0.005 |

The exact committors (discrete-chain grid solutions in `examples/figures/exact_committors/`) are used for
validation only: never to fit or select.

**Splits.** All splits are whole paths: a path's forward and mirrored pieces stay together. They are made by
MC cycle:
- Cycles before 40% are not used (burn-in).
- 40–85% is cut into blocks of 5% of the cycles. Blocks 1 and 5 are validation (checkpoint and model
  selection), blocks 3 and 7 are the test set, and the rest is training.
- Cycles ≥ 85% are the OOD test block.

The q_np labels of the OOD block come from fits that also saw those frames (accepted for now; to be checked
later with a q_np refit that leaves the block out).

**Models.** All models share `fit / predict / export`; the exported numpy forward pass is the one evaluated.
- M1: L1 logistic regression on soft labels q_np, and an additive P-spline GAM on logit q_np.
- M2: gradient-boosted trees, used for feature screening only (permutation importance on validation paths).
  The top-K features (K = 2, 4 for 2D) and the raw CVs are swept for M1 and M3. The 2D feature library holds
  monomials of y, x, u = (y + x)/√2 and v = (y − x)/√2.
- M3: SiLU MLP (32; 64×64; 128×128×128), weight decay 1e-4, 24 epochs. Loss: weighted MSE on logit q_np
  (×5 for 0.05 < q_np < 0.95) + β × Σ over lags 1, 2, 4, 8 of the weighted MSE of within-piece increments,
  with β ∈ {0, 0.1, 1, 10}. The checkpoint is selected by the SD over q of the stopped Z_q on the validation
  paths, never by validation MSE.

**Calibration and the exported λ** (`lambda_runtime.py`).
- z(x) is mapped to m = g(z) by isotonic regression of q_np on z, over 300 bins of equal width in z, with a
  monotone cubic through the isotonic blocks. Equal widths keep low and high q resolved; bins of equal weight
  would put everything below q ≈ 1e-3 into one bin.
- A/B come from the state CV s (y, or x in 1D), not from the model. λ = s outside (λ_A, λ_B). Inside,
  λ = λ_A + (λ_B − λ_A)·log1p(h/ε)/log1p(1/ε) with h = (1 − a(s))(1 − b(s))·m + b(s).
  - a and b are smoothstep ramps of width 5% of λ_B − λ_A, a fixed fraction of the state-to-state range,
    not tied to any interface.
  - So s < λ_A ⇔ λ < λ_A and s > λ_B ⇔ λ > λ_B exactly (also with ≤ and ≥), and λ is continuous at both
    boundaries. Outside the ramps h = m.
  - The log scale (ε = 1e-3) only sets where interfaces land: `interface_for_q`. Crossings and turns
    depend only on the order of λ and the interfaces.
- The OOD flag combines a per-CV training range check (counted per frame by the adapter) with a Mahalanobis
  distance on the standardized features (`LambdaModel.ood`).

**Metrics** (path weights throughout; Z_q as in the notebooks: `w_traj` = path weight, `g_traj` = path):
- *Isosurface spread*: 12 planned interfaces equally spaced in ln q (Z, cc: 1e-4 … 0.95; 1D: 1e-2 … 0.95).
  - For each, the weighted variance of logit q_np (and of logit q_exact) over the frames with
    |logit g(λ) − logit q_k| < 0.2. A λ whose isosurfaces are isocommittors scores ≈ 0.2²/3 = 0.013.
  - The table mean is over the levels where the exact committor's slab holds ≥ 500 frames and logit q_np on it
    varies by less than 10× that baseline, i.e. where q_np resolves the committor; the levels used are listed.
  - Below about q = 1e-3 the q_np labels are noise. There, alignment can only be judged against the exact
    committor (per-level figure).
- *Smoothness*:
  - step ratio: mean squared lag-1 increment of logit g(λ) over that of logit q_np;
  - RMSE of the increments against q_np's at lags 1–8;
  - sign-change ratio of consecutive lag-1 increments;
  - turns and interface crossings per 1000 frames under the sampler's own turn rule (`pgmoves.turn_detected`,
    both directions, restarted after each turn), relative to q_np and to the exact committor used as the
    order parameter.
- *Dynamical*: the stopped Z_q of g(λ), max |Z/s.e.| (zqs_z) and its SD over q, at lags 1 … 4096; and the
  stopped Δr² relative to q_np's. N_AB is not defined for path pieces, so Δr²/2N_AB is reported as this ratio.
- *Static*: Spearman correlation with q_np, and the RMSE of logit g(λ) against logit q_np (and logit q_exact)
  after averaging over a 40×40 grid of CV-space bins.
- *Cost*: µs per frame of the exported model: one-frame calls (as the sampler makes them) and batch.

## Using it in the sampler

```python
import sys; sys.path.insert(0, 'examples/param_lambda')
import toytis_adapter
toytis_adapter.install('examples/param_lambda/exported/z_br_m3.npz')    # before Simulation(settings)
settings['interfaces'] = toytis_adapter.interfaces_for_q('examples/param_lambda/exported/z_br_m3.npz', q_levels)
settings['v_ord'] = False
```

`install` replaces `ensemble.OrderX` and `ensemble.OrderParameter` and wraps `ensemble.kick_star`; no sampler
file changes.
- calculate(ph) returns `[λ, y, x]` as numpy floats, as OrderX does. toytis' `validate_staple` relies on that,
  and order.txt keeps the positions.
- The wrapper places each initial kick point on the learned interface instead of at y = interface value.
- The adapter caches the extension start point, which `ext_propagate` asks for again at every step.
- A short fresh staple run of the Z potential with the adapter is a test (`tests/test_param_lambda.py`).

Runs analysed with `tis_data` read λ as the order parameter. The states there must still come from y:
order.txt rounds λ to 6 decimals, and `tis_data` tests `lam <= lambda_A`.

## Caveats

- The OOD block's q_np labels come from fits that saw those frames (see Splits).
- q_np level errors are inherited through the calibration: Z plateau +0.03; 1D flat up to +0.017 in the
  middle, at zqs_z 1.7. That fit is a statistical level error the stopped Z_q cannot see. The exact columns
  show it.
- The turn replay can only see turns before the point where the y-based rule of the actual run stopped the
  piece.
- zqs_z is a maximum over 200 bins × 7 lags: it grows with the number of frames. Compare it with the exact
  committor's value on the same split, not with 2.
- In 1D every strictly monotone λ has the committor's isosurfaces, so all models tie there. The 1D cases check
  the pipeline: monotone fits, calibration, A/B, export.
- The curved-channel run is the 2000-cycle test run (the production run is not on this machine).
