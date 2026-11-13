# Parametric order parameter λ(x) for StapleTIS

A parametric λ(x) was fitted to the nonparametric committor q_np of staple runs, to replace the order
parameter of the sampler. Priorities: (1) cost per frame, (2) smoothness along trajectories, (3) λ isosurfaces
close to isocommittors, (4) pointwise accuracy.

The terms, data and metrics are explained in [How to read this report](#how-to-read-this-report) below the
findings.

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

## How to read this report

### Symbols

| symbol | meaning |
|---|---|
| x | the CVs of one frame (y, x in 2D; x in 1D) |
| q_np(x) | the nonparametric committor from the NPNEq fit of the run. It is the training label, defined per frame only, not as a function you can evaluate at a new x |
| q_exact(x) | the exact committor from a grid solution; used for validation only, never for fitting or selection |
| logit q | ln(q / (1 − q)). For small q it is ≈ ln q, so logit units are relative (factor) errors in q: 0.1 logit ≈ 10% |
| z(x) | the raw output of a fitted model (M1 or M3), trained to approximate logit q_np. An unbounded score, not exactly calibrated |
| g | the calibration map: a monotone increasing function fitted after training (isotonic regression of q_np on z, smoothed by a monotone cubic). It does not change the ordering of frames, so it does not change any isosurface; it only says which value of z belongs to a given committor level |
| m(x) = g(z(x)) | the model's committor estimate in [0, 1]. All metrics are computed on m (or on z, which orders frames the same way). Earlier versions of this text wrote g(λ) for m |
| λ(x) | the exported order parameter: m on a log scale between λ_A and λ_B, with short ramps at A and B (see Calibration). Away from the ramps λ and m have the same isosurfaces; the interface of committor level q_k is λ_k = `interface_for_q(q_k)` |
| s | the state CV (y in 2D): A is s ≤ λ_A, B is s ≥ λ_B, as in the sampler |

The stopped Z_q code (`optimalrcs/metrics_tis.py`) takes its inputs as per-frame arrays:
- `r_traj`: the candidate's m, set to 0 in A and 1 in B.
- `b_traj`: 1 in A or B.
- `i_traj`: the piece a frame belongs to. Lag windows never run from one piece into another.
- `w_traj`: the Monte Carlo weight of the frame's path.
- `g_traj`: the path a frame belongs to. A path's forward and mirrored pieces share frames, so for the standard
  error they count as one independent unit.

### Data: pieces and weights

- Each staple path is cut at its first crossing of the ensemble's interface M (the interface below the ensemble)
  and gives two pieces:
  - a **forward piece**: the frames after M up to the end of the path;
  - a **mirrored piece**: the frames from the start of the path up to M, read backwards. Brownian dynamics at
    equilibrium are time-reversible, so a reversed segment is a valid trajectory.
- Every frame carries its path's **rejection weight**: the number of MC cycles the path stayed the current path
  (tistools `get_weights`). It is the same for both pieces.
- These are exactly the frames, pieces and weights of the q_np fit (`tis_data.prepare_data --reversed`); the
  loader checks that frame by frame.
- **Every metric below uses both piece kinds and these weights.** The turn replay scans each piece in its own
  time order, the mirrored ones backwards.

### Test paths and the OOD block

Both are whole paths that no model was trained or selected on. They differ in *when* in the MC chain they were
sampled:
- **Test**: two blocks of 5% of the MC cycles (blocks 3 and 7) between 40% and 85% of the run, interleaved with
  the training blocks. They come from the same stage of sampling as the training data: an in-distribution test.
- **OOD** (out of distribution): every path from the last 15% of the cycles, all later than any training path.
  If the MC chain drifts (paths through regions it had not reached yet), λ has to extrapolate there. Similar
  numbers on test and OOD mean λ does not depend on the sampling stage it was trained on.
- One caveat: the q_np labels of the OOD frames come from fits that also saw those frames. The λ models never
  did, and the exact-committor columns are unaffected.

### Metrics, and how to read the numbers

Table columns, in order:

| metric | what it is | how to read it |
|---|---|---|
| spread (exact), spread (q_np) | **Isosurface alignment.** For each of the 12 planned interfaces q_k, take the frames on that λ isosurface (the slab \|logit m − logit q_k\| < 0.2) and compute the weighted variance of logit q_exact (or logit q_np) over them. The table gives the mean over levels | If λ's isosurfaces were isocommittors, the slab would hold only frames with q within ±0.2 logit of q_k: variance 0.2²/3 = 0.013, the **floor** (the exact committor scores 0.012–0.013). Above the floor, √(spread − 0.013) is the RMS misalignment in logit units: 0.043 (Z, M3) → 0.17, so the committor along one interface varies by about ±17%. 1.26 (Z, y) → 1.1, about a factor 3. Against q_np the value also contains q_np's own noise |
| zqs_z | stopped Z_q profile divided by its standard error; the maximum of \|Z/s.e.\| over q (first 5 bins skipped) and over lags 1 … 4096 | ≈ 2 or less for the committor *if paths are independent*. Here paths from one MC block are correlated, so even the exact committor scores 7–11 on Z. Compare with the exact committor's value on the same split, not with 2 |
| SD(Z_q) | the standard deviation over q (200 bins on [0, 1]) of the stopped Z_q profile itself, maximum over lags | 0 for the committor up to sampling noise; the exact committor's value is the noise floor (Z ≈ 2.2, cc ≈ 3.5). Several times that = dynamically inconsistent (y on Z: 93). This is the selection metric (on the validation paths, lags 1 … 256) |
| step ratio | Σ w (Δ logit m)² / Σ w (Δ logit q_np)² over lag-1 steps | Jitter of λ along trajectories. < 1: smoother than q_np (q_np is noisy; the exact committor has 0.64 on Z, 0.89 on cc). Clearly > 1: λ jitters |
| Δ RMSE lag 1 / 8 | RMS of (Δ logit m − Δ logit q_np) over steps of 1 and 8 frames | How closely λ's increments follow q_np's, in logit units; includes q_np's noise. Smaller is better |
| sign-change ratio | fraction of consecutive lag-1 increments with opposite sign, divided by the same fraction for q_np | ≈ 1 for every λ here: the lag-1 motion is thermal noise. It does not separate the candidates |
| turns / q_np, turns / exact | turns per 1000 frames when the sampler's turn rule (`pgmoves.turn_detected`) is replayed on the pieces with λ and the planned interfaces, divided by the same count with q_np or the exact committor as order parameter | 1 = as many turns as the ideal order parameter. > 1: spurious turns, where λ re-crosses interfaces the committor would not (y on Z: 4.1). < 1: λ turns less, because it is smoother than the reference |
| crossings / q_np | interface crossings per 1000 frames, relative to q_np | 1 = the same crossing rate; > 1 = extra back-and-forth crossings |
| Δr² / q_np (lag 1) | stopped Δr² of m, Σ w (r(min(t + lag, τ_t)) − r(t))² / lag, divided by q_np's | The committor minimizes Δr² (the NPNEq criterion). ≈ 1 = as good as q_np; > 1 = worse (y on Z: 11.6). It can be slightly below 1 because q_np's noise adds to its own Δr² |
| Spearman | rank correlation of z with q_np over all frames | Whether λ orders the frames like q_np. 0.95–1.00 for every λ here, y included (0.994 on Z), so it hardly separates candidates |
| logit RMSE (q_np / exact) | bin the frames on a 40 × 40 grid in CV space (quantile bins), average logit m and logit q_ref per bin, then the weighted RMS of the difference | Pointwise accuracy in logit units, as a function of position: 0.04 ≈ 4% error in q. Against q_exact every candidate scores at least 0.12–0.13, the exact committor included (Z 0.12, cc 0.13). Every candidate is calibrated on q_np (g is fitted to q_np), so q_np's level errors, mostly at small q, enter m. Compare with the exact committor's row |
| µs/frame (one / batch) | the exported model: one frame per call, as in the sampler, and a batch of 200k frames | Cost. A toytis MD step takes 8.6 µs |
| training s | wall time of the fit, including the checkpoint evaluations | |

### The stopped Z_q is the library's

- It is computed with `optimalrcs/metrics_tis.py`: `stop_index`, `comp_zq_stopped` (the profile Z_q) and
  `comp_zq_stopped_zscore` (Z / s.e.). That is the numpy twin of `optimalrcs-cupy/metrics_tis.py`, whose
  `max_z_zq_stopped` and `max_sd_zq_stopped` are the zqs_z and sdzqs that the NPNEq fits print.
- What it does: a lag window [t, t + Δt] that would run past the end of a piece (in A/B, at an L/R exit or a
  completed turn) is stopped at that end instead of dropped. Dropping those windows biases the ordinary Z_q of
  path pieces, even for the exact committor.
- The inputs are as in the notebooks: `w_traj` = path weight, `g_traj` = path, `i_traj` = piece, 200 bins, the
  first 5 bins skipped for the maximum.
- What differs from the numbers a fit prints:
  - **Lags:** this report uses 1, 4, 16, …, 4096; checkpoint selection 1, 4, 16, 64, 256; the library default
    is 1, 2, 4, …, 32768.
  - **r** is the candidate's calibrated m.
  - **Frames:** one split only (test, OOD or validation), not all frames.

  So the q_np row's zqs_z here is not the value its fit printed.


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

The exact committors (discrete-chain grid solutions in the `exact/` folder of each potential in `examples/figures_by_potential/`) are used for
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

**Metrics**: defined in [How to read this report](#how-to-read-this-report). Specifics:
- The planned interfaces are 12 levels equally spaced in ln q (Z, cc: 1e-4 … 0.95; 1D: 1e-2 … 0.95).
- The mean spread is taken over the levels where the exact committor's slab holds at least 50 frames and logit
  q_np on it varies by less than 10× the floor, i.e. where q_np resolves the committor. The levels used are
  listed under each table.
- Below about q = 1e-3 the q_np labels are noise. There, alignment can only be judged against the exact
  committor (per-level figure).
- N_AB is not defined for path pieces, so Δr²/2N_AB is reported as the ratio to q_np.

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
