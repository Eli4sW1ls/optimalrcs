# NPNEq on path-sampling data: what was changed and why

Notes on adapting the NPNEq committor optimization (`CommittorNE`) and its validation
metrics to (RE)TIS, REPPTIS and staple (iSTAR) path data. Covers the reasoning, every
modification, the evidence for it, and where it lives in the code.

---

## 1. Summary

NPNEq fits the committor from short trajectories. It does not need equilibrium data,
but it does need **every step it learns from to follow the real dynamics**. Path
sampling breaks that in two ways, and the standard validation metrics break in a third:

| Problem | Fix | Section |
|---|---|---|
| Each ensemble keeps only paths that cross its interface M, so frames **before** that crossing are conditioned on their future | Cut every path at its first M crossing (t\*) | [3](#3-cut-at-the-first-m-crossing-t) |
| Accepted paths are not a fair sample until rejections are counted | Monte Carlo path weights (repeat counts) in the fit and the metrics | [4](#4-monte-carlo-path-weights) |
| Segments end where the sampler stops them (L/R exit, staple turn, A/B), so Z_q and obs-vs-pred silently drop or select frames by their future | **Stopped** Z_q and stopped obs-vs-pred, with a standard error | [5](#5-stopped-z_q), [6](#6-standard-error-and-the-convergence-criterion), [7](#7-stopped-observed-vs-predicted) |
| The library's Z_q drops windows that run into A/B at a segment end | Treat A/B ends as traps (the library's existing padding) | [8](#8-ab-ends-as-traps-padding-and-the-original-z_q-at-short-lags) |
| The cut throws away ~40% of the frames | Optional: reuse them read backwards (time reversal) | [9](#9-optional-time-reversed-pieces) |
| Underdamped (low-friction) dynamics: the committor depends on the velocity, which `order.txt` may not contain | Record velocities (`v_ord`); otherwise `--fd-velocity` and a fit at lag ~4 | [10](#10-underdamped-low-friction-dynamics) |

**Convergence criterion for path data:** `zqs_z` (worst |stopped Z_q / standard error|
over q and all lag times) ≲ 2, together with a levelled-off `dr2` and a small `|dx|`.

---

## 2. Background: what NPNEq actually needs

NPNEq solves the Galerkin form of the committor's defining property,

$$\sum_t w_t\, f_k(x_t)\,\big[q(x_{t+1}) - q(x_t)\big] = 0 \quad \text{for all basis functions } f_k ,$$

i.e. $\mathbb E[q(x_{t+1}) \mid x_t] = q(x_t)$: the committor is a **fair bet**, on average it
does not drift.

- This holds **for any distribution of the starting frames** $x_t$, so non-equilibrium
  data, mixed ensembles and arbitrary amounts of data per region are fine.
- It requires that, **given $x_t$, the next frame follows the true dynamics**. Any
  selection that depends on what happens *after* $t$ tilts the steps and biases the fit.

The validation criterion Z_q tests the same property at longer lags: summing
$q(x_{t+\Delta t}) - q(x_t)$ over frames grouped by $q(x_t)$ must give a flat profile, for
every $\Delta t$.

**General rule used throughout:** a condition on a path is harmless if it only involves
the past of a frame, or if it is a *stopping rule* (where the path ends is decided by the
path up to that point). It is harmful if it involves the frame's future.

| Ensemble condition | Involves | Harmful? |
|---|---|---|
| Starts at A / L / R | past | no |
| Ends at the first A/B, L/R exit or completed turn | stopping rule | no |
| Crosses M | future of frames before the crossing | **yes** → cut (§3) |
| Max path length rejection (FTL) | future | mildly (few paths) |
| Path weight depending on the whole path (e.g. length) | future | **yes** → weights (§4) |

How a path was *generated* (shooting point, backward propagation) does not matter: the
Monte Carlo acceptance makes the chain sample the path ensemble regardless. What matters
is the ensemble definition.

---

## 3. Cut at the first M crossing (t\*)

Every ensemble requires its paths to cross an interface M. Before that crossing, each step
is conditioned on reaching M later: the dynamics is tilted by $h(x_{t+1})/h(x_t)$ with $h$
the probability to reach M first. Those frames estimate $q/h$, not $q$.

**Fix:** use each path only from its first frame past M onward. After that frame the
remaining conditions are stopping rules, so the steps are unbiased.

| Folder | Ensemble | M |
|---|---|---|
| 000 | [0⁻] | λ₀ (skipped unless `--include-zero-minus`) |
| 001 | [0⁺⁻] | λ₀ |
| 00k (k ≥ 2) | [(k−1)±] / [(k−1)\*] | λ_{k−1} |

Details:

- **Staple paths:** the search starts at the **core** (the part inside [L, R]), whose start
  and end are recorded in `staridx` in `pathensemble.txt`. Backward extensions can cross M
  before the core starts, so "first crossing of the whole path" would be wrong.
  `staridx = (0, 0)` means the whole path is core (REPPTIS, RETIS, [0⁻]).
- t\* is the first core frame on the other side of M from the frame the core starts from.
- Coverage is not lost: a region below M_i is covered by the lower ensembles' post-crossing
  frames. TIS's enhanced sampling near the barrier is kept, because all kept steps count
  equally.
- [0⁺⁻] RML paths cross M = λ₀ only when entering A at their last frame, so nothing is
  left after the cut (expected).

**Evidence** (`sim_istar0br10intf`, flat 1D overdamped, exact committor linear):
without the cut the fitted q is off by up to 0.22 (RMS 0.12–0.13); with the cut,
0.005–0.024 depending on the weights.

---

## 4. Monte Carlo path weights

Keeping each accepted path once is not a fair sample: in the MC average a rejected move
counts the **old** path again. Without that, paths the chain leaves easily are
over-represented. How easily it leaves depends on the whole path (length, how much time it
spends near M), so this is again a condition on the future of frames, also after t\*.

**Fix:** weight every step of a path by its repeat count (cycles until the next accepted
path), via tistools `get_weights` on `pathensemble.txt`. Weights are scaled to average 1
so `--gamma` keeps its meaning. The initial `ld` path is skipped.

Weights that do **not** matter for the committor (NPNEq tolerates them): per-ensemble
normalization, crossing probabilities, equilibrium reweighting, random subsampling of paths.

**Staple high-acceptance factor:** tistools `get_weights_staple` additionally doubles shot
LML/RMR paths. Tested as a per-path weight it **fails** (below), so the default is
rejection counts only (`--weights rejections`).

**Evidence** (exact committor, stopped Z_q z-score, same path subset):

| Weights | max \|z\| |
|---|---|
| none | 9.1 (fails near A, where shooting moves are rejected most) |
| rejection counts | **2.0** |
| staple (rejections × HA factor) | 5.5 |

Fits agree: RMS vs exact 0.013/0.005 (rejections), 0.016/0.011 (none), 0.024/0.014 (staple).

---

## 5. Stopped Z_q

**Problem.** The standard Z_q drops every window $[t, t+\Delta t]$ that runs past the end
of its segment. That is harmless for an MD run cut at an arbitrary time, but path segments
end *because of what they did*: they exit through L/R, complete a turn, or reach A/B. The
dropped windows are exactly those of frames about to exit, so the remaining windows
drift away from the exits on average. The standard Z_q is non-flat and lag-dependent
**even for the exact committor**. obs-vs-pred, ROC, cross-entropy, mse and AUC have the
same problem: they only count frames whose segment reaches A or B.

**Fix: stop the window instead of dropping it.** With $\tau_t$ the first frame $\ge t$ that
is in A/B or ends the segment,

$$Z(x, \Delta t) = \frac{1}{\Delta t}\sum_{t:\; q(x_t) \le x,\; x_t \notin A\cup B} w_t\,\Big[q\big(x_{\min(t+\Delta t,\ \tau_t)}\big) - q(x_t)\Big].$$

Picture it as **padding every segment after its end with copies of its last frame**: after
the end, q stays where it was. For the committor, $Z \approx 0$ (flat) at every $\Delta t$.

**Why this is not an assumption that the system stops moving.** Split the real change:

$$q(x_{t+\Delta t}) - q(x_t) = \underbrace{\big[q(x_\tau) - q(x_t)\big]}_{\text{in the data}} + \underbrace{\big[q(x_{t+\Delta t}) - q(x_\tau)\big]}_{\text{not observed}}.$$

The committor is a fair bet from any point on, including $\tau$, so the unobserved part has
zero mean whatever the system does after $\tau$ (optional stopping theorem). Leaving it out
does not shift the average; it only removes noise that was never measured. The library
already does the same at A/B (q counted as 0/1 after entering); L/R and turn ends just use
$q(x_\tau)$ instead of 0/1.

Requirements: the segment end must be a stopping rule (true for L/R exit, turn, A/B), and
the steps up to it must be unbiased (§3, §4).

What is lost: at lag $\Delta t$ a window hitting $\tau$ early only covers $\tau - t$
frames, so long-lag memory effects are probed only by segments long enough to have them.
That is a limit of the data, not a bias.

**Why not just use short lags of the standard Z_q?** At lag $\Delta t$ every segment loses
its last $\Delta t$ frames, so the bias grows with $\Delta t$ / segment length. On
`sim_istar0br10intf` (median segment 59 frames) it is invisible up to $\Delta t \approx 4$–8
and clear from 16 on (z = 6.3 at 16, 10.1 at 32). Short lags are also the weakest test (the
fit enforces $\Delta t = 1$), and the safe cutoff changes with every data set. Up to
$\Delta t \approx 4$ the two versions give identical numbers: the stopped Z_q is the same
quantity, corrected where the old one breaks.

---

## 6. Standard error and the convergence criterion

Z sums over ~10⁶ frames, so even the exact committor gives a wiggly Z. To judge it:

- Segments are independent, so $\mathrm{Var}\,Z(x) = \sum_{\text{segments}} S_{\text{seg}}(x)^2$,
  with $S_{\text{seg}}(x)$ the segment's own contribution up to $x$.
- **z-score** = Z / standard error; it stays within about ±2 for an RC consistent with the
  committor, at every q and lag.
- Computed sparsely (sort contributions by (segment, bin), running sums), so memory grows
  with the number of frames, not segments × bins.
- The first 5 bins next to A are left out of the maximum (few frames, unreliable error).

New metrics (usable in `metrics_print`):

| Name | Printed as | Meaning |
|---|---|---|
| `max_z_zq_stopped` | `zqs_z` | worst \|z\| over q and lags — **the convergence criterion** (≲ 2) |
| `max_sd_zq_stopped` | `sdzqs` | worst standard deviation of stopped Z over q |

`fit_transform(save_min_metric='max_z_zq_stopped')` saves the RC with the lowest `zqs_z` in
`r_traj_min_sd_zq`.

**Evidence** (synthetic: ±1 random walk on a lattice, segments ending at random window
exits, exact committor x/N, 4 seeds): exact q `zqs_z` 1.1–1.7; wrong RC (q²) ≈ 10; the
standard drop-the-window Z_q on the **exact** q 9–13.

---

## 7. Stopped observed-vs-predicted

The usual plot needs an A/B outcome per frame, which most path segments do not have. The
stopped version uses the value where the window stops as the outcome: 0 or 1 if the
segment reaches A or B, else q at the segment end. Bin by $q(x_t)$; for the committor the
mean outcome equals the prediction. When every segment ends in A or B, it reduces to the
usual plot. It is much less sensitive than the z-scored Z_q (both the exact and the fitted
q sat on the diagonal within 0.011), so it is a sanity check, not the criterion.

---

## 8. A/B ends as traps ("padding") and the original Z_q at short lags

The library's own `comp_zq` also drops windows that **reach A or B** near the end of a
segment: path segments end at their first A/B frame, and the A/B term requires the
segment to continue for $\Delta t$ more frames (`delta_i_to_end >= dt`). This produced
steps at q ≈ 0 and q ≈ 1 even at $\Delta t = 2$.

**Fix:** the scripts call the library's existing
`set_fixed_traj_length_trap(boundary0 | boundary1, 2**16)`, which treats segments ending
in A/B as staying there (padding their distance-to-end). This also makes the printed `dr2`
count transitions into A/B.

With that, the original Z_q is plotted at short lags (Δt = 1, 2, 4, 8) next to the stopped
one, each lag labelled with the share of windows it drops (`dropped_fraction`). On
`sim_istar0br10intf` it is usable up to Δt ≈ 2; beyond that the L/R/turn-end bias appears
(−5 at Δt = 4, −18 at Δt = 8).

---

## 9. Optional: time-reversed pieces

The cut discards the frames before t\*. For **time-reversible dynamics at equilibrium**
they can be recovered by reading the path backwards: in reversed time the M condition is
met at the **last** M crossing of the core, and the path start is a stopping rule (window
exit, completed turn of the backward extension, or A). So each path gives

```
forward : [first M crossing of the core → end]     read forward
reversed: [start → last M crossing of the core]     read backwards
```

- Both pieces get the path's weight. They share frames, so they form one **independence
  group** (`group_traj`) for the standard error and the random-halves split (without
  grouping the z-scores came out ~10% too large).
- Velocity columns (`v_ord`) are negated in reversed pieces; other CVs must be even under
  time reversal.
- Not valid for driven / non-equilibrium dynamics.

**Evidence** (exact committor and fits, 4 subsets each):

| Run | max \|z\| exact q: forward → +reversed | fit RMS vs exact (mean) |
|---|---|---|
| REPPTIS `sim_repptis0br_ys` | no worse (fwd 1.7–3.5, rev 1.6–2.5) | 0.0128 → **0.0064** |
| staple `sim_istar0br10intf` | 2.1 → 2.9 (size-matched forward-only control: 2.1) | 0.0086 → 0.0066 (mixed) |

So: **use `--reversed` for REPPTIS; keep it off for staple** until understood. A likely
cause is that the staple ensemble is not symmetric under time reversal (LML/RMR are
extended only at their start), so reversed pieces may be mis-weighted.

---

## 10. Underdamped (low-friction) dynamics

With inertia the state is (position, **current** velocity) and the committor is q(x, v):
two particles at the same x moving in opposite directions can have very different q. A
position-only RC is at best an average over velocities, and the stopped Z_q shows it as
lag dependence.

**Checks** (brute-force reference q(x, v) from shooting with the engine's own integrator,
600 trajectories per grid point; fits compared frame by frame):

| Run | Setup | Result |
|---|---|---|
| `sim_istar0lgv15intf` (flat, γ = 1, Δt = 2·10⁻⁴, paths ≈ 8000 frames, no velocities in `order.txt`) | `--history 0 1`, `0 8 16 32` | Staircase in q(λ), `zqs_z` ≈ 18: the history basis cannot use the velocity (see below) |
| same | `--fd-velocity`, lag-1 fit | Right for fast particles and near B; **slow particles near A too high by +0.12 to +0.15**. Started at the true committor, NPNEq drifts to the same biased answer |
| same | `--fd-velocity`, lag-4 fit | Near-A bias +0.07 at 100 paths/ensemble (+0.02 to +0.11 on three 30-path subsets), overall RMS 0.046; all lags ≥ 4 pass |
| `sim_istarwell2rv` (cosine well, γ = 1, Δt = 0.005, paths ≈ 220 frames, `v_ord`) | recorded velocity, lag-1 fit | **RMS 0.019, no near-A/B bias, passes every lag** |
| same | `--fd-velocity`, lag-1 fit | RMS 0.032, no near-A bias, but fails lags 2–256 (\|z\| 6–10) |
| same | `--fd-velocity`, lag-4 fit | RMS 0.021, lags ≥ 4 pass |

**Why histories fail.** The history basis is a polynomial in (x(t), x(t−d)), each scaled to
its range. At low friction and small Δt, x(t) − x(t−d) is ~10⁻⁴ of that range, so the
velocity is only reachable through huge, cancelling coefficients. `--fd-velocity` gives
the same information (the one-frame change of the order parameter, computed before the
cut and negated in reversed pieces) as its own O(1) input. Same information, usable form.

**Why an estimated velocity hurts a lag-1 fit.** NPNEq enforces "q one frame later equals
q now on average", which is exact only if the CVs are the full state. x(t) − x(t−1) is
the velocity of the *previous* step plus that step's random position kick, not the current
velocity. Over one frame these small differences are a sizeable part of the change NPNEq
sees, so the lag-1 equations target a slightly wrong function: consistent one frame ahead,
wrong over longer times. Over several frames the real change dominates, so fitting at lag
~4 removes most of it. A velocity over more frames (`--fd-velocity D` with D > 1) is
staler and makes it worse.

**What is not understood.** On the flat low-friction run the error concentrated near A
(slow particles), on the well it was spread out. The runs differ a lot (Δt 25× smaller,
paths ~40× longer and nearly ballistic, no force), and which of these matters is open.

**Statistics near A.** Few independent paths inform slow particles near A. Fits on
different 30-path subsets differed by ~0.1 there and all passed the stopped Z_q; use
≥ 100 paths per ensemble.

**Test caveat.** A grid reference interpolated linearly fails the stopped Z_q at lags 1–4
even with the exact velocity (well run: \|z\| 12 at lag 1, while the fit passes), because
the short-lag test resolves q on scales finer than the grid. Judge references at lags
≥ 4–16 only.

**Recommendations**

1. **Record velocities** (`v_ord = True`) and train on (x, v) with the standard lag-1 fit.
   Accurate, and `zqs_z` can be trusted at all lags.
2. **Positions only:** `--fd-velocity` (default D = 1), fit at **lag ~4**, read `zqs_z`
   only at lags ≥ 4. The lag-Δ fit (stopped windows, Σ f_k(t)[q(min(t+Δ, τ_t)) − q(t)] = 0)
   is not in the library yet: it exists only in the test scripts
   (`examples/figures/lgv_check/lag4_stats.py`, `examples/figures/well_check/well_test.py`).
3. Not helpful (tested): smaller γ, higher basis degree, a joint (r, x, v) basis, extra CVs
   (x + v/γ, arrival time at A).

---

## 11. What was dropped or is still open

- **ROC/AUC, cross-entropy, mse:** no longer printed or plotted for path data; they need a
  real A/B outcome per segment. Possible fix (not implemented): use the iSTAR committor of
  the (interface, direction) state where each segment ends as an RC-independent soft
  outcome; mse/ce then remain fair, ROC via sample weights y and 1−y. Valid under iSTAR's
  memory-loss assumption.
- **`feps.png`:** F(q) reflects path-sampling density, not equilibrium; its Z_q panel is
  the old all-lag version. Use `tis_diagnostics.png` instead.
- **Staple weights:** whether the high-acceptance factor has a correct per-path form, and
  the reversal asymmetry, depend on the staple move details.
- Validated on 1D toy systems with a known committor: flat overdamped, flat underdamped,
  and an underdamped cosine well.
- **Lag-Δ fit and `zqs_z` from lag Δ on:** needed for positions-only underdamped data
  (§10), not yet in the library.

---

## 12. Code map

**Library (both backends: `optimalrcs/` TensorFlow, `optimalrcs-cupy/` CuPy)**

| File | What |
|---|---|
| `metrics_tis.py` | `stop_index`, `comp_zq_stopped`, `comp_zq_stopped_zscore`, `comp_obs_pred_stopped`, `dropped_fraction`, metrics `max_z_zq_stopped` / `max_sd_zq_stopped` |
| `plots_tis.py` | `plot_zq_stopped`, `plot_zq_short_lags`, `plot_obs_pred_stopped` |
| `nonparametrics.py` | `npneq(..., w_traj=)` per-step weights |
| `optimalrcs.py` | `CommittorNE(..., path_weights=, group_traj=)`, `fit_transform(save_min_metric=)`, `plots_tis()`, metric registry includes `metrics_tis` |

**Examples (`examples/`)**

| File | What |
|---|---|
| `tis_data.py` | shared loader: interfaces, run settings (`dim`, `v_ord`), weights, cut at M (`first_frame_past_m`), reversed pieces (`last_frame_before_m`), data preparation, figures |
| `load_tis_data.py` / `load_tis_data_cupy.py` | fit scripts (TF / CuPy) |

**Tests:** `tests/test_metrics_tis.py`, plus CuPy-vs-TF checks in `tests/test_cupy.py`.

**Command-line options added**

| Option | Default | Meaning |
|---|---|---|
| `--weights rejections\|staple\|none` | rejections | MC path weights |
| `--no-cut` | off | keep whole paths (comparison only, biased) |
| `--reversed` | off | add time-reversed pieces (REPPTIS) |
| `--fd-velocity [D]` | off (D = 1 if given) | add the change of the order parameter over D frames as a velocity CV (§10) |
| `--burn-in N` | 0 | skip the first N cycles per ensemble |
| `--paths-per-ensemble N --seed S` | all | random subset, weights kept |
| `-a/-b`, `--interfaces-file` | first/last interface | state definitions |

**Figures:** `tis_diagnostics.png` (original Z_q at short lags with dropped share, stopped
Z_q / std. error, stopped obs-vs-pred), `metrics.png` (`dr2`, `zqs_z`, `|dx|`),
`committor_vs_order.png`, `committor_on_potential.png` (1D curve or 2D contours).

**Example**

```bash
conda activate test
cd examples
python load_tis_data_cupy.py /path/to/sim --paths-per-ensemble 2000 --seed 0 \
    --max-iter 2000 --print-step 250 --figure-dir figures_cupy/mysim
```

---

## 13. How the checks were done

- Toy runs with a **known committor**: flat 1D potential, overdamped Langevin. The
  exact committor is linear, corrected for the discrete time step (absorption is only
  checked at saved frames): $q(\lambda) = (\lambda - \lambda_A + s)/(\lambda_B - \lambda_A + 2s)$,
  $s = 0.5826\,\sigma$, $\sigma = \sqrt{2D\,\Delta t}$ (Siegmund correction).
- Underdamped runs: no closed form, so a brute-force q(x, v) from shooting on a grid
  (600 trajectories per point, the engine's integrator and force), interpolated at each
  frame's (x, v); finer grid near A where q changes fast.
- Each variant on several independent random path subsets, to separate effects from noise;
  size-matched controls where a difference could be due to data size.
- A metric is trusted only if the **exact** committor passes it and a wrong RC fails it.
