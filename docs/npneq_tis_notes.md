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
| Which value Z_q should converge to: N_AB for MD, 0 for paths cut at t\* | The level is set by the A→B departures in the data | [8.1](#81-which-value-z_q-converges-to-n_ab-0-or-something-else) |
| The cut throws away ~40% of the frames, and for staple paths that biases the sample | Add the mirrored pieces (`--reversed`), with rejection weights; required for the 2D staple run | [9](#9-optional-time-reversed-pieces), [9.1](#91-staple-paths-need-both-pieces-2d-brownian-z-sim_istarz_br_g10) |
| Underdamped (low-friction) dynamics: the committor depends on the velocity, which `order.txt` may not contain | Record velocities (`v_ord`); otherwise `--fd-velocity` | [10](#10-underdamped-low-friction-dynamics) |
| Even with the full state (x, v), a lag-1 fit converges extremely slowly at low friction (energy is nearly conserved per step) | Fit at lag Δ with stopped windows, γ·Δ·Δt ≈ 0.05–0.1 | [10.1](#101-full-state-x-v-why-lag-1-converges-so-slowly-and-the-lag-δ-fit) |

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

Earlier evidence on the 2D Z staple run `sim_istarz_2603` (18 Sep, chat only, no saved
script): with the full paths, the fit gave q = 0.78 / 0.98 / 0.98 / 0.99 at λ = 0.13 /
0.21 / 0.45 / 0.69, while the TIS crossing probabilities give 1.4·10⁻⁴ / 3.6·10⁻⁴ /
1.8·10⁻² / 0.21, and the ensemble's own LML/LMR counts bound q(0.13) ≤ 0.40. Using only
the frames after the M crossing gave q(0.13) = 3·10⁻⁴. (An earlier version of that test
fitted a constant q ≈ 0.17 because the loader left λ out of the CVs; that run is void.)
A fit also needs transitions *into* B: with none, q ≡ 0 is an exact fixed point (§2).

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
LML/RMR paths. Tested as a per-path weight it **fails** (below, and in both staple systems
of §9.1), so the default is rejection counts only (`--weights rejections`). For staple
paths, use them **with the mirrored pieces** (`--reversed`, §9.1): with forward pieces
only the shot LML/RMR paths are under-represented.

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

### 6.1 What `zqs_z` can and cannot tell (passes while the committor is wrong)

`zqs_z` tests whether the fitted q is a fair bet *on the data it was fitted to*: for every
window (start t, stopped at A/B or at the end of the piece), q at the stop minus q at the
start averages to zero, per q bin and lag. That is necessary for the committor, but on
path pieces it is far from sufficient. Cases in this project where it passed and the
committor was wrong, or the other way round:

| Case | `zqs_z` (own data) | Error vs exact / shooting | Why the test misses it |
|---|---|---|---|
| 2D Z staple, both pieces truncated at 200 frames (§9.3) | **0.70** (best of all) | RMS 0.075 | 75% of the frames removed; few windows can reach A/B within 200 frames (not counted), so q is mostly compared with itself |
| 2D Z staple, forward pieces, `staple` weights (§9.3) | 1.25 | RMS 0.055 | the data are biased; the fit is consistent with the biased data |
| 2D Z staple, forward pieces, rejection weights (§9.3) | 1.76 | RMS 0.045 | same (the exact q fails its data test, §9.1) |
| 2D Z staple, both pieces, rejection weights (§9.3) | **2.73** (fails) | **RMS 0.005** (best) | more data, smaller error bars: the same small residuals give larger z |
| 1D flat staple, forward vs both pieces (§9.3) | 1.12 vs 1.42 | RMS 0.0068 vs 0.0022 | same as above, in reverse order |
| 2D Z staple, both pieces, 1500 paths/ens. (§9.2) | 1.36 | plateau +0.031 | a uniform shift of q where few pieces reach A/B does not change the increments |
| 2D Z long MD, NPq | Z_q flat at N_AB, `max_sd_zq` ≈ 1 | plateau +0.05 (16 seeds) / −0.09 (new run) | the data (37 or 16 transitions) fix the plateau level only to ±0.05–0.1 |
| 1D well staple, mirrored, lag 1 (§10) | worst \|z\| 1.2–20 between chunks | RMS 0.030–0.052, not monotone | the test fluctuates with the fit; the exact q itself fails at short lags (grid) |

**Why.**
1. *Self-consistency only.* The test uses the same data as the fit. Biased data (wrong
   weights, forward pieces only) give a biased fit that is consistent with them. Only a
   test with an external reference, e.g. the **exact committor on the data** (§9.1), shows
   that the data are off.
2. *Power comes only from windows that reach A or B.* A window that ends inside the
   interface range is compared with the fit's own value there. An error that the
   increments do not see is invisible: a uniform shift of q in a region that pieces
   rarely leave towards A/B (the Z plateau), errors along slow directions (energy at low
   friction, §10.1), and everything beyond the longest piece (truncation). Truncating
   pieces makes `zqs_z` *look better* while removing exactly the information that fixes
   the levels.
3. *It is a z-score.* Dividing by the standard error makes it depend on the data size and
   on the spread of the weights. More frames → smaller error bars → larger z for the same
   residual; heavy or uneven weights → larger error bars → smaller z. `zqs_z` is therefore
   **not comparable between data sets** (data preparations, subsets, truncations), only
   between fits of the same data.
4. *It is the maximum over ~200 q bins × ~10 lags* of a heavy-tailed quantity (a few
   heavy paths dominate the variance, §9.2): the exact committor reaches |z| ≈ 3 on clean
   synthetic data, and a single chunk can jump (well staple). The threshold 2 is not
   calibrated.

**How to use it.** `zqs_z` ≫ 2 still flags a fit that is not converged or a CV set that is
not Markovian; `zqs_z` ≲ 2–3 means "consistent with these data", not "correct". Add:
- the exact committor (or shooting at a few hundred frames, `toy2d.shoot`) put through the
  same test, as a calibration of the data and of the threshold;
- independent halves of the paths, or several path seeds: their spread is the
  uncertainty of the regions the test cannot see (plateaus);
- an independent quantity that pins the levels, e.g. the TIS crossing probabilities
  (open, §11);
- never rank data preparations by their own `zqs_z`.

### 6.2 How the stopped Z_q, its error and `zqs_z` are computed (`metrics_tis.py`)

For a fitted q per frame, a lag Δt (frames) and 200 bins of q on [0, 1]:

1. **Stop index.** τ_t = the first frame ≥ t of the same piece that is in A or B, or the
   piece's last frame (`stop_index`, a reverse running minimum).
2. **One contribution per frame outside A and B:**
   c_t = w_t [q(min(t + Δt, τ_t)) − q(t)] / Δt, with w_t the path weight. It goes into the
   bin of q(t).
3. **Profile:** Z(x, Δt) = Σ_{t: q(t) ≤ x} c_t, the cumulative sum over the bins. For the
   committor E[c_t | x_t] = 0 (optional stopping), so Z ≈ 0 at every x.
4. **Standard error:** the frames are grouped into independent units g (`g_traj`: one
   group per path, so a path's forward and mirrored pieces count as one unit; default one
   per piece). With S_g(x) = Σ_{t ∈ g, q(t) ≤ x} c_t, the group's own part of Z,
   σ²(x) = Σ_g S_g(x)². This is a cluster-robust variance: any correlation within a path
   is included exactly; groups are assumed independent, and the mean is taken as 0 (the
   null hypothesis), so a real bias also enlarges σ.
5. **z(x, Δt) = Z/σ**, and **`zqs_z` = max |z|** over the bins (leaving out the 5 next to
   A, q < 0.025) and over the lags 1, 2, 4, …, 2¹⁵. `sdzqs` (`max_sd_zq_stopped`) is the
   largest standard deviation of the unnormalized Z over x.

What the weights and groups imply:
- A path that is repeated in the MC chain (rejections) enters once with its repeat count as
  weight: correct for the mean, and its variance counts it as one unit, as it should.
- Paths are **not** fully independent in an MC chain: consecutive paths share shooting
  points and history, and the per-block z-scores in §9.2 move together from block to
  block. σ is then too small and z too large. Using blocks of consecutive MC cycles as
  groups (instead of single paths) would give an honest σ.
- Even for the exact committor, the maximum of |z| over ~200 correlated bins × 16 lags is
  expected around 3, not ≲ 2 (an estimate: for independent normals it would be about
  √(2 ln N) ≈ 4 with N ≈ 3000; the correlations along x and between lags lower it). That
  agrees with |z| ≈ 3 for the exact committor on clean synthetic data (§9.2). A calibrated threshold is about 3–3.5, or better the value of the exact
  committor on the same data.

### 6.3 The dropped-window Z_q, and Z without the standard error

**Dropped windows (the original `comp_zq`).** It leaves out every window that runs past
the end of its piece. Pieces end because of what they did (interface exit, turn, A/B), so
the dropped windows are those of frames about to exit: the kept ones are selected on
their future. It is biased **for the exact committor too**, and the bias grows with
Δt / piece length: lattice test |z| 9–13 for the exact q (§6); `sim_istar0br10intf` −5 at
Δt = 4 and −18 at Δt = 8 (§8). It is reliable only at lags where the dropped fraction
(printed in the TIS plot legend) is a few percent, in practice Δt ≤ 2, and those short
lags are the weakest test, since the fit itself enforces Δt = 1. With A/B traps the A/B
ends are no longer dropped, but the interface ends still are. Use it only for the shortest
lags, and never as a convergence criterion for path data.

**Z without dividing by σ** (`sdzqs`, or max |Z|). Dividing by σ does not cause the blind
spots of §6.1; they come from testing the fit on its own data. Without σ:
- Z has no scale. It grows with the amount of data, the weights and the lag, so there is
  no threshold for "too large".
- It cannot tell noise from bias: in a sparsely sampled q range one heavy path makes Z
  large although nothing is wrong.
- It is fine for comparing **fits of the same data** (seeds, iterations, envelope): lower
  is more consistent, and the noise level is the same for all of them.
- A scale that is interpretable: divide Z by the weighted number of windows that end in B
  (or A). Then Z ≈ the bias in q times the share of those windows that started in the
  region. That is close to what the per-band martingale test of §9.2 measures.

### 6.4 More rigorous checks for path data

No test on the fitted data can detect what the data do not determine (a plateau level that
few pieces fix, §9.2). A rigorous assessment therefore needs either an error bar on q, or
information the fit did not use.

1. **Shooting at sampled frames (the reference standard).** Choose a few hundred frames
   between A and B, run 100–200 new trajectories from each with the engine, and compare:
   χ² = Σ_i (q_fit,i − q_shoot,i)² / se_i², which should be about the number of frames.
   Model-free, with exact error bars, and it checks levels as well as shapes. Cheap for
   toy systems (`toy2d.shoot`: 300 frames × 200 shots in about 5 s); for real systems it
   is a short extra simulation with the same engine.
2. **Error bars by resampling.** Fit independent subsets of the paths (halves, path
   seeds) or blocks of MC cycles (block bootstrap). The spread is the honest uncertainty,
   including the plateau. Costs a few fits.
3. **TIS crossing probabilities as a level check.** For trajectories leaving A, the
   flux-weighted average of q over the points where they first cross λ_i equals
   P_A(λ_B | λ_i), the TIS crossing probability from λ_i to B. TIS estimates it without
   any committor fit, so it pins the levels exactly where the stopped Z_q is blind. The
   first-crossing points from A are directly available in RETIS [i⁺] ensembles; for
   staple data it is implemented with the tistools iSTAR MSM (§6.5), which relies on its
   memory-loss assumption.
4. **The stopped Z_q, improved:** groups = blocks of MC cycles (honest σ); a calibrated
   threshold (the exact or a shooting-checked q on the same data, or about 3–3.5); and
   report it per lag rather than only the maximum. It remains a necessary condition, not
   a sufficient one.

### 6.5 The level check with the iSTAR turn-to-turn MSM (`examples/crossing_check.py`)

**Idea.** By the strong Markov property, the committor averaged over the phase points where
trajectories cross an interface (in a given direction, coming from a given turn) is exactly
the probability that, after such a crossing, B is reached before A. TIS estimates that
probability from the path ensembles without any committor: the tistools iSTAR MSM
(`istar_analysis`) has states "turned at λ_j, heading right/left", estimates from the turn
interfaces of the staple paths
q_t[j][k] = P(reach λ_k | last turn at λ_j, λ_{k∓1} reached), and solves for the
committor z of each state. So a fitted q can be compared with numbers it was not fitted to.
These numbers come from how far whole paths get, which is exactly what fixes the level of q
on a plateau.

**Computation.**
1. *MSM*: `istar_msm`: tistools `compute_weight_matrices` → `get_transition_probs_weights`
   (gives q_t) → `construct_M_istar` → `global_pcross_msm_star` (gives z). Use the tistools
   default **staple** weights here. On the 1D test the rejection weights gave errors up to
   0.015; the staple weights reproduce the exact values. For the frames of the fit it is the
   other way round (§4, §9.3).
2. *Crossings in the committor data*: every piece starts at a crossing of its ensemble's
   middle interface λ_m. The direction is the side of λ_m the start frame lies on. The turn
   the trajectory came from is the extremum of the partner piece (the same path read the
   other way): j with min ∈ [λ_{j−1}, λ_j] for upward crossings, or max ∈ [λ_j, λ_{j+1}] for
   downward ones (`crossings`).
3. *MSM prediction per crossing* (`predicted_committor`), for an upward crossing coming
   from turn j < m:
   C = Σ_{k=m}^{N−2} Π_{l=m+1}^{k} q_t[j][l] · (1 − q_t[j][k+1]) · z[k, heading left]
   + Π_{l=m+1}^{N−1} q_t[j][l]. Downward crossings are the mirror image, with reaching A
   counting 0.
4. *Comparison* (`compare`): per (interface, direction), the path-weighted mean of q over
   the crossing frames against the mean of C. Errors: bootstrap over paths for the means;
   for the MSM, 4 blocks of MC cycles. Summary: χ² of (mean q − MSM) over the cells.

**Results** (`examples/figures/crossing_check/`):

| Run | Exact committor − MSM | What the check can tell |
|---|---|---|
| 1D flat staple (`run_1d.py`), 1000 paths/ens. | ≤ 0.005 at all 15 (interface, direction) cells | memory loss is exact in 1D: the implementation is right, and levels are tested to ~0.005 |
| curved channel, Brownian, 2000-cycle test run (`run_cc.py`) | RMS 0.020, χ² 27 for 17 cells; largest −0.05 at the two interfaces next to the saddle (MSM noise ±0.02 per cell) | the fit (RMS 0.015 vs exact): χ² 20. The exact committor shifted by +0.03 where 0.5 < q < 0.95: χ² 34.5 (not separable from the exact one); shifted by +0.06: χ² 79 (detected) |
| 2D Brownian Z (`run_z.py`), all cycles | up to +0.43 (e.g. at λ = 0.61: exact 0.69, MSM 0.30) | nothing: y-interfaces cut across the 45° ridge, so where a trajectory crosses (x) matters and the turn-to-turn MSM loses it. The MSM, not the fit, is off |

**Use and limits.** This is the only check found so far that tests the level of q from the
path data alone, without extra simulation. Its resolution is the noise of the MSM (it
shrinks with more cycles; ±0.02 per cell for 2000 cycles) plus its memory-loss error,
which depends on how well the order parameter's interfaces separate the dynamics. Where
the MSM is valid (1D, and roughly the curved channel away from the saddle), a fit that
disagrees with it at many interfaces has a level error. Where the MSM is not valid (the Z
potential), disagreement says nothing about the fit. On the Z run this also implies that
the iSTAR crossing probability of that run is probably biased (not checked against the
rate directly).

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

### 8.1 Which value Z_q converges to: N_AB, 0, or something else

The criterion is always the same: for the committor, Z_q is **flat in q and the same at every
lag**. The *level* it is flat at is not a property of the method (NPq or NPNEq) but of the
data: which boundary departures the segments contain.

**Why.** Z_q(x, Δt) adds up the forward change of q over windows of length Δt, from all
frames at or below the level x (windows that reach A or B stop there). For the committor:

- A window that starts **outside A and B** has zero expected change (fair bet). Over many
  windows these frames add only noise, whatever their distribution.
- A window that starts **inside A** (q = 0) and runs into an excursion has a nonzero change:
  summed over the windows of one departure, an excursion that reaches B adds **1**, one that
  falls back into A adds 0. These frames are at q = 0, so they count for every level x > 0.
- Frames inside B have q = 1 and only enter at x = 1, so B→A departures do not count.

So the converged value is the **(weighted) number of A→B departures whose last A frame is
in the data**, at every q and lag:

| Data | Converged Z_q of the committor | Why |
|---|---|---|
| Long equilibrium MD, one or several seeds | **N_AB**, the number of A→B transitions (summed over seeds) | every A→B transition starts from an A frame in the data |
| MD cut into excursions (frames outside A/B plus the A/B frame before and after each), with `set_boundary_traps()` | **N_AB**, identical to the continuous run | frames dwelling inside A/B contribute exactly 0, so removing them changes nothing |
| Same, **without** traps | falls below N_AB at large lags (artifact) | the boundary terms of frames within Δt of a segment edge are dropped (§8) |
| TIS / REPPTIS / staple paths cut at t\* | **0** | no segment starts inside A, so only zero-mean terms remain. The stopped Z_q (§5) leaves A/B frames out by construction and is 0 too |
| Uncut paths (`--no-cut`) | some positive value, not a valid test | segments do start at A, but the frames before the M crossing are conditioned on crossing M (§3), so the departures are not a fair sample |
| Short trajectories started from arbitrary points (shooting, swarms) | number of segments that start in A and reach B, usually ≈ 0 | same rule |

With path weights every count is weighted.

**Noise.** The level is a count, so at short lags it fluctuates by roughly its own square
root. With N_AB = 37 the exact committor's Z_q ranged over 25–42 at Δt = 1–256. With a
target of 0, the spread is all there is to judge, which is why path data use the z-score
(§6) and not the raw value.

**Telling a real failure from an artifact.**

- *Converged:* flat in q, all lags at the same level (N_AB, 0, …) within the noise.
- *Truncation artifact:* curves fall towards 0 as Δt grows, more strongly near q = 1 or
  near segment ends, also for the exact committor. Check `dropped_fraction` (path data) or
  use traps (MD excursions, A/B ends of paths).
- *Not the committor:* a slope in q, or lags that disagree beyond the noise, after the
  artifacts above are removed. Examples: `zqs_z` ≈ 10 for q², ≈ 18 for positions-only
  underdamped fits (§10).

**Evidence** (2D Brownian Z potential, 16-seed long MD, A: y ≤ 0.1, B: y ≥ 0.85, 37 A→B and
37 B→A transitions, 4.6M rows in 205k excursions;
`examples/figures/z_brownian/npq_md/zq_traps.png`):

- Without traps the plotted Z_q sat at 9–36 depending on Δt and dropped towards 0 above
  q ≈ 0.7 for Δt = 4–256.
- With `set_boundary_traps()`: NPq Z_q flat in q, ≈ 30 at Δt = 1 rising to 36.9 at
  Δt = 32768. The exact committor gives 36.8–37.15 at Δt = 32768.
- Excursions with traps reproduce the continuous trajectory's Z_q to every printed digit
  (seed 1, all lags; `tests/test_boundary_traps.py` checks it to 1e-6 in both backends).

**Code.** `CommittorNE.set_boundary_traps(trap_boundary=None)` (both backends) treats every
segment that starts or ends in A/B as continuing there indefinitely
(`FutureBoundary.set_distance_to_end_trap`, `PastBoundary.set_distance_from_start_trap`).
It changes the Z_q-based metrics and plots, not the fit. `set_fixed_traj_length_trap`
(above) pads only the end side, which is enough for paths cut at t\* (they never start in
A/B), but not for MD excursions.

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

The staple row above was on a small subset and is within noise. Larger tests (§9.1)
reverse that conclusion: **use `--reversed` with rejection weights for staple paths as
well**. On the 2D Brownian Z run it is required; on the 1D flat run it is harmless.

### 9.1 Staple paths: forward vs both pieces (2D Brownian Z, `sim_istarz_br_g10`)

> **Revised by §9.2.** The z-score evidence below is weaker than it looks: the test's error
> is dominated by a few heavy paths, and on clean data the exact committor already reaches
> |z| ≈ 3. What holds: adding the mirrored pieces with rejection weights improves the fit,
> and the `staple` weighting is wrong. What changed: the remaining plateau offset is mostly
> statistical (§9.2).

**Symptom.** The staple NPNEq fit (forward pieces only, rejection weights, 3000 paths per
ensemble) put the plateau behind the 45° ridge 0.05–0.07 too high. Shooting with the toytis
engine confirmed the reference committor (1200 trajectories per point, ±0.014). The fit's
stopped Z_q sat at ≈ −3 at the longest lags for every q below the plateau.

**Diagnosis: test the data with the exact committor.** Put the exact committor into the
stopped Z_q z-score of the *data*. If the exact committor fails, the data (weights, cut,
pieces) are biased and no fit can be right. This takes about a minute and needs no fit
(`examples/figures/z_brownian/staple_check.py`, `staple_split.py`, `staple_wscan.py`,
`staple_reversed.py`).

| Exact committor on the staple data | Worst z, plateau (lags 1, 16, 256, 4096) | Worst z, next to A/B |
|---|---|---|
| rejection weights, forward only (as fitted) | +3.0 +2.8 +3.3 +2.6 | +2.1 +1.9 +1.8 +2.4 |
| **rejection weights, forward + mirrored** | **+1.6 +1.2 +0.7 +1.4** | **−0.9 −1.2 −0.8 +1.3** |
| staple weights, forward only | −2.1 −2.2 +2.3 +1.5 | −3.2 −3.2 −2.6 −1.6 |
| staple weights, forward + mirrored | −3.9 −4.1 −3.9 −1.9 | −5.6 −5.6 −5.0 −2.0 |
| all weights 1, forward only | about +6 | |

(1500 paths per ensemble, seed 0, for the four main rows.)
- Forward pieces only: the excess on the plateau comes from the ensembles past the ridge
  (006–009, M = 0.45–0.69). It is present in both halves of the paths, so it is systematic.
- A high-acceptance factor f on shot LML/RMR paths (about half of the paths in ensembles
  003–009), applied to forward pieces only, moves the result steadily from +3 (f = 1) to −3
  (f = 2); f ≈ 1.25–1.5 nearly passes. Rejection weights with both pieces pass without a
  tuned factor. Reading: each accepted path must contribute both halves around its M
  crossings, and the factor 2 of `get_weights_staple` is the whole-path counterpart of the
  same correction. This reading is not yet derived from the staple move rules.
- 1D flat staple run (`sim_istar0br10intf`, 6000 paths per ensemble, exact linear committor):
  rejection weights pass with and without mirrored pieces (within ±2 at lags ≥ 16; both
  +3.7 at lag 1 right next to A). Staple weights fail either way (+6 to +9).

**Refit** with `--reversed`, rejection weights, 1500 paths per ensemble (20.9M frames),
otherwise the same settings (`examples/figures/istarz_br_g10_reversed/`). Settings of both
fits: CVs (y, x), library lag-1 update, gamma 0.1, envelope 0.01, ny 6, A/B traps;
`load_tis_data_cupy.py`, no random seed (forward only: 3000 paths/ens.; mirrored: subset
seed 0). The forward-only fit ran twice; 3.23 is the first run, whose q was lost; the q
used for every comparison is the rerun's (`zqs_z` 3.05).

| | Forward only (3000 paths/ens.) | **Forward + mirrored (1500 paths/ens.)** |
|---|---|---|
| `zqs_z` after 10000 iterations | 3.23 (first run; rerun 3.05) | **1.36** |
| RMS − exact per bin | 0.058 | **0.025** |
| mean (fit − exact) on the plateau | +0.061 | **+0.023** |
| q at the shooting points (0.639, 0.664, 0.698, 0.747) | 0.707 0.738 0.754 0.799 | 0.665 0.700 0.715 0.767 |

Continued to 25000 iterations the fit converged at +0.031 (RMS 0.029, `zqs_z` 1.36): §9.2.

**The long MD of the same system** also overshot the plateau (+0.05–0.07 with NPq). There
the cause is the data: 32768 independent GPU walkers give 4.30·10⁷ ± 4% steps per A→B
transition, matching the reference (4.4·10⁷), while the 16-seed MD had 37 A→B transitions
where ~25 were expected (+2.5σ). Fits on the two halves of the MD give +0.031 and +0.079,
independent of the fit lag (1, 10, 50 rows) and of the envelope schedule
(`examples/figures/z_brownian/lag_md.py`, `rate_walkers.py`). With 37 transitions the
plateau level is uncertain by several hundredths.

**Reproduced from scratch with other seeds (5 Nov,
`examples/base_documented_2d_brownian_staple.ipynb`, figures `examples/figures/nb_brownian_staple/`).**
Forward pieces only (3000 paths/ens.), 1000 warm-up iterations at envelope 1, then 15000 at
envelope 0.01, gamma 0.1: q at the shooting points 0.718 / 0.749 / 0.763 / 0.807 (shooting
0.639 / 0.664 / 0.698 / 0.747), RMS 0.061 per row: the forward-only overshoot does not
depend on the fit seed or on a warm-up. A later run of the same notebook with mirrored
pieces (2000 paths/ens., path-subset seed 1, 27.9M frames, fit seed 3) shows RMS 0.039 and
`zqs_z` ≈ 1.9 in its saved figures (the notebook outputs were not saved). Note: path seed 1
draws its 2000 paths from the same ~3000-path copy as the other fits, so it is **not an
independent data set**.

**Long MD with NPq, longer and at a smaller envelope (5 Nov,
`examples/base_documented_2d_brownian_md.ipynb`, `examples/figures/nb_zbrownian_md/seeds/`).**
Same 16 seeds and kept rows as before (4.62M rows, 37 A→B), `set_boundary_traps`, ny 6;
1000 warm-up iterations at envelope 1, then 15000 at envelope 0.001, gamma 0.1 (NPq ignores
gamma). RMS(fit − exact) per row for 0.05 < q < 0.95: 0.053 after the warm-up, 0.062 at
16000 iterations (per bin 0.049; seed 3: 0.050). The plateau stays ~+0.05 above exact, as
with envelope 0.01: a smaller envelope and more iterations do not remove it, consistent
with a fluctuation of this data set (§9.2) rather than a fit problem.

**A new MD run of the same system (6 Nov, same notebook):** 1.06·10⁹ steps generated on the
GPU (one row per 20 steps, 4.46M kept rows), **16 A→B transitions**. NPq (envelope 0.001,
1000 + 15000 iterations, seed 6) gives RMS 0.084 per row against the exact committor, and
here the fit is **below** the exact committor on the plateau. The model-free fractions of
excursions entering a band of exact q that reach B are below the exact values too:

| Band of exact q | Excursions | Reach B (data) | NPq at entry | Exact at entry |
|---|---|---|---|---|
| 0.20–0.40 | 53 | 0.11 ± 0.04 | 0.21 | 0.30 |
| 0.40–0.55 | 39 | 0.33 ± 0.08 | 0.38 | 0.50 |
| 0.55–0.70 | 65 | 0.54 ± 0.06 | 0.56 | 0.65 |
| 0.70–0.85 | 92 | 0.73 ± 0.05 | 0.76 | 0.80 |

So one MD sample overshoots the plateau (+0.05, 37 transitions) and another undershoots it
(−0.09, 16 transitions): the NPq fits follow their data, and a few tens of transitions
pin the plateau level only to about ±0.05–0.1. This confirms the statistical reading of §9.2.

### 9.2 How well do cut path pieces determine the committor? (2D Brownian Z)

Overnight tests (`examples/figures/z_brownian/synth*`, `martingale_*.py`,
`istarz_br_g10_reversed/half_fits.py`, `lag_halves.py`). Summary first:

- **The method works:** on clean synthetic data (30000 GPU walkers started uniformly between A
  and B, run until absorbed, saved every 50 steps; exact committor known), NPNEq with the
  staple settings converges within 2500 iterations to the exact committor (plateau −0.007,
  the reference's own interpolation error).
- **Pieces that end at interfaces fix the plateau level only weakly.** The plateau is set
  by the pieces that reach A or B, and few do: of the staple pieces entering the plateau,
  3% (q 0.55–0.7) and 19% (q 0.7–0.85) reach A or B. A uniform shift of q on the plateau
  hardly changes the one-step increments of the other pieces.
- **Consequence:** fits of the staple data scatter by about ±0.02–0.03 in the plateau
  level between independent halves of 7500 paths; this is statistical, not a bias of
  the method, and a longer fit lag does not reduce it.

| Plateau (fit − exact) | lag 1 | longer lag |
|---|---|---|
| synthetic walkers, run until absorbed (5.2M frames) | −0.007 | |
| synthetic walkers **cut at interfaces** like path pieces (all, 30000 pieces) | −0.015 | |
| — even half / odd half | −0.040 / +0.009 | −0.031 / +0.031 (lag 10 rows = 500 steps) |
| staple, forward + mirrored, 1500 paths/ens. | +0.031 | |
| — even half / odd half (7500 paths each) | +0.049 / +0.014 | +0.034 / +0.016 (lag 50) |
| staple, forward + mirrored, 2250 paths/ens. (31.5M frames) | | +0.022 (lag 50) |
| staple, forward pieces only, 3000 paths/ens. | +0.061 | |

Settings (all: CVs (y, x), gamma 0.1, envelope 0.01, ny 6; §14): lag-1 fits use the library
update with lazy basis and A/B traps (synthetic uncut 2500, cut 10000 iterations; staple
1500 paths/ens. 25000; staple halves 25000 / 22500, stopped when the plateau changed by
< 0.001); longer lags use the stopped-window update without traps (synthetic lag 10:
7500 iterations; staple lag 50: 3000).

The cut synthetic pieces have no weights, no staple moves and no mirrored pieces, so the
spread there comes from "pieces end at interfaces" alone. (They also have only 10% of the
frames of the uncut walkers; less data adds to the spread.)

**Calibrating the tests.**
- Stopped Z_q z-score of the exact committor on the clean synthetic data: worst |z| 2.8–2.9
  (at q ≈ 0.61, the ridge edge). |z| ≈ 3 is therefore normal here, and the forward-only
  staple value of +3 (§9.1) is not significant on its own.
- On the staple data the z-score's variance is dominated by a few long, heavily weighted
  paths (5 of 12893 carry 41% of it below q = 0.65; each ridge crossing contributes about
  ±0.6 × weight). Subsets (single ensembles, halves) then give heavy-tailed z values (±10).
- **Martingale test at the first entry into a q band**: per piece,
  D = q_exact(stop) − q_exact(first frame in the band), stop = first A/B frame or the end of
  the piece; E[D] = 0 for unbiased data (optional stopping), one value per piece, standard
  error over paths. On the synthetic walkers |E[D]| ≤ 0.010 ± 0.007 in every band. On the staple data
  with rejection weights and both pieces: +0.0015 ± 0.0012 (q 0.55–0.7), +0.0010 ± 0.0018
  (q 0.7–0.85); forward pieces alone +0.004 ± 0.002, mirrored −0.001 ± 0.002; early and late
  halves of the run agree (no equilibration problem). The `staple` weighting gives
  −0.011 ± 0.002 (−6σ): wrong. But the fitted q (+0.031) passes this test just as well
  (E[D] = +0.0001 ± 0.0012), because so few pieces reach A or B: the test cannot measure
  the plateau level either.

**The long MD** of the same system (§I.2 of the results overview) behaves the same way: the
16-seed MD had 2.5σ more transitions than average (GPU walkers: 4.30·10⁷ ± 4% steps per A→B
transition, reference 4.4·10⁷), and its two halves gave +0.031 and +0.079 at every fit lag
and envelope setting.

**Overall estimate.** The fits with both pieces give +0.031 (1500 paths/ens., lag 1),
+0.025 (mean of the two halves at lag 50) and +0.022 (2250 paths/ens., lag 50), against an
expected statistical uncertainty of about ±0.015–0.02. So the remaining offset is about
+0.02 ± 0.015: mostly statistical, but all estimates are positive, so a small residual
bias of order +0.01 (for instance the slight lean of the forward pieces in the martingale
test, +0.004 ± 0.002) cannot be excluded at this sample size.

Figures of all these fits and tests: `examples/figures/z_brownian/summary_*.png`, per-case
plots in `cases/` (results overview, II.3 "The plateau investigation in figures").

**Per-ensemble and per-block z-scores** (`examples/figures/z_brownian/staple_cycle_blocks.py`,
`.log`; exact committor on the forward-only data, 29975 paths, rejection weights; no fit).
Stopped Z_q z-score at q = 0.3 / 0.63 / 0.75: all data +1.0 / +1.9 / +2.7 at lag 16
(−1.3 / +1.1 / +3.1 at lag 4096). Per ensemble and per quarter of the MC cycles, single
blocks reach −13 to +7 while the whole ensemble is near 0 (e.g. ensemble 003: −11.0 and
−12.7 in two quarters at lag 4096; ensemble 009: +6.3 / +6.5 in the middle quarters). The
MC chain is strongly correlated from block to block (long-lived heavy paths), so block
z-scores are not independent and their error bars are too small. The pooled z at q = 0.75
is positive in all four quarters (+1.2 to +1.6 at lag 16): weak evidence (about 1 in 16 by
chance) for a small systematic plateau effect. To settle it, repeat on the full run
(~50000 paths per ensemble on the workstation).

**What to do with path data:**
- Use rejection weights with both pieces (`--reversed`), not the `staple` weighting.
- Expect the level of flat regions far from A and B (plateaus) to be uncertain by a few
  hundredths unless many pieces reach A or B; estimate it from independent halves of the
  paths. More paths reduce it as 1/√N.
- The z-score and the obs-vs-pred plot do not detect this uncertainty.
- Open: information that pins the plateau level directly, e.g. the crossing probabilities
  between interfaces that TIS itself provides.

---

### 9.3 Data preparations compared (`examples/staple_data_choices.ipynb`)

Same paths, same fit settings and seed for every preparation; only the weights, the
mirrored pieces, the cut at M and a piece-length limit differ. Settings: `CommittorNE`,
warm-up 1000 iterations at envelope 10⁶ (a flat envelope), then envelope 0.01, gamma 0, ny 6,
lazy basis, A/B traps, fit seed 0. RMS of fit − exact over frames with 0.05 < q_exact <
0.95 (path-weighted) and per bin. Results: `examples/figures/staple_compare/*/comparison.csv`.

| Preparation | 1D flat `sim_istar0br10intf` (1000 paths/ens., seed 0, 4000 it.): `zqs_z` / RMS frames | 2D Z `sim_istarz_br_g10` (1000 paths/ens., seed 12, 15000 it.): `zqs_z` / RMS frames / RMS bins |
|---|---|---|
| forward, rejection weights | 1.12 / 0.0068 | 1.76 / 0.045 / 0.046 |
| **both pieces, rejection weights** | **1.42 / 0.0022** | **2.73 / 0.0047 / 0.0135** |
| forward, `staple` weights | 1.63 / 0.0149 | 1.25 / 0.055 / 0.053 |
| both, `staple` weights | 1.37 / 0.0098 | 2.81 / 0.093 / 0.088 |
| forward, no weights | 2.62 / 0.0176 | 1.46 / 0.054 / 0.049 |
| both, no weights | 2.28 / 0.0113 | 3.23 / 0.018 / 0.016 |
| no cut at M, rejection weights | 3.73 / 0.106 | 14.1 / 0.204 / 0.196 |
| both, rejection, **truncated** at 200 frames | 1.46 / 0.0020 (1.24M of 1.27M frames) | **0.70 / 0.075 / 0.073** (3.5M of 14.0M frames) |
| both, rejection, long pieces **discarded** (> 200 frames) | – | 8.5 / 0.380 / 0.381 (0.31M frames, 4172 pieces) |
| both, no weights, truncated at 200 (1D only) | 2.31 / 0.0105 | – |

- **Both pieces with rejection weights is the best preparation on both systems** (2D: 10×
  better than forward-only). The `staple` weighting and no weights are worse; no cut at M
  is wrong (§3).
- **Truncation:** valid in principle (a stopping rule), and harmless on the 1D run, where
  only 2% of the frames are in pieces longer than 200 frames. On the 2D Z run 200 frames
  removes 75% of the frames (and with them, presumably, most of the windows long enough to
  reach A or B; not counted): RMS 0.075, 16× worse,
  while `zqs_z` (0.70) looks best of all (§6.1). Truncate only far in the tail of the
  piece-length distribution, and check it with the exact committor or two limits.
- **Discarding** long pieces selects pieces on their future: strongly biased (RMS 0.38).
- On this subset (1000 paths/ens., path seed 12, gamma 0) the both-pieces fit has RMS 0.0047
  per frame and 0.013 per bin, smaller than the 1500-path fits of §9.2 (0.03 per bin); the
  plateau offset was not printed. It is one subset; the ±0.02 spread of §9.2 applies.
- In the 2D notebook the configuration named `rev_rej_none200` uses rejection weights
  (truncate); in the 1D one it uses no weights.

## 10. Underdamped (low-friction) dynamics

With inertia the state is (position, **current** velocity) and the committor is q(x, v):
two particles at the same x moving in opposite directions can have very different q. A
position-only RC is at best an average over velocities, and the stopped Z_q shows it as
lag dependence.

**Checks** (brute-force reference q(x, v) from shooting with the engine's own integrator,
600 trajectories per grid point; fits compared frame by frame). Settings of all fits in this
table: `CommittorNE`, gamma 0.1, envelope 0.01, ny 6, 4000 iterations, fit seed 0, paths
subset seed 0, rejection weights. `sim_istar0lgv15intf`: CVs x and Δx; the lag-1 row is the
library update (non-lazy basis, A/B traps, 100 paths/ens.), the lag-4 rows the stopped-window
update (no traps; 100 or 30 paths/ens.); the history rows' settings were not recorded.
`sim_istarwell2rv` (`figures/well_check/well_test.py`): 2000 paths/ens., stopped-window update
also at lag 1, full basis, no traps.

| Run | Setup | Result |
|---|---|---|
| `sim_istar0lgv15intf` (flat, γ = 1, Δt = 2·10⁻⁴, paths ≈ 8000 frames, no velocities in `order.txt`) | `--history 0 1`, `0 8 16 32` | Staircase in q(λ), `zqs_z` ≈ 18: the history basis cannot use the velocity (see below) |
| same | `--fd-velocity`, lag-1 fit | Right for fast particles and near B; **slow particles near A too high by +0.12 to +0.15**. Started at the true committor, NPNEq drifts to the same biased answer |
| same | `--fd-velocity`, lag-4 fit | Near-A bias +0.07 at 100 paths/ensemble (+0.02 to +0.11 on three 30-path subsets), overall RMS 0.046; all lags ≥ 4 pass |
| `sim_istarwell2rv` (cosine well, γ = 1, Δt = 0.005, paths ≈ 220 frames, `v_ord`) | recorded velocity, lag-1 fit | **RMS 0.019, no near-A/B bias, passes every lag** |
| same | `--fd-velocity`, lag-1 fit | RMS 0.032, no near-A bias, but fails lags 2–256 (\|z\| 6–10) |
| same | `--fd-velocity`, lag-4 fit | RMS 0.021, lags ≥ 4 pass |
| same, **+ mirrored pieces** (`--reversed`) | recorded velocity, lag-1 fit | RMS 0.030, fails lags 4–256 (\|z\| 3.4–4.7): worse than forward only |
| same, + mirrored pieces | recorded velocity, lag-4 fit | RMS 0.017 (forward only 0.015), lags ≥ 4 within 2.4 |
| same, + mirrored pieces | `--fd-velocity`, lag 1 and 4 | not run yet (`examples/base_documented_1d_langevin_staple.ipynb`) |
| same, + mirrored pieces, notebook (6 Nov) | recorded velocity, library lag-1 update, gamma 0, envelope 0.01, warm-up 2000 at envelope 10⁶, 10000 it., seed 1 | RMS 0.033 (near A +0.009, near B +0.025); RMS 0.030–0.052 and worst \|z\| 1.2–20 from chunk to chunk, not monotone; the exact q on these data: 19.1 17.6 15.1 4.2 2.5 1.3 at lags 1–256 |

**Mirrored pieces on the well** (`figures/well_check/run_rev_recorded.log`). Unlike on the 2D
Brownian Z (§9.1), they do not help here. The data test with the exact committor shows no
bias of the forward pieces to remove: it fails at short lags with and without them
(lags 1, 2, 4, 16, 64, 256: 12.0 11.1 9.6 3.5 2.5 2.1 forward only; 15.4 14.2 12.2 5.2 2.4
2.2 with mirrored pieces). That failure grows by about √2 with twice the frames, as a fixed
error does: most likely the interpolation error of the coarse 29 × 29 reference grid,
which short lags see most. At lags ≥ 64 both pass.

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

### 10.1 Full state (x, v): why lag 1 converges so slowly, and the lag-Δ fit

**Test** (`examples/figures/well_md_fit/`): long equilibrium MD of the cosine well
(`md_istarwell2_100M`, γ = 1, kT = m = 1, Δt = 0.005, one frame per step, recorded
velocities), first 6M frames (5007 A→B transitions), A: x ≤ −0.35, B: x ≥ 0.35. CVs (x, v),
`CommittorNE`, ny 6, envelope 0.001, lazy basis. Compared per (x, v) bin with the
brute-force reference q(x, v) (§13) and with the MD-outcome committor (fraction of frames
that reach B first, no fit). The two references differ by RMS 0.012 per bin: the noise
floor of the comparison.

| Fit | Iterations | Cross-entropy | AUC | RMS − exact (bin / frame) | On the band \|E − 0.15\| < 0.05 | Largest bin error |
|---|---|---|---|---|---|---|
| exact reference | | 0.3948 | 0.8735 | | | |
| history `[0, 1]`, CV x only, γ_reg 0.2 | 3000 | 0.3972 | 0.8731 | | | |
| (x, v), lag 1, γ_reg 0.2 | 10000 | 0.3963 | 0.8717 | 0.034 / 0.038 | 0.080 | 0.20 |
| (x, v, E), lag 1, γ_reg 0.2, then 0.05 | 10000 + 5000 | 0.3952 | 0.8730 | 0.020 / 0.024 | 0.045 | |
| (x, v), lag 1, γ_reg 0.1 (stopped by its own criterion) | 95000 | 0.3955 | 0.8727 | 0.022 / 0.027 | 0.064 | 0.17 |
| **(x, v), lag 16, γ_reg 0.1** | **3000** | **0.3951** | **0.8734** | **0.014 / 0.022** | **0.037** | **0.09** |
| **(x, v), lag 16, γ_reg 0.1** | **12000** | **0.3947** | **0.8737** | **0.011 / 0.018** | **0.027** | **0.06** |

(γ_reg is the `gamma` regularization of the fit, not the friction.) The lag-16 fit reaches
the noise floor of the comparison; its cross-entropy and AUC are slightly better than the
grid-interpolated reference's. The lag-1 fits keep their error in two thin bands along the
energy contour E = ½v² + V(x) ≈ 0.15 through the bump tops (the energy a particle needs to
cross a bump), with sharp vertical edges at x ≈ ±0.29 in the (x, v) map that the data do
not have (the MD-outcome committor is smooth there). The lag-1 run's cross-entropy kept
creeping down (2.7·10⁻⁵ per 15000 iterations at the end), so its own metrics cannot tell
this from convergence. (On the iSTAR path data of the same well, a lag-1 fit with recorded v
gave a per-frame RMS of 0.019, §10 table. That number is averaged over path frames, which
weight the region near the bump tops differently, and was not broken down by energy; it is
not directly comparable and should be rechecked with a lag-Δ fit.)

**Why lag 1 is slow even with the full state.** NPNEq solves "q one step later equals q now,
on average". Over one step, low-friction Langevin dynamics is almost Newtonian: the energy
E changes only by about γ·Δt (0.5% here) through friction and noise. An error in q that
depends only on E is therefore nearly invisible to the one-step equations: it is only seen
through the small friction/noise part. In the eigenmodes of the lag-Δ transition operator
P_Δ, such a slowly relaxing error has weight |1 − e^{−μΔ}| ≈ μΔ in (P_Δ − I)q = 0, while
fast modes saturate near 1. The equations are badly conditioned at Δ = 1, and the slowly
relaxing directions are exactly where this committor changes fastest (crossing a bump or
bouncing back at the bump-top energy).

**Why the lag-Δ fit works.** With the full state the dynamics is Markov, so the committor
is a fair bet over any stretch of time: E[q(x_{min(t+Δ, τ_t)}) | x_t, v_t] = q(x_t, v_t),
with τ_t the first A/B frame at or after t (windows must be stopped there, not dropped).
The answer is the same at every lag, but the slow directions become ~Δ times more visible
while the fast ones do not grow, so the conditioning improves by roughly a factor Δ.
Choose γ·Δ·Δt ≈ 0.05–0.1 (here Δ = 16: 0.08), well below the velocity memory
1/(γ·Δt) = 200 frames and the typical time to reach A or B. From a cold start the first
~2000 lag-16 iterations overshoot (error 0.34, then 0.20 per bin) before converging.

**What "lag Δ" means.** It is not subsampling: every frame is still a window start. Each
frame t is compared with the frame Δ steps later, min(t + Δ, τ_t), where τ_t is the first
A/B frame or the end of the piece; the windows overlap. Subsampling every Δ-th frame
would target the same condition with 1/Δ of the windows, and would miss A/B visits between
kept frames. (For the 2D Brownian Z MD, "lag 10 / 50" counts saved rows of 20 steps.)

**Diagnostics of a lag-Δ fit.** It only enforces the Δ-step condition, so q can carry
small wiggles that cancel over Δ steps. Z_q (target N_AB = 5007) was flat at 5005–5020 for
all lags ≥ 16, but overshot at lags 1–8 around q ≈ 0.3–0.6 (up to ~6100 at lag 1), and
`max_sd_zq` rose to ~340 (lag-1 fit: ~50) although q became more accurate. Judge a lag-Δ
fit by Z_q and `zqs_z` at lags ≥ Δ, by cross-entropy, and by obs-vs-pred (on the diagonal,
AUC 87.37 %). A grid reference has the same short-lag artifact (§10, test caveat).

**What did not cure it (lag 1).**
- `gamma` (regularization): only scales the step. At the fixed point the Galerkin condition
  holds whatever its value, so it does not bias the result. γ_reg = 0 diverged from a cold
  start; 0.1–0.2 are stable; lowering it later (two stages) gained little.
- Envelope 0.01 instead of 0.001: same speed early, then stalled at a worse point
  (`sdzq` ~72 vs ~47 at 10000 iterations).
- An extra CV aligned with the slow directions (E = ½v² + V(x)) roughly halved the error in
  the same number of iterations, but did not remove the bands.
- NPq (`Committor`): not valid for (x, v). It minimizes Σ (q(t+1) − q(t))², which gives the
  committor only for dynamics that is reversible in the CVs; under time reversal v → −v and
  q(x, v) ≠ q(x, −v). A larger lag does not change that. NPq at any lag is fine for
  positions-only RCs of reversible (e.g. Brownian) dynamics.

**Histories.** A history pair (x(t−d), x(t)) supplies velocity *information* when v is not
recorded, in a worse form than v itself (a difference of nearly equal numbers, half a step
or more stale); it does not touch the conditioning. The cure for the slow convergence is
the lag of the *fit*, not the lag of the *features*. Coarser histories (d = 4, 8, 16, or
0 1 4 16) on the 2D Langevin Z staple run did not help either: `zqs_z` 30–37 vs 33 for
`[0, 1]` (`examples/figures/zlgv_history/`).

### 10.3 Seed and start dependence of the lag-1 (x, v) fit (1D well MD)

Notebooks `examples/base_documented_1d_langevin.ipynb`, `-diff.ipynb`,
`_dt5e-4.ipynb` with the shared loop `examples/seed_runs.py`; figures
`examples/figures/langevin_seeds/`, `langevin_seeds_dt5e-4/`. All: `CommittorNE`, CVs
(λ, recorded v), library lag-1 update, gamma 0.1, envelope 0.01, ny 6, lazy basis, 15000
iterations; the seed sets which CV and envelope centre each iteration draws. RMS per frame
against the brute-force reference:

| Data | Warm-up (1000 it., envelope 1) | Seeds → RMS |
|---|---|---|
| dt 0.005, 6M frames | none | 0: 0.052; 1: 0.019; 2: 0.041; 3: 0.041 |
| same | λ only | 4: 0.031; 5: 0.023 |
| dt 0.0005 BBK, 1 row per 10 steps, 5–7M rows | v only | 4: 0.020; 8: 0.026; 9: 0.022; 10: 0.025 (at 11000 it.) |
| dt 0.005, 6M frames, **gamma 0** in the main run | 1000 or 2000 it. | 99: 0.035 / 0.033 (RMS rises again after 6000 it. with the 1000-it. warm-up: 0.026 → 0.035) |

- **The lag-1 result depends on the start.** Seed 1, whose first iterations drew v, reached
  RMS 0.014 per bin and 0.025 on the E ≈ 0.15 band (`examples/figures/well_md_fit/compare_seed.py`,
  `compare_seed1.png`): as good as the lag-16 fit (0.011 / 0.027), cross-entropy 0.3949
  (exact 0.3948). Seeds that drew λ first keep the sharp edges at λ ≈ ±0.29 and stall.
- A warm-up that updates only along v narrows the spread to RMS 0.020–0.026.
- So the §10.1 statement "lag 1 is impractically slow" holds for unlucky starts; a lucky
  or v-first start converges at lag 1. Treat disagreement between seeds as non-convergence,
  and compare by cross-entropy or `zqs_z`. The lag-16 fit was run with one seed only;
  a multi-seed check is open.

### 10.4 Integrator and time step (1D well MD)

- **10× smaller time step:** a new BBK run with dt 0.0005 (`md_istarwell2_dt5e-4_10M`,
  saved every 10 steps) gives the same MD-outcome committor as dt 0.005: RMS per bin
  0.016 against a combined noise of 0.017 (4211 vs 4162 A→B in 5M rows;
  `examples/figures/langevin_dt5e-4/md_committor_check.png`). The integrator step does not
  cause the stalled seeds.
- **Euler–Maruyama vs BBK** (`examples/figures/well_md_fit/compare_md_committors.py`,
  `md_committor_euler_vs_bbk.png`, 3M rows each): MD-outcome committors agree within noise
  (EM − BBK RMS 0.020 vs noise 0.023; each − reference 0.017).
- **Why histories work on Euler–Maruyama data but not on BBK data:** in Euler–Maruyama,
  x(t) − x(t−1) = v·dt exactly, so the history pair (x(t−1), x(t)) is the full state. In BBK
  the position step carries its own noise (~6% of the step) and tracks the half-step
  velocity, so a history fit converges to a slightly wrong fixed point (§10).
- The harmonic walls of the engine outside the well do not affect the results: fit errors
  and MD-outcome − exact (±0.002) do not depend on whether a frame was bounced off a wall
  (chat analysis, no saved script).

### 10.5 Curved-channel toy potential (MD, Brownian and Langevin)

A smooth 2D test system made for checking the method without the Z potential's flat
plateau and hard walls: `toytis/potentials/curved_channel.py`, `CurvedChannelDoubleWell`
(4 kT barrier at kT 0.05; wells at y = 0.1 and 0.9; the channel runs diagonally and bows
out, so the isocommittor lines are curved and tilted relative to y). Tools:
`examples/toy2d.py` (toytis integrators for many walkers, exact overdamped committor,
shooting, lag-Δ update); notebooks `toy2d_md.ipynb` (MD) and `toy2d_staple_analysis.ipynb`
(staple runs). MD (6 Nov, `toy2d_md.ipynb`, `toy2d_md-lgv.ipynb`): 128 walkers × 10⁶ steps,
dt 0.005, saved every 50 steps (2.56M frames), A: y ≤ 0.15, B: y ≥ 0.85.

| Dynamics | Transitions | Fit | Result |
|---|---|---|---|
| Brownian, γ 10 | 494 A→B; MD/TPT rate 0.96 ± 0.04 | NPq on (y, x), envelope 0.01, gamma 0.1, 1000 + 6000 it. | RMS vs exact 0.0067 per bin, 0.0055 per frame; vs shooting 0.011 (shooting error 0.008) |
| Langevin, γ 2 | 1607 A→B | NPNEq on (y, x, v_y, v_x), lag 1 saved frame (γ·dt_saved = 0.5), warm-up at envelope 10⁶ | vs shooting 0.014 (shooting error 0.008); vs MD outcome per bin 0.042, below its own noise (≥ 0.048) |

The Brownian fit is converged after the warm-up; the Langevin fit after ~3000 iterations (vs shooting 0.070 after the warm-up, 0.016 at 3000). Staple runs of this potential (Brownian
and Langevin, launch commands in the notebook header) have not been analysed yet; on
2000-cycle test runs the staple notebook gave `zqs_z` 1.05 and RMS 0.015 vs exact
(Brownian, 1200 iterations) and RMS 0.027 vs shooting (Langevin, lag 5).

### 10.2 2D Langevin Z potential, staple paths, positions only

`sim_istarz_2603` (γ = 2, kT = 0.05, Δt = 0.002, no velocities in `order.txt`), 1000 paths per
ensemble (8M frames), no mirrored pieces, `examples/figures/zlgv_fd/`. Settings: gamma 0.2,
envelope 0.001, ny 6, A/B traps, seed 0; history rows: library update, lazy, 10000
(`[0, 1]`, notebook) or 2000 iterations (coarser, `figures/zlgv_history/history_test.py`);
finite-difference rows: 3000 iterations, lag 1 library update, lag 4 stopped-window update
(`fd_test.py`):

| CVs | Fit lag | `zqs_z` (max over lags) | Worst \|z\| at lags 1, 2, 4, 8, 16, 64, 256, 1024 |
|---|---|---|---|
| y, x with history `[0, 1]` | 1 | 33 | (lag 1 fine, lags ≥ 2 fail) |
| y, x, Δy, Δx (finite differences) | 1 | 29 | 4.3 6.1 11.8 17.2 21.6 21.8 16.9 28.9 |
| y, x, Δy, Δx | 4 | 30 | 3.4 5.2 6.4 7.5 8.6 11.7 23.7 30.2 |
| y, x, Δy, Δx, **y + x, Δy + Δx** | 4 | **14** | 2.3 1.3 2.1 3.7 6.8 14.1 10.7 8.2 |

The one-frame change of a position is ~4·10⁻⁴ of its range here (1D well: 2·10⁻³), so a
history basis barely reaches the velocity. The 45° ridge needs y and x combined: with
the ridge-normal coordinate and velocity as CVs the stopped Z_q stays within ±2.5 except
at the plateau level (q ≈ 0.73, up to +14 at lags 16–1024). Not converged at 3000
iterations; no exact reference for this run.

**Mirrored pieces with the best setup** (finite-difference velocities, y + x, lag 4): not run
yet. `examples/base_documented_2d_langevin.ipynb` now runs exactly this setup with
`REVERSED = True` (switches `FD_VELOCITY`, `ADD_DIAG`, `FIT_LAG`); compare its worst |z| per
lag with the last row of the table.

**Recommendations**

1. **Record velocities** (`v_ord = True`) and train `CommittorNE` on the full state (x, v)
   (no history). Not NPq.
2. **Fit at lag Δ with stopped windows**, γ·Δ·Δt ≈ 0.05–0.1 (§10.1); judge `zqs_z` / Z_q at
   lags ≥ Δ. A lag-1 fit has the right answer but converges impractically slowly at low
   friction. `lazy_basis=True` (much faster and less GPU memory on millions of frames).
3. **Positions only:** finite-difference velocities as explicit CVs (`--fd-velocity`, which
   now adds one per position column), with a lag-Δ fit; no histories. In 2D, add CVs aligned
   with sharp features (ridge-normal coordinate and velocity).
4. The lag-Δ fit (Σ f_k(t)[q(min(t+Δ, τ_t)) − q(t)] = 0) is not in the library yet: it
   exists only in test scripts (`examples/figures/well_md_fit/lag_xv.py`,
   `examples/figures/zlgv_fd/fd_test.py`, `examples/figures/well_check/well_test.py`).
5. Not helpful (tested): smaller friction γ, higher basis degree, a joint (r, x, v) basis,
   extra CVs (x + v/γ, arrival time at A), coarser histories, envelope 0.01, lowering the
   `gamma` regularization (§10.1).

---

## 11. What was dropped or is still open

- **ROC/AUC, cross-entropy, mse:** no longer printed or plotted for path data; they need a
  real A/B outcome per segment. Possible fix (not implemented): use the iSTAR committor of
  the (interface, direction) state where each segment ends as an RC-independent soft
  outcome; mse/ce then remain fair, ROC via sample weights y and 1−y. Valid under iSTAR's
  memory-loss assumption.
- **`feps.png`:** F(q) reflects path-sampling density, not equilibrium; its Z_q panel is
  the old all-lag version. Use `tis_diagnostics.png` instead.
- **Staple weights:** rejection weights with both pieces pass on the 2D Brownian Z and 1D
  flat staple runs (§9.1). Why (the relation to the high-acceptance factor) is not yet
  derived from the staple move rules. Check other staple systems with the exact-committor
  test before relying on it.
- Validated on 1D toy systems with a known committor: flat overdamped, flat underdamped,
  and an underdamped cosine well.
- **Lag-Δ fit (`fit_lag`) and `zqs_z` from lag Δ on:** needed for every low-friction system,
  also with the full state (x, v) (§10.1); only in test scripts, not yet in the library.
- **Seeds:** the lag-1 (x, v) fit depends on the start (§10.3); check the lag-16 fit with
  several seeds.
- **Mirrored pieces on underdamped staple data:** recorded v on the well done (§10 table);
  finite-difference velocities on the well and the best 2D Z setup (fd, y + x, lag 4) not run
  yet (notebooks `base_documented_1d_langevin_staple.ipynb`, `base_documented_2d_langevin.ipynb`).
- **2D Brownian Z staple:** repeat the per-block z-score test on the full run (§9.2), and
  fit an independent subset (the path-seed-1 subset is not independent).
- **A convergence test that sees levels** (§6.1): `zqs_z` only checks self-consistency;
  combine it with shooting at a few hundred frames, independent halves, or the TIS
  crossing probabilities.
- **Curved-channel staple runs** (§10.5): analyse the full Brownian and Langevin runs.

---

## 12. Code map

**Library (both backends: `optimalrcs/` TensorFlow, `optimalrcs-cupy/` CuPy)**

| File | What |
|---|---|
| `metrics_tis.py` | `stop_index`, `comp_zq_stopped`, `comp_zq_stopped_zscore`, `comp_obs_pred_stopped`, `dropped_fraction`, metrics `max_z_zq_stopped` / `max_sd_zq_stopped` |
| `plots_tis.py` | `plot_zq_stopped`, `plot_zq_short_lags`, `plot_obs_pred_stopped` |
| `nonparametrics.py` | `npneq(..., w_traj=)` per-step weights |
| `optimalrcs.py` | `CommittorNE(..., path_weights=, group_traj=)`, `fit_transform(save_min_metric=)`, `plots_tis()`, `set_boundary_traps()` (§8.1), metric registry includes `metrics_tis` |
| `boundaries.py` | `FutureBoundary.set_distance_to_end_trap`, `PastBoundary.set_distance_from_start_trap` |

**Examples (`examples/`)**

| File | What |
|---|---|
| `tis_data.py` | shared loader: interfaces, run settings (`dim`, `v_ord`), weights, cut at M (`first_frame_past_m`), reversed pieces (`last_frame_before_m`), data preparation, figures |
| `load_tis_data.py` / `load_tis_data_cupy.py` | fit scripts (TF / CuPy) |

**Tests:** `tests/test_metrics_tis.py`, `tests/test_boundary_traps.py`, plus CuPy-vs-TF checks in `tests/test_cupy.py`.

**Command-line options added**

| Option | Default | Meaning |
|---|---|---|
| `--weights rejections\|staple\|none` | rejections | MC path weights |
| `--no-cut` | off | keep whole paths (comparison only, biased) |
| `--reversed` | off | add the mirrored pieces; recommended for REPPTIS and staple (§9, §9.1) |
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
- 1D overdamped MD: the engine has harmonic walls (k = 100) outside [−0.2, 0.4], so the FEP
  rises beyond them; this does not change the committor between A and B.

## 14. Fit settings of the runs

**Defaults** (checked in `optimalrcs-cupy/optimalrcs.py`): `envelope_scale` 0.01 (module
global); `CommittorNE.fit_transform`: `gamma` 0, `ny` 6, `lazy_basis` False,
`gamma_all_frames` False. NPq (`Committor`) ignores `gamma`. `save_min_metric` only stores an
extra copy of q that no script reads: every reported q is the last iterate.
`load_tis_data_cupy.py` defaults: gamma 0.1, ny 6, 10000 iterations, `min_delta_x` 1e-6, no
history, envelope 0.01, A/B traps, rejection weights, cut at M, forward pieces, no seed.

**Terms.** *Update*: "library" = the library's lag-1 NPNEq/NPq step; "stopped Δ" = the
hand-written stopped-window update Σ_t w_t f_k(t)[q(min(t+Δ, τ_t)) − q(t)] = 0 (§10.1), also
used at Δ = 1 in some scripts. *Basis*: lazy = `lazy_basis=True`; chunked = the scripts'
`LazyRY`; full = `basis_poly_ry` (same numbers, more memory). *Traps*: FTL =
`set_fixed_traj_length_trap(A|B)`, BT = `set_boundary_traps`; both change only the metrics.
"fwd" = forward pieces only, "+mir" = with mirrored pieces (`--reversed`). Paths are subsets
of each ensemble drawn with seed 0 unless noted.

| Run (section) | Data | CVs | Update, lag | gamma | Envelope | ny | Iterations | Basis | Traps | Seed | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1D overdamped MD, flat / two bumps (overview I.1) | 50M frames, 5M-frame blocks | λ | NPq library, 1 | – | 0.01 | 6 | 1500 per block | full | – | block | `figures/longmd_npq/npq_longmd.py` |
| 1D flat staple `sim_istar0br10intf` (II.1) | 12000 paths/ens., rej., fwd | λ | library, 1 | 0.1 | 0.01 | 6 | 3000 | lazy | FTL | 0 | `figures/overview_plots/regen_all.py flat` |
| 1D flat staple, §3/§4 fits; REPPTIS and small staple subsets (§9 table) | not recorded | | | | | | | | | | no script found |
| 1D well MD, x + history `[0, 1]` (§10.1) | 6M frames | x, `y(t-d),y(t)` | library, 1 | 0.2 | 0.001 | 6 | 1500 (overview) / 3000 (§10.1, not recorded) | lazy | – | 0 | `regen_all.py hist01` |
| 1D well MD, (x, v) (§10.1) | 6M frames | (x, v) | library, 1 | 0.2 | 0.001 | not recorded | 10000 | not recorded | – | not recorded | earlier `base_documented_1d_langevin.ipynb` |
| 1D well MD, (x, v, E) (§10.1) | 6M frames | (x, v, E) | library, 1 | 0.2, then 0.05 | 0.001 | 6 | 10000 + 5000 | lazy | – | 0 per stage | `figures/well_md_fit/energy_cv.py` |
| 1D well MD, (x, v), long (§10.1) | 6M frames | (x, v) | library, 1 | 0.1 | 0.001 | 6 | 95000 (stop: cross-entropy gain < 3e-5 over 3 × 5000) | lazy | – | chunk | `long_xv.py` |
| **1D well MD, (x, v), lag 16** (§10.1) | 6M frames | (x, v) | stopped Δ = 16 frames | 0.1 | 0.001 | 6 | 12000 | chunked | – | chunk | `lag_xv.py` |
| Flat low-friction staple `sim_istar0lgv15intf` (§10 table) | 100 or 30 paths/ens., rej., fwd | x, Δx | library, 1 / stopped Δ = 4 | 0.1 | 0.01 | 6 | 4000 | full | FTL / – | 0 | `figures/lgv_check/fit_and_test.py`, `lag_fit.py`, `lag4_stats.py` |
| 1D well staple `sim_istarwell2rv` (§10 table) | 2000 paths/ens., rej., fwd or +mir | (x, v) recorded or fd | stopped Δ = 1 or 4 | 0.1 | 0.01 | 6 | 4000 | full | – | 0 | `figures/well_check/well_test.py` |
| 1D well staple, library refit (overview II.2) | 2000 paths/ens., rej., fwd | (x, v) recorded | library, 1 | 0.1 | 0.01 | 6 | 4000 | lazy | FTL | 0 | `regen_all.py wellpath` |
| 2D Brownian Z MD, NPq (overview I.2) | 16 seeds, rows every 20 steps, excursions (4.6M rows) | (y, x) | NPq library, 1 row | – | 0.01 | 6 | 8000 | full | BT | 0 (halves 1, 2) | `figures/z_brownian/npq_md.py` |
| 2D Brownian Z MD, NPNEq lag fits (I.2) | seeds 1–8 (2.4M rows) or 9–16 | (y, x) | stopped Δ = 1, 10, 50 rows | 0.1 | 0.01 | 6 | 5000 (seeds 9–16: not logged) | chunked | – | chunk | `lag_md.py` |
| 2D Brownian Z MD, envelope schedule (I.2) | seeds 1–8 | (y, x) | stopped Δ, value not logged | 0.1 | 1 → 0.1 → 0.01 → 0.001 | 6 | 1000 + 1000 + 1000 + 2000 | chunked | – | stage | `lag_md.py`, `env_sched.log` |
| 2D Brownian Z staple, fwd (§9.1, II.3) | 3000 paths/ens. (subset seed not recorded), rej., fwd, 20.8M frames | (y, x) | library, 1 | 0.1 | 0.01 | 6 | 10000 | not recorded | FTL | none | `load_tis_data_cupy.py`, `istarz_br_g10/run.log` |
| 2D Brownian Z staple, y + x (II.3) | the fwd fit's frames | (y, x, y + x) | library, 1 | 0.1 | **1** | 6 | 22752 (stopped by hand) | lazy | FTL | 0 | `base_documented_2d_brownian_staple_diag.ipynb` |
| **2D Brownian Z staple, +mir** (§9.1, §9.2, II.3) | 1500 paths/ens., rej., +mir, 20.9M frames | (y, x) | library, 1 | 0.1 | 0.01 | 6 | 10000 + 15000 | not recorded, then lazy | FTL | none, then 100 + chunk | `load_tis_data_cupy.py`, `istarz_br_g10_reversed/continue_fit.py` |
| 2D Brownian Z staple halves, lag 1 (§9.2) | even / odd paths of the +mir data | (y, x) | library, 1 | 0.1 | 0.01 | 6 | 25000 / 22500 (stop: plateau change < 0.001 and \|dx\| < 2e-4) | lazy | FTL | chunk | `half_fits.py` |
| 2D Brownian Z staple, lag 50 (§9.2) | halves of 1500 paths/ens., or 2250 paths/ens. (31.5M frames) | (y, x) | stopped Δ = 50 frames | 0.1 | 0.01 | 6 | 3000 | chunked | – | chunk | `lag_halves.py` |
| Synthetic walkers (§9.2) | 30000 walkers, every 50 steps (5.2M frames), uncut or cut at interfaces (0.5M) | (y, x) | library, 1 / stopped Δ = 10 rows | 0.1 | 0.01 | 6 | uncut 2500, cut 10000, lag 10 7500 | lazy / chunked | FTL | chunk | `figures/z_brownian/synth_fit.py` |
| 2D underdamped Z staple, history (§10.2) | 1000 paths/ens., rej. (`[0, 1]`: possibly +mir, see II.4) | y, x, history | library, 1 | 0.2 | 0.001 | 6 | 10000 (`[0, 1]`), 2000 (coarser) | lazy | FTL | 0 | `base_documented_2d_langevin.ipynb`, `figures/zlgv_history/history_test.py` |
| 2D underdamped Z staple, fd (§10.2) | 1000 paths/ens., rej., fwd, 8M frames | y, x, Δy, Δx (+ y + x, Δy + Δx) | library, 1 / stopped Δ = 4 | 0.2 | 0.001 | 6 | 3000 | lazy / chunked | FTL | 0 | `figures/zlgv_fd/fd_test.py` |
| 1D well MD, seed runs (§10.3) | 6M frames (dt 0.005) or 5–7M rows (dt 0.0005) | (λ, v) | library, 1 | 0.1 (warm-up 0.1) | 0.01 (warm-up 1) | 6 | 15000 (+ 1000 warm-up) | lazy | – | 0–10 | `base_documented_1d_langevin*.ipynb`, `seed_runs.py` |
| 2D Brownian Z MD, NPq seed runs (§9.1) | as NPq above | (y, x) | NPq library, 1 row | – | 0.001 (warm-up 1) | 6 | 15000 + 1000 | full | BT | 3, 6 | `base_documented_2d_brownian_md.ipynb` |
| 2D Brownian Z staple, notebook seed runs (§9.1) | fwd 3000 paths/ens., or +mir 2000 paths/ens. (path seed 1) | (y, x) | library, 1 | 0.1 | 0.01 (warm-up 1) | 6 | 15000 + 1000 | lazy | FTL | 0, 3 | `base_documented_2d_brownian_staple.ipynb` |
| Data preparations (§9.3) | 1D flat 1000 paths/ens. (seed 0); 2D Z 1000 paths/ens. (seed 12) | λ / (y, x) | library, 1 | **0** | 10⁶ (warm-up 1000), then 0.01 | 6 | 1D 4000, 2D 15000 | lazy | FTL | 0 | `staple_data_choices.ipynb` |
| 2D Z MD, new 1.06·10⁹-step run (§9.1) | 4.46M kept rows, 16 A→B | (y, x) | NPq library, 1 row | – | 0.001 (warm-up 1) | 6 | 1000 + 15000 | full | BT | 6 | `base_documented_2d_brownian_md.ipynb` |
| Curved channel MD (§10.5) | 128 × 10⁶ steps, every 50 (2.56M frames) | (y, x) / (y, x, v_y, v_x) | NPq / NPNEq library, 1 | 0.1 | 1 or 10⁶ (warm-up 1000), then 0.01 | 6 | 1000 + 6000 | full / lazy | BT | 0 | `toy2d_md.ipynb`, `toy2d_md-lgv.ipynb` |
| Notebooks for the +mir tests (not run yet) | well: 2000 paths/ens.; 2D Z: 1000 paths/ens.; +mir | as above | stopped Δ = 4 (or library, 1) | 0.1 (well) / 0.2 (2D Z) | 0.01 / 0.001 | 6 | 4000 / 3000 | full / chunked | FTL | 0 | `base_documented_1d_langevin_staple.ipynb`, `base_documented_2d_langevin.ipynb` |
