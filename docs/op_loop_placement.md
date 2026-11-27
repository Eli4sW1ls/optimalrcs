# The loop and its interface placement, step by step (state of 2026-11-27)

This note lists every step of the order-parameter loop and every criterion that places, keeps or rejects an
interface, with the exact quantity that is measured, its limit, the config key and the code that does it. It
describes the code in `examples/param_lambda/loop/` as it is now, with the values of `configs/loop_z_dp.toml`
(Brownian Z potential, dp placement). Changes of 2026-11-26/27 (separate skip limits for one and two skipped
interfaces, a free gap multiple for the placement, the uniformity rule of the evening step, the reasons for refused
splits in the log) are marked **[11-27]**; the settings that differ in `configs/loop_z_dp_u.toml` are collected in
2.7. Where path length enters the placement, and where it does not: 2.8. Overview and results of all systems: [`op_loop_report.md`](op_loop_report.md).

Notation:
- λ: the learned order parameter. λ_A and λ_B are the state boundaries (fixed: λ equals the state CV y in the states).
- λ_0 = λ_A < λ_1 < … < λ_{n-1} = λ_B: the interfaces. The pair (i, i+1) is two neighbouring interfaces.
- h(λ): the model's committor as a function of λ (`LambdaModel.h_of_lambda`); λ is a monotone rescaling of h.
- Ensemble e (infretis column): [0−] is e = 0, [0+] is e = 1 with window (λ_0, λ_1). For e ≥ 2 the staple ensemble
  has middle interface M = λ_{e−1} and window (L, R) = (λ_{e−2}, λ_e).
- Frame: one saved MD frame (here every saved step). An RMS step: the root-mean-square change of the state CV per
  frame, σ_s, measured on the paths.
- Counts of paths are unweighted. Fractions that the gate tests (skip rates, empty regions) are weighted by the
  path weight (path_f).

## Part 1: the loop

| Step | What it does | Main settings (loop_z_dp) | Code |
|---|---|---|---|
| 0. Pass 0 | inftools infinit: RETIS iterations with λ = y until 20 paths reached B | 200 steps/iteration, ≤ 12 iterations, pL 0.3 | `backends/infretis_backend.py`, `pass0_infinit.py` |
| 1. Pool | every path of every pass, cut at its first crossing of the interface below its ensemble, with its CVs and weight | weight path_f, burn-in 20% after a λ switch, ≤ 1500 paths per ensemble | `pool.py` |
| 2. Teacher | non-parametric committor fit (NPNEq) on all pooled frames | 6000 iterations, degree 6, ridge 0.1, CVs y, x, u, v | `teacher.py` |
| 3. Student | MLP fit of logit q to the teacher, exported as λ(x) | 64×64 SiLU, 24 epochs | `student.py`, `lambda_runtime.py` |
| 4. Diagnostics | SD(Z_q) of the new and the previous model, stability, out-of-distribution share, exact error (Z has an exact committor) | | `diagnostics.py` |
| 5. Interfaces | placement on all stored paths, see Part 2 | `method = "dp"` | `interfaces.py`, `passes.py: _place_dp` |
| 6. Seeds | a valid initial path per ensemble, cut from stored paths, newest passes first | | `seed.py` |
| 7. Sampling pass | infretis StapleTIS, infinite swap, λ frozen during the pass | 1000 cycles, 4 workers | `passes.py: run_and_check` |
| 8. Re-check | the gate of Part 2.4 on the pass's own accepted paths, plus accepted paths with an empty shooting region | | `passes.py: run_and_check` |
| 9. Pause | figure of the pass, then wait for `python -m loop.confirm loop_z_dp continue\|stop` | `[run] confirm_passes`, `pass_figures` | `passes.py: _confirm` |
| 10. Stop test | all five criteria of 3.1 hold (from pass 2 on); otherwise another pass, at most `max_passes` | `[stop]` (loop_z_dp: 7 passes; loop_z_dp_u: 10 passes, `interface_tol` 10%) | `passes.py: stop_test` |
| 11. Freeze | λ fixed; interfaces re-placed and verified, see Part 3 | loop_z_dp: 2 rounds of 2000 cycles; loop_z_dp_u: 3 rounds of 3000 | `passes.py: freeze` |
| 12. Production | StapleTIS run with the frozen λ, interfaces and seeds; P_cross and rate from inftools iSTAR | 20 000 cycles | `controller.py: run_production`, `rates.py` |

Steps 0–4 and 6–12 are described in more detail in [`op_loop_report.md`](op_loop_report.md), section "The method".
Part 2 covers step 5 and the gate used in steps 8 and 11.

## Part 2: interface placement (`[interfaces] method = "dp"`)

Placement runs after every fit, on all stored paths of the pool, each replayed under the new λ
(`interfaces.path_arrays`, at most 5000 paths). It has four stages:

1. **dp:** dynamic programming chooses a whole set at once, aiming at crossing probability `target_p` per pair.
2. **evening step:** splits ensembles whose shooting region is much longer than typical, and merges ones much shorter.
3. **balance step:** moves single interfaces to make the shooting-region lengths more even.
4. **gate:** the full check. If the result fails it, the loop falls back to the older plan-and-thin method (Part 2.6).

### 2.1 What a pair's crossing probability is

The local crossing probability of a pair is p_i = P(a path that crossed λ_i reaches λ_{i+1} before A). With
committor-level interfaces it follows from the committor levels h_i = h(λ_i) (`q_ratio`, `_pair_p`):

- both levels below ½: p = h_i / h_{i+1};
- both above ½: p = (1 − h_{i+1}) / (1 − h_i);
- a pair that straddles ½: the smaller of the two.

The end pairs use effective levels for the states (`state_levels`). h_A is the weighted mean of h at the first frames
past λ_A; h_B is the same for the first frames below λ_B. Since h(λ_A) = 0 itself would make the first pair look
infinitely wide.

**The target is per pair.** `target_p` = 0.3 is the aim for each p_i separately. The product of all p_i is the total
crossing probability P_A(λ_B), which is fixed by the system, so the target sets the number of interfaces:
n ≈ ln P_A / ln target_p.

Why about 0.2–0.3: with N paths split evenly over n ensembles and equal cost per path, the relative variance of P_A
is roughly Σ_i (1 − p_i) / (p_i N/n) = n² (1 − p) / (p N). With n = ln P_A / ln p this is proportional to
(1 − p) / (p ln² p). That is smallest at p ≈ 0.2 and flat: within 5% for p between 0.15 and 0.3. Longer paths in
some ensembles and correlated paths (the statistical inefficiency, which varies by ensemble) shift this. The
**evening** and **balance** steps (2.3, 2.4) handle the path-length side.

### 2.1b Measured levels (`[interfaces] levels = "measured"`, added 2026-11-24)

The model's committor levels can be off even where the surfaces are good: in z_lgv by up to 10× in the middle; in
the Brownian Z by about 5× at the first interface. With `levels = "measured"`, placement re-maps them before dp,
evening and balance (`level_model`, `measured_levels`, `LevelMap`).

1. **Measured levels.** inftools iSTAR on the last run (`rates.staple_rate`) gives P_A(λ_k) at its interfaces. For
   isocommittor interfaces the committor is a martingale, so q_k = P_A(λ_B) / P_A(λ_k) exactly (gambler's ruin).
2. **Model levels of the same surfaces.** The median, in logit, of the model's level h(λ(x)) at the frames just past
   each up-crossing of the run's interface k. This also works when λ changed since that run.
3. **The map.** Piecewise linear in logit through the points (model level, measured level), both made
   non-decreasing. Outside the measured range: the model's logit, shifted by the offset of the nearest point. The
   surfaces stay; only the levels assigned to them change. Interfaces with fewer than 20 up-crossings are left out.

Which run is used:
- in a pass: the newest pool pass's run (the previous pass);
- in a freeze: the failed verification run, else `levels_run` (a run directory in the loop directory, e.g. a finished
  production with the same λ), else the newest pool pass.

If the run can't be analysed, placement falls back to the model's levels and logs why.

Caveats:
- Every measured level is divided by the run's P_A(λ_B), so it carries that run's P_cross error: 30–100% for a
  2 000-cycle run. A pair's p, below ½, is P_A(λ_j) / P_A(λ_i) and doesn't depend on P_A(λ_B). But the logit map does,
  and so does the p of pairs above ½.
- Brownian Z, calibrated on the 12-interface production: measured/model = 0.20 at λ 0.067, 0.65–0.89 from 0.094 to
  0.822, 1.01 near B. On two independent short runs the map halved the error on one and changed nothing on the other.
  On the old 9-interface production it was worse (logit error 0.57 against 0.33).

**Forward vs. B-side p.** Above q = ½, `q_ratio` measures a pair by the B-side ratio (1 − q_i) / (1 − q_j). That is
the crossing probability from B, not the forward one. A production's measured local p is forward: q_i / q_j for
isocommittor surfaces. So near B a pair placed at B-side p ≈ 0.3 shows forward p of 0.8–0.98. This is by design, not
a calibration error.

### 2.2 Stage 1: dp (`place_dp`)

**Candidate levels.**
- 120 committor levels, equally spaced in logit h between h_A and h_B, mapped to λ.
- Two duplicates closer than 1e-6 × (λ_B − λ_A) count as one.
- λ_A and λ_B are always in the set.

**Objective.** Over a chain λ_A → … → λ_B, minimise Σ_pairs (ln(p_i / target_p))². A pair with p twice the target
costs as much as one with half of it.

**Hard limits on a pair (i → j).** A forbidden pair is never used. All limits use the placement margin (2.5).
- **skip up:** of the up-crossings of λ_i, the weighted fraction whose next frame lands at or past λ_j must be
  ≤ `skip_eps` / `place_margin` (default `eps` / 1.25 = 0.8%; loop_z_dp_u: 5% / 1.5 = 3.3%). Applied only where λ_i
  has ≥ 100 up-crossings.
- **skip down:** of the down-crossings of λ_j, the fraction whose next frame lands below λ_i must be within the same
  limit, where ≥ 100 down-crossings exist.
- **[11-27] two skipped:** a window (λ_i, λ_k, λ_j) is banned when a step that crosses λ_i upward lands at or past
  λ_j, or one that crosses λ_j downward lands below λ_i, in more than `skip2_eps` / `place_margin` of the crossings
  (loop_z_dp_u: 1% / 1.5 = 0.67%; ≥ 100 crossings). This is the same quantity as the gate's `skip2_up` / `skip2_dn`
  (2.5). It is applied as a window ban, so the dp must route around it.
- **gap:** λ_j − λ_i ≥ `place_c` × δ_i if `place_c` is set, otherwise c × `place_margin` × δ_i (loop_z_dp: 1.25 × 1 =
  1.25; loop_z_dp_u: 1.5 × 0.6 = 0.9). δ_i is the 95% quantile of |Δλ| over the steps that cross λ_i.
  - It is waived where λ_i has ≥ 100 up-crossings (`gap_if_unmeasured = true`): there the measured skip rate
    decides. The gap rule is only a proxy for the skip rate where it can't be measured.
  - **[11-27]** `place_gap_always = true` keeps the gap rule in the placement even where the skip rate is measured.
    The re-check (2.5) keeps its waiver either way.
- **distance from A** (first pair, A → λ_j): the 5% quantile of (y − λ_A) at the first frames past λ_j, divided
  by σ_s, ≥ n_a × 1.25 = 3.75 RMS steps.
- **distance from B** (last pair, λ_i → B): the 5% quantile of (λ_B − y) at the first frames below λ_i ≥ 3.75 RMS
  steps.

**Window check on the solution.**
- The full gate (2.5) runs on the solved set.
- A window that fails the empty-region or region-frames test is banned: the DP tracks (previous, current) pairs, so a
  window (λ_{e−2}, λ_e) can be excluded. Then the problem is solved again.
- At most 25 rounds.

### 2.3 Stage 2: evening step (`even_regions`)

**Shooting-region length** of ensemble e (`window_frames`, `region_frames`; changed 2026-11-24).

*Which paths:* the newest run's own accepted paths, i.e. the last pass or the failed verification run
(`region_paths`, `[interfaces] region_passes` = 1), replayed under the new λ. The gate still uses the whole pool.

*How each path is measured:*
1. Cut each stored path into the staple paths it would contain under the trial interfaces. From every run of frames
   inside the window (λ_{e−2}, λ_e) that crosses λ_{e−1}, follow the path outward on both sides until the sampler's
   turn rule fires, or A or B is reached (`seed.extend_turn`, the vectorised infretis `turn_detected`).
2. The turn rule sees the window plus one interface beyond each side. That covers paths that go up to two interfaces
   past the window before turning.
3. A staple path's region is its number of frames strictly inside the window.

*The statistic:* the unweighted mean over the staple paths (`region_stat = "mean"`).

*Why each choice:*
- Unweighted: an ensemble's sampled paths are what cost MD.
- Cut at the trial set's turns: an extra interface ends staple paths sooner. Without this, no split ever looks useful.
- Mean, not median: on the plateau, regions are either ~10 or ~5 000 frames, and a median from a short run flips
  between the two.

*Validation:* the new set's own run, measured this way, against the ensemble paths infretis actually sampled
(mean frames inside the window):

| Ensemble | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|
| sampled | 181 | 466 | 921 | 1 156 | 214 | 19 | 8 373 | 11 044 | 166 | 55 |
| this measure | 208 | 460 | 925 | 951 | 177 | 19 | 8 421 | 9 655 | 164 | 63 |

- Predicted from the old production's paths (other interfaces, same λ): within 1.7× everywhere.
- The former measure (path_f-weighted median per run inside the window, over the whole pool) was up to 100× short
  on the plateau: 82 predicted against 8 210.

**Repeat, at most 12 moves:**
- typical = the median over the ensembles of their region lengths.
- **Split** the longest ensemble if its region > `long_ratio` (2) × typical.
  - Take the wider of its two pairs (in logit h; a pair touching a state is capped at h = 1e-4 so it isn't
    infinitely wide).
  - Candidates: 20 per pair, for both pairs of the ensemble. They are equally spaced in logit h, or in λ for a pair
    touching a state.
  - Drop candidates that make either new pair p > `p_max` (0.9). A pair crossed almost surely adds an ensemble and no
    information.
  - Rank the rest by the longest region of the windows the split creates or changes, shortest first (the measure
    above).
  - Take the first candidate in that order that passes the local gate and the full gate, if its longest region is
    ≤ (1 − `split_gain`) × the ensemble's region.
  - If no candidate gains that much, log a `skip` with the best value.
  - Changed 2026-11-24: before, only the first gate-passing candidate nearest the logit midpoint was judged, and the
    pair was given up if that one didn't gain enough.
  - `split_gain` is now 0.15 in loop_z_dp, was 0.3. In the intermediate basin each split shortens the region by only
    15–30%, but four of them together take it from ~12 000 to ~2 000 frames (Part 4).
- **[11-27] `split_rule = "uniform"`** replaces the `split_gain` test. A candidate's score is the variance over all
  ensembles of ln(region frames) after the split (windows without staple paths are left out, at least two windows
  needed). The best-scoring candidate that passes both gates is taken if its variance is below the current one,
  however small the gain. A split that leaves two long windows instead of one raises the variance and is refused. The
  evening step keeps going (`even_max_moves`, default 12) while an ensemble stays above `long_ratio` × typical, so
  several splits can follow each other. The ensembles tried, the merge rule and the candidate positions are as in
  `"gain"`. With `split_rule = "gain"` (default) nothing changes.
- **[11-27] Log of a skip.** Each split or skip line shows the frames before and after (and, for `"uniform"`, the
  variance before and after). A skip also lists up to three improving candidates that the gate refused, with the check
  that failed (e.g. `gap at 0.3123`), or "no candidate improves".
- **Merge** the shortest ensemble if its region < typical / `long_ratio`: remove its middle interface, if
  - the merged pair keeps p ≥ `merge_p_min` (now **0.2**; the default target_p / 3 = 0.1 let three merges
    in the first freeze drop pairs to p ≈ 0.11), and
  - the full gate passes.
- Each interface is tried once for split and once for merge.

### 2.4 Stage 3: balance step (`balance_regions`)

Moves one interior interface at a time to one of 14 positions equally spaced between its neighbours.

**Objective**, lower is better:
- Σ_e (ln region_e − mean ln region)², the spread of the region lengths on a log scale,
- plus `balance_p_weight` (0.3) × Σ_pairs (ln(p_i / target_p))², which keeps the p's near the target.

**A move is kept only if all hold:**
- it lowers the objective by at least 1%;
- both adjacent pairs keep `balance_p_min` (now **0.2**; default `merge_p_min`) ≤ p ≤ `p_max` (0.9), unless the
  pair was above 0.9 already before the move;
- the local gate passes, including distance from A and from B for the first and last interface (`end_dist`).

At most 3 sweeps over all interfaces; it stops early when a sweep moves nothing.

### 2.5 The gate (`check`, local form `_local_check`)

The gate is the definition of "this interface set can be sampled". It is applied:
- to the stored paths during placement, with stricter limits: eps, skip_eps, skip2_eps and max_empty divided by
  `place_margin`; c, n_a and n_sh multiplied by it (`_place_limits`; `place_c`, if set, replaces c × margin as is). This
  margin is the hysteresis: a set that passes on the pool should still pass on the next pass's own paths;
- to the pass's own paths after each pass, the re-check, with the bare limits;
- to the verification runs of the freeze, also with the bare limits.

| Check | Quantity | Fails if | Default (loop_z_dp) |
|---|---|---|---|
| `skip_up` (per interface k) | weighted fraction of up-crossings of λ_k whose next frame is at or past λ_{k+1} | > `skip_eps` (default `eps`), with ≥ 100 up-crossings | 0.01 (loop_z_dp_u: 0.05) |
| `skip_dn` (per interface k) | the same for down-crossings landing below λ_{k−1} | > `skip_eps`, with ≥ 100 down-crossings | 0.01 (0.05) |
| `skip2_up` (per interface k) **[11-27]** | fraction of up-crossings of λ_k whose next frame is at or past λ_{k+2}: two interfaces skipped in one step | > `skip2_eps` (default `eps`), with ≥ 100 up-crossings | 0.01 (0.01) |
| `skip2_dn` (per interface k) **[11-27]** | the same for down-crossings landing below λ_{k−2} | > `skip2_eps`, with ≥ 100 down-crossings | 0.01 (0.01) |
| `gap` (per interface k) | (λ_{k+1} − λ_k) / δ_k, δ_k = 95% quantile of \|Δλ\| on crossing steps of λ_k (borrowed from the nearest interface with crossings if λ_k has none) | < `c`, and not waived (waived with ≥ 100 up-crossings) | c = 1 (loop_z_dp_u: 0.6) |
| `dist_a` (first interface) | 5% quantile of (y − λ_A) at the first frames past λ_1, in RMS steps σ_s | < `n_a` | 3 |
| `dist_b` (last interior interface) | 5% quantile of (λ_B − y) at the first frames below λ_{n−2}, in σ_s | < `n_a` | 3 |
| `empty_region` (per ensemble) | weighted fraction of excursions through the window (L, R) that cross M with no frame strictly inside; a single step from ≤ L to ≥ R counts as empty | > `max_empty`, with ≥ 20 excursions | max_empty = eps = 0.01 |
| `region_frames` (per ensemble e ≥ 2) | 5% quantile of the frames inside the window per excursion | < `n_sh`, with ≥ 20 excursions | n_sh = 2 |
| `single_step_turns` (whole set) | fraction of staple turns, replayed on λ, completed within one step of the extremum | > `eps`, with ≥ 20 turns | 0.01 |

The local form (`_local_check`, used in evening, balance and refill) tests only what moving or inserting interface k
changes: skip, skip2 and gap of interfaces k−1, k and k+1; the windows whose middle interface is one of those three;
optionally the distances from A and B. An interface with no crossings at all fails: an interface is only placed
where the paths verify it.

**Re-check only:** accepted paths with an empty shooting region (`empty_shooting_regions`, infretis rejects every
shooting move from such a path).
- Without `max_empty` in the config: one such path anywhere fails.
- With `max_empty` set: fails only if an ensemble's fraction exceeds it.

**[11-27] Why 5% and 0.6 in loop_z_dp_u.** A skipped interface costs the sampler a little efficiency and infretis
handles it; two skipped are more delicate and keep the 1% limit. The own-path re-checks of loop_z_dp and loop_z_dp_c
(rows with ≥ 50 crossings, both directions pooled) give the skip rate by gap in units of δ:

| gap / δ | 0–0.6 | 0.6–0.8 | 0.8–1.0 | 1.0–1.2 | 1.2–1.4 | 1.4–1.7 | 1.7–2 | 2–3 | ≥ 3 |
|---|---|---|---|---|---|---|---|---|---|
| rows | 11 | 15 | 20 | 18 | 21 | 21 | 16 | 50 | 126 |
| skip rate | 1.3% | 1.8% | 1.4% | 1.1% | 0.13% | 0.11% | 0.05% | 0.05% | 0.02% |

Below 1.2 δ the skip rate is 1.1–1.8%, above it ≤ 0.13%. With a 5% limit, no skip failure remains in the re-checks of
loop_z_dp_c (the largest was 3.8%). loop_z_dp still has three above 5% (pass 2: 5.2% and 9.8%; pass 3: 7.2%).
A loop_z_dp_c pass-1 row at 0.69 δ had 11% on 69 crossings, below the test's 100-crossing threshold. The gap rule
only matters where crossings are rare.

### 2.6 Fallback: the plan-and-thin method (`place`, `method = "plan"`)

Used when dp finds no set, or the result fails the gate.
1. **Plan** (`plan_natural`): interior levels equally spaced in the natural coordinate u(h) = ∫ dh / √(2 D(h)).
   - D(h) is the per-frame diffusion of h, so each gap takes about the same diffusive time to cross.
   - Pairs more than `max_ratio` (10) apart in crossing-probability ratio are split.
   - In dp mode: `n_interfaces` levels, no growth by path-length budget.
2. **Thin, forward pass:** walk up from λ_A and drop each planned level that fails skip (`skip_eps`), gap or distance
   from A against the last kept one. The rounds of step 3 then also apply the double-skip check. Then drop the last kept interface while it fails against B.
3. **Thin, rounds:** run the full gate and remove one interface per round from the first failing check:
   - a failing window loses its upper interface;
   - `skip_up`, `skip2_up` or `gap`: the next interface goes;
   - `skip_dn`, `skip2_dn` or `dist_a`: that interface goes;
   - `dist_b`: the last interior interface goes;
   - single-step turns: the upper member of the closest pair goes.
4. **Refill:** a pair whose crossing-probability ratio still exceeds 10 is split at the best of 40 logit-spaced
   candidates that passes the local and full gate. At most 10 are added.

### 2.7 Config keys of placement (`[interfaces]`)

| Key | Meaning | loop_z_dp |
|---|---|---|
| `method` | `"dp"` (Part 2.2–2.4) or `"plan"` (2.6) | dp |
| `target_p` | crossing probability aimed at per pair | 0.3 |
| `eps`, `c`, `n_a`, `n_sh` | gate limits (2.5) | 0.01, 1, 3, 2 |
| `skip_eps`, `skip2_eps` **[11-27]** | gate limits for one and for two skipped interfaces (default `eps`) | not set (= eps) |
| `place_c` **[11-27]** | placement gap multiple of δ, applied as is (default: c × `place_margin`) | not set |
| `place_gap_always` **[11-27]** | keep the gap rule in the placement where the skip rate is measured | false |
| `split_rule` **[11-27]** | evening step: `"gain"` (`split_gain`) or `"uniform"` (lowers the variance of ln region frames) | gain |
| `even_max_moves` **[11-27]** | evening step: at most this many moves | 12 |
| `max_empty` | allowed fraction of excursions with an empty shooting region; also switches the re-check of accepted empty paths to a per-ensemble fraction | not set (= eps, strict re-check) |
| `place_margin` | factor of the stricter placement limits | 1.5 (default 1.25) |
| `gap_if_unmeasured` | waive the gap rule where the skip rate is measured (≥ 100 crossings) | true |
| `long_ratio` | evening step: split above / merge below this factor of the typical region | 2 |
| `split_gain` | evening step: a split must shorten the longest region by this fraction | 0.15 |
| `region_passes` | region lengths are measured on the newest this-many pool passes (own paths of the last run) | 1 |
| `region_stat` | `"mean"` or `"median"` of the region per staple path | mean |
| `merge_p_min` | evening step: the merged pair keeps p ≥ this | 0.2 |
| `p_max` | no split or balance move creates a pair with p above this | 0.9 |
| `balance` | run the balance step | true |
| `balance_p_weight` | weight of the p term in the balance objective | 0.3 |
| `balance_p_min` | balance: no move pushes a pair below this p | 0.2 |
| `n_interfaces`, `max_ratio` | fallback plan and refill (2.6) | 10, 10 |
| `levels` | `"model"` or `"measured"` (2.1b) | measured |
| `levels_run` | freeze round 1: the run to calibrate on (sub-directory of the loop directory) | production_refill |
| `levels_burn_in` | burn-in fraction for the iSTAR analysis of that run | 0.1 |

Settings that differ in `configs/loop_z_dp_u.toml` (inherits loop_z_dp_c, which adds the C hold-out of the student):

| Key | loop_z_dp | loop_z_dp_u |
|---|---|---|
| `[run] cycles_per_pass`, `confirm_passes` | 1000, true | 2000, false |
| `[stop] max_passes`, `interface_tol` | 7, 0.02 | 10, 0.1 |
| `[teacher] max_iter` | 6000 | 10000 |
| `[student] epochs`, `holdout_resid` | 24, off | 48, 1.0 |
| `[interfaces] c` | 1 | 0.6 |
| `[interfaces] skip_eps`, `skip2_eps` | 0.01, 0.01 | 0.05, 0.01 |
| `[interfaces] place_gap_always`, `place_c` | not set | false, not set (placement gap 0.9 δ) |
| `[interfaces] split_rule`, `even_max_moves` | gain (`split_gain` 0.15), 12 | uniform, 12 |
| `[interfaces] place_margin`, `merge_p_min`, `balance_p_min`, `p_max`, `target_p`, `long_ratio`, `n_sh`, `balance` | 1.5, 0.2, 0.2, 0.9, 0.3, 2, 2, true | same |
| `[production] verify_rounds`, `verify_cycles` | 2, 2000 | 3, 3000 |

### 2.8 Where path length enters the placement **[11-27]**

Two lengths exist for an ensemble e: the **shooting region** (frames strictly inside the window (λ_{e−2}, λ_e); these
are the frames a shooting point can be drawn from) and the **whole staple path** (frames from its first to its last
frame, including the legs outside the window until the turn; this is what a move costs in MD). The placement uses the
first, not the second.

| Step | Path length used | Where |
|---|---|---|
| dp (2.2) | none. The objective is only (ln(p / target_p))². A window is banned if it fails the empty-region or the region-frames test of the gate | `place_dp` |
| evening (2.3) | the region length per ensemble: unweighted mean over the staple paths cut from the last run's own paths at the trial set's turns. Splits above `long_ratio` × the median; merges below median / `long_ratio`; `"uniform"`: variance of ln region | `even_regions`, `window_frames` |
| balance (2.4) | the same region length: spread of ln region over the ensembles, plus the p term | `balance_regions` |
| gate (2.5) | only a lower bound: 5% quantile of frames inside the window ≥ `n_sh`, and the empty-region fraction | `check`, `window_stats` |
| fallback plan (2.6) | `length_budget` with predicted median path lengths (`natural.predict_lengths`) adds interfaces. Off in the dp fallback (`no_growth`) | `plan_natural` |
| not used in the loop | the whole staple path length and the predicted acceptance A_len × (1 − NCR) of a shot. Both are computed by `window_profile` and used only by `interfaces.refine_measured`, run separately on finished production runs (`python -m loop.refine_cmd`). Its default objective minimises the predicted variance of ln k per MD step, with L_e = the mean whole staple path length | `refine_measured` |

Consequences:
- An ensemble whose staple paths are long mostly outside its window (long legs before the turn) is not seen by the
  evening and balance steps.
- The per-pass path lengths (`state.json` `lengths`, panel (c) of `pass_<k>/diagnostics.png`) are logged and plotted, not
  used for a decision.
- The adaptive pass length (`[run] pass_stop = "paths"`, off in the Z configs) counts accepted paths per ensemble,
  not frames.

## Part 3: freeze and production

- **Freeze** (`passes.freeze`, or `python -m loop.freeze_cmd <config> --name <name> --from-freeze` to re-freeze into a
  new directory). λ of the last pass is fixed.
  - If the last pass's re-check failed, or a re-freeze is forced, it runs up to `verify_rounds` (loop_z_dp: 2,
    loop_z_dp_u: 3) verification rounds. Each round:
    1. re-places the interfaces with Part 2: on the pool in round 1, on the failed run's own paths after that;
    2. runs `verify_cycles` (2000; loop_z_dp_u: 3000) cycles with that set;
    3. re-checks the gate on the run's own paths.
  - The first set that passes is frozen. Its seeds are cut and the production directory is written (not started).
  - If none passes, the freeze is refused (`freeze.json` with `ok = false`).
- **Production:**
  `run.py loop <config> --production [--production-name <dir>]`: 20 000 cycles StapleTIS, then the inftools iSTAR
  analysis (P_cross, rate, block errors) into `<dir>/rates.json`.

### 3.1 The stop test (`passes.stop_test`, step 10)

Run after every pass from pass 2 on. All five must hold; the log prints `stop test: all criteria met` or
`continue (reasons)`. When it passes, the loop goes to the freeze, otherwise it continues up to `max_passes` (or
`max_md_steps`).

| # | Criterion | Limit (`[stop]` key) |
|---|---|---|
| 1 | stability: weighted spread of logit m on the slabs of the other model's levels, new on old and old on new, at every resolved level | ≤ `stability`, default 3 × floor = 3 × 0.2² / 3 = 0.04 (floor: the spread of a slab of half-width 0.2 logit when the two models agree) |
| 2 | prospective SD(Z_q): the previous model on the new pass's paths against the refit | ≤ (1 + `zq_tolerance`) × refit; `zq_tolerance` = 0.2 |
| 3 | interfaces: same number as in the previous pass, and none moved by more than a fraction of λ_B − λ_A | `interface_tol` = 0.02 (loop_z_dp_u: 0.1) |
| 4 | the pass's own-path re-check (2.5) passes and no interface is unverified | none |
| 5 | share of frames outside the training range, over all ensembles | ≤ `ood` = 0.01, and ≤ 1.5 × the previous pass + 1e-4 |

In loop_z_dp and loop_z_dp_c the test never passed (checked on `state.json`, passes 2–7 of both):
- criterion 3 failed in every pass: the number of interfaces changed in 11 of the 12 transitions, and in the
  twelfth the largest move was 30% of λ_B − λ_A;
- criterion 1 failed in every pass: the largest spread was 0.04–0.17 in loop_z_dp (limit 0.04) and 0.17–1.6 in
  loop_z_dp_c;
- criterion 4 failed in 4 of 6 passes of loop_z_dp (passes 5 and 6 passed) and in all 6 of loop_z_dp_c;
- criterion 2 failed in passes 3 and 5 of loop_z_dp and in passes 3, 5 and 7 of loop_z_dp_c; criterion 5 never failed.

## Part 4: results of loop_z_dp (Brownian Z)

Figures: `examples/figures_by_potential/_shared/param_lambda/op_loop/loop_z_dp/figures/`. Made by
`python -m loop.plots loop_z_dp` and `python -m loop.plot_lengths loop_z_dp <run>=<label> …`.

**Passes.** Pass 0 (infinit) took 8.6M MD steps; passes 1–7 took 0.6–1.0M each, 13.8M in total. The loop stopped at
the pass limit (7), not by the stop test.

| Pass | Interfaces | SD(Z_q) | Logit RMSE vs exact | Isosurface spread vs exact | Re-check on own paths |
|---|---|---|---|---|---|
| 1 | 7 | 24.3 | 0.60 | 0.152 | pass |
| 2 | 17 | 31.6 | 0.77 | 0.217 | fail (skip on the ridge) |
| 3 | 15 | 137 | 1.09 | 0.153 | fail (skip, gap) |
| 4 | 13 | 97.4 | 0.71 | 0.157 | fail (one accepted path with empty region) |
| 5 | 9 | 39.0 | 0.63 | 0.097 | pass |
| 6 | 9 | 24.9 | 0.35 | 0.103 | pass |
| 7 | 13 | 17.6 | 0.31 | 0.097 | fail (gap, empty region) |

- `convergence.png`: accuracy per pass.
- `calibration.png`: model against exact committor; from pass 5 on the fit follows the exact committor between
  logit −8 and +4, and flattens above logit q ≈ 4, near B.
- `interfaces.png`: the interfaces of every pass on the potential, with the exact isocommittors at the same levels.

**Free-energy profiles** (`feps_pass<k>.png`, `feps_pass7_exact.png`): the path-weighted F(q) and Z_q of pass k's
model on pass k's own paths (optimalrcs TIS diagnostics). Pass 7 against the exact committor on the same paths:
- F(q) has the same shape as the exact one: a barrier region up to q ≈ 0.6, the deep minimum at q ≈ 0.6 (the
  plateau basin), then a rise to B;
- Z_q, which is flat for the true committor, has spikes up to about 33 for the model (about 9 for the exact
  committor), at q ≈ 0.6–0.8. That is the plateau, where the model is least accurate.

**Interface sets of the freezes:**

| | First freeze (`production/`) | Re-freeze (`production_refill/`) |
|---|---|---|
| Settings | merge/balance floor p 0.1 | merge_p_min = balance_p_min = 0.2 |
| Interfaces | 9: 0.05, 0.072, 0.162, 0.197, 0.422, 0.597, 0.813, 0.843, 0.85 | 12: 0.05, 0.067, 0.094, 0.148, 0.233, 0.376, 0.543, 0.662, 0.822, 0.834, 0.845, 0.85 |
| Inner p (on the model) | 0.11 – 0.64, Σ(1−p)/p = 28.1 | 0.23 – 0.61, Σ(1−p)/p = 19.6 |
| Verification | round 2 passed | round 1 passed |
| Production rate (20 000 cycles) | 2.32e-5, rel. error 0.31; 2.1× the exact 1.13e-5 | 1.27e-5, rel. error 0.39; 1.12× exact |
| Measured local p (production) | 0.026, 0.07, 0.61, 0.061, 0.20, 0.32, 0.75, 0.92 | 0.053, 0.10, 0.37, 0.27, 0.20, 0.24, 0.36, 0.27, 0.83, 0.90, 0.98 |

`pcross_convergence.png`: the 12-interface run's running rate settles at the exact rate. The 9-interface run's
settles at about 2× the exact rate. The inftools block error of the 12-interface run has not reached a clear plateau
(0.53).

The measured p differs from the model's p ≈ 0.3 near the states:
- A end: 0.05 and 0.10, against the model's 0.27 and 0.35;
- B end: 0.83 – 0.98, against 0.3 – 0.6.

That is the known ramp caveat (weak point 4).

**Basin re-freeze** (`production_basin_*`, refused). New region measure, `split_gain` 0.15.

*Verification 1*, 15 interfaces with four in the basin (0.784, 0.796, 0.804, 0.809):
- The path lengths became about uniform. Median 600–2 100 frames in ensembles 3–13, 90th percentile ≤ 4 400
  everywhere, against 29 000–38 000 in the plateau ensembles of both productions. Largest share of frames 0.14.
- It failed the re-check on two skip rates just over the 1% limit:

  | Interface | Skip | Rate | Crossings |
  |---|---|---|---|
  | 0.442 | down | 1.5% | 184 |
  | 0.784 | up | 1.7% | 493 |

*Verification 2* re-placed on round 1's own paths. It lost the basin interfaces: a 7 600-frame region
came back. It failed skip_dn 2.4% at 0.659 (155 crossings, about 4 events), so the freeze was refused.

**Path lengths per ensemble** (`path_lengths.png`, own accepted paths): median length / median shooting-region frames.

| Ensemble | Old, production | New, verification run |
|---|---|---|
| [0+] | 5 / – | 5 / – |
| 2 | 295 / 237 | 328 / 93 |
| 3 | 813 / 437 | 577 / 279 |
| 4 | 760 / 177 | 977 / 506 |
| 5 | 1 074 / 111 | 1 417 / 684 |
| 6 | 1 630 / 30 (ridge) | 1 017 / 15 (ridge) |
| 7 | 15 960 / 15 014 (plateau) | 1 129 / 13 (ridge) |
| 8 | 1 214 / 228 | 5 455 / 4 270 (plateau) |
| 9 | | 8 981 / 8 210 (plateau) |
| 10 | | 488 / 94 |
| 11 | | 61 / 38 |

The new set splits the old plateau ensemble in two, and the longest region drops from 15 014 to 8 210 frames.
The share of stored frames in the most expensive ensemble drops from 0.28 to 0.19. The plateau ensembles are still
50–600 times longer than the ridge ensembles; see weak point 1.

**Measured-levels re-freeze** (`production_measured`, 2026-11-24). Settings: `levels = "measured"` calibrated on
`production_refill`, `place_margin` 1.5, `split_gain` 0.15. It passed verification 1 and was frozen.
- Interfaces (14): 0.05, 0.075, 0.097, 0.157, 0.174, 0.238, 0.417, 0.567, 0.691, 0.807, 0.817, 0.822, 0.841, 0.85.
- Verification run (2 000 cycles): P_cross block error 0.50. The 12- and 15-interface verification runs of the same
  length had 10.3 and 6.8, but errors from runs this short are rough.
- 90th-percentile path length ≤ 14 000 frames. That is between the 12-interface production (29 500) and the
  basin set (4 400): the stricter margin left room for only one interface inside the basin.
- Production (20 000 cycles, 15.3 M MD steps against 23.2 M for `production_refill`; 59 min wall time against
  120 min): P_cross 5.5e-6, rate 3.8e-5 = **3.4× exact**, rel. error 0.69 (P_cross) and 0.65 (rate).
  - The running P_cross starts ~20× high and falls steadily over the whole run without levelling off
    (`figures/pcross_convergence.png`). The run has not reached its steady state, so the error bar understates the
    error.
  - P_cross agrees with `production_refill` up to λ ≈ 0.24 (5.05e-4 against 5.28e-4). The excess builds up from
    λ 0.42 on and mostly in the basin-entry pair 0.691 → 0.807: p 0.50, against 0.27 for 0.662 → 0.822 in
    `production_refill`.
  - Local p: 0.013, 0.37, 0.35, 0.81, 0.39, 0.25, 0.30, 0.39, 0.50, 0.97, 0.93, 0.89, 0.94. The first pair
    (0.050 → 0.075) is still far below target (0.013); the measured-levels map did not fix the A end.

| Settings | Interfaces | Verification | Longest region (90th percentile) | Production |
|---|---|---|---|---|
| model levels, margin 1.25 (`production_refill`) | 12 | passed round 1 | 29 500 | 1.12× exact, rel. error 0.39 |
| model levels, `split_gain` 0.15, margin 1.25 (`production_basin`) | 15 | refused: skip 1.5–2.4% | 4 400 | none |
| measured levels, margin 1.5 (`production_measured`) | 14 | passed round 1 | 14 000 | 3.4× exact, rel. error 0.65, still drifting down; 0.66× the MD steps |

## Part 5: known weak points (for review)

1. **Region-length estimate. Fixed 2026-11-24** (2.3). The former estimate was up to 100× short on the plateau
   (82 predicted against 8 210 measured in ensemble 9 of the new set).
   - The cause was the measure, not the pool. It weighted each run by path_f and counted every short run inside the
     window separately, while a real staple path holds a basin visit of thousands of frames.
   - The new measure uses the last run's own paths, cut into staple paths at the trial set's turns, unweighted mean.
2. **The ridge** (λ ≈ 0.38–0.62) is crossed in a few frames. Every level inside fails the skip check (1–44%). The
   ridge ensembles have the shortest regions (13–15 frames median) and this cannot change by moving interfaces:
   the number of frames between two committor levels does not depend on how λ labels them.
3. **The plateau is the intermediate basin.**
   - Of the frames above λ 0.5 in the new set's run, 96% lie in λ 0.78–0.82. The exact committor there is 0.61–0.69,
     across y 0.47–0.62: this is the free-energy minimum at q ≈ 0.6.
   - The committor is nearly flat in a metastable basin, so a single interface cannot cut the dwell. A path wanders
     back and forth across it without making a staple turn. Single splits gain 15–30%, which the former
     `split_gain` 0.3 rejected.
   - Several interfaces in the basin together make paths turn often.
     - Offline result with `split_gain` 0.15, on the frozen λ: 15 interfaces, four of them at λ 0.784, 0.796, 0.804,
       0.809, with p 0.59, 0.86, 0.88, 0.90.
     - Longest region 1 998 against 9 654 for the current 12-interface set. Max/min 84× against 507×.
     - Sum of regions 8 359 against 21 040. Σ(1−p)/p 20.3 against 19.6.
   - `p_max` 0.9 stops further splits there.
   - Not yet verified by a run.
4. **End-pair p** uses the effective state levels. The export's ramps near the states hold h above the committor
   there, so the end pairs' p is overestimated (`state_levels` docstring).

### Added 2026-11-27 (from loop_z_dp_c and the replays of its placements)

5. **The last window of pass 1 cannot be split.** In loop_z_dp_c pass 1 the last window (λ 0.673 → B) had a predicted
   region of 31 000 frames against ~1 000 elsewhere. The evening step logged `skip 0.6899`: no candidate shortened it
   by `split_gain`. A replay with `"uniform"` also skipped it (variance 4.78 → 4.45 at best, then refused by the
   gate). λ₁ cannot resolve the basin; in the old loop the basin interfaces came from pass 2 on, when λ resolved it
   (replay of pass 2: region 27 000 → 2 300 frames by one split near 0.835).
6. **The gap rule blocks useful splits where crossings are measured** when `place_gap_always = true`. In a replay of
   the old pass 7 (gap 0.9 δ), three improving splits were refused, all by `gap`: two next to an interface just
   inserted (0.004 apart), and 15 candidates between 0.19 and 0.31, because δ at 0.3123 is large on the barrier flank.
   `loop_z_dp_u` has `place_gap_always = false` for this reason.
7. **Measured levels failed in pass 1** of loop_z_dp_c: `KeyError: 'start_intf'` while reading pass 0, so the model's
   levels were used (the log says so). Pass 0 is a RETIS-mode infinit run without the staple columns. Not investigated
   whether the old loop had the same.
8. **The C hold-out is inactive in the first passes.** Held fraction of the labels at 1.0 logit, with each pass's own
   teacher labels (old pool): 0.02% (pass 1), 0.05% (2), 14% (4), 24% (7), because early labels have a small logit
   spread (SD 1.2 in pass 1, 3.7 in pass 7). The threshold is absolute.
9. **The placement sees the region, not the whole staple path** (2.8): no term for the MD cost of a move or for the
   acceptance outside `refine_measured`.
