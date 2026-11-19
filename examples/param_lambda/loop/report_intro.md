# Order-parameter learning loop for path-sampling initialization

A loop that learns the order parameter lambda(x) over several short sampling passes and hands a frozen lambda,
interfaces that pass a spacing gate, and valid initial paths to a production StapleTIS run. Backend so far:
infretis with the ASE engine (overdamped Langevin, the toy potentials of the toytis runs), with inftools' infinit
as pass 0. Plan and design: [`op_loop_plan.md`](op_loop_plan.md). Code: `examples/param_lambda/loop/`.

## Findings

**StapleTIS weights in infretis (notes section 15): committor pieces need path_f, in every mode including infinite
swap; path_f / path_w stays right for crossing probabilities and rates.** Flat 1D system with its exact committor,
three StapleTIS modes, two seeds of 20 000 cycles each.

Why the two weights answer different questions. The staple extension flips a coin to decide which end of an
LML/RMR core to extend, so one core produces two stored staples (previous turn + core, core + next turn), each
generated with probability 1/2. Shooting therefore samples LML/RMR staples at w = 1/2, consistent with the W matrix
(infretis' own reasoning, confirmed here), and path_f / path_w recovers the natural measure of the stored staple
paths: right for the iSTAR crossing probability and the rate. For the committor, what matters is the measure of the
pieces cut at M, and both stored versions contain the same core. path_f counts each version at 1/2, so the core
counts once, like an LMR core; path_f / path_w counts it twice. Whether a core is LML or LMR is decided after the
M crossing, so doubling LML cores selects on the pieces' future and biases them (as the factor 2 did in toytis,
notes section 9.3). The coin-flipped turn frames enter at 1/2, which is harmless: the coin is independent of the
dynamics.

Test 1, level test: the path-weighted mean of the exact committor at the first frames past M (forward pieces),
two seeds pooled, bootstrap over paths:

| mode | path_f, [1*] / [2*] / [3*] | path_f / path_w, [1*] / [2*] / [3*] |
|---|---|---|
| infinite swap | 0.2567 / 0.4998 / 0.7438 (s.e. 0.0004) | 0.2544 / 0.4998 / 0.7462 |
| nearest-neighbour swaps | 0.2569 / 0.5002 / 0.7437 | 0.2546 / 0.5002 / 0.7460 |
| shooting only | 0.2561 / 0.4994 / 0.7443 | 0.2539 / 0.4993 / 0.7466 |

The three modes agree within one s.e. with path_f; path_f / path_w shifts the two outer ensembles by 0.0023 (about
5 s.e.), toward the side where LML/RMR cores dominate, and leaves the symmetric middle ensemble unchanged.

Test 2, the stopped Z_q of the exact committor on the pieces (forward and mirrored), worst |z| over lags 1-256
(unbiased data stay at about 3 or below, notes section 6.1):

| mode | path_f (seed 1 / seed 2) | path_f / path_w |
|---|---|---|
| infinite swap | 2.9 / 3.6 | 5.6 / 5.1 |
| nearest-neighbour swaps | 1.8 / 2.1 | 4.5 / 4.4 |
| shooting only | 1.8 / 2.2 | 5.9 / 5.5 |

Infinite swap with path_f is slightly above the other modes here, but the level test shows no bias; the likely
cause is stronger correlation between pieces of swapped paths, which the path groups do not fully capture.

Consequence for the loop: training passes and production both run with infinite swap; the training pieces use
path_f (`pool.weighting = "rejections"`); the crossing probability and rate of the production run use path_f /
path_w through inftools' iSTAR analysis. The loader default (`infretis_data.default_weighting`: path_f / path_w
with infinite swap) is unchanged; for committor fits on infretis staple data pass `--weights rejections`.

**Interface thinning on the Z ridge is a sampler-resolution limit, not a lambda defect; placement now refills
wide gaps where the gate allows.** The frozen Z set left one ensemble spanning q = 0.007 -> 0.645 (crossing
probability ratio about 90, the plan caps neighbours at 10). The exact committor on the frames of pass 7 shows why
every planned ridge level was removed:

| exact q | frames | median per-step change of logit q | median per-step change of lambda |
|---|---|---|---|
| 0.001-0.01 (basin side) | 112k | 0.03 | 0.002 |
| 0.01-0.2 (ridge) | 1.5k | 0.4 | 0.015-0.03 |
| 0.2-0.65 (plateau) | 421k | 0.003 | 0.00008 |

The ridge is crossed in about 10 saved frames (subcycles = 1, dt = 0.02 time units). The number of frames between
two committor levels is set by the dynamics, and no monotone relabelling of lambda changes it, so interfaces inside
the ridge fail the gap and shooting-region tests whatever lambda is. A smaller timestep would add ridge frames but
multiply the cost of the plateau paths (8k-13k frames), which dominate the MD budget. The fixable part was the
thinning itself: it removes one planned level at a time and does not put anything back, so a stretch where the
planned levels sat badly ends up empty even when another level in between would pass.

`interfaces.refill` (run at the end of `interfaces.place`) therefore re-inserts, into every interior pair of
neighbours whose crossing-probability ratio exceeds `max_ratio`, the candidate (40, equally spaced in logit h) that
splits the pair best and passes the same margined limits as the thinning; the next pass's re-check on its own
paths, or the freeze's verification rounds, decides whether it survives. A first version refilled against the bare
gate (no margin). It re-added what thinning had just removed (0.8455, "too close to B", in every round), and all
three verification rounds of that set failed on marginal checks: 2 accepted paths with an empty shooting region,
skip_up 2.2% (limit 1%) at 0.7994, gap 0.77 delta between 0.7994 and 0.8142 (`failed_refill_attempts/attempt1_bare_gate_500cycles_freeze.json`,
1.08M MD steps). Where no candidate passes, the pair stays as it
is. For the pairs touching a state, h(lambda_A) = 0 is replaced by the mean h at the first frames past lambda_A
(and likewise at B): with committor-level interfaces P(reach lambda_1 | crossed lambda_A) = E[q at those frames] /
q(lambda_1). The exact committor confirms the identity on Z (1.8e-6 / 1.78e-4 = 0.010; production run 0.0098),
but the exported lambda's ramps hold the model's h near the states too far from 0 and 1, so the model's end-pair
ratios are lower bounds (11 at A and 13 at B, against about 100 and 64): refill adds fewer interfaces there than it
should, never more. On the stored Z paths under the frozen lambda, refill adds lambda = 0.5377 (model q 0.066)
between 0.2925 and 0.7994 (ratio 91 -> 9.7), 0.0614 next to A and 0.8439 next to B: 13 interfaces, all
placement limits pass on the stored paths. Its verification was refused as well (`failed_refill_attempts/attempt2_margined_500cycles_freeze.json`,
3 rounds of 500 cycles): shooting-region q05 of 2 frames (limit 3) in the ensembles (0.5377, 0.7994, 0.8142) and
(0.7994, 0.8142, 0.8219), then gap 0.97 delta between 0.7994 and 0.8142. The failures sit at the closely spaced
plateau-entry interfaces of the original set, not at the refilled ones, and are marginal 5%-quantile and gap
estimates from a few dozen excursions per 500-cycle run. Re-placement on the pooled paths keeps all 13
interfaces every round, so the rounds cannot settle: the pool dilutes the failing run's own paths.

Longer verification (2000 cycles per round, `production.verify_cycles`) did not settle it, so the cause is not only
sample size. The same 13-interface set gave: round 1 (last interface 0.8446) all checks pass; round 2 (0.8449)
`region_frames` fails again in the ensemble (0.5377, 0.7994, 0.8142), q05 = 2 frames (`production_refill2_freeze.json`,
3.0M MD steps for both rounds). In that ensemble each 2000-cycle run holds about 105 excursions from about 85 paths, of
which 1 (round 1) or 4 (round 2) have fewer than 3 frames inside. Infinite-swap path weights are heavy-tailed
(median 9, maximum 56-90), so a single short excursion of weight 76 carries 4.4% of the ensemble's weight, and
the weighted 5% quantile hinges on one or two paths: the short ones hold 2.1% of the weight in round 1 and 5.7%
in round 2, a true value close to the limit. A longer run converges to that value, not away from it. The geometry
behind it: the step size of lambda at 0.7994 (delta 0.0115-0.0153 per saved frame, from the steep part of lambda
that the teacher's plateau overshoot creates) is 7 to 9 times that at 0.8142 (0.0017), and the spacing between
them is 0.0148, i.e. 1.0-1.3 delta at 0.7994 (the gap check's limit is 1). Interface 0.8142 (model q 0.734) is not
needed for the ratio cap: without it, 0.7994 -> 0.8219 has ratio 1.6 and a spacing of 1.5-2 delta.

**Result of the refilled set** (option "thin the verification set on the failed run's own paths"). The second
verification set was thinned on the own paths of the failed 2000-cycle run: 0.8142 removed (gap 1.12 against the
placement limit 1.25), the last interface moved 0.8449 -> 0.8434, giving 12 interfaces
`[0.05, 0.0615, 0.0859, 0.1396, 0.2023, 0.2925, 0.5377, 0.7994, 0.8219, 0.8343, 0.8434, 0.85]`. It passed the gate in one
2000-cycle round, and its 20 000-cycle production (`production_refill3`) gave:

| Z production | interfaces | MD steps | initialization MD steps | rate / exact | rel. error of the rate |
|---|---|---|---|---|---|
| learned lambda, 10 interfaces | 10 | 18.6M | 9.5M | 2.03x | 0.49 |
| learned lambda, refilled and thinned | 12 | 17.8M | 19.2M | 0.97x | 1.79 (inftools P_cross plateau 2.19) |
| seed lambda (control) | 11 | 12.5M | 0 | 3.36x | 3.4 |

The point estimate is on the exact rate, but the error bar is larger than the 10-interface run's (which the convergence
plot shows to be understated), so this does not show an improvement. The refilled interface 0.5377 split the wide gap, yet
the upper half (0.5377 -> 0.7994) now carries 90% of the variance of P_cross: its local crossing probability is 0.026
(block error 1.08). The model's committor predicts 0.10 for this pair (q 0.067 at 0.5377, 0.661 at 0.7994), but the exact
committor at those lambda values is 0.012 and 0.618 (on the training frames, medians within +-0.004 of each lambda), a ratio
of 50, which agrees with the measured 0.026. So the failure is the teacher's overestimate at the low end of the plateau
(model q 5.5x the exact value at 0.5377): refill measured the pair with the model's committor, found it at the cap
(ratio 9.7), and the interface it chose is in exact terms still a factor of 50 from its upper neighbour.
The initialization column includes every verification run of the refill attempts, 9.7M MD steps in all (two refused
sets of three 500-cycle rounds, 2.2M; two 2000-cycle rounds of the 13-interface set, 3.0M; the accepted round, 2.0M, and
the original verification, 0.4M).

**Extending the refilled run to 50 000 cycles exposed an infretis crash, now fixed in the sampler.** At cycle 23 069 a
path with an empty shooting region (`{0: (1, 0)}`) in [1*] was picked for shooting; the staple fallback found no cut path
and the worker died (`AttributeError` on `None`); a plain restart crashed again. Such paths are not rare (7
(path, ensemble) pairs in 13 691 accepted paths; the 2000-cycle verification had none by chance), so the gate's "zero empty
regions" cannot protect a long run. `staple_sh` now rejects the move with the existing status NSG (core change authorized,
`test_no_shooting_point.py`). Effect of the alternative of giving those pairs zero weight, measured on the data up to the
crash (10% burn-in): P_cross 2.764e-6 -> 2.744e-6 (-0.7%), rate 1.652x -> 1.642x exact, a negligible change against the
statistical error. The same data give 1.65x the exact rate at 23k cycles against 0.97x at 20k: the point value moves 70%
within 3000 cycles, inside the error.

Result at 50 000 cycles (45.4M production MD steps, one NSG rejection path active): rate 1.247e-5 per time unit = 1.10x the
exact 1.13e-5, relative error 0.26 (inftools block-error plateau of P_cross 0.37). Estimates against exact over the run:
0.97x at 20k, 1.65x at 23k, 1.10x at 50k; the running estimate has settled within about 10% of the exact rate over the
last 10M steps, and the error band is about a factor 7 narrower than at 20k (1.79 -> 0.26). The 10-interface run (20k
cycles) stays at 2.03x with error 0.49; the seed control (12.5M steps) at 3.36x with error 3.4. The plain-step comparison
at equal MD steps is therefore dominated by the refilled run's 19.2M-step initialization cost (verification rounds),
which has to be amortized over a long production.

Where the production error comes from: the block error of each local crossing probability P(lambda_k+1 |
lambda_k) of the Z production, as a share of the variance of P_cross (`production/ensemble_errors.json`):

| learned lambda, step | local P | rel. error | share |
|---|---|---|---|
| lambda_A -> 0.0859 | 0.0098 | 0.30 | 0.27 |
| 0.0859 -> 0.2925 (3 steps) | 0.30-0.43 | 0.06-0.10 | 0.06 |
| **0.2925 -> 0.7994 (the ridge gap)** | **0.0087** | **0.41** | **0.50** |
| 0.7994 -> 0.8142 | 0.88 | 0.23 | 0.16 |
| 0.8142 -> lambda_B (3 steps) | 0.89-0.97 | 0.02-0.03 | 0.00 |

The two steps with a local crossing probability near 1/100 carry three quarters of the variance; these are exactly
the pairs refill now splits. The seed-lambda control spreads its variance over many steps (largest: the first
step, 0.42). Config keys under `[interfaces]`: `refill` (default true), `refill_candidates` (40). Added and
removed interfaces are recorded in each pass record and freeze round (`refilled`, `removed`).

The production rates below (learned lambda: 2.03x the exact rate, rel. error 0.49, i.e. about one error bar; seed
lambda: 3.36x, rel. error 3.4) were run with the frozen set before refill existed. A second calibration problem is
separate from the thinning: lambda between 0.7 and 0.8 covers exact q from 0.05 to 0.55, from the teacher's
overshoot on the plateau; it shifts where the natural-coordinate levels land but does not cause the removals.

**Convergence of the Z productions** (`figures/pcross_convergence.png` of the loop directory; inftools'
`error_analysis_staple`, run by `python -m loop.pcross_errors loop_z`, drawn by `python -m loop.plots loop_z
--pcross-only`):

![Z production convergence](../examples/figures_by_potential/_shared/param_lambda/op_loop/loop_z/figures/pcross_convergence.png)

- Seed lambda, extended to 50 000 cycles (31.3M MD steps): the rate drifts down over the whole run and ends at
  2.8x exact (relative error 0.45, inftools P_cross plateau 0.72); it has not converged. At equal production MD
  (31M) the refilled learned lambda is within about 10% of exact (it ends at 1.10x, error 0.26, at 45.4M), after
  19.2M MD steps of initialization; the seed lambda needs none.
- Learned lambda: the block-error curve is flat at 0.47, but the running P_cross rises in steps: about 0.65x the
  exact value from 7M to 14M MD steps, then a jump of x3 near 14M to 2x exact. Steps like these come from rare
  heavy paths in the wide ridge ensemble (half of the variance, table above), so 0.47 probably understates the
  error. The inftools rate is defined only from about 14M steps on (a negative MFPT before, rate reported as 0).
- Exact P_cross (dashed) = exact rate x time per frame / the run's flux (flux depends only on lambda_A).

## M3: robustness systems (infretis + ASE)

Three systems, each with the loop of loop_z (1000 cycles per pass, at most 7 passes, gated freeze) and a
20 000-cycle production with lambda frozen (infinite swap). Exact references: the 3D grid committor of the gate
system (`loop/exact_zw.py`, k_AB = 2.115e-5 per time unit, checked against six shooting points and a GPU MD scan)
and the 2D grid committor of the two-channel system (`loop/exact_stag.py`, k_AB = 1.148e-4, half the flux per
channel); the embedding has the committor and rate of Z.

| system | what it tests | CVs | interfaces | initialization MD | production MD | rate / exact | rel. error | seed lambda: rate / exact (rel. error) |
|---|---|---|---|---|---|---|---|---|
| gate (`loop_zw`, ZChannelsW, dV_w 3 kT) | a slow w that moves the ridge | y, x, w | 14 | 11.1M | 12.8M | **1.14x** | 0.33 | 1.61x (0.70) |
| embedding (`loop_zu`, uncoupled double well in w) | a slow w the committor does not depend on | y, x, w | 5 | 12.0M | 5.4M | 3.02x | 2.69 | 5.75x (0.35) |
| two channels (`loop_stag`, staggered channels, overdamped) | a channel missing from the initial data | x, y | 12 | 17.4M | 1.6M | **0.54x** | 0.50 | 15.6x (0.44) |
| embedding without w (`loop_zu_xy`, same pass 0 as loop_zu) | what carrying the uncoupled w costs | y, x | 7 | 9.8M | 6.4M | 0.17x | 0.95 | (as loop_zu) |
| embedding, frame budget fixed (`loop_zu_b`, same pass 0) | loop_zu after the frame-budget fix | y, x, w | 10 | 13.6M | 17.6M | **1.85x** | 0.51 | (as loop_zu) |
| embedding without w, frame budget fixed (`loop_zu_xy_b`) | what carrying the uncoupled w costs | y, x | 13 | 12.9M | 12.7M | **1.08x** | 0.36 | (as loop_zu) |
| replicate of loop_zu_b (`loop_zu_b_s2`, seed 100) | seed-to-seed spread | y, x, w | 10 | 11.2M | 8.6M | **1.50x** | 0.36 | |
| replicate of loop_zw (`loop_zw_s2`, seed 100) | seed-to-seed spread | y, x, w | 14 | 8.5M | 13.1M | **0.65x** | 0.51 | |
| replicate of loop_zu_xy_b (`loop_zu_xy_b_s2`, seed 100) | seed-to-seed spread | y, x | 10 | 17.7M | (freeze refused) | – | – | |
| gate, 12 passes (`loop_zw_x12`, loop_zw extended) | do more passes help | y, x, w | 14 | 13.4M | 13.0M | **0.94x** | 0.69 | |
| gate, pass stop on accepted paths (`loop_zw_paths`) | 100 paths per ensemble per pass instead of 1000 cycles | y, x, w | 14 | 14.1M | 12.8M | **1.19x** | 0.54 | |
| loop_zu_xy_b_s2 extended to 12 passes (`loop_zu_xy_b_s2_x12`) | do more passes help | y, x | 9 | 22.0M | 23.4M | 0.37x | 0.62 | |

Seed-lambda controls (`python -m loop.experiments.seed_control setup|run <loop>`): lambda = the state CV, seeded from
the same pool, 20 000 cycles with infinite swap; zu and zw use the 11 equally spaced y levels of the Z control (12.7M
MD steps each, no initialization cost beyond pass 0), stag its 13 seed levels in x (1.3M MD steps). On Z the seed
control at 20 000 cycles was 3.36x and still 2.8x at 50 000, so the zu and zw controls are not converged either.

Figures (`python -m loop.plots <loop>`; 3D systems: one slice per w well, w = -0.1 and +0.1):

![gate, interfaces at w = -0.1](../examples/figures_by_potential/_shared/param_lambda/op_loop/loop_zw/figures/interfaces_w-0.1.png)
![gate, interfaces at w = +0.1](../examples/figures_by_potential/_shared/param_lambda/op_loop/loop_zw/figures/interfaces_w+0.1.png)
![embedding, interfaces at w = -0.1](../examples/figures_by_potential/_shared/param_lambda/op_loop/loop_zu/figures/interfaces_w-0.1.png)
![two channels, interfaces](../examples/figures_by_potential/_shared/param_lambda/op_loop/loop_stag/figures/interfaces.png)
![two channels, committor maps](../examples/figures_by_potential/_shared/param_lambda/op_loop/loop_stag/figures/maps.png)

Further figures in each loop's `figures/`: `paths*.png` (the frozen lambda with a few production paths), `maps*.png` (model and teacher against the exact committor),
`calibration.png`, `convergence.png`, `tis_pass<k>.png`, `feps_pass<k>.png`.

Findings:
1. **Gate: the loop learns a w-dependent lambda and the rate comes out right** (1.14x, within its error). The
   isosurfaces follow the ridge in each well, which is mirrored between w = -0.1 and +0.1. It
   needed both verification rounds of the freeze (round 1: empty shooting regions in ensemble 1, 1.2% of the paths).
   Pass 1 ran with lambda(y, x) from an early screening attempt; the teacher's RMS against the exact committor went
   from 0.056 (pass 1) to 0.13 (pass 2) and the loop recovered over passes 3-7 (stability 0.79 -> 0.16).
2. **Embedding: an uncoupled slow CV makes the production useless.** In the figure the isosurfaces wander off
   the exact isocommittors from pass 1 on, and the sets shrink to 5 interfaces from pass 3. With w among the inputs,
   lambda moves in large steps along the paths; the spacing gate removed every level between 0.17 and 0.85
   (downward skips of 15% at 0.24 and 0.46) and refill found nothing that passes, so one ensemble spans most of
   the barrier. The teacher's RMS against exact stayed at 0.19-0.20 (gate: 0.056). The CVs a lambda sees therefore
   have to be screened; see 4.
   **Correction (2026-11-18): w is not the main cause.** The same loop on (y, x) only (`loop_zu_xy`, same pass 0)
   failed the same way: 7 interfaces with one ensemble over lambda 0.165-0.85, teacher RMS against exact 0.16-0.24,
   production 0.17x (rel. error 0.95). The pool view thinned whole paths uniformly to its 8M-frame budget (the
   infinit passes hold 22.7M frames), so the teacher saw about a third of the few paths that reach B. In loop_zu_xy
   the teacher also counted B paths in the thinned view: 14 of 43, below n_b = 20, so pass 1 fitted to a provisional
   B at y >= 0.372 (logit RMSE 12). loop_zu kept enough for the count, but its teacher was as poor (RMS 0.20 and 0.17
   at passes 1-2). The poor lambdas gave collapsed interface sets, and the passes produced few transitions (B paths per
   loop pass, zu: 73, 43, 2, 1, 1, 7, 0; zu_xy: 71, 3, 13, 1, 1, 39, 0). The budgets now thin in two strata (paths
   that reach B get at most half of it) with inverse-inclusion weights, and the B count is the pool's estimate (pass
   1 of the same data: 42). loop_z and loop_zw did not hit this: their infinit passes had enough B paths for the
   thinned view.
   **Rerun with the fix, w kept (`loop_zu_b`): the loop works.** Production 1.85x (rel. error 0.51; loop_z's first
   production: 2.03x, 0.50), 10 interfaces, teacher RMS against exact 0.035-0.12 from pass 2 and 0.056 at pass 7
   (loop_z 0.064), student logit RMSE 0.33 at pass 7 (loop_z 0.44), and 325 B paths over the 7 loop passes (loop_z
   279, loop_zw 390, loop_zu 127). So the uncoupled w did little harm here; the failure came from the frame budget. One
   run per variant, so this is one realization, not a cost estimate. The interface sets still dropped to 7 at passes 2
   and 4 (pass 5: 2 B paths) and recovered.
   **The same without w (`loop_zu_xy_b`):** production 1.08x (rel. error 0.36), 13 interfaces from pass 4 on, teacher
   RMS against exact 0.066-0.10 from pass 3 (0.086 at pass 7), student logit RMSE 0.26-0.53, 237 B paths over the 7
   loop passes, spread evenly (26-41 per pass). The rates of the two reruns agree within their errors (1.85x +- 0.51
   rel. against 1.08x +- 0.36). The run without w gave steadier interface sets (13 from pass 4; with w 7-13), and the
   run with w a better final teacher (0.056 against 0.086). On this evidence the cost of an uncoupled slow CV is in
   the stability of the interface sets, not in the rate. Pass 1 needed an infretis fix (plan, 2026-11-18).
3. **Two channels: the loop does not find a channel that the initial data miss.** State A cuts the strip next to the
   channel mouths, so the RETIS paths of pass 0 stay on the side of infinit's initial path (upper channel: 139551
   frames past x = -0.9 at y > 0, none at y < 0, while plain dynamics from A enters both channels about equally).
   Over 7 passes the lower channel appeared twice (2 and 1895 frames, never past its barrier); the production enters
   it (176k of 856k channel frames) but no path crosses its barrier at x = 0.6, and the rate is half the exact one,
   as expected from one channel. The maps show it: the pass-7 model matches the exact committor in the upper channel
   and is too low before the lower channel's barrier; the interfaces at q 0.2-0.94 are placed for the lower channel
   and lie where q ~ 1 in the upper one (harmless for the estimate, wasted sampling). The relative error (0.50) does not flag the factor 2. Neither the learned lambda nor
   the trust-region blend can create transitions that the sampler does not make.
   **The seed lambda (x) is worse: 15.6x the exact rate, with a relative error of 0.44.** With lambda = x, the same x
   level lies before the barrier in the lower channel and past it in the upper one. Shooting does not move a path
   between channels, and in 20 000 cycles no lower-channel path reached ensemble 7: ensembles 3-6 (x -0.6 to 0) hold
   46-85% lower-channel paths (weight share after a 10% burn-in), ensembles 7-11 (x >= 0.2) only upper-channel paths.
   The crossing probability chains the easy pre-barrier progress of the lower channel (ratios 0.93, 0.87, 0.66) with
   the post-barrier ratios of the upper one (about 1), so it is about 30 times too high. The learned lambda puts the
   upper-channel barrier between two of its interfaces and gets the one-channel rate.
4. **CV screening from the pool data fails; a short shooting test works.** Three data-only tests could not separate
   the coupled w of the gate from the uncoupled w of the embedding:
   - the validation SD(Z_q): insensitive (-1% when leaving out x on the gate);
   - the held-out Dirichlet energy, single block or 5-fold: kept both (+41%, 5/5 folds against +24%, 4/5 folds).
     With the path-ensemble weights it is not the equilibrium Dirichlet form, so the committor need not minimize it;
   - split-half reproduction of the teacher's well difference (`loop/experiments/w_split.py`: the teacher on two
     disjoint halves of the pool, the bin-mean q difference between w < 0 and w >= 0 frames over (y, x) compared
     between the halves): on the full pools the correlation is 0.59 (MC-block halves) and 0.37 (even/odd passes) for
     the gate, 0.00 and **0.68** for the embedding. On the infinit passes alone (what a pass-1 screening sees) only
     2-7 paths per half reach B and the fits are 0.07-0.19 RMS off the exact committor.

   The reason is in the data: a single teacher fit on (y, x, w) already gives the uncoupled w a well difference of
   0.044 (RMS over the bins; exact: 0.0075, the binning floor) against 0.070 for the gate (exact 0.106). The pool of
   the embedding holds 50 paths that reach B over 12 passes, clustered by pass (22 in pass 4, 4 in passes 6-11), and
   the share of w < 0 frames swings from 0.27 to 0.97 between passes. The outcome information about w is a few
   lineages, and every data split keeps them on both sides.

   **Shooting test** (`loop/experiments/w_shoot.py`, the "fixed-length shooting" of E7):
   - (y, x) of 400 pool frames with teacher q in 0.2-0.8;
   - from each, 16 short overdamped trajectories with w = -0.1 and 16 with w = +0.1;
   - the well difference d of the mean (y, x) displacement after 10, 50 and 200 steps, from two independent halves
     of the shots;
   - S = mean over points of d1 . d2, which is exactly 0 in expectation when w does not enter the (y, x) dynamics.

   Result (z = S / s.e. over points; seeds 0-3 at 400 points, 1-3 at 100):

   | system | 400 points, 5.1M walker steps (6-15 s) | 100 points, 4 shots per half |
   |---|---|---|
   | gate | max z 14.4-15.3 | 4.9-5.2 |
   | embedding | max z 0.0-1.0 | 1.3-1.5 |

   With a threshold of z = 3 it keeps the gate's w and drops the embedding's at both budgets. Two limitations:
   - It conditions on w by setting it, which a toy system allows. A molecular system needs restrained starts or
     binning by w.
   - It tests whether w enters the dynamics of the other CVs near the transition region, which is sufficient for a
     committor dependence only in combination with the fit. A coupled w that the committor does not need would
     still be kept (the conservative error).

   What dropping w gains in the loop: little in the rate. With the fixed frame budget the loop gave 1.85x on (y, x, w)
   (`loop_zu_b`) and 1.08x on (y, x) (`loop_zu_xy_b`), equal within errors. Without w the interface sets were steadier
   (finding 2).
5. **Fixes found on the way** (none could show on Z or cc, which reach B in pass 0 and have small pools): the
   teacher's provisional B now reaches the export (it was always None, so lambda saturated past the data front);
   the blend's seed table ends at the states (it held its last value beyond the data); a pass writes its fit,
   lambda and interfaces to `fit.json` before its run and an interrupted run is resumed to its full length (it was
   taken as finished); the pool view drops whole paths at random above `pool.max_frames` (the 3D pools hold 21-23M
   frames, 5.9 GB as a view); the GPU lock exempts every process that has the lock file open (a deadlock between
   an offline script and loop_stag).
6. **An infretis edge case, now understood and fixed (2026-11-18)**: `REPEX_state_staple.treat_output` raised
   `ValueError: Ensemble ... has invalid pptype RML` (loop_zu pass 1, the gate production, loop_zu_xy_b pass 1). The
   path is a shot in the last ensemble that the extender carried back into B: both turns at B, but labelled RML from
   the segment before the extension. `treat_output` now accepts every path type in the last ensemble (plan,
   2026-11-18). The case in ensemble 1 (gate production) is presumably the mirror image at A; it has not recurred.
7. **Seed replicates (seed 100, same pass 0): the rates agree within their errors, but the learned lambda varies a
   lot between seeds.** Figure: `op_loop/figures/replicates.png` (`python -m loop.plot_replicates`).

   | system | rate / exact, seed 0 | seed 100 | student logit RMSE at pass 7, seed 0 / 100 |
   |---|---|---|---|
   | embedding with w (`loop_zu_b`) | 1.85x (0.51) | 1.50x (0.36) | 0.33 / 0.94 |
   | gate (`loop_zw`) | 1.14x (0.33) | 0.65x (0.51) | 0.79 / 0.81 |
   | embedding without w (`loop_zu_xy_b`) | 1.08x (0.36) | freeze refused | 0.53 / 1.37 |

   All five finished productions lie within about 1 sigma of the exact rate. loop_zu_xy_b_s2 started badly: student
   RMSE 2.55 at pass 1 against 1.24 for seed 0, from the same pass-0 data and a teacher of similar quality (RMS 0.28
   against 0.22). It improved every pass but was still at 1.37 at pass 7, and both verification rounds failed
   (ensemble 5, downward skips 3.7%), so the freeze refused and there is no production. Across all seven loops of M3
   and M5 the student RMSE stops improving by pass 3 in five (later passes move it by +-0.3, sometimes upwards:
   loop_zu_xy_b 0.26 -> 0.53, loop_zu_b_s2 0.56 -> 0.94). The stop test never fired: every loop ran to max_passes = 7,
   because "interfaces changed" is true at every pass (placement re-plans every time).

## M5: underdamped dynamics, lambda(x, v) (`loop_well_lgv`)

1D cos well with walls (toytis CosWellWalls), ASE Langevin, kT 1, gamma 1, dt 0.005, A x <= -0.35, B x >= 0.35; the
CVs are (x, v), the states stay in x. Exact committor of the ASE chain on a 701 x 501 grid
(`loop/exact_well_lgv.py`), reference rate from GPU MD of the same step (`loop/md_well_lgv.py`): k_AB = 0.33709 +-
0.00014 (4096 walkers x 2M steps, seed 1; the 1M-step run gave 0.33703 +- 0.00023), one transition per 593 steps.

**The loop learns the velocity dependence and the rate is right: 1.008x the MD rate (rel. error 0.061).**

| | interfaces | initialization MD | production MD | P_cross (rel. error) | rate / MD rate | rel. error |
|---|---|---|---|---|---|---|
| learned lambda(x, v) | 8 | 0.95M | 1.67M | 0.641 (0.018) | **1.008x** | 0.061 |

- *MD check of the exact committor* (`python -m loop.plot_well_lgv`, 50 saved walkers x 2M steps, every 25th
  frame): the fraction of frames whose next state is B lies on q_exact in every logit bin. Binned against the exact
  committor, the z-scores have RMS 2.35 with no pattern, which fits about 8 correlated frames per relaxation time
  1/gamma. The MD density of x matches exp(-V).
- *Learning*: student logit RMSE 1.56, 1.10, 0.76, 0.72, 0.68, 0.63, 0.74 over passes 1-7 (plateau from pass 3),
  teacher RMS 0.03-0.06. The re-check passed on every pass. The largest errors lie in two corners: next to A at
  v << 0, where q < 1e-5 and the fit is too high, and next to B at v >> 0, where q > 1 - 1e-5 and the fit is too low.
  Both are extreme logits that no interface uses. By pass 7 the isosurfaces follow the spiral of q(x, v) in the
  central well.
- *Cost*: the rate error comes from the flux (6%), not from P_cross (1.8%). This barrier is not a rare event
  (P_cross = 0.64, one transition per 593 steps): plain MD with the same 1.67M steps sees about 2800 transitions and
  gets a rate error of about 2%. So this system tests the velocity machinery: v read from the frame with the sign
  of vel_rev, reversed paths recomputed, no mirrored pieces. It says nothing about efficiency. An efficiency test
  needs a higher barrier.
- *Long lambda = x control* (`production_seed`, 100 000 cycles, seed interfaces -0.35 ... 0.35 in steps of 0.1): at
  52k cycles when pulled. Its rate has to be computed on the workstation, where `load/` is (`loop.rates.staple_rate`).


## How to read the numbers

- *teacher RMS vs exact*: path-weighted RMS of q_np against the exact committor over all pooled frames between
  the states (validation only, never used for fitting, selection or stopping).
- *spread vs exact*: the isosurface spread of the exported lambda (variance of logit q_exact in slabs of
  |logit m - logit q_k| < 0.2 at 10 levels from 1e-3 to 0.95); a lambda whose isosurfaces are isocommittors scores
  0.013.
- *SD(Z_q)*: stopped Z_q of the model's committor on the newest pass, maximum SD over q across lags 1-256;
  *previous model*: the same for lambda_{k-1} before the refit (the prospective test).
- *re-check*: every spacing check (skips both ways, gap, distance from A/B, shooting-region occupancy,
  single-step turns, accepted paths with an empty shooting region) on the pass's own paths, at the saved-frame
  resolution at which infretis evaluates lambda.
- Rates and crossing probabilities come only from production runs with lambda frozen.
