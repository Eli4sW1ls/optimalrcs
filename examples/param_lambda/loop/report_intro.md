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

- Seed lambda: P_cross drifts down over the whole run, from 2e-4 to 3.4x the exact value, and its block-error curve
  has no plateau (3 to 11 across block lengths). The quoted error (6.2) is not usable; the run is not converged.
- Learned lambda: the block-error curve is flat at 0.47, but the running P_cross rises in steps: about 0.65x the
  exact value from 7M to 14M MD steps, then a jump of x3 near 14M to 2x exact. Steps like these come from rare
  heavy paths in the wide ridge ensemble (half of the variance, table above), so 0.47 probably understates the
  error. The inftools rate is defined only from about 14M steps on (a negative MFPT before, rate reported as 0).
- Exact P_cross (dashed) = exact rate x time per frame / the run's flux (flux depends only on lambda_A).

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
