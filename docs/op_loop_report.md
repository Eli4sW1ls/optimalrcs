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
placement margin (`place_margin` 1.25): the last removal, lambda 0.473 (exact q about 0.006), had 3 frames in its
shooting region, enough for the gate (n_sh = 3) but not for the margin (3.75).

`interfaces.refill` (run at the end of `interfaces.place`) therefore re-inserts, into every interior pair of
neighbours whose crossing-probability ratio exceeds `max_ratio`, the candidate (40, equally spaced in logit h) that
splits the pair best and passes the gate itself, without the margin; the next pass's re-check on its own paths,
or the freeze's verification rounds, decides whether it survives. Where no candidate passes, the pair stays as it
is. For the pairs touching a state, h(lambda_A) = 0 is replaced by the mean h at the first frames past lambda_A
(and likewise at B): with committor-level interfaces P(reach lambda_1 | crossed lambda_A) = E[q at those frames] /
q(lambda_1). The exact committor confirms the identity on Z (1.8e-6 / 1.78e-4 = 0.010; production run 0.0098),
but the exported lambda's ramps hold the model's h near the states too far from 0 and 1, so the model's end-pair
ratios are lower bounds (11 at A and 13 at B, against about 100 and 64): refill adds fewer interfaces there than it
should, never more. On the stored Z paths under the frozen lambda, refill adds lambda = 0.5377 (model q 0.066)
between 0.2925 and 0.7994 (ratio 91 -> 9.7), 0.0616 next to A and 0.8455 next to B: 13 interfaces, full gate
passes.

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

## Per-system tables (generated by `python -m loop.report`)

### StapleTIS weights in infretis (notes section 15), flat 1D, exact committor

| run | weighting | pieces | exact q: stopped Z_q max abs z at lags 1, 4, 16, 64, 256 | mean exact q at the first frames past M, per ensemble |
|---|---|---|---|---|
| inf | path_f | 19432 | 2.91, 2.38, 2.42, 1.89, 1.14 | 1: 0.0274, 2: 0.2567, 3: 0.5002, 4: 0.7439 |
| inf | path_f / path_w | 19432 | 4.71, 4.89, 4.95, 5.59, 5.20 | 1: 0.0274, 2: 0.2544, 3: 0.5002, 4: 0.7462 |
| inf_s2 | path_f | 19788 | 2.76, 2.58, 3.59, 2.81, 2.70 | 1: 0.0274, 2: 0.2567, 3: 0.4993, 4: 0.7438 |
| inf_s2 | path_f / path_w | 19788 | 4.95, 4.76, 4.72, 5.05, 4.33 | 1: 0.0274, 2: 0.2544, 3: 0.4994, 4: 0.7461 |
| nn | path_f | 16293 | 1.48, 1.44, 1.73, 1.18, 1.76 | 1: 0.0271, 2: 0.2564, 3: 0.5001, 4: 0.7435 |
| nn | path_f / path_w | 16293 | 4.38, 4.51, 4.40, 4.36, 3.94 | 1: 0.0271, 2: 0.2540, 3: 0.5003, 4: 0.7459 |
| nn_s2 | path_f | 16441 | 1.60, 1.50, 2.12, 1.61, 1.33 | 1: 0.0274, 2: 0.2575, 3: 0.5003, 4: 0.7439 |
| nn_s2 | path_f / path_w | 16441 | 4.15, 4.00, 4.24, 4.40, 4.20 | 1: 0.0274, 2: 0.2553, 3: 0.5000, 4: 0.7461 |
| sh | path_f | 16874 | 1.02, 1.14, 1.25, 1.24, 1.78 | 1: 0.0274, 2: 0.2568, 3: 0.4991, 4: 0.7454 |
| sh | path_f / path_w | 16874 | 4.86, 5.06, 5.54, 5.90, 5.88 | 1: 0.0274, 2: 0.2545, 3: 0.4992, 4: 0.7475 |
| sh_s2 | path_f | 16602 | 2.20, 1.99, 1.90, 1.27, 1.01 | 1: 0.0274, 2: 0.2555, 3: 0.4997, 4: 0.7436 |
| sh_s2 | path_f / path_w | 16602 | 4.53, 4.58, 4.56, 5.48, 4.55 | 1: 0.0274, 2: 0.2534, 3: 0.4995, 4: 0.7460 |

### loop_cc

Pass 0 (infinit, seed order parameter): reached B = True; 2 iterations stored. Total MD steps of the initialization: 1,529,275.

| pass | interfaces | teacher RMS vs exact | SD(Z_q) new | SD(Z_q) previous model | stability max | spread vs exact | logit RMSE vs exact | longest / shortest median length [1*]+ | re-check | MD steps | fit s | pass s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.190 | 7.79 | – | – | 0.0662 | 1.555 | 442 / 172 | OK | 169,387 | 118 | 258 |
| 2 | 12 | 0.085 | 9.23 | 43.16 | 0.167 | 0.0240 | 0.661 | 409 / 232 | OK | 178,363 | 150 | 296 |
| 3 | 12 | 0.067 | 11.31 | 15.05 | 0.040 | 0.0161 | 0.507 | 402 / 224 | OK | 183,407 | 184 | 340 |
| 4 | 12 | 0.044 | 12.14 | 18.78 | 0.040 | 0.0285 | 0.346 | 431 / 250 | OK | 204,918 | 294 | 414 |
| 5 | 12 | 0.033 | 19.29 | 24.86 | 0.145 | 0.0180 | 0.313 | 421 / 249 | OK | 186,295 | 357 | 588 |

Last interfaces: `[0.15, 0.3272, 0.46141, 0.56327, 0.651, 0.72203, 0.77536, 0.81359, 0.83322, 0.84167, 0.84639, 0.85]`
Re-check of the last pass on its own paths: all checks pass.
Accepted paths with an empty shooting region per ensemble: 1: 0/142, 2: 0/118, 3: 0/130, 4: 0/135, 5: 0/137, 6: 0/127, 7: 0/107, 8: 0/96, 9: 0/109, 10: 0/115, 11: 0/74.

| production run | interfaces | cycles | MD steps | MD steps of the initialization | P_cross (rel. error) | rate per time unit (rel. error) | rate / exact |
|---|---|---|---|---|---|---|---|
| learned lambda (frozen) | 12 | 20,000 | 3,845,786 | 1,529,275 | 0.000761 (0.24) | 0.00184 (0.19) | 1.19x |
| seed lambda (control) | 11 | 20,000 | 4,319,110 | 0 | 0.000706 (0.24) | 0.00158 (0.20) | 1.02x |

Exact rate 0.00155 per time unit (toy2d.exact_committor_brownian (TPT on a grid, h = 0.004)).

### loop_z

Pass 0 (infinit, seed order parameter): reached B = True; 3 iterations stored. Total MD steps of the initialization: 9,064,996.

| pass | interfaces | teacher RMS vs exact | SD(Z_q) new | SD(Z_q) previous model | stability max | spread vs exact | logit RMSE vs exact | longest / shortest median length [1*]+ | re-check | MD steps | fit s | pass s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 12 | 0.126 | 11.23 | – | – | 0.0432 | 1.139 | 17284 / 233 | fails | 356,473 | 452 | 624 |
| 2 | 8 | 0.143 | 32.89 | 58.25 | 0.719 | 0.1612 | 2.044 | 13115 / 370 | OK | 449,949 | 501 | 669 |
| 3 | 11 | 0.094 | 12.76 | 22.73 | 0.176 | 0.1069 | 1.477 | 4374 / 435 | OK | 636,718 | 576 | 800 |
| 4 | 11 | 0.055 | 66.96 | 53.41 | 2.445 | 0.1261 | 0.859 | 10647 / 400 | fails | 1,027,420 | 528 | 863 |
| 5 | 12 | 0.103 | 37.32 | 34.41 | 0.309 | 0.2097 | 0.719 | 8969 / 537 | fails | 750,182 | 528 | 792 |
| 6 | 11 | 0.077 | 36.69 | 70.61 | 0.078 | 0.2171 | 0.671 | 9402 / 366 | fails | 696,832 | 445 | 685 |
| 7 | 11 | 0.064 | 10.56 | 12.49 | 0.109 | 0.1761 | 0.440 | 15208 / 397 | fails | 637,408 | 463 | 689 |

Last interfaces: `[0.05, 0.08594, 0.13956, 0.20231, 0.29253, 0.47275, 0.79936, 0.81418, 0.82187, 0.83431, 0.85]`
Re-check of the last pass on its own paths: [['region_frames', 5, 2.0]].
Accepted paths with an empty shooting region per ensemble: 1: 0/123, 2: 0/58, 3: 0/66, 4: 0/72, 5: 0/60, 6: 0/45, 7: 0/17, 8: 0/21, 9: 0/33, 10: 0/47.

Frozen set (verification 1): `[0.05, 0.0859, 0.1396, 0.2023, 0.2925, 0.7994, 0.8142, 0.8219, 0.8343, 0.85]`.
Verification round 1: removed [(0.4728, 'region_frames 3 (ensemble/interface 5)')]; passes; 398,006 MD steps.

| production run | interfaces | cycles | MD steps | MD steps of the initialization | P_cross (rel. error) | rate per time unit (rel. error) | rate / exact |
|---|---|---|---|---|---|---|---|
| learned lambda (frozen) | 10 | 20,000 | 18,648,151 | 9,064,996 | 3.15e-06 (0.50) | 2.3e-05 (0.49) | 2.03x |
| seed lambda (control) | 11 | 20,000 | 12,474,255 | 0 | 5.85e-06 (4.47) | 3.8e-05 (3.36) | 3.36x |

Exact rate 1.13e-05 per time unit (toy2d.exact_committor_brownian (TPT on a grid, h = 0.004)).
