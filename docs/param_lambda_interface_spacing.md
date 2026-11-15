# Interface spacing with a learned λ and fixed states

Recommendation: keep the states and λ as they are, and place the interfaces with a minimum-spacing rule measured
in MD steps of the sampled paths. Change λ near A only if the rule removes too many interfaces. Leave a
model-defined A for later.

## The problem

The exported λ keeps the existing states: A is s < λ_A on the state CV s. The interfaces are committor levels,
equally spaced in ln q from q = 1e-4 to 0.95. Two things can then bring interfaces closer together than one MD
step:

1. **Near A.** A's boundary s = λ_A is not an isocommittor surface. Where the committor on that boundary is above
   a planned level, the isosurface for that level runs into the boundary. The A ramp of the exported λ (width
   0.05·(λ_B − λ_A)) then squeezes it into a thin band next to λ_0.
2. **Where the committor is steep.** At the Z barrier ridge, several isocommittor surfaces lie within one step of
   each other.

For the path ensembles this means:
- Crossings jump over interfaces.
- The regions between interfaces hold almost no frames, although every phase point is assumed to belong to the
  ensemble region of the interfaces its path crosses.
- The staple turn rule (back below the interface below the highest one crossed) fires on single steps.
- Kick points placed on an interface have no frames around them.

## Measurements

Exported M3 model, interfaces at the planned levels, test paths:
- **skip**: P(a step that crosses λ_k upward also crosses λ_{k+1}), path-weighted.
- **distance to A**: the 5% quantile of s − λ_A over first frames past the interface, in RMS steps of s.

| | cc (planned) | Z (planned) |
|---|---|---|
| exact committor within one step outside A (median / 99% / max) | 6e-4 / 8e-3 / 0.02 | ≈ 0 |
| first interface (q = 1e-4): distance to A | 0.6 steps | 9.4 steps |
| largest skip | 57% (q 1e-4 → 2.3e-4) | 18% (q 0.034 → 0.078, ridge) |
| interfaces with skip > 1% | 7 of 12 (q ≤ 6.4e-3) | 3 of 12 (q 0.034–0.41) |

The exact committor, put through the same construction, gives the same numbers (cc: skip 65%, 44%, 29%, …). The
cause is the geometry, not the fit.

## The rule

Place the interfaces greedily, starting from λ_0 = λ_A and going through the planned levels (any plan works:
equal ln q, equal flux, …). Accept the next planned level as λ_{k+1} only if it passes all three tests below;
otherwise skip that level and try the one after it. The tests use the paths the model was trained on:

1. **Skip.** P(skip λ_{k+1} | crossing λ_k upward) ≤ ε, the direct criterion. It is used only where there are at
   least 100 upward crossings of λ_k.
2. **Gap.** λ_{k+1} − λ_k ≥ c · δ(λ_k), where δ(ℓ) is the 95% quantile of the per-step |Δλ| over the steps that
   cross ℓ. This proxy for the skip test also works where crossings are rare (the Z ridge has 28–140 per
   interface in the test set).
3. **Distance from A** (first interface only). The 5% quantile of s − λ_A on frames crossing λ_1 is at least n_A
   RMS steps of s.

Finally, drop the last interface while it fails the gap test against λ_B.

The tests use only the sampled dynamics and the λ values, so the rule works the same in any dimension and with any
monotone rescaling of λ. It changes the interfaces, not λ, and the states stay as defined.

**Defaults:** ε = 1%, c = 1, n_A = 3. With c = 3 the rule was too strict: on cc it kept only q ≥ 0.034, while
the skip rate is already 0 at a gap of about 1.8δ.

## What the rule gives (prototype, M3)

Placement uses the training paths; the check below is on the test paths. The prototype is at
`eval/interface_spacing_m3.json` in each system's `param_lambda` folder of `examples/figures_by_potential/`
(e.g. `2d_zpot/brownian_g10/staple/param_lambda/z_br/eval/`; flat1d and bumps1d: `1d_flat/brownian_staple/param_lambda/eval/`,
`1d_two_bumps/brownian_staple/param_lambda/eval/`).

| | planned | placed | largest skip (test) | smallest gap / δ (test) | first interface: distance to A |
|---|---|---|---|---|---|
| cc | 12 (q ≥ 1e-4) | 6: q = 0.0064, 0.034, 0.078, 0.18, 0.41, 0.95 | 0% | 1.5 | 2.9 steps |
| Z | 12 (q ≥ 1e-4) | 9: q = 1e-4 … 0.0148, 0.078, 0.95 | 0.8% | 1.1 | 9.4 steps |

- **cc.** The low-q levels go: those isocommittor surfaces cannot be separated from A's boundary. The first
  ensemble now covers q = 0 → 6.4e-3, so its crossing probability is lower. That costs sampling effort in [0⁺]
  but nothing in correctness.
- **Z.** The levels near A stay. The three ridge levels (0.034, 0.18, 0.41) are merged, and the regions between
  them held about 1e-4 of the frames anyway.

## If the rule removes too much

Change λ only next to A: λ = λ_A + smin(λ_q − λ_A, κ(s − λ_A)), with smin a smooth minimum. Near A the λ
isosurfaces then run parallel to A's boundary, at least (λ_k − λ_A)/κ from it, until the committor isosurfaces
leave the boundary. κ follows from n_A and the planned levels. This gives up committor alignment only where the
isocommittor surfaces cross the state boundary. It replaces the current multiplicative ramp, which is what causes
the squeezing.

## Why not a dynamic A (A = {q < q_A}) now

It would make λ_0 itself a committor isosurface and remove problem 1. But it:

- changes the state definition, which the current setup fixes to the existing definitions;
- needs q_A above the noise of q_np (≥ 1e-3), and the model to be right inside the old A, where it has no
  training frames;
- changes the committor's boundary condition, so q_np has to be refit with the new A and the runs redone.

It is worth a separate test once the fixed-A version is in production.

## Implementation (not done yet)

- `evaluate.place_interfaces(model, frames, plan, eps, c, n_A)` returns the placed levels plus the per-interface
  diagnostics (skip, n_up, gap/δ, distance to A).
- The export stores the placed interfaces next to the planned ones, and the adapter reads them.
- The report gets a table and the interface map with the placed interfaces.
- Tests: the placed interfaces are increasing, inside (λ_A, λ_B), and satisfy the three tests on the data used.
- After a production run with the new interfaces, re-run the placement: those paths sample the new interfaces
  better.

## Staple runs with the learned λ: path lengths and the natural coordinate (2026-11-15)

The rules above keep interfaces apart in MD steps. For StapleTIS the path length per ensemble matters as well: some ensembles must not run far longer than the rest. This section is from test runs of the sampler with the learned λ. `staple_run.py --check` measures the skip and distance-to-A tests above; the `place_interfaces` function listed under Implementation is still not written.

`examples/param_lambda/staple_run.py` runs toytis StapleTIS with an exported λ (`toytis_adapter`) and the settings of
the run it was trained on (Z: `sim_istarz_br_g10`; cc: `test_cc_br`). `--model none` is the control with plain y.
Test runs of 2000 cycles: `toytis/simulations/test_*lam-*`; `examples/param_lambda/staple_stats.py` tabulates
them.

**States.** A and B stay those of y: λ = y outside (λ_A, λ_B) and strictly inside otherwise, and the first and
last interfaces are λ_A and λ_B. In 2.6M frames of learned-λ runs, A/B from λ and from y never differed.

**Interface placement sets the path lengths.** All interfaces are λ isosurfaces; only their levels differ.
Median accepted path length per ensemble [k*] (frames). Runs of 2000 cycles, except those marked: I stopped
them early because their long paths made them slow, so their statistics are rougher.

| system | placement | median path length over the ensembles | longest / shortest |
|---|---|---|---|
| cc | plain y (control) | 220–493 | 2.2 |
| cc | equal in λ (log1p scale, ≈ ln q), from q = 0.007 (stopped at 1033 cycles) | 66–834 | 12.6 |
| cc | `y`: isosurfaces at equally spaced median y | 209–590 | 2.8 |
| Z | plain y (control) | 617–1391 | 2.3 |
| Z | equal in λ (from q = 0.001) (stopped at 46 cycles) | 15 … 24 200 | > 100 |
| Z | `blend` (α = 0.5) (stopped at about 330 cycles) | 414 … 11 100 | 27 |
| Z | `max(y, λ)`, 11 / 15 interfaces | 670 … 9 365 / 200 … 5 660 | 14 / 28 |

- **cc:** with equal λ spacing the low-q isosurfaces hug the A boundary, so those paths are short. The q ≈ 0.5
  ensemble's paths are long.
- **Z:** past the tilted ridge lies a plateau where q ≈ 0.6–0.85 changes very slowly. There λ is flat, and a
  staple turns only after a long excursion. More interfaces in the plateau (15 instead of 11) shorten its paths
  only to about 5000 frames.
- **Too close to A:** interfaces below q ≈ 3·10⁻³ on cc lie 1–2 MD steps from A for 5% of their crossings, and up
  to 9% of upward crossings skip the next interface (`staple_run.py --check`, on the training run's test paths).
- **A kick bug, fixed:** with interfaces closer than 2% of their value (dense near λ_B), toytis' `kick_star` put
  the kick point beyond the neighbouring interface and retried forever. `toytis_adapter` now uses a tenth of the
  gap in that case.

**Why: the committor is a martingale.** For overdamped dynamics q has no mean drift along a trajectory; it only
diffuses, with D(q) = ⟨Δq²⟩/2 per frame. Two consequences:
- **Crossing probabilities** follow from the levels alone: from q_k, the chance of reaching q_{k+1} before q_{k−1}
  is (q_k − q_{k−1})/(q_{k+1} − q_{k−1}). At small q this needs a bounded ratio q_{k+1}/q_k.
- **Path lengths** follow from D(q): crossing a gap Δq takes about Δq²/D frames.

On Z, D(h) (h = the export's calibrated committor after the A/B ramps) is about 2·10⁻³ per frame on the ridge but
only about 10⁻⁶ in the plateau (`natural.png`). At small q, D ∝ q², so the natural coordinate behaves like ln q
there, as the log1p scale already does.

**Natural coordinate.** u(h) = ∫ dh/√(2D(h)), scaled to [0, 1], is the coordinate in which diffusion is uniform.
Interfaces equally spaced in u take about equally long to cross.
- **Reparametrizing alone changes nothing.** A monotone reparametrization of λ leaves the isosurfaces unchanged, so
  with the same interfaces the sampler does exactly the same. What u changes is where "equally spaced" interfaces
  land.
- **The export.** `run.py natural <system>` estimates D(h) from the training run's lag-1 increments (path weights)
  and exports the same model with λ = λ_A + (λ_B − λ_A)·u(h): a monotone table in `LambdaModel`, the
  `exported/<sys>_<fam>-nat.npz` files.
- **Below the data** D is continued as (h + ε)², as in the log1p scale; **above it**, as constant.
- **Ratio cap.** `staple_run.py --max-ratio R` inserts interfaces until neighbouring levels differ by at most a
  factor R (so crossing probabilities are at least about 1/R). On Z this adds one interface on the ridge.

**Predicting path lengths from the training run alone, without long MD.** A 1D Markov chain in h resamples the
observed increments of each h bin and applies the sampler's turn rule (`pgmoves.turn_detected`). An ensemble
[k*]'s path length is modelled as two independent walks, started just past λ_k and stopped at a turn or in A/B
(`natural.predict_lengths`). Against the test runs (`natural.png`, right panel):
- **Long paths: within 1.3–1.4×.** The predictor finds the plateau ensembles of every Z run: observed / predicted
  median 24 200 / 17 600, 11 100 / 8 600, 8 900 / 7 700, 5 750 / 4 200.
- **Low q: within about 2×,** on both systems.
- **Ridge: far too short (10–60×).** The training run has only about 30–50 frames per h bin there, and h is not
  Markov across the ridge. The real ridge paths are moderate, 200–1400 frames.

So it is reliable as a warning for long paths, which is what placement needs, but not as an absolute cost model.

**Predicted for the natural placement** (11 interfaces requested, plus the ratio cap R = 10):
- **cc:** 174–405 frames, as balanced as the y run, with the first interface at q = 0.012.
- **Z:** 135–3144. With 15 requested (16 after the cap) the longest becomes 1794, and with 19 requested, 1352. More
  ensembles also lower the total cost per cycle.

Tests of the natural placement (all 2000 cycles; medians over the ensembles [1*] …):

| system | placement | median path length | longest / shortest | longest 90th percentile |
|---|---|---|---|---|
| cc | plain y (control) | 220–493 | 2.2 | 933 |
| cc | natural, 11 interfaces | 228–496 | 2.2 | 902 |
| Z | plain y (control) | 617–1391 | 2.3 | 2556 |
| Z | `max(y, λ)`, 11 interfaces | 670–9365 | 14 | 22 695 |
| Z | `max(y, λ)`, 15 interfaces | 200–5660 | 28 | 13 819 |
| Z | natural, 16 interfaces | 388–2940 | 7.6 | 5 734 |

- **cc:** as balanced as plain y, with interfaces starting at q = 0.012.
- **Z:** the natural placement's longest median is half that of the best earlier placement (2940 against 5660),
  and its 90th percentile 2.4× smaller. It is still 2.1× the plain-y run's longest median (1391).
- **Predictor vs. run:** predicted 1794 for the longest median, observed 2940, so the plateau factor is about 1.6
  here, not the 1.3 of the earlier runs. Multiply the predicted longest median by 1.3–1.6.

**Recipe.** `run.py natural <system>`, then `staple_run.py --model <fam>-nat --max-ratio 10`. Raise
`--n-interfaces` until the predicted longest median (printed by `run.py natural`; multiply by 1.3–1.6) is
acceptable.

**What placement cannot fix.** The Z plateau is a property of the committor's shape: its isosurfaces there are far
apart in time. Placement can only add interfaces. The alternative is to change the isosurfaces where q is flat:
blend λ toward y there, steered by D(h) or |∇q|. Where q carries no gradient, its isosurfaces matter least for the
alignment criterion.

Figures (per system: `natural.png` for D(h), the two scales and predicted vs observed; `interfaces.png` for the
potential with the interfaces of each placement):
- Z: `examples/figures_by_potential/2d_zpot/brownian_g10/staple/param_lambda/z_br/natural/`
- cc: `examples/figures_by_potential/2d_curved_channel/brownian_g10/param_lambda/cc_br/natural/`
