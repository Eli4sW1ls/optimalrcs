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
