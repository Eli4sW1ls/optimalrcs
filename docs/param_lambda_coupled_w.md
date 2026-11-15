# Coupled slow coordinate w (tunnelling case): first results

The tunnelling case of the embedding test (`param_lambda_embedding.md`): a slow coordinate w, orthogonal to the
order parameter y, that changes the committor. The system, its parameters and the staple run are described in
`examples/figures_by_potential/3d_zpot_w/README.md`.

In brief:
- **Potential.** Two mirror-image Z channels, chosen by w through a double well (minima at w = ±0.1, barrier 3 kT).
- **Where w matters.** Between the two ridges, the committor at a given (y, x) depends strongly on w. Neither y nor
  (y, x) resolves it.
- **Run.** One StapleTIS seed, 100 000 cycles, with the settings of the 2D Z run and y as the order parameter.
  The pipeline (`run.py fit-qnp / extract / screen / fit --lean / evaluate / export zw_br`) ran on another machine
  (commit `bb2dce9`).

## Result (2026-11-15): the labels miss w and x, so nothing downstream can find them

**The run itself is fine.** Every ensemble [1*] … [9*] holds w > 0 in 45–55% of its frames in every fifth of the
cycles, and 6–14% of its paths change channel (`3d_zpot_w/check_run.py`). The ensembles [0⁻] and [0*] wander
(0.23–0.65), because their paths are 6–35 frames long and w hardly moves within them. In A the two channels are
identical by symmetry, and A frames are not training data.

**q_np is close to a function of y.**
- Its plot against y (`zw_br/qnp/committor_vs_order.png`) is a narrow band. The true committor at y = 0.45
  ranges from 0.02 to 0.30 with x and w (shooting, below).
- Against q_np, the plain y seed has isosurface spread 0.030 (floor 0.013) and logit RMSE 0.11. On the 2D Z run
  the seed has 1.38 and 1.50.
- M2 screening agrees: permutation importance y 6.85, everything else ≤ 0.06.

**The NPNEq fit was not converged** (`zw_br/qnp/metrics.png`). Over its 30 000 iterations Δr² fell from about
880 to 700, still steadily at the end, and the maximum stopped Z_q is still 17–22 and noisy. The converged 2D Z
fit (25 000 iterations, 2 CVs) reached zqs_z 1.36.

**Exact check** (`3d_zpot_w/check_vs_shooting.py`). The shooting committor at 27 (x, y) points in both channels,
4000 trajectories each (`zpotw_gpu.py shoot`). Points with 0.003 < q < 0.997:

| model | logit RMSE against shooting | channel contrast (RMS logit q(+w0) − q(−w0)) |
|---|---|---|
| shooting (truth) | – | 1.82 |
| M3 (y, x, w; 128×128×128) | 1.22 | 0.12 |
| M3-fast (64×64) | 1.22 | 0.11 |
| M3-yx (without w) | 1.22 | 0 |
| M1-GAM | 1.28 | 0.11 |
| seed (y) | 1.27 | 0 |

- **No model sees the channels or x.**
  - At y = 0.45, x = 0.05 the truth is 0.020 / 0.286 (w = −w0 / +w0). M3 gives 0.071 / 0.068.
  - Along x at fixed w the truth varies 15-fold (0.02 → 0.30). M3 stays at 0.07–0.08.
- **They are also off in absolute value.** At y = 0.6–0.65, M3 gives 0.19–0.29 where the truth is 0.38–0.51.
  Overall the logit RMSE is the same as for y alone.
- **M3's small w dependence is a bias, not structure.** q(−w0) > q(+w0) holds at every point, also where the truth
  is reversed. That is a channel offset carried by path-correlated label errors, as in the uncoupled test
  (`param_lambda_embedding.md`, finding 4), not the mirrored dependence.

**Why the pipeline's own metrics did not flag it.**
- **They are all relative to q_np.** Isosurface spread is against q_np (M3 0.015, M3-yx 0.017, seed 0.030). The
  selection metric also compares to q_np: test SD(Z_q) is 4.9–5.0 for M3, 5.1 for M3-yx, 4.8 for q_np and 7.2 for
  y. The models reproduce the labels faithfully, so these numbers look good.
- **The only internal warning was in the NPNEq log:** Δr² still falling, and zqs_z far above 1–2.

## What follows

1. **The w question is still open.** This run tests the labels, not the parametric λ. The comparison M3 against
   M3-yx is meaningful only once q_np resolves the channels.
2. **Refit q_np until it converges.** Run longer, until Δr² flattens and zqs_z drops to 1–2, as on the 2D runs.
   With 3 CVs and a w–x interaction this likely needs well beyond 30 000 iterations, which is a workstation job.
   Watch for the w path-label effect: check the channel contrast against shooting, not only the relative
   metrics.
3. **Keep an external reference in the loop.** Run `check_vs_shooting.py` after every q_np or model change. A
   denser reference needs the exact 3D committor (a 3D extension of `exact_tools.solve_brownian_2d`), still to
   do.
4. **For real systems without a reference:** report the NPNEq convergence trace (Δr², stopped Z_q) with every
   q_np. Treat a still-falling Δr² as "labels not usable". A good spread or SD(Z_q) relative to q_np says nothing
   about the committor itself.
