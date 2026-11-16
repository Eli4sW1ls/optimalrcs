# 3D Z potential with a coupled slow coordinate w (tunnelling test)

The tunnelling case of the param_lambda embedding test: a slow coordinate w, orthogonal to y, that changes the
committor. The uncoupled case (w appended to the 2D runs) is in `docs/param_lambda_embedding.md`.

## Potential (toytis `potentials/zpot_w.py`, `ZChannelsW`)

Positions (y, x, w). Two copies of the Z potential (Lx = 0.3, 45° ridge) whose ridges are mirror images, selected
by w:

U = C(y, x) + s(w) R(y, x) + (1 − s(w)) R(y, Lx − x) + dV_w ((w/w0)² − 1)²,  s(w) = (1 − tanh(w/δ))/2

- C is the Z potential without its ridge: walls, state wells, slope, gravity. R is the Z ridge (line x + y = 0.6),
  and its mirror image lies on y − x = 0.3.
- Channel w ≈ −w0 is the Z potential itself; channel w ≈ +w0 is its mirror image.
- Defaults: w0 = 0.1, δ = 0.02, dV_w = 3 kT (kT = 0.05).
- Between the ridges (0.3 < y < 0.6) a point is on the A side of one channel's ridge and on the B side of the
  other's. So the committor depends on w there, and neither y nor (y, x) resolves it.
- A path can also pass a ridge through w: climb it, hop into the other channel, and carry on. In the (y, x)
  projection it crosses the ridge without paying for it.
- Symmetry: U(y, x, w) = U(y, Lx − x, −w). The two channels are equivalent, so every ensemble should hold w > 0
  half of the time. That is a built-in check that the sampler has equilibrated the channels.

Checks (2026-11-13):
- The written-out force equals the composition of the Z class to 1e-13.
- force = −∇(U + gravity·y) holds, and so does the mirror symmetry.
- One call costs 2.1 µs (the 2D Z class: 5.0 µs).
- The CUDA force in `zpotw_gpu.py` matches it to 2e-14.

## Choosing dV_w (`zpotw_gpu.py`, GPU Brownian dynamics, the run's settings)

`scan`: 20000 walkers × 2·10⁶ steps from A for each barrier (`explore/scan_w0.1_d0.02.json`, `logs/scan.log`).

| dV_w | steps between hops in A | A↔B transitions per 10⁹ steps | transition paths ending in the other channel |
|---|---|---|---|
| 2 kT | 4 100 | 96 | 0.56 |
| **3 kT** | **7 800** | **84** | **0.58** |
| 4 kT | 16 000 | 74 | 0.56 |
| 5 kT | 35 000 | 60 | 0.44 |
| 8 kT (w frozen: plain Z) | 430 000 | 47 | 0.07 |

- Lower barriers open the route through w: more transitions, and most transition paths change channel on the way.
- The Z staple paths are about 1000 steps long (median 1032, 99th percentile 3400). At 3 kT, w is slow on that
  scale, and the 300-cycle pilot confirms it: 3–14% of the accepted staple paths change channel.
- The w dynamics are still fast enough for the sampler to equilibrate the channels over 10⁵ cycles.
- At 5 kT and above, equilibrating the channels becomes the bottleneck.

`shoot` (4000 trajectories per point, `logs/shoot.log`), committor at y = 0.45:

| x | q (w = −w0) | q (w = +w0) |
|---|---|---|
| 0.05 | 0.020 | 0.29 |
| 0.15 | 0.16 | 0.15 |
| 0.25 | 0.30 | 0.017 |

- In both triangles between the ridges the channels differ by 10–15× (100× with w frozen, at 8 kT).
- On the centre line x = 0.15 they agree, and q(y, x, −w0) = q(y, 0.3 − x, +w0) within the error bars.

## The staple run (toytis `notebooks/zpot_w_run.py`)

The settings are those of `sim_istarz_br_g10` (overdamped, γ = 10, dt = 0.002, kT = 0.05, the same 11
interfaces on y, 10⁵ cycles), with dim = 3. order.txt gets the columns y, x and w.

The sampler is unchanged; `notebooks/nd_patches.py` makes two changes at run time:
- The order parameter writes all three positions; op[0] = y is still the order parameter.
- `kick_star` and `kick_retis` make 3D kick points. They are the original functions with two checked edits to
  their source text: the dim-2 branch is taken for every dim ≥ 2, and w is drawn near ±w0, channel at random.

```bash
cd /mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis
systemd-run --user --unit=zw-s1 --collect -p MemoryMax=8G -p MemorySwapMax=0 --working-directory=$PWD \
    /home/elias/anaconda3/envs/test/bin/python -u notebooks/zpot_w_run.py --seed 1
# -> simulations/sim_istarzw_br_g10_dV3_s1 (a fresh run deletes that directory first)
# continue / extend:  ... notebooks/zpot_w_run.py --seed 1 --restart --max-cycles 150000
```

Pilot (`simulations/test_zw_pilot`, 300 cycles):
- 1.96·10⁶ MD steps in 47 s: 23 µs per step including the sampler, and 6.5·10³ steps per cycle (as in Z).
- 10⁵ cycles: about 4 h on one core.
- order.txt grows by about 0.3 MB per cycle, so about 30 GB per 10⁵ cycles unthinned. The Z run's files hold
  3000 paths per ensemble, and the loader keeps 1500 (`--paths-per-ensemble`).
- Several seeds can run in parallel, one core each.

## Analysis (optimalrcs `examples/param_lambda`, config `zw_br`)

Results so far: `docs/param_lambda_coupled_w.md`. After any refit, check the models against the exact
shooting committor with `python check_vs_shooting.py`.

```bash
python run.py fit-qnp zw_br && python run.py extract zw_br && python run.py screen zw_br
python run.py fit zw_br --lean && python run.py evaluate zw_br
```

- The tis_data loader reads the three columns; checked on the pilot.
- Burn-in: paths before 40% of the cycles are not used (`split.id_from`), which also covers the channel equilibration.
- Still to do: the exact 3D committor. It needs a 3D version of `exact_tools.solve_brownian_2d`, with two ridge
  families and about 60–80 w planes (a few million grid points): GPU hours. Until then the spread is measured
  against q_np, and `zpotw_gpu.py shoot` gives point checks.
