# 2D staggered channels (toytis `potential_staggered_channels.PotentialStaggeredChannels`)

Settings of `notebooks/staggered_channels_run.py`: V_peak = 0.5/0.07·kT (7.1 kT), V_dip = 3 kT, x_kink = 0.4, walls a = 100,
c = 0.15; kT = 0.25, m = 1. The order parameter is toytis position 0 (x); states A: x ≤ -1.6, B: x ≥ 1.6. Barrier tops at
x = -0.6 (upper channel) and +0.6 (lower channel).

| Folder | Contents |
|---|---|
| `exact/langevin_g0.1/` | Langevin (BBK, friction 0.1, dt 0.02: the script defaults) reference by GPU shooting (`_shared/exact_committors/lgv2d_exact.py stag`, `lgv2d_stag_cuts.py`): velocity-averaged q̄ and q at fixed v_op = -s, 0, +s on a 0.1 grid (10000 shots/point, s.e. ≤ 0.005), and fine cuts through the two barrier tops along the channel centres (`cuts.png`), and `fq.png`: F(q) of the Langevin phase-space density (18000 phase points × 1000 shots, a second batch at the barrier tops; plateau ≈ 15 kT, ±1–2 kT). In the maps "y" is the order parameter (toytis x) and "x" the transverse coordinate (toytis y) |

Per-frame references for a run with recorded velocities: `lgv2d_tools.shoot(lgv2d_tools.SYSTEMS['stag'], op, transverse, v_op, v_transverse, n_shots)`.
The vectorized force (`lgv2d_tools.stag_force`) matches the toytis class to 1e-13. Other frictions or dt: change `SYSTEMS['stag']`.
