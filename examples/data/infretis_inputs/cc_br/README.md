# 2D curved channel, overdamped: infretis StapleTIS input

Toytis `curved_channel:CurvedChannelDoubleWell` (defaults) on the particle's (y, x) positions; gamma = 10, kT = 0.05, dt = 0.005, mass 1. A: y <= 0.15, B: y >= 0.85, 11 interfaces 0.15, 0.22, ..., 0.85. Order parameter = y (CV columns of order.txt: y, x). Exact committor of the discrete chain: `figures_by_potential/2d_curved_channel/exact/` (`q_exact_h0.0015.npz`).

| | |
|---|---|
| sampler | infretis `mode = "staple"`, `staple_infinite_swap = True`, 4 workers, `steps = 20000` cycles |
| engine | ASE `overdamped_lgv`, timestep 0.0509025 fs, friction 0.982269 / fs, T = 580.226 K (reduced units: sigma = 1 A, epsilon = 1 eV, m = 1 amu, time unit 10.18 fs) |
| interfaces | [0.15, 0.22, 0.29, 0.36, 0.43, 0.5, 0.57, 0.64, 0.71, 0.78, 0.85] |
| toytis run | sim_istar_cc_br / test_cc_br |
| initial paths | `load_copy/0..10`: one valid staple path per ensemble (copied from the order-parameter loop: figures_by_potential/_shared/param_lambda/op_loop/loop_cc/production_seed_seeds) |

Start: `bash runner.sh` (cleans, copies `load_copy` to `load`, runs `infretisrun -i infretis.toml`). The paths of the
checkouts are set in `runner.sh`; `ToytisPotential` loads the potential from the toytis checkout (`$TOYTIS`).
`delete_old = false`: every accepted path is kept in `load/`.

Analysis: `python examples/load_tis_data_cupy.py <this folder> --reversed --h5 ...` or `examples/infretis_staple_1d.ipynb`
(`tis_data.prepare_data` reads infretis runs; weights and the cut at M: `docs/npneq_tis_notes.md` section 15; the loop's
choice of weights: `docs/op_loop_report.md`). Rates: `inftools.istar`.

The ASE overdamped integrator makes the same step as toytis' Brownian engine, x += dt F / (m gamma) + N(0, 2 kT dt / (m gamma)),
so the exact committors of the discrete toytis chains apply.
