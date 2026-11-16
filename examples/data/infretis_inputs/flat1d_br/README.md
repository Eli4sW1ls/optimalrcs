# 1D flat potential with harmonic walls, overdamped: infretis StapleTIS input

One particle on a flat potential (V = 0 on [-0.2, 0.4], harmonic walls k = 100 outside), the infretis `examples/ase/staple_flat_walls_*` system in reduced units: kT = 1, gamma = 5, dt = 2e-4, mass 1. A: x <= -0.1, B: x >= 0.3, interfaces -0.1, 0, 0.1, 0.2, 0.3. The toytis flat staple run `sim_istar0br10intf` has gamma = 1 and B = 0.8 instead. Exact committor of the discrete chain: `figures_by_potential/1d_flat/exact/` (`ase_systems.exact_committor`).

| | |
|---|---|
| sampler | infretis `mode = "staple"`, `staple_infinite_swap = False`, 4 workers, `steps = 20000` cycles |
| engine | ASE `overdamped_lgv`, timestep 0.0020361 fs, friction 0.491135 / fs, T = 11604.5 K (reduced units: sigma = 1 A, epsilon = 1 eV, m = 1 amu, time unit 10.18 fs) |
| interfaces | [-0.1, 0.0, 0.1, 0.2, 0.3] |
| toytis run | sim_istar0br10intf (same potential; other gamma and B) |
| initial paths | `load_copy/0..4`: one valid staple path per ensemble (copied from the order-parameter loop: figures_by_potential/_shared/param_lambda/op_loop/staple_weights/seeds) |

Start: `bash runner.sh` (cleans, copies `load_copy` to `load`, runs `infretisrun -i infretis.toml`). The paths of the
checkouts are set in `runner.sh`; `ToytisPotential` loads the potential from the toytis checkout (`$TOYTIS`).
`delete_old = false`: every accepted path is kept in `load/`.

Analysis: `python examples/load_tis_data_cupy.py <this folder> --reversed --h5 ...` or `examples/infretis_staple_1d.ipynb`
(`tis_data.prepare_data` reads infretis runs; weights and the cut at M: `docs/npneq_tis_notes.md` section 15; the loop's
choice of weights: `docs/op_loop_report.md`). Rates: `inftools.istar`.

The ASE overdamped integrator makes the same step as toytis' Brownian engine, x += dt F / (m gamma) + N(0, 2 kT dt / (m gamma)),
so the exact committors of the discrete toytis chains apply.
