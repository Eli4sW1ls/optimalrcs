# 1D cos well (toytis `potentials/cos_well.py`, `CosWellWalls`), Langevin

A: λ ≤ -0.35, B: λ ≥ 0.35. Long MD `md_istarwell2_100M` (dt = 0.005, positions and velocities); staple run with recorded
velocities `sim_istarwell2rv`.

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `exact/` | `exact_committors/well`, `well_exact.py`, `well_run.log` | exact q(x, v) for the Euler and BBK integrators |
| `langevin_md/seeds/` | `langevin_seeds` | `base_documented_1d_langevin.ipynb`, per fit seed |
| `langevin_md/seeds_dt5e-4/` | `langevin_seeds_dt5e-4` | the same at dt = 5e-4 |
| `langevin_md/history_seeds/` | `langevin_hist_seeds` | fits with a history CV (y(t-d), y(t)) |
| `langevin_md/history01/` | `langevin_hist01` | history 0,1 fit checks (gamma, envelope) |
| `langevin_md/dt5e-4_check/` | `langevin_dt5e-4` | MD committor check at dt = 5e-4 |
| `langevin_md/fits/` | `well_md_fit` | longer fits on 6M MD frames (energy CV, lagged (x, v)) |
| `langevin_md/overview_plots/` | `overview_plots/md_well_*` | plots for `docs/results_overview.md` |
| `langevin_staple/checks/` | `well_check` | fits on the staple paths: recorded vs finite-difference velocity, mirrored pieces, lag; brute-force reference |
| `langevin_staple/notebook/` | `nb_well_staple` | staple notebook fits |
| `langevin_staple/overview_plots/` | `overview_plots/wellpath_*` | plots for `docs/results_overview.md` |
