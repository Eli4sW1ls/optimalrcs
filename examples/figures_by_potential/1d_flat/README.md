# 1D flat potential

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `exact/` | `exact_committors/brownian1d` (flat files), `brownian1d_exact.py`, `br1d_run.log` | exact committors of the flat runs |
| `brownian_staple/first_fits/` | `sim_istar0br10intf` | first fits on the overdamped staple run `sim_istar0br10intf` (γ = 1, kT = 1, dt = 2e-4; A x ≤ -0.1, B x ≥ 0.8) |
| `brownian_staple/staple_compare_p1000_s0/` | `staple_compare/sim_istar0br10intf_p1000_s0` | `staple_data_choices.ipynb`: weights, mirrored pieces, cut, piece length |
| `brownian_staple/crossing_check/` | `crossing_check/run_1d.py` | MSM crossing check |
| `brownian_staple/overview_plots/` | `overview_plots/flat_*` | plots for `docs/results_overview.md` |
| `brownian_staple/param_lambda/` | `param_lambda/flat1d` | parametric λ models |
| `brownian_long_md/npq/`, `plots/` | `longmd_npq`, `longmd_plots` (flat files) | NPq on the long overdamped MD (A λ ≤ -0.1, B λ ≥ 0.3) |
| `langevin_infretis_retis/gamma5_fd_velocity/` | (new, 2026-11-13) | infretis RETIS run `F_cond_analysis/1D_ASE_infretis/flat_lgv/gamma5` (ASE Langevin; A x ≤ -0.1, B x ≥ 0.1; interfaces -0.1, 0, 0.1), read with `infretis_data.py` from an h5 archive: CommittorNE on (x, Δx) with `--fd-velocity 1` (`fd1/`, zqs_z 2.2) and `4` (`fd4/`, 4.4), reversed pieces, 5000 iterations; `q_x_dx.png` = q(x, Δx) |
| `langevin_staple/lgv_check/` | `lgv_check` | Langevin staple run `sim_istar0lgv15intf` (γ = 1, kT = 1, m = 1, dt = 2e-4; A -0.1, B 1.3): lag, basis and velocity checks |
