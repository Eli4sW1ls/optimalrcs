# 2D Z potential (toytis `sjoelbak_angle`, `RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)`)

y is the order parameter, x the transverse coordinate; A: y ≤ 0.05, B: y ≥ 0.85 (interfaces 0.05, 0.13, ..., 0.85).

## brownian_g10: overdamped, γ = 10, kT = 0.05, dt = 0.002 (staple `sim_istarz_br_g10`, MD `md_zbr_g10_s1..16`)

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `reference_and_analyses/` | `z_brownian` | finite-volume reference committor (`committor_reference_h*.npz`, read by many notebooks), engine shooting, MD and synthetic-data tests, staple splits |
| `md/` | `nb_zbrownian_md` | fits on the long MD |
| `staple/original/` | `istarz_br_g10` | original staple fit (3000 paths/ensemble, forward pieces) |
| `staple/original_diag/` | `istarz_br_g10_diag` | the same with the ridge-normal CV y + x |
| `staple/reversed/` | `istarz_br_g10_reversed` | forward + mirrored pieces, continued fits, even/odd halves, lag 50 |
| `staple/subset_test/` | `istarz_test` | first tests on a subset of the run |
| `staple/notebook/` | `nb_brownian_staple` | `base_documented_2d_brownian_staple.ipynb` (per seed, new fixed-length trajectories) |
| `staple/staple_compare_p1000_s12/` | `staple_compare/sim_istarz_br_g10_p1000_s12` | `staple_data_choices.ipynb` |
| `staple/selection_test/` | `selection_test` | balancing the TIS data over regions (path selection, blind starts) |
| `staple/param_lambda/` | `param_lambda/z_br`, `z_emb` | parametric λ models |
| `short_trajectories/` | `nb_zpot_short_traj` | `zpot_short_trajectories.ipynb`: short trajectories from path data, fixed/geometric/turn stopping |
| `checks/crossing_check/` | `crossing_check/run_z.py`, `z_check.*` | MSM crossing check |
| `checks/reference_check/` | `reference_check/z_*` | reference committor check |
| `checks/poisson_trap/`, `checks/zq_weights/` | `poisson_trap`, `zq_weights` | Z_q trap and weight tests |
| `overview_plots/` | `overview_plots/zmd*`, `zstaple*` | plots for `docs/results_overview.md` |

## langevin_g2: Langevin, γ = 2, kT = 0.05, dt = 0.002 (staple `sim_istarz_2603`, positions only)

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `fd_velocity/` | `zlgv_fd` | finite-difference velocities as CVs, lag 1 and 4 |
| `history/` | `zlgv_history` | history CVs |
| `short_trajectories/` | `nb_zpot_short_traj_langevin` | `zpot_short_trajectories_langevin.ipynb` |
| `overview_plots/` | `overview_plots/zlgv*` | plots for `docs/results_overview.md` |

`exact/` (`exact_committors/zpot`, `zpot_exact.py`): exact committor of the staple dynamics.
