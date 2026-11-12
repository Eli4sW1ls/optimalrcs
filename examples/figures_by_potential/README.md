# Figures and results, by potential

A structured copy of `examples/figures/`, made by `examples/reorganize_figures.sh`. Sorted by **potential**, then by
**dynamics** (overdamped `brownian_g<gamma>` or `langevin_g<gamma>`), then by kind of run (`staple` = staple/iSTAR TIS path
data, `md` = plain MD, `exact` = reference committors, `checks` = method checks).

| Folder | Potential | Runs |
|---|---|---|
| `1d_flat/` | flat 1D potential | staple `sim_istar0br10intf` (overdamped), staple `sim_istar0lgv15intf` (Langevin), long overdamped MD |
| `1d_two_bumps/` | two cosine bumps of 1 kT on [-0.1, 0.3] | staple `sim_istar2cosbr_2108` (param_lambda), long overdamped MD |
| `1d_cos_well/` | toytis `cos_well.CosWellWalls` | Langevin MD `md_istarwell2_100M`, Langevin staple `sim_istarwell2rv` |
| `2d_zpot/` | toytis `sjoelbak_angle` (Z potential, Lx = 0.3, 45° ridge) | overdamped γ = 10 (`sim_istarz_br_g10`, MD `md_zbr_g10_s*`), Langevin γ = 2 (`sim_istarz_2603`) |
| `2d_curved_channel/` | toytis `curved_channel.CurvedChannelDoubleWell` (defaults) | overdamped γ = 10 (`sim_istar_cc_br`), Langevin γ = 2 (`sim_istar_cc_lgv`), MD |
| `_shared/` | several | tools and results that cover more than one potential |
| `_unsorted/` | ? | files the script had no rule for, under their old path |

Each potential folder has its own README with the old folder of every subfolder.

**Old paths.** `examples/figures/` is gone: notebooks, scripts and docs use the paths in this tree. `MANIFEST.tsv`
(per machine, not in git) lists every file as `old path (under figures/)` and `new path (under figures_by_potential/)`,
for finding where an old path went.

**Git.** As in `figures/`, numpy files (`*.np*`) are not tracked except under `*/exact/`; see `.gitignore`. The
workstation and the laptop therefore hold different files in the same folders; run the script on each machine.
