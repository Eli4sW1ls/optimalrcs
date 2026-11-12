# 2D curved channel (toytis `curved_channel.CurvedChannelDoubleWell`)

Defaults: Vb = 0.2, eps = 0, y0 = 0.5, w = 0.4, xA = -0.15, xB = 0.15, b = 0.2, k = 15, a = 0. y is the order parameter;
interfaces 0.15, 0.22, ..., 0.85 (A: y ≤ 0.15, B: y ≥ 0.85); kT = 0.05, dt = 0.005.

| Folder | Old folder (figures/) | Contents |
|---|---|---|
| `exact/` | `exact_committors/cc`, `cc_exact.py`, `cc_run.log` | exact committor of the overdamped dynamics |
| `toy2d_check/` | `toy2d_check` | potential and committor check |
| `brownian_g10/staple/` | `nb_toy2d_staple/sim_istar_cc_br` | `toy2d_staple_analysis.ipynb` on `sim_istar_cc_br` (γ = 10) |
| `brownian_g10/staple_test/` | `nb_toy2d_staple/test_cc_br` | the same on the test run |
| `brownian_g10/staple_compare_p2000_s0/` | `staple_compare/sim_istar_cc_br_p2000_s0` | `staple_data_choices.ipynb` |
| `brownian_g10/md/`, `md_test/` | `nb_toy2d_md/CurvedChannelDoubleWell/(TEST_)brownian_*` | `toy2d_md.ipynb`: MD and fits |
| `brownian_g10/checks/` | `crossing_check/run_cc.py`, `reference_check/cc_*` | MSM crossing check, reference check |
| `brownian_g10/param_lambda/` | `param_lambda/cc_br`, `cc_emb`, `cc_emb_np` | parametric λ models |
| `langevin_g2/staple/`, `staple_test/` | `nb_toy2d_staple/sim_istar_cc_lgv`, `test_cc_lgv` | `toy2d_staple_analysis-lgv.ipynb` (γ = 2) |
| `langevin_g2/md/`, `md_test/` | `nb_toy2d_md/CurvedChannelDoubleWell/(TEST_)langevin_*` | `toy2d_md-lgv.ipynb` |
