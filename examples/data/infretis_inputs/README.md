# infretis StapleTIS inputs

Self-contained run folders (`bash runner.sh`), made by `../make_infretis_inputs.py` from the order-parameter loop's
seed-lambda production runs and their seed paths. Positional order parameter, infinite swap, ASE overdamped Langevin.

| Folder | System | toytis run |
|---|---|---|
| `flat1d_br/` | 1D flat potential with harmonic walls, overdamped | sim_istar0br10intf (same potential; other gamma and B) |
| `z_br/` | 2D Z potential, overdamped | sim_istarz_br_g10 |
| `cc_br/` | 2D curved channel, overdamped | sim_istar_cc_br / test_cc_br |

Not included (no ASE system definition and no seed paths in the loop yet): the 1D two-bump and cosine-well runs,
the Langevin versions, and the staggered channels.
