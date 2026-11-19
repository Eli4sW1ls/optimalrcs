# Running loop_z_lgv (underdamped Z) on the workstation

The learning loop on the Z potential with ASE's Langevin integrator, at the settings of the toytis run
`sim_istarz_2603`: γ = 2, kT = 0.05, m = 1, dt = 0.002. The states are A: y ≤ 0.05 and B: y ≥ 0.85, the CVs are
(y, x, v_y, v_x), and the order parameter is λ(x, v).

| file | what it is |
|---|---|
| system | `ase_systems.z_lgv` |
| loop config | `examples/param_lambda/configs/loop_z_lgv.toml` (inherits `loop_z`) |
| MD reference rate | `loop/md_z_lgv.py` |

Tested on the laptop: pass 0 (infinit) starts, and order.txt holds [λ = y, y, x, v_y, v_x]. The vectorized force of
the MD script matches the toytis potential that the ASE calculator uses.

## Steps

All from `examples/param_lambda`, in the environment with the infretis checkout first on `sys.path`, as for the
other loops.

1. **Reference rate.** This system has no exact phase-space committor, so the rate comes from long MD of the same
   chain:

   ```bash
   python -m loop.md_z_lgv --walkers 16384 --steps 4000000
   ```

   - The script writes `examples/figures_by_potential/2d_zpot/langevin_md_ase/rate.json`, which `z_lgv` reads as
     its reference rate.
   - Check `n_ab` in the output: the error is about 1/√n_ab, so aim for at least 1000 transitions. If there are
     fewer, rerun with a larger `--steps`.

2. **The loop** (pass 0, the passes, and the gated freeze):

   ```bash
   systemd-run --user --unit=loop-zlgv --collect -p MemoryMax=16G -p MemorySwapMax=0 --working-directory=$PWD \
       python -u run.py loop loop_z_lgv
   ```

3. **Production and its rate:**

   ```bash
   python -u run.py loop loop_z_lgv --production
   ```

4. **Seed-λ control.** The same production with λ = y, for the comparison:

   ```bash
   python -m loop.experiments.seed_control setup loop_z_lgv --cycles 20000
   python -m loop.experiments.seed_control run loop_z_lgv
   ```

5. **Block errors and convergence:**

   ```bash
   python -m loop.pcross_errors loop_z_lgv
   ```

## What to bring back

From `examples/figures_by_potential/_shared/param_lambda/op_loop/loop_z_lgv/`:
- `state.json`, `freeze.json`, `loop.log`;
- `models/`;
- every `pass_*/manifest.json` and `pass_*/fit.json`;
- `production/` and `production_seed/`: `rates.json`, `infretis_data.txt`, `error_analysis/`.

The `pool/` is only needed for refits and figures that read the frames. Also bring back
`examples/figures_by_potential/2d_zpot/langevin_md_ase/rate.json`.

For the well (`loop_well_lgv`), which also ran on the workstation:
- Bring back `loop_well_lgv/models/lambda_7.npz`, the frozen model. `loop/results.py` finds it there and draws the
  well's panel of the results figure.
- Its seed-λ control (`production_seed`) has no `rates.json` here; run step 4 for `loop_well_lgv` if it was not run.

## Not supported yet

`loop.plots` draws maps for 2D systems and for the 1D (x, v) plane. A 2D system with velocity CVs needs a slice,
for example v = 0, or the velocity average. Its per-pass figures therefore fail. The rate figures
(`pcross_errors`, `plots --pcross-only`) and `loop.results` work.
