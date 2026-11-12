# param_lambda: a parametric order parameter λ(x) from q_np

Fits λ(x) to the nonparametric committor of a staple run, evaluates it on held-out paths, and exports a numpy
forward pass for the toytis sampler. Results: [`docs/param_lambda_report.md`](../../docs/param_lambda_report.md).

| file | step |
|---|---|
| `configs/*.toml` | one per system: data, splits, features, sweeps, metrics, export (fixed seeds) |
| `data.py` | step 2: frame cache identical to `tis_data.prepare_data` + cycles, ensembles, M crossings; splits, pairs |
| `models.py` | step 3: M1 (L1 logistic, P-spline GAM), M2 (GBT screening), M3 (MLP), `fit / predict / export` |
| `evaluate.py` | step 4: isosurface spread, smoothness, turn replay, stopped Z_q, Δr², static metrics |
| `lambda_runtime.py` | step 5: the exported λ (numpy only): features, score, calibration, A/B handling, OOD flag |
| `toytis_adapter.py` | step 5: plugs λ into toytis without changing it |
| `report.py`, `report_intro.md` | step 6: figures and the report |
| `run.py`, `steps.py`, `run_all.sh` | command line; `./run_all.sh` regenerates everything |
| `variants.py`, `configs/loss_variants.toml` | training variants (lean sweep, several seeds), every run on the test paths |
| `embed.py`, `embed_report.py`, `embed_intro.md`, `run_embedded.sh`, `configs/*_emb*.toml` | embedding test: the 2D runs plus 29 non-interacting coordinates ([`docs/param_lambda_embedding.md`](../../docs/param_lambda_embedding.md)) |
| `tests/` | `python -m pytest tests -q` (export = model, continuity, A/B, reversal, adapter, loader, toytis run) |

```bash
python run.py extract z_br      # frame cache (needs the run and its q_np)
python run.py screen z_br       # M2
python run.py fit z_br          # M1 + M3 sweeps (resumes; --fresh to restart, --only m1|m3, --lean: 4 M3 runs)
python run.py evaluate z_br     # test + OOD metrics, TIS plots
python run.py export z_br       # exported/z_br_<family>.npz, µs per frame
python run.py report            # docs/param_lambda_report.md
python run.py variants z_br --seeds 0 1 2    # loss variants (configs/loss_variants.toml), test metrics per run
./run_embedded.sh z_emb cc_emb cc_emb_np     # embedding test; then: OMP_NUM_THREADS=1 python embed_report.py
```

Interface spacing near fixed states: [`docs/param_lambda_interface_spacing.md`](../../docs/param_lambda_interface_spacing.md).

Training time (laptop, RTX 2000 Ada, 8 GB; Z: 5.2M training frames). The checkpoints are scored by the network
on the GPU; the metrics themselves take ~0.7 s per checkpoint (before, the numpy forward pass over 2.7M frames
took ~13 s per checkpoint and ~75% of the sweep). M3 runs take 10–40 s each on Z.

| | Z | cc | flat 1D | bumps 1D |
|---|---|---|---|---|
| full M3 sweep (36 / 36 / 12 / 12 runs) | 12.5 min | 4.5 min | 0.7 min | 3.1 min |
| `fit --lean` M3 (4 runs: raw CVs, 64×64 and 128×128×128, β ∈ {0, 10}) | 80 s | 28 s | 14 s | 60 s |
| whole `fit` step (M1 + M3 + loading), full sweep | 15 min | 5 min | 1 min | 3.5 min |

The lean sweep selects the same M3 as the full one on Z (and M3-fast 64×64 β=10); on cc and 1D all M3 runs tie
within a few % in the selection metric. Training time grows about linearly with frames × network size.
Run long steps in their own unit:
`systemd-run --user --unit=pl-z --collect -p MemoryMax=12G -p MemorySwapMax=0 --working-directory=$PWD ./run_all.sh z_br`.

## On the workstation (RTX 5080)

1. Environment. Blackwell (sm_120) needs torch ≥ 2.7 built for CUDA 12.8; the laptop's torch 2.5.1+cu121 will not
   run on it.
   ```bash
   conda create -n pl python=3.12 numpy scipy scikit-learn pandas matplotlib pytest
   conda activate pl
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install cupy-cuda12x                      # only for fit-qnp and the TIS plots
   pip install -e /path/to/tistools              # only for extract
   ```
2. Data. The frame caches hold everything that `screen`, `fit`, `evaluate` and `export` need. They are git-ignored,
   so copy them:
   ```bash
   rsync -a --include='*/' --include='frames.npz' --include='importance.json' --exclude='*' \
       laptop:.../examples/figures_by_potential/ examples/figures_by_potential/
   ```
   (each system's folder is `common.out_dir(<system>)`, e.g. `2d_zpot/brownian_g10/staple/param_lambda/z_br/`; the exact
   committors are in each potential's `exact/` folder and are in git)
   `extract` and `fit-qnp` also need the toytis runs and, for Z, `examples/figures_by_potential/2d_zpot/brownian_g10/staple/reversed/`:
   set `tis_dir` in the configs.
3. Run: `PY=$(which python) ./run_all.sh z_br`, or the steps above, from `examples/param_lambda`.
   `MLP` picks `cuda` automatically.
