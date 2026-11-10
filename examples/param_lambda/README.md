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
| `tests/` | `python -m pytest tests -q` (export = model, continuity, A/B, reversal, adapter, loader, toytis run) |

```bash
python run.py extract z_br      # frame cache (needs the run and its q_np)
python run.py screen z_br       # M2
python run.py fit z_br          # M1 + M3 sweeps (resumes; --fresh to restart, --only m1|m3)
python run.py evaluate z_br     # test + OOD metrics, TIS plots
python run.py export z_br       # exported/z_br_<family>.npz, µs per frame
python run.py report            # docs/param_lambda_report.md
```

On the laptop (RTX 2000 Ada, 8 GB) the Z sweep takes ~2 h: 36 M3 runs, of which the 128×128×128 ones take
~5 min each including the checkpoint evaluations. The other systems take minutes. Run long steps in their own unit:
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
   so copy them, together with the exact committors used by the report's maps:
   ```bash
   rsync -a --include='*/' --include='frames.npz' --include='importance.json' --exclude='*' \
       laptop:.../examples/figures/param_lambda/ examples/figures/param_lambda/
   rsync -a laptop:.../examples/figures/exact_committors/{zpot,cc,brownian1d} examples/figures/exact_committors/
   ```
   `extract` and `fit-qnp` also need the toytis runs and, for Z, `examples/figures/istarz_br_g10_reversed/`:
   set `tis_dir` in the configs.
3. Run: `PY=$(which python) ./run_all.sh z_br`, or the steps above, from `examples/param_lambda`.
   `MLP` picks `cuda` automatically.
