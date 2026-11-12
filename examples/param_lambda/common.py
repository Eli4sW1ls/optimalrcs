"""Paths, config loading and imports shared by the parametric-order-parameter pipeline.

The pipeline fits a parametric lambda(x) to the nonparametric committor q_np of a staple run,
evaluates it on held-out paths, and exports a numpy forward pass for the toytis sampler:

    run.py extract  -> frame cache (CVs, q_np, ids, cycles, weights, exact q)
    run.py screen   -> M2 gradient-boosted-tree feature ranking
    run.py fit      -> M1 (sparse logistic, spline GAM) and M3 (MLP) sweeps
    run.py evaluate -> metrics table on the test paths and the OOD block
    run.py export   -> calibrated numpy model (lambda_runtime.py) for toytis_adapter.py
    run.py report   -> figures and the markdown tables
"""
import importlib.util
import os
import sys
import tomllib

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXAMPLES = os.path.join(REPO, 'examples')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(EXAMPLES, 'figures', 'param_lambda')
TOYTIS = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis'

Q_CLIP = 1e-6

if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)


def _merge_tables(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge_tables(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_toml(name_or_path):
    path = name_or_path if name_or_path.endswith('.toml') else os.path.join(HERE, 'configs', name_or_path + '.toml')
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    # `inherit = "<config>"`: start from that config and override table by table (embedded systems, variants)
    if 'inherit' in cfg:
        cfg = _merge_tables(_read_toml(cfg.pop('inherit')), cfg)
    return cfg


def load_config(name_or_path):
    """Config dict from configs/<name>.toml (or a path). Relative paths in [data] (and [embed] labels) are taken from
    the repo root."""
    cfg = _read_toml(name_or_path)
    for key in ('q_np', 'check_against', 'exact_file'):
        if key in cfg['data'] and not os.path.isabs(cfg['data'][key]):
            cfg['data'][key] = os.path.join(REPO, cfg['data'][key])
    emb = cfg.get('embed', {})
    if emb.get('labels', 'base') != 'base' and not os.path.isabs(emb['labels']):
        emb['labels'] = os.path.join(REPO, emb['labels'])
    cfg['out_dir'] = os.path.join(OUT_ROOT, cfg['name'])
    os.makedirs(cfg['out_dir'], exist_ok=True)
    return cfg


def metrics_tis():
    """optimalrcs/metrics_tis.py (numpy), loaded by its path: `import optimalrcs` would import TensorFlow."""
    if 'metrics_tis_np' not in sys.modules:
        spec = importlib.util.spec_from_file_location('metrics_tis_np', os.path.join(REPO, 'optimalrcs', 'metrics_tis.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules['metrics_tis_np'] = mod
    return sys.modules['metrics_tis_np']


def logit(q):
    q = np.clip(q, Q_CLIP, 1 - Q_CLIP)
    return np.log(q) - np.log1p(-q)


def say(*a, log=None):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    if log is not None:
        with open(log, 'a') as f:
            f.write(s + '\n')
