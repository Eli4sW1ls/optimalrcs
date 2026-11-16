#!/usr/bin/env python3
"""Resume an interrupted production run from its restart.toml and analyse it:
python -m loop.resume_production <loop config> [--dir production]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, controller, manifest, rates  # noqa: E402,F401
from loop.backends import infretis_backend as be  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--dir', default='production')
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    out = os.path.join(cfg['loop_dir'], a.dir)
    steps = int(be.read_toml(os.path.join(out, 'infretis.toml'))['simulation']['steps'])
    if be.cost(out)['cycles'] < steps:
        be.resume(out, steps)
    r = rates.staple_rate(out, burn_in=float(cfg.get('production', {}).get('burn_in', 0.1)))
    r['cost'] = be.cost(out)
    if cfg['system_spec'].get('dim') == 2:
        r['exact'] = rates.exact_rate(cfg['system_spec'])
    manifest.write(os.path.join(out, 'rates.json'), **r)
    print({k: v for k, v in r.items() if k not in ('interfaces', 'pcross')})
