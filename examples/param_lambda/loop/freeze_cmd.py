#!/usr/bin/env python3
"""Run (or retry) the gated freeze of a finished loop: python -m loop.freeze_cmd <loop config>

    python -m loop.freeze_cmd <config> --name production_refill --from-freeze
        a second production directory: start from the interfaces of freeze.json, re-place them with the current
        placement (thinning + refill), verify (forced round), write <name>/ (not started)
    python -m loop.freeze_cmd <config> --name production_refill3 --from-run production_refill2_verify_2/run
        the same from a finished verification run: thin its interface set on its own paths, verify, write"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, controller, passes, pool as pool_mod  # noqa: E402,F401

if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--name', default='production')
    ap.add_argument('--from-freeze', action='store_true', help='start from the interfaces of freeze.json')
    ap.add_argument('--from-run', help='start from a finished verification run (a directory with infretis.toml and '
                    'its data): its interface set, thinned on its own paths in the first round')
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    st = controller.State(cfg)
    start = None
    if a.from_freeze:
        with open(os.path.join(cfg['loop_dir'], 'freeze.json')) as f:
            start = json.load(f)['interfaces']
    start_paths = None
    if a.from_run:
        from loop.backends import infretis_backend as be
        run_dir = os.path.join(cfg['loop_dir'], a.from_run)
        start = [float(v) for v in be.read_toml(os.path.join(run_dir, 'infretis.toml'))['simulation']['interfaces']]
        start_paths = passes.own_paths(be.raw_paths(run_dir), cfg['system_spec']['state_cv'])
    passes.freeze(cfg, st, pool_mod.Pool(cfg['loop_dir']), os.path.join(cfg['loop_dir'], 'loop.log'),
                  name=a.name, start=start, start_paths=start_paths)
