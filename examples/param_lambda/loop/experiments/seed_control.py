#!/usr/bin/env python3
"""Production control with the seed order parameter (lambda = the state CV, the interfaces of the toytis training
run), seeded from the same pool as the learned-lambda production run, for the comparison at equal MD steps.

    python -m loop.experiments.seed_control setup <loop config> [--cycles N]
    python -m loop.experiments.seed_control run   <loop config>
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from loop import env, controller, manifest, pool as pool_mod, rates, seed as seed_mod  # noqa: E402
from loop.backends import infretis_backend as be  # noqa: E402
import staple_run  # noqa: E402  (SEED_INTERFACES of the toytis runs)


class Identity:
    def __init__(self, col=0):
        self.col = col

    def __call__(self, cv):
        return np.atleast_2d(cv)[:, self.col].astype(float)


def setup(cfg, cycles):
    system = cfg['system_spec']
    intf = staple_run.SEED_INTERFACES[system['name']]
    out = os.path.join(cfg['loop_dir'], 'production_seed')
    pool = pool_mod.Pool(cfg['loop_dir'])
    seeds = os.path.join(cfg['loop_dir'], 'production_seed_seeds')
    dirs, tainted, report = seed_mod.build_seeds(pool, Identity(system['state_cv']), intf, cfg['run'].get('mode', 'staple'), seeds)
    run = dict(cfg['run'], **cfg.get('production', {}).get('run', {}))
    be.write_run(out, system, run, intf, be.seed_order_table(system), cycles, cfg['run'].get('mode', 'staple'),
                 dirs, tainted_inodes=tainted, shooting=cfg['run'].get('shooting', 'sh'), seed=int(cfg.get('seed', 0)) + 2000,
                 note='production control with the seed order parameter')
    print('prepared', out, intf)


def run(cfg):
    out = os.path.join(cfg['loop_dir'], 'production_seed')
    cost = be.run(out) if not os.path.isfile(os.path.join(out, 'restart.toml')) else be.cost(out)
    r = rates.staple_rate(out, burn_in=float(cfg.get('production', {}).get('burn_in', 0.1)))
    r['cost'] = cost
    if cfg['system_spec'].get('dim') == 2:
        r['exact'] = rates.exact_rate(cfg['system_spec'])
    manifest.write(os.path.join(out, 'rates.json'), **r)
    print({k: v for k, v in r.items() if k not in ('interfaces', 'pcross')})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['setup', 'run'])
    ap.add_argument('config')
    ap.add_argument('--cycles', type=int)
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    if a.cmd == 'setup':
        setup(cfg, a.cycles or int(cfg.get('production', {}).get('cycles', 20000)))
    else:
        run(cfg)


if __name__ == '__main__':
    main()
