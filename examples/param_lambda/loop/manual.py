#!/usr/bin/env python3
"""The loop's steps one at a time (manual use). Run from examples/param_lambda:

    python -m loop.manual place  <loop_dir> <model.npz> [--mode staple] [--config loop_z]
        plan (natural coordinate, ratio cap, length budget) + spacing gate on the loop's stored paths under the
        model; prints the interfaces and the interface-health table.
    python -m loop.manual check  <run_dir> --interfaces a b c ... [--mode staple]
        every spacing check on a finished infretis run's own paths (the lambda recorded in its order.txt), plus
        the accepted paths with an empty shooting region.
    python -m loop.manual seed   <loop_dir> <model.npz> <out_dir> --interfaces a b c ... [--mode staple]
        cut one valid initial path per ensemble from the stored paths (infretis load layout).
    python -m loop.manual toml   <out_dir> <model.npz> --system z_br --interfaces ... --seeds <seed_dir> [--cycles N]
        an infretis run directory with the learned lambda, the interfaces and the seeds (not started).

The exported models (format 1 or 2) also work in toytis: copy one to exported/<system>_<tag>.npz and run
`python staple_run.py <system> --model <tag> --interfaces ...` (toytis_adapter evaluates LambdaModel.one).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from loop import env, ase_systems, interfaces as itf, pool as pool_mod, seed as seed_mod  # noqa: E402
from loop.backends import infretis_backend as be  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402
import common  # noqa: E402


def _ic(name):
    if not name:
        return dict(n_interfaces=10, max_ratio=10.0, eps=0.01, c=1.0, n_a=3, n_sh=3, length_budget=np.inf)
    return common._read_toml(name)['interfaces']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('place'); a.add_argument('loop_dir'); a.add_argument('model')
    a.add_argument('--mode', default='staple'); a.add_argument('--config')
    c = sub.add_parser('check'); c.add_argument('run_dir'); c.add_argument('--interfaces', type=float, nargs='+', required=True)
    c.add_argument('--mode', default='staple'); c.add_argument('--config')
    s = sub.add_parser('seed'); s.add_argument('loop_dir'); s.add_argument('model'); s.add_argument('out_dir')
    s.add_argument('--interfaces', type=float, nargs='+', required=True); s.add_argument('--mode', default='staple')
    t = sub.add_parser('toml'); t.add_argument('out_dir'); t.add_argument('model'); t.add_argument('--system', required=True)
    t.add_argument('--interfaces', type=float, nargs='+', required=True); t.add_argument('--seeds', required=True)
    t.add_argument('--cycles', type=int, default=1000); t.add_argument('--mode', default='staple'); t.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    if args.cmd == 'place':
        model = LambdaModel(args.model)
        pool = pool_mod.Pool(args.loop_dir)
        ic = _ic(args.config)
        view = pool.select(passes='all', lineage='exclude', need_labels=False)
        plan = itf.plan_natural(model, view, ic, log=None)
        paths = itf.path_arrays(pool, model)
        placed = itf.place(model, paths, plan['lam'], ic, args.mode)
        print('planned  ', np.round(plan['lam'], 6).tolist())
        print('removed  ', [(round(v, 6), r) for v, r in placed['removed']])
        print('refilled ', [(round(r['lam'], 6), round(r['q'], 4)) for r in placed.get('refilled', [])])
        print('interfaces = [' + ', '.join(f'{v:.6f}' for v in placed['interfaces']) + ']')
        print(itf.health_table(placed['check']))
    elif args.cmd == 'check':
        raw = be.raw_paths(args.run_dir)
        own = [(o[:, 0], o[:, 1], float(W.max())) for o, W in zip(raw['ops'], raw['W']) if W.max() > 0]
        res = itf.check(own, args.interfaces, _ic(args.config), args.mode)
        print(itf.health_table(res))
        print('accepted paths with an empty shooting region:', itf.empty_shooting_regions(raw, args.interfaces, args.mode))
    elif args.cmd == 'seed':
        pool = pool_mod.Pool(args.loop_dir)
        dirs, tainted, report = seed_mod.build_seeds(pool, LambdaModel(args.model), args.interfaces, args.mode, args.out_dir)
        for e, r in report.items():
            print(e, r)
        print(f'{len(dirs)} seed paths in {args.out_dir}; {len(tainted)} tainted trajectory files')
    elif args.cmd == 'toml':
        system = ase_systems.get(args.system)
        dirs = [os.path.join(args.seeds, str(i)) for i in range(len(args.interfaces))]
        be.write_run(args.out_dir, system, dict(workers=args.workers), args.interfaces,
                     be.learned_order_table(system, args.model), args.cycles, args.mode, dirs)
        print(f'{args.out_dir}: infretis.toml written; run with  cd {args.out_dir} && infretisrun -i infretis.toml')


if __name__ == '__main__':
    main()
