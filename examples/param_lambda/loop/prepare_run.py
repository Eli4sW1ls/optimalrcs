#!/usr/bin/env python3
"""Prepare (not start) a run with seeds from the pool:
  --like <run>                     the interfaces and lambda of a finished run (a repeat with another seed), or
  --exact-grid <npz> --q q1 ...    the exact committor as lambda (exact_lambda.py), interfaces at committor levels
                                   (lambda_A, lambda(q_1), ..., lambda(q_k), lambda_B).
Start it with python -m loop.resume_production <config> --dir <name>.

    python -m loop.prepare_run loop_z_dp --like production_balanced --name production_balanced_rep --seed-offset 4400
    python -m loop.prepare_run loop_z_dp --exact-grid models/exact_lambda_h0.001.npz --q 1e-5 ... --name production_exactq"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, controller, manifest, ase_systems, pool as pool_mod, seed as seed_mod  # noqa: E402,F401
from loop.backends import infretis_backend as be  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--name', required=True)
    ap.add_argument('--like', help='finished run directory whose interfaces and lambda are used')
    ap.add_argument('--exact-grid', help='exact_lambda npz (path relative to the loop directory or absolute)')
    ap.add_argument('--q', type=float, nargs='+', help='committor levels of the interfaces (with --exact-grid)')
    ap.add_argument('--mode', default='staple', choices=['staple', 'retis'])
    ap.add_argument('--shooting', default=None)
    ap.add_argument('--cycles', type=int, default=40000)
    ap.add_argument('--seed-offset', type=int, default=4400)
    ap.add_argument('--workers', type=int, help='infretis workers (default: the config [run] / [production.run] value)')
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    system, pc = cfg['system_spec'], cfg.get('production', {})
    run = dict(cfg['run'], **pc.get('run', {})) if a.mode == 'staple' else dict(cfg['run'])
    if a.workers:
        run['workers'] = a.workers
    shooting = a.shooting or ('wf' if a.mode == 'retis' else run.get('shooting', 'sh'))
    out = os.path.join(cfg['loop_dir'], a.name)
    if os.path.isdir(out):
        sys.exit(f'{out} exists, left as is')
    if a.like:
        man = manifest.read(os.path.join(cfg['loop_dir'], a.like, 'manifest.json'))
        from lambda_runtime import LambdaModel
        model_path, intfs = man['model'], [float(v) for v in man['interfaces']]
        model = LambdaModel(model_path)
        table = be.learned_order_table(system, model_path)
        note = f'{a.name}: repeat of {a.like} (its interfaces and lambda), mode {a.mode}'
    elif a.exact_grid and a.q:
        from exact_lambda import ExactLambdaModel
        model_path = a.exact_grid if os.path.isabs(a.exact_grid) else os.path.join(cfg['loop_dir'], a.exact_grid)
        model = ExactLambdaModel(model_path)
        intfs = [model.lambda_a] + [float(model.lam_of_q(q)) for q in sorted(a.q)] + [model.lambda_b]
        table = {'class': 'ExactLambda', 'module': be.ORDERP, 'grid': os.path.abspath(model_path),
                 'cvs': ase_systems.cv_tables(system)}
        note = f'{a.name}: exact committor lambda ({os.path.basename(model_path)}), q levels {sorted(a.q)}'
    else:
        sys.exit('give --like, or --exact-grid with --q')
    pool = pool_mod.Pool(cfg['loop_dir'])
    dirs, tainted, seeds = seed_mod.build_seeds(pool, model, intfs, a.mode, os.path.join(cfg['loop_dir'], f'{a.name}_seeds'))
    be.write_run(out, system, run, intfs, table, a.cycles, a.mode, dirs, tainted_inodes=tainted, shooting=shooting,
                 seed=int(cfg.get('seed', 0)) + a.seed_offset, note=note)
    manifest.write(os.path.join(out, 'manifest.json'), model=os.path.abspath(model_path), interfaces=intfs,
                   verified_by=note, seeds=seeds)
    print(f'{len(intfs)} interfaces {[round(v, 5) for v in intfs]}, {a.cycles} cycles, {a.mode}/{shooting}; '
          f'run prepared in {out} (not started)')
