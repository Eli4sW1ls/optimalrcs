#!/usr/bin/env python3
"""Prepare (not start) a regular infretis RETIS run on the interfaces and lambda of a finished run, as a
reference against the staple runs (same lambda, same interfaces, other path ensemble). Seeds come from the pool,
as for the staple runs (seed.build_seeds, mode 'retis'). Start it with a plain `infretisrun -i infretis.toml` in
the run directory, or `python -m loop.backends.infretis_backend`-style be.run(dir).

    python -m loop.prepare_retis loop_z_dp --like production_var --name production_retis [--cycles 20000]
        [--shooting wf|sh] [--seed-offset 5000]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, controller, manifest, pool as pool_mod, seed as seed_mod  # noqa: E402,F401
from loop.backends import infretis_backend as be  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--like', required=True, help='finished run directory (under the loop directory) whose '
                    'interfaces and lambda are used')
    ap.add_argument('--name', default='production_retis')
    ap.add_argument('--cycles', type=int, default=20000)
    ap.add_argument('--shooting', default='wf', choices=['wf', 'sh'],
                    help="'wf': wire fencing in all but the first two ensembles (as pass 0); 'sh': shooting everywhere")
    ap.add_argument('--seed-offset', type=int, default=5000)
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    system = cfg['system_spec']
    run = dict(cfg['run'])                                    # not [production.run]: no staple options
    src = os.path.join(cfg['loop_dir'], a.like)
    man = manifest.read(os.path.join(src, 'manifest.json'))
    model_path, intfs = man['model'], man['interfaces']
    out = os.path.join(cfg['loop_dir'], a.name)
    if os.path.isdir(out):
        sys.exit(f'{out} exists, left as is')
    model = LambdaModel(model_path)
    pool = pool_mod.Pool(cfg['loop_dir'])
    dirs, tainted, seeds = seed_mod.build_seeds(pool, model, intfs, 'retis', os.path.join(cfg['loop_dir'], f'{a.name}_seeds'))
    be.write_run(out, system, run, intfs, be.learned_order_table(system, model_path), a.cycles, 'retis', dirs,
                 tainted_inodes=tainted, shooting=a.shooting, seed=int(cfg.get('seed', 0)) + a.seed_offset,
                 note=f'{a.name}: regular RETIS on the interfaces of {a.like}')
    manifest.write(os.path.join(out, 'manifest.json'), model=model_path, interfaces=intfs,
                   verified_by=f'interfaces and lambda of {a.like}; regular RETIS, shooting {a.shooting}', seeds=seeds)
    print(f'{len(intfs)} interfaces, {a.cycles} cycles, shooting {a.shooting}; run prepared in {out} (not started)')
