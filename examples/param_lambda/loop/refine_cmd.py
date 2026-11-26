#!/usr/bin/env python3
"""Refine the interfaces of finished production runs for the sampler's acceptance (interfaces.refine_measured):
split pairs crossed with low probability and shooting regions far longer than the rest, then balance the region
lengths, all measured on the runs themselves. Writes <name>_freeze.json; with --prepare also the run directory
<name>/ with seeds from the pool (not started; start it with python -m loop.resume_production <config> --dir <name>).

    python -m loop.refine_cmd loop_z_dp --runs production_measured production_measured_rep/run \
        --name production_balanced [--prepare] [--cycles 40000]"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, controller, manifest, interfaces as itf, pool as pool_mod, seed as seed_mod  # noqa: E402,F401
from loop.backends import infretis_backend as be  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402


def _fmt(x, nd=3):
    return '   -  ' if x is None or not np.isfinite(x) else f'{x:6.{nd}f}'


def _rows(t, acc=None):
    lv = t['interfaces']
    hdr = '  ens  window                  ' + ('obs acc  NCR  long | ' if acc is not None else '') + \
          'pred acc  A_len  NCR   spread  path len  region | p below middle | n_eff (w = wider than the runs sampled)'
    print(hdr)
    for e in range(2, len(lv)):
        x = t['ensembles'].get(e) or {}
        r = (acc or {}).get(e, {})
        obs = (f'{_fmt(r.get("acc"), 2)} {_fmt(r.get("ncr"), 2)} {_fmt(r.get("too_long"), 2)} | ' if acc is not None else '')
        print(f'  {e:3d}  ({lv[e - 2]:.3f},{lv[e - 1]:.3f},{lv[e]:.3f})  {obs}  {_fmt(x.get("acc_pred"), 2)}  '
              f'{_fmt(x.get("acc_len"), 2)} {_fmt(x.get("ncr"), 2)} {x.get("spread", np.nan):7.1f} '
              f'{x.get("path_len", np.nan):8.0f} {x.get("frames", np.nan):7.0f} | {t["p"][e - 2]:.3f} | '
              f'{x.get("n_eff", np.nan):6.0f} {"w" if x.get("widened") else ""}')
    a_ = np.array([(t['ensembles'].get(e) or {}).get('acc_pred', np.nan) for e in range(2, len(lv))])
    l_ = np.array([(t['ensembles'].get(e) or {}).get('path_len', np.nan) for e in range(2, len(lv))])
    print(f'  predicted acceptance: mean {np.nanmean(a_):.2f}, min {np.nanmin(a_):.2f}, max {np.nanmax(a_):.2f}; '
          f'path length max/min {np.nanmax(l_) / np.nanmin(l_):.1f}x; objective {t["objective"]:.3f}')
    if t.get('var_terms'):
        v = np.array(t['var_terms'])
        print('  pairs: ' + '  '.join(f'{lv[j]:.3f}->{lv[j + 1]:.3f} p {t["p"][j]:.3f} share {v[j] / v.sum():.2f}'
                                      for j in range(len(v))))
        print(f'  predicted variance x cost (arb. units): {t["var_x_cost"]:.4g}; sum of path lengths {t["cost"]:.0f}; '
              f'{len(lv) - 1} ensembles')


def report(res):
    print('\nbefore (observed from the logs of the first run; predicted from the runs\' own paths and levels):')
    _rows(res['before'], res['acceptance'])
    print('\nsteps:')
    for s_ in res['steps'] or [{'change': 'none'}]:
        print('  ', s_)
    print('\nafter (predicted):')
    _rows(res['after'])
    print('  gate on the runs\' own paths: before', 'ok' if res['check_before']['ok'] else res['check_before']['fails'],
          '; after', 'ok' if res['check']['ok'] else res['check']['fails'])
    ck = res['check']
    print('  after: nonzero skip rates / A distance on the runs\' own paths: ' + ', '.join(
        f'{r["lam"]:.4f} up {r["skip_up"]:.4f} dn {r["skip_dn"]:.4f}' for r in ck['rows']
        if (r['skip_up'] or 0) > 0 or (r['skip_dn'] or 0) > 0) +
        f'; dist_a {ck["rows"][0].get("dist_a", np.nan):.2f}; single-step turns {ck["turns"].get("single_step", np.nan):.4f}')
    for pt in res.get('probes', []):
        print(f'\nprobe: ' + ('full set' if pt.get('full') else f'start set + {pt["added"]}') + ' (predicted):')
        _rows(pt)
    print('  interfaces:', [round(v, 5) for v in res['interfaces']])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--runs', nargs='+', required=True, help='finished run directories (under the loop directory) '
                    'with the same interfaces and lambda')
    ap.add_argument('--name', default='production_balanced')
    ap.add_argument('--prepare', action='store_true', help='also write the run directory <name>/ (not started)')
    ap.add_argument('--cycles', type=int, help='cycles of the prepared run (default: [production] cycles)')
    ap.add_argument('--seed-offset', type=int, default=4000)
    ap.add_argument('--probe', action='append', default=[], help='comma-separated interfaces to add to the start '
                    'set and score the same way, for comparison (repeatable)')
    ap.add_argument('--set', action='append', default=[], help='a comma-separated full interface set to score the '
                    'same way (e.g. another run\'s), for comparison (repeatable)')
    ap.add_argument('--objective', choices=['variance', 'acceptance'], help='refine_objective (default: config, '
                    'else variance)')
    ap.add_argument('--eps', type=float, help='gate: skip limit per interface (default: config, else 0.01)')
    ap.add_argument('--n-a', type=float, help='gate: A/B distance of the first/last interface in RMS steps of s '
                    '(default: config, else 3)')
    ap.add_argument('--place-margin', type=float, help='placement limits this much stricter than the gate '
                    '(default: config, else 1.25)')
    ap.add_argument('--dt-factor', type=int, default=1, help='prepared run: integration step dt / f with f times '
                    'the subcycles (same frames, finer integrator)')
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    if a.objective:
        cfg['interfaces']['refine_objective'] = a.objective
    for k, v in (('eps', a.eps), ('n_a', a.n_a), ('place_margin', a.place_margin)):
        if v is not None:
            cfg['interfaces'][k] = v
    system, ic, pc = cfg['system_spec'], cfg['interfaces'], cfg.get('production', {})
    run = dict(cfg['run'], **pc.get('run', {}))
    if a.dt_factor != 1:
        run['dt_factor'] = a.dt_factor
    mode = run.get('mode', 'staple')
    run_dirs = [os.path.join(cfg['loop_dir'], d) for d in a.runs]
    model_path = manifest.read(os.path.join(run_dirs[0], 'manifest.json'))['model']
    model = LambdaModel(model_path)
    log = os.path.join(cfg['loop_dir'], 'loop.log')
    probes = [[float(x) for x in pr.split(',')] for pr in a.probe] + \
             [('set', [float(x) for x in s_.split(',')]) for s_ in a.set]
    res = itf.refine_measured(model, run_dirs, ic, mode, state_cv=system['state_cv'], log=log, probes=probes)
    report(res)
    intfs = res['interfaces']
    manifest.write(os.path.join(cfg['loop_dir'], f'{a.name}_freeze.json'), model=model_path, interfaces=intfs,
                   ok=res['check']['ok'], fails=res['check']['fails'], source=f'refine_measured on {a.runs}',
                   steps=res['steps'], before=res['before'], after=res['after'],
                   acceptance={str(k): v for k, v in res['acceptance'].items()}, levels=res['levels'])
    if a.prepare:
        out = os.path.join(cfg['loop_dir'], a.name)
        if os.path.isdir(out):
            sys.exit(f'{out} exists, left as is')
        pool = pool_mod.Pool(cfg['loop_dir'])
        dirs, tainted, seeds = seed_mod.build_seeds(pool, model, intfs, mode, os.path.join(cfg['loop_dir'], f'{a.name}_seeds'))
        be.write_run(out, system, run, intfs, be.learned_order_table(system, model_path),
                     int(a.cycles or pc.get('cycles', 10000)), mode, dirs, tainted_inodes=tainted,
                     shooting=run.get('shooting', 'sh'), seed=int(cfg.get('seed', 0)) + a.seed_offset,
                     note=f'{a.name}: refine_measured on {a.runs}')
        manifest.write(os.path.join(out, 'manifest.json'), model=model_path, interfaces=intfs,
                       verified_by=f'refine_measured on {a.runs} (gate on their own paths: '
                                   f'{"ok" if res["check"]["ok"] else res["check"]["fails"]})', seeds=seeds)
        print(f'\nrun prepared in {out} (not started): python -m loop.resume_production {a.config} --dir {a.name}')
