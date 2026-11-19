#!/usr/bin/env python3
"""Does the teacher converge within its iteration budget? Refit the teacher on a finished loop's pool (the data a
pass k + 1 would see) in chunks, each chunk warm-started from the previous one, and record the RMS of q against the
exact committor after every chunk. Read-only on the loop (no labels are written back).

    python -m loop.experiments.teacher_iters <loop config> [--start warm|cold] [--chunks 8] [--chunk-iter 3000]
    python -m loop.experiments.teacher_iters <loop config> --ensemble 6 [--chunk-iter 6000]

--start warm: the pool's current labels (the last pass's teacher) and the last model on unlabelled frames, as
passes.teach does; cold: the linear ramp in the state CV (as pass 1). Output: <loop_dir>/experiments/
teacher_iters_<start>.json.

--ensemble N: N independent fits of chunk-iter iterations each from the same start (seeds differ), and the mean
of their logits over the first 1, 2, ..., N members: does averaging fits beat iterating one? Output:
teacher_ensemble_<start>.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from loop import controller, gpu, passes, pool as pool_mod, teacher  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402


def rms(view, q, mask=None):
    ok = np.isfinite(q) & view.inside & np.isfinite(view.q_exact)
    if mask is not None:
        ok &= mask
    return float(np.sqrt(np.average((q[ok] - view.q_exact[ok]) ** 2, weights=view.w[ok])))


def logit_rmse(view, q):
    """Path-weighted RMS of logit q - logit q_exact over frames with exact q in (1e-4, 1 - 1e-4)."""
    qe = view.q_exact
    ok = np.isfinite(q) & view.inside & (qe > 1e-4) & (qe < 1 - 1e-4)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.sqrt(np.average((lg(q[ok]) - lg(qe[ok])) ** 2, weights=view.w[ok])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--start', choices=['warm', 'cold'], default='warm')
    ap.add_argument('--chunks', type=int, default=8)
    ap.add_argument('--chunk-iter', type=int, default=3000)
    ap.add_argument('--ensemble', type=int, default=0)
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    with open(os.path.join(cfg['loop_dir'], 'state.json')) as f:
        st = json.load(f)
    pool = pool_mod.Pool(cfg['loop_dir'])
    view = passes._view(pool, cfg)
    assert view.has_exact, 'needs the exact committor'
    tc = dict(cfg['teacher'], max_iter=a.chunk_iter)
    k = len(st['passes']) + 1
    if a.start == 'warm':
        seed_q = view.q_np.astype(float).copy()
        new = ~np.isfinite(seed_q) & view.inside
        if new.any():
            seed_q[new] = LambdaModel(st['passes'][-1]['model']).q(view.cv[new])
    else:
        seed_q = None
    out = dict(loop=cfg['name'], start=a.start, chunk_iter=a.chunk_iter, n_frames_view=int(view.n), chunks=[])
    newest = view.pass_of_frame == pool.passes()[-1]
    if seed_q is not None:
        out['start_rms'] = rms(view, seed_q)
        out['start_logit_rmse'] = logit_rmse(view, seed_q)
    print(f'{cfg["name"]} {a.start}: {view.n} frames' + (f', start RMS {out["start_rms"]:.4f}' if seed_q is not None else ''), flush=True)
    od = os.path.join(cfg['loop_dir'], 'experiments')
    os.makedirs(od, exist_ok=True)
    path = os.path.join(od, f'teacher_iters_{a.start}.json')
    if a.ensemble:
        path = os.path.join(od, f'teacher_ensemble_{a.start}.json')
        out['members'], out['mean'] = [], []
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6))
        zsum = None
        for c in range(a.ensemble):
            t0 = time.time()
            with gpu.exclusive(what=f'teacher_ensemble {cfg["name"]} member {c + 1}'):
                res = teacher.fit(view, tc, seed_q=seed_q, seed=int(cfg.get('seed', 0)) + k + 1000 * c)
            q = res['q']
            fin = {key: v[-1] for key, v in res['history'].items() if v}
            out['members'].append(dict(rms=rms(view, q), logit_rmse=logit_rmse(view, q), seconds=time.time() - t0,
                                       max_z_zq_stopped=fin.get('max_z_zq_stopped')))
            zsum = lg(q) if zsum is None else zsum + lg(q)
            qm = 1 / (1 + np.exp(-zsum / (c + 1)))
            out['mean'].append(dict(n=c + 1, rms=rms(view, qm), logit_rmse=logit_rmse(view, qm)))
            print(json.dumps(out['members'][-1]), json.dumps(out['mean'][-1]), flush=True)
            with open(path, 'w') as f:
                json.dump(out, f, indent=1)
        print(path)
        return
    q = seed_q
    for c in range(a.chunks):
        t0 = time.time()
        with gpu.exclusive(what=f'teacher_iters {cfg["name"]} chunk {c + 1}'):
            res = teacher.fit(view, tc, seed_q=q, seed=int(cfg.get('seed', 0)) + k + 1000 * c)
        q = res['q']
        fin = {key: v[-1] for key, v in res['history'].items() if v}
        rec = dict(iterations=(c + 1) * a.chunk_iter, rms=rms(view, q), rms_newest=rms(view, q, newest),
                   logit_rmse=logit_rmse(view, q), seconds=time.time() - t0,
                   max_sd_zq_stopped=fin.get('max_sd_zq_stopped'), max_z_zq_stopped=fin.get('max_z_zq_stopped'),
                   delta_r2=fin.get('delta_r2'))
        out['chunks'].append(rec)
        print(json.dumps(rec), flush=True)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
    print(path)


if __name__ == '__main__':
    main()
