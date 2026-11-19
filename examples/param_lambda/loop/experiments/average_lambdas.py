#!/usr/bin/env python3
"""Does averaging the last lambdas reduce the pass-to-pass noise? (offline: stored models, final pool, exact q)

    python -m loop.experiments.average_lambdas [loop ...] [--json out.json]

Every model lambda_k of a loop and the averages of m consecutive models ending at k (mean of logit q over
lambda_{k-m+1..k}, m = 2, 3, 4) are scored on one common view, the loop's final pool (the loop's own record scores
lambda_k on the pool of pass k, which grows from pass to pass, so those numbers are not comparable across k). Score:
the binned logit RMSE against the exact committor, as in `diagnostics.exact_metrics`. Reported per loop: single
and averaged scores per pass, and over the second half of the passes the mean ratio average / single at the same k.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from loop import env, controller, pool as pool_mod  # noqa: E402,F401
from loop.passes import _view  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402
import common  # noqa: E402
import evaluate as ev  # noqa: E402

DEFAULT = ['loop_z', 'loop_zw_x12', 'loop_zw_s2', 'loop_zw_paths', 'loop_zu', 'loop_zu_xy', 'loop_zu_b',
           'loop_zu_b_s2', 'loop_zu_xy_b', 'loop_zu_xy_b_s2_x12', 'loop_stag', 'loop_cc']
WINDOWS = (2, 3, 4)


def model_file(cfg, rec):
    p = rec['model']
    if os.path.isfile(p):
        return p
    q = os.path.join(cfg['loop_dir'], 'models', os.path.basename(p))
    return q if os.path.isfile(q) else None


def run_loop(name):
    cfg = controller.load_loop_config(name)
    st = json.load(open(os.path.join(cfg['loop_dir'], 'state.json')))
    pool = pool_mod.Pool(cfg['loop_dir'])
    t0 = time.time()
    view = _view(pool, cfg)
    print(f'{name}: view {view.n} frames in {time.time() - t0:.0f} s', flush=True)
    if not view.has_exact:
        print(f'{name}: no exact committor on the view, skipped', flush=True)
        return None
    idx = np.flatnonzero(view.inside)
    cv = view.cv[idx]
    zx = common.logit(view.q_exact[idx].astype(float))
    w = view.w[idx].astype(float)
    cv2 = cv[:, :min(view.dim, 2)]
    mask = np.ones(len(idx), bool)
    def score(lg):
        t = time.time()
        v = float(ev.binned_logit_rmse(cv2, lg, zx, w, mask))
        if time.time() - t > 30:
            print(f'  (score took {time.time() - t:.0f} s)', flush=True)
        return v
    ks, logits = [], []
    for rec in st['passes']:
        f = model_file(cfg, rec)
        if f is None:
            continue
        t = time.time()
        m = np.clip(LambdaModel(f).q(cv), 1e-6, 1 - 1e-6)
        print(f"  model {rec['k']}: q in {time.time() - t:.0f} s", flush=True)
        ks.append(rec['k'])
        logits.append(common.logit(m).astype(np.float32))
    rows = []
    for i, k in enumerate(ks):
        r = dict(k=k, single=score(logits[i].astype(float)))
        for m in WINDOWS:
            if i + 1 >= m:
                r[f'avg{m}'] = score(np.mean(np.stack(logits[i - m + 1:i + 1]), axis=0).astype(float))
        rows.append(r)
    print(f'\n== {name}: {len(idx)} frames inside, {len(ks)} models', flush=True)
    print('  k   single   ' + '   '.join(f'avg{m}' for m in WINDOWS))
    for r in rows:
        print(f"  {r['k']:<2d}  {r['single']:6.3f}   " + '   '.join(f"{r[f'avg{m}']:.3f}" if f'avg{m}' in r else '  -  '
                                                             for m in WINDOWS), flush=True)
    half = rows[len(rows) // 2:]
    summ = dict(loop=name, n=len(rows), last=rows[-1]['single'], best=min(r['single'] for r in rows))
    for m in WINDOWS:
        rat = [r[f'avg{m}'] / r['single'] for r in half if f'avg{m}' in r]
        summ[f'ratio_avg{m}'] = float(np.mean(rat)) if rat else np.nan
        summ[f'last_avg{m}'] = rows[-1].get(f'avg{m}', np.nan)
    return dict(summary=summ, rows=rows)


def main():
    argv = sys.argv[1:]
    out_json = None
    if '--json' in argv:
        i = argv.index('--json')
        out_json, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    loops = [a for a in argv if not a.startswith('--')] or DEFAULT
    res = []
    for L in loops:
        r = run_loop(L)
        if r is not None:
            res.append(r)
            if out_json:
                json.dump(res, open(out_json, 'w'), indent=1)
    print('\n== final pass: single lambda_K against the average of the last m (common final view); best single pass')
    print('loop                    best   last   ' + '  '.join(f'last{m}' for m in WINDOWS) + '   | second half: mean avg/single ' + ' '.join(f'm={m}' for m in WINDOWS))
    for r in res:
        s = r['summary']
        print(f"{s['loop']:22s} {s['best']:6.3f} {s['last']:6.3f}  " + '  '.join(f"{s[f'last_avg{m}']:5.3f}" for m in WINDOWS)
              + '   |                               ' + ' '.join(f"{s[f'ratio_avg{m}']:.2f}" for m in WINDOWS))
    for m in WINDOWS:
        a = [r['summary'][f'last_avg{m}'] / r['summary']['last'] for r in res if np.isfinite(r['summary'][f'last_avg{m}'])]
        b = [r['summary'][f'ratio_avg{m}'] for r in res if np.isfinite(r['summary'][f'ratio_avg{m}'])]
        print(f'  m={m}: final pass avg/last mean {np.mean(a):.2f} (min {np.min(a):.2f}, max {np.max(a):.2f}); '
              f'second-half mean {np.mean(b):.2f}')


if __name__ == '__main__':
    main()
