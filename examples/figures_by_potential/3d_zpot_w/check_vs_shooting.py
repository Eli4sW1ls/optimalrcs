"""Exported zw_br models against the shooting committor (explore/shoot_*.json, zpotw_gpu.py shoot): an exact
reference at 27 (x, y) points in both channels, independent of q_np.

    python check_vs_shooting.py [--dV 3] [--models m3 m3-fast m3-yx m1-gam seed]

Prints q at every point and two numbers per model:
* logit RMSE against the shooting committor (points with 0.003 < q < 0.997);
* the channel contrast, RMS over (x, y) of logit q(w = +w0) - logit q(w = -w0). The true contrast is large in the
  triangles between the two ridges; a model that ignores w has 0, and a w-dependence of one sign everywhere is
  a channel bias, not the mirrored structure (q(y, x, w) = q(y, Lx - x, -w)).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PL = os.path.join(HERE, '..', '..', 'param_lambda')
sys.path.insert(0, PL)
from lambda_runtime import LambdaModel  # noqa: E402


def logit(q):
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dV', type=float, default=3.0)
    ap.add_argument('--system', default='zw_br')
    ap.add_argument('--models', nargs='+', default=['m3', 'm3-fast', 'm3-yx', 'm1-gam', 'seed'])
    a = ap.parse_args()
    shoot = json.load(open(sorted(glob.glob(os.path.join(HERE, 'explore', 'shoot_*.json')))[0]))
    rows = [r for r in shoot['rows'] if r['dV_kT'] == a.dV]
    w0 = 0.1
    pts = np.array([[r['y'], r['x'], s * w0] for r in rows for s in (-1, 1)])
    ref = np.array([v for r in rows for v in (r['q_minus'], r['q_plus'])])
    models = {k: LambdaModel(os.path.join(PL, 'exported', f'{a.system}_{k}.npz')) for k in a.models}
    q = {k: m.q(pts) for k, m in models.items()}
    print('  x     y   |  shooting w-   w+  | ' + ' | '.join(f'{k:>15s}' for k in models))
    for i in range(0, len(pts), 2):
        print(f'{pts[i, 1]:.2f}  {pts[i, 0]:.2f} | {ref[i]:7.4f} {ref[i + 1]:7.4f} | '
              + ' | '.join(f'{q[k][i]:7.4f} {q[k][i + 1]:7.4f}' for k in models))
    ok = (ref > 0.003) & (ref < 0.997)
    okp = ok[0::2] & ok[1::2]
    contrast = lambda v: float(np.sqrt(np.mean((logit(v[1::2]) - logit(v[0::2]))[okp] ** 2)))
    print(f'\npoints with 0.003 < q < 0.997: {ok.sum()}; channel pairs: {okp.sum()}')
    print(f'{"":10s} {"logit RMSE":>11s} {"contrast":>9s}   (shooting contrast {contrast(ref):.2f})')
    for k, v in q.items():
        print(f'{k:10s} {np.sqrt(np.mean((logit(v[ok]) - logit(ref[ok])) ** 2)):11.3f} {contrast(v):9.2f}')


if __name__ == '__main__':
    main()
