#!/usr/bin/env python3
"""E7 split-half test for a slow CV w: does the w-dependence of the teacher's committor reproduce on independent data?

    python -m loop.experiments.w_split <loop config> [--data infinit|all] [--splits blocks,parity] [--cv w]

A single teacher fit on (y, x, w) gives the uncoupled w of loop_zu a well difference two thirds as large as the coupled
w of loop_zw (RMS of the bin-mean q difference between w < 0 and w >= 0 frames over the (y, x) bins: 0.044 against
0.070, exact 0.0075 and 0.106), so the size of the fitted dependence cannot decide. Here the teacher is fitted on two
disjoint halves of the pool (whole pieces) and the well-difference profiles Delta_1(y, x), Delta_2(y, x) are compared:
a real dependence reproduces (correlation near 1), a fitted one from finite data should not.

Halves: `blocks` = MC blocks 0-4 against 5-9 of every pass (pool.View.piece_block); `parity` = even against odd passes.
The frames are conditioned on the sign of the CV (the two wells of the double well); profiles on a 25 x 25 grid of
(state CV, other CV) quantile bins with at least 30 frames per well in both halves. Output: <loop dir>/w_split/<data>.json.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from loop import env, controller, gpu, passes, teacher  # noqa: E402,F401
from loop import pool as pool_mod  # noqa: E402

FRAME_ATTRS = ('cv', 'lam', 't', 'piece', 'path', 'w', 'ens', 'pass_of_frame', 'src_idx', 'q_np', 'q_exact', 'A', 'B',
               'inside', 'valid', 'q', 'z', 'frame_split', 'incl')


def subview(v, frame_mask):
    """The view restricted to whole pieces (frame_mask must be constant on every piece)."""
    s = copy.copy(v)
    for a in FRAME_ATTRS:
        if hasattr(v, a):
            setattr(s, a, getattr(v, a)[frame_mask])
    s.n = int(frame_mask.sum())
    s.path_id = np.unique(s.path)
    return s


def bins(v, cols, nb=25):
    ok = v.inside
    edges = [np.quantile(v.cv[ok, c], np.linspace(0, 1, nb + 1)) for c in cols]
    ij = [np.clip(np.searchsorted(e, v.cv[:, c], 'right') - 1, 0, nb - 1) for e, c in zip(edges, cols)]
    return ij[0] * nb + ij[1], nb * nb


def profile(v, q, b, nbins, wcol, frames):
    """Bin means of q in each well over `frames` (bool mask), and the per-bin minimum count."""
    m_ok = frames & v.inside & np.isfinite(q)
    out = []
    for m in (m_ok & (v.cv[:, wcol] < 0), m_ok & (v.cv[:, wcol] >= 0)):
        n = np.bincount(b[m], minlength=nbins)
        sw = np.bincount(b[m], v.w[m], minlength=nbins)
        out.append((n, np.bincount(b[m], v.w[m] * q[m], minlength=nbins) / np.maximum(sw, 1e-300)))
    return out[0][1] - out[1][1], np.minimum(out[0][0], out[1][0])


def stats(d1, d2, n):
    rms = lambda d: float(np.sqrt(np.average(d ** 2, weights=n)))
    # uncentred: the null hypothesis is Delta = 0, not a constant
    c = float(np.average(d1 * d2, weights=n))
    r = c / (rms(d1) * rms(d2))
    return dict(rms_1=rms(d1), rms_2=rms(d2), corr=r, reproducible_rms=float(np.sqrt(max(c, 0.0))), n_bins=int(len(n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--data', default='all', choices=['all', 'infinit'])
    ap.add_argument('--splits', default='blocks,parity')
    ap.add_argument('--cv', default='w')
    ap.add_argument('--max-frames', type=int, default=4_000_000)
    ap.add_argument('--min-n', type=int, default=30)
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    pool = pool_mod.Pool(cfg['loop_dir'])
    if a.data == 'infinit':
        cfg['pool']['passes'] = [k for k in pool.passes() if pool.meta(k).get('kind') == 'infinit']
    cfg['pool']['max_frames'] = a.max_frames
    v = passes._view(pool, cfg)
    names = list(map(str, v.cv_names))
    wcol, other = names.index(a.cv), [i for i in range(len(names)) if i not in (v.state_cv, names.index(a.cv))][0]
    b, nbins = bins(v, (v.state_cv, other))
    tc = dict(cfg['teacher'])
    out = dict(config=a.config, data=a.data, passes=sorted(set(v.interfaces_of_pass)), n_frames=int(v.n),
               cvs=names + list(tc.get('features', [])), max_iter=int(tc.get('max_iter', 3000)), splits={})
    print(a.config, a.data, 'frames', v.n, 'passes', out['passes'], flush=True)
    d = os.path.join(cfg['loop_dir'], 'w_split')
    os.makedirs(d, exist_ok=True)
    for split in a.splits.split(','):
        if split == 'blocks':
            half = v.piece_block[v.piece] < 5
        elif split == 'parity':
            half = v.pass_of_frame % 2 == 0
        else:
            raise SystemExit(f'unknown split {split}')
        if half.all() or not half.any():
            print(split, 'one half is empty, skipped', flush=True)
            continue
        qs = []
        for k, m in enumerate((half, ~half)):
            cache = os.path.join(d, f'{a.data}_{split}_{k}.npz')
            if os.path.isfile(cache) and np.load(cache)['q'].shape == (int(m.sum()),):
                r_q = np.load(cache)['q']
                print(f'  {split} half {k}: cached fit {cache}', flush=True)
            else:
                sv = subview(v, m)
                n_b = len(np.unique(sv.path[sv.B]))
                with gpu.exclusive(what=f'w_split {a.config} {split} half {k}'):
                    res = teacher.fit(sv, tc, seed_q=None, seed=int(cfg.get('seed', 0)) + 17 + k)
                r_q = res['q']
                np.savez_compressed(cache, q=r_q)
                print(f'  {split} half {k}: {sv.n} frames, {n_b} paths reach B, {res["seconds"]:.0f} s', flush=True)
            q = np.full(v.n, np.nan)
            q[m] = r_q
            qs.append(q)
        # profiles on bins with enough frames per well in both halves; the exact committor on the same frames
        d1, n1 = profile(v, qs[0], b, nbins, wcol, half)
        d2, n2 = profile(v, qs[1], b, nbins, wcol, ~half)
        keep = (n1 >= a.min_n) & (n2 >= a.min_n)
        n = np.minimum(n1, n2)[keep]
        row = stats(d1[keep], d2[keep], n)
        if v.has_exact:
            qe = v.q_exact.astype(float)
            e1, _ = profile(v, qe, b, nbins, wcol, half)
            e2, _ = profile(v, qe, b, nbins, wcol, ~half)
            ex = stats(e1[keep], e2[keep], n)
            row['exact'] = dict(rms_1=ex['rms_1'], rms_2=ex['rms_2'],
                                corr_fit_exact_1=float(np.corrcoef(d1[keep], e1[keep])[0, 1]),
                                corr_fit_exact_2=float(np.corrcoef(d2[keep], e2[keep])[0, 1]))
            qa = np.where(half, qs[0], qs[1])
            ok = v.inside & np.isfinite(qa)
            row['exact']['rms_q_vs_exact'] = float(np.sqrt(np.average((qa[ok] - qe[ok]) ** 2, weights=v.w[ok])))
        out['splits'][split] = row
        print(' ', split, json.dumps(row), flush=True)
    with open(os.path.join(d, f'{a.data}.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
