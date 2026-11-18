"""Teacher: the nonparametric committor (NPNEq, optimalrcs-cupy CommittorNE) on the pooled pieces.

- Warm start: `seed_q` per frame (old frames keep their previous q, new frames start from the current
  parametric model's m(x), or from a linear ramp in the state CV on the first pass).
- Fixed iteration budget; frames capped by dropping whole pieces at random in two strata (paths that reach B, at
  most half the budget, and the rest), the kept pieces weighted by their inverse inclusion probability.
- Provisional B: when fewer than `n_b` paths of the data reach B, the committor is fitted to
  B_k = {s >= s_front}, the highest interface with at least `n_front` (weighted) arrivals.
- Returns q on every frame of the view (NaN for frames dropped by the cap, which the student never uses),
  the metric history and the cost.

The CVs of the fit are the pool CVs plus the configured extra linear forms (teacher.features, e.g. "u", "v" of
lambda_runtime.Features), standardized to unit range as the notebooks do (the polynomial basis in y).
"""
import os
import sys
import time

import numpy as np

import common
from lambda_runtime import Features

from loop import env

_CUPY_DIR = os.path.join(env.REPO, 'optimalrcs-cupy')


def _optimalrcs_cupy():
    import importlib.util
    if _CUPY_DIR not in sys.path:
        sys.path.insert(0, _CUPY_DIR)
    if 'optimalrcs_cupy' in sys.modules:
        return sys.modules['optimalrcs_cupy']
    spec = importlib.util.spec_from_file_location('optimalrcs_cupy', os.path.join(_CUPY_DIR, 'optimalrcs.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules['optimalrcs_cupy'] = mod
    return mod


def front(view, interfaces, n_front=50):
    """(s_front, arrivals): arrivals[k] = number of distinct paths of the view whose maximum state CV reaches
    interfaces[k], each path weighted by its inverse inclusion probability under the pool's frame budget (an
    estimate of the pool's count); s_front = the highest interface above lambda_A with at least n_front arrivals
    (None if none)."""
    s = view.cv[:, view.state_cv]
    ip = np.searchsorted(view.path_id, view.path)
    smax = np.full(len(view.path_id), -np.inf)
    np.maximum.at(smax, ip, s)
    # paths counted with their inverse inclusion probability (pool frame budget): an estimate of the pool's count
    incl = np.ones(len(view.path_id))
    if hasattr(view, 'incl'):
        incl[ip] = view.incl
    arrivals = [int(round(float(incl[smax >= lv].sum()))) for lv in interfaces]
    ok = [k for k, a in enumerate(arrivals) if a >= n_front and interfaces[k] > view.lambda_a]
    return (float(interfaces[max(ok)]) if ok else None), arrivals


def select_frames(view, max_frames, seed=0):
    """(frame mask, per-frame weight factor): whole pieces, thinned at random until about max_frames remain. Two
    strata as in pool.Pool.select: pieces of paths that reach B (at most half the budget) and the rest; the kept
    pieces are weighted by their inverse inclusion probability."""
    if view.n <= max_frames:
        return np.ones(view.n, bool), np.ones(view.n)
    from loop.pool import thin_strata
    n_pieces = int(view.piece.max()) + 1
    b_piece = np.zeros(n_pieces, bool)
    b_piece[np.unique(view.piece[np.isin(view.path, np.unique(view.path[view.B]))])] = True
    sizes = np.bincount(view.piece, minlength=n_pieces)
    n_b = int(sizes[b_piece].sum())
    f_b, f_o = thin_strata(n_b, view.n - n_b, max_frames)
    rng = np.random.default_rng(seed)
    f = np.where(b_piece, f_b, f_o)
    keep = rng.random(n_pieces) < f
    return keep[view.piece], (1.0 / f)[view.piece]


def fit(view, cfg, seed_q=None, provisional_b=None, seed=0, log=None):
    """NPNEq on the view. cfg: the [teacher] table. seed_q: per-frame start values (NaN = unknown).
    provisional_b: s_front or None. Returns dict(q, mask, history, iterations, seconds, ...)."""
    import cupy as cp
    orc = _optimalrcs_cupy()
    mask, wscale = select_frames(view, int(cfg.get('max_frames', 4_000_000)), seed)
    idx = np.flatnonzero(mask)
    # the arrays of the fit: contiguous pieces (the mask keeps whole pieces)
    piece = view.piece[idx]
    t = view.t[idx].astype(float)
    w = view.w[idx].astype(float) * wscale[idx]
    w = w / w.mean()
    group = view.path[idx]
    s = view.cv[idx, view.state_cv]
    A = view.A[idx]
    if provisional_b is not None and not view.lambda_a < provisional_b < view.lambda_b:
        provisional_b = None
    B = view.B[idx] if provisional_b is None else (s >= provisional_b)
    cv = view.cv[idx]
    specs = list(cfg.get('cvs') or view.cv_names) + list(cfg.get('features', []))
    feat = Features(list(view.cv_names), specs)
    X = feat(cv)
    lo, hi = np.quantile(X, 0.001, axis=0), np.quantile(X, 0.999, axis=0)
    X = (X - lo) / np.where(hi > lo, hi - lo, 1.0)          # unit range, as the raw CVs of the toy runs
    if seed_q is None:
        r0 = np.clip((s - view.lambda_a) / ((provisional_b or view.lambda_b) - view.lambda_a), 0, 1)
    else:
        r0 = np.asarray(seed_q, float)[idx]
        lin = np.clip((s - view.lambda_a) / ((provisional_b or view.lambda_b) - view.lambda_a), 0, 1)
        r0 = np.where(np.isfinite(r0), r0, lin)
    r0 = np.clip(r0, 0, 1)
    r0[A], r0[B] = 0, 1
    t0 = time.time()
    q = orc.CommittorNE(boundary0=A, boundary1=B, i_traj=piece, t_traj=t, path_weights=w, group_traj=group,
                        seed_r=r0)
    q.set_fixed_traj_length_trap(A | B, 2 ** 16)
    X_gpu = cp.asarray(X, dtype=q.prec)
    rng = np.random.default_rng(seed)

    def comp_y():
        return X_gpu[:, int(rng.integers(X_gpu.shape[1]))]

    max_iter = int(cfg.get('max_iter', 3000))
    q.fit_transform(comp_y, gamma=float(cfg.get('gamma', 0.1)), ny=int(cfg.get('ny', 6)), max_iter=max_iter,
                    print_step=int(cfg.get('print_step', 250)),
                    metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped', 'delta_x', 'time_elapsed'),
                    save_min_metric='max_z_zq_stopped', lazy_basis=True)
    r = cp.asnumpy(q.r_traj)
    secs = time.time() - t0
    history = {k: [float(v) for v in vals] for k, vals in q.metrics_history.items()}
    out = np.full(view.n, np.nan)
    out[idx] = r
    del X_gpu, q
    cp.get_default_memory_pool().free_all_blocks()
    res = dict(q=out, mask=mask, history=history, iterations=max_iter, seconds=secs, n_frames=int(len(idx)),
               provisional_b=provisional_b, features=specs)
    if log:
        common.say(f'teacher: {len(idx)} frames, {max_iter} iterations, {secs:.0f} s'
                   + (f', provisional B at s >= {provisional_b}' if provisional_b is not None else ''), log=log)
    return res
