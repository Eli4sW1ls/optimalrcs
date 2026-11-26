"""Do histories recover the committor's dependence on the hidden w?  Cluster solve + NPNEq at lag 50 on the gate data.

CVs are (y, x) only; w is never used (except in the upper-bound rows). Everything is scored against the exact committor
(validation only): plateau bias, RMS, and r2_w, the share of the w-driven part of the committor recovered
(0 for any function of the current (y, x), 1 for the exact committor; galerkin/galerkin_solve.py).

Cluster solves (10000 k-means clusters, one linear solve at --lag, phi = 0 in A, B; galerkin_solve.solve_kmeans) on
    yx      (y, x)                                          baseline: a function of the current (y, x)
    lag64   (y, x, y(t-64), x(t-64))                        one lagged copy
    lag3    (y, x) + lagged copies at 16, 64, 256           delay embedding
    mean    (y, x) + means of (y, x) over the last 64 and 256 frames   history summarized: averages the noise
    yxw     (y, x, w)                                       upper bound
Frames with fewer earlier frames than the lag use the piece's first frame (the library's shift 'r(t0)'; for the means
the window starts at the piece start). Every CV is scaled by the std of y (resp. x) over the frames between the states.

NPNEq at the same lag (the library's update with the lag-Delta moments, as fit_lag/fit_lag_test.py), from each cluster
solve (envelope 0.01, --iters iterations), with CVs y, x, u, v and
    nohist      no history
    hist        history_delta_t 0 1 2 4 8 16 32 64 128 256, history type 'y(t-d),r(t-d)', shift 'r(t)'
and from q = 1/2 (envelope 1e6) with history.

Usage: python history_fit.py [--lag 50] [--iters 1000] [--step 250]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'galerkin'))
import hidden_w_history as h  # noqa: E402
import galerkin_solve as g  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--lag', type=int, default=50)
ap.add_argument('--iters', type=int, default=1000)
ap.add_argument('--iters-half', type=int, default=2000)
ap.add_argument('--step', type=int, default=250)
ap.add_argument('--clusters', type=int, default=10000)
ap.add_argument('--chunk', type=int, default=1_000_000)
a = ap.parse_args()

import cupy as cp  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402

orc = h._optimalrcs_cupy()
import nonparametrics  # noqa: E402

HIST = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
d = h.load_data()
n = len(d['A'])
inner = ~d['A'] & ~d['B']
qp = h.projection(d)
same = d['i_traj'][1:] == d['i_traj'][:-1]
plateau = inner & (d['q_exact'] > 0.2) & (d['q_exact'] < 0.4)
t0n, t1n = g.transitions(d, a.lag)
T0, T1 = cp.asarray(t0n), cp.asarray(t1n)
WT = cp.asarray(d['path_weights'])[T0]
it = d['i_traj']
new_piece = np.r_[True, it[1:] != it[:-1]]
starts = np.flatnonzero(new_piece)
pstart = starts[np.cumsum(new_piece) - 1]                   # first frame of every frame's piece
y, x, w = (d['X'][:, c].astype(np.float64) for c in (1, 2, 3))
sy, sx, sw = y[inner].std(), x[inner].std(), w[inner].std()
idx = np.arange(n)


def lagged(col, dl):
    return col[np.maximum(idx - dl, pstart)]


def window_mean(col, win):
    cs = np.cumsum(np.r_[0.0, col])                          # cs[k] = sum of col[:k]
    lo = np.maximum(idx - win + 1, pstart)
    return (cs[idx + 1] - cs[lo]) / (idx - lo + 1)


EMB = {
    'yx': [y / sy, x / sx],
    'lag64': [y / sy, x / sx, lagged(y, 64) / sy, lagged(x, 64) / sx],
    'lag3': [y / sy, x / sx] + [f(dl) for dl in (16, 64, 256) for f in
                                (lambda k: lagged(y, k) / sy, lambda k: lagged(x, k) / sx)],
    'mean': [y / sy, x / sx] + [f(win) for win in (64, 256) for f in
                                (lambda k: window_mean(y, k) / sy, lambda k: window_mean(x, k) / sx)],
    'yxw': [y / sy, x / sx, w / sw],
}
rows = []


def emit(**kw):
    row = dict(lag=a.lag, **kw)
    rows.append(row)
    print('RESULT ' + json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(HERE, f'history_fit_lag{a.lag}.json'), 'w'), indent=1)


def cluster_solve(name):
    S = np.column_stack(EMB[name])
    rng = np.random.default_rng(0)
    sub = rng.choice(np.flatnonzero(inner), size=400_000, replace=False)
    t = time.time()
    km = MiniBatchKMeans(n_clusters=a.clusters, random_state=0, batch_size=8192, n_init=3).fit(S[sub])
    q, _ = g.solve_kmeans(d, km.predict(S), a.clusters, t0n, t1n)
    q = np.clip(q, 0, 1)
    emit(kind='cluster', embedding=name, dim=S.shape[1], minutes=(time.time() - t) / 60,
         **g.score(d, q, qp, same, plateau))
    return q


def npneq_lag(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk=None, w_traj=None, **kw):
    akj = cp.zeros((fk.shape[0],) * 2)
    b = cp.zeros(fk.shape[0])
    for s in range(0, len(T0), a.chunk):
        i0, i1, wt = T0[s:s + a.chunk], T1[s:s + a.chunk], WT[s:s + a.chunk]
        fa, fb = fk[:, i0], fk[:, i1]
        akj += fa @ ((fb - fa * (1 + gamma)) * wt).T
        b += fa @ (-(r_traj[i1] - r_traj[i0]) * wt)
    al = cp.linalg.lstsq(akj, b[:, None], rcond=None)[0][:, 0]
    return cp.clip(r_traj + cp.tensordot(al, fk, 1), 0, 1)


nonparametrics.npneq = npneq_lag
XU = cp.asarray(h.cv_matrix(d['X'], ('y', 'x', 'u', 'v')))


def npneq_run(label, r0, envelope, n_iter, history):
    np.random.seed(0); cp.random.seed(0)
    q = orc.CommittorNE(boundary0=d['A'], boundary1=d['B'], i_traj=d['i_traj'], t_traj=d['t_traj'],
                        path_weights=d['path_weights'], group_traj=d['group_traj'], seed_r=r0)
    q.set_fixed_traj_length_trap(d['A'] | d['B'], 2 ** 16)
    orc.envelope_scale = envelope
    kw = dict(gamma=0.1, ny=6, print_step=a.step, metrics_print=('iter', 'max_z_zq_stopped', 'delta_x'),
              save_min_metric='max_z_zq_stopped')
    if history:
        kw.update(history_delta_t=HIST, history_shift_type='r(t)')
    done, t = 0, time.time()
    while done < n_iter:
        m = min(a.step, n_iter - done)
        q.fit_transform(lambda: XU[:, np.random.randint(XU.shape[1])], max_iter=m - 1 if done == 0 else m, **kw)
        done += m
        r = cp.asnumpy(q.r_traj).astype(float)
        emit(kind='npneq', start=label, history=bool(history), iter=done, minutes=(time.time() - t) / 60,
             zqs_z=float(q.metrics_history['max_z_zq_stopped'][-1]), **g.score(d, r, qp, same, plateau))
    del q
    cp.get_default_memory_pool().free_all_blocks()
    return r


starts_q = {}
for name in ('yx', 'lag3', 'mean', 'lag64', 'yxw'):
    starts_q[name] = cluster_solve(name)
    if name in ('yx', 'lag3', 'mean'):
        npneq_run(f'cluster {name}', starts_q[name], 0.01, a.iters, history=False)
npneq_run('cluster yx', starts_q['yx'], 0.01, a.iters, history=True)
npneq_run('cluster lag3', starts_q['lag3'], 0.01, a.iters, history=True)
npneq_run('q = 1/2', None, 1e6, a.iters_half, history=True)
print('DONE', flush=True)
