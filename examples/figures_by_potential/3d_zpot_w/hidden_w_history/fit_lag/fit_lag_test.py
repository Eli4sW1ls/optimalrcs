"""NPNEq at fit lag Delta on the gate (zw) staple data, `full` variant (y, x, u, v, w), scored against the exact committor.

The library's fit_transform always updates at lag 1. As in figures_by_potential/1d_flat/langevin_staple/lgv_check/
lag_fit.py, nonparametrics.npneq is swapped for a lag-Delta version: with t' = t + Delta, or the piece's last frame when
the piece ends in A or B before that (transitions running past the end of a piece that ends in neither are dropped:
galerkin.galerkin_solve.transitions, the same bookkeeping as the cluster solve),

    sum_t w_t phi_k(t) [ r(t') - r(t) ] = 0,    akj = sum_t w_t phi_k (phi_j(t') - (1 + gamma) phi_j(t)),

with the library's gamma convention (gamma weighted like the transition). The basis is that of the library (polynomials
in (r, one CV) times the envelope, ny 6). The exact committor satisfies the equation at every lag (martingale).

Stages (one run each of --lag):
    half     start from q = 1/2, envelope 1e6 (global), --iters-half iterations, checks every --step
    cluster  start from the 10000-cluster solve at the same lag (../galerkin/q_kmeans10000_lag<lag>.npy), envelope 0.01,
             --iters-cluster iterations

Usage: python fit_lag_test.py --lag 50 [--iters-half 2000] [--iters-cluster 1000] [--step 250] [--stages half cluster]
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
ap.add_argument('--lag', type=int, required=True)
ap.add_argument('--iters-half', type=int, default=2000)
ap.add_argument('--iters-cluster', type=int, default=1000)
ap.add_argument('--step', type=int, default=250)
ap.add_argument('--stages', nargs='+', default=['half', 'cluster'], choices=['half', 'cluster'])
ap.add_argument('--chunk', type=int, default=1_000_000, help='transitions per block')
a = ap.parse_args()

import cupy as cp  # noqa: E402

orc = h._optimalrcs_cupy()
import nonparametrics  # noqa: E402

d = h.load_data()
qp = h.projection(d)
same = d['i_traj'][1:] == d['i_traj'][:-1]
inner = ~d['A'] & ~d['B']
plateau = inner & (d['q_exact'] > 0.2) & (d['q_exact'] < 0.4)
t0n, t1n = g.transitions(d, a.lag)
T0, T1 = cp.asarray(t0n), cp.asarray(t1n)
WT = cp.asarray(d['path_weights'])[T0]
print(f'lag {a.lag}: {len(t0n)} transitions', flush=True)


def npneq_lag(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk=None, w_traj=None, **kw):
    fkk = fk if not callable(fk) else None
    akj = cp.zeros((fkk.shape[0],) * 2)
    b = cp.zeros(fkk.shape[0])
    for s in range(0, len(T0), a.chunk):
        i0, i1, w = T0[s:s + a.chunk], T1[s:s + a.chunk], WT[s:s + a.chunk]
        fa, fb = fkk[:, i0], fkk[:, i1]
        akj += fa @ ((fb - fa * (1 + gamma)) * w).T
        b += fa @ (-(r_traj[i1] - r_traj[i0]) * w)
    al = cp.linalg.lstsq(akj, b[:, None], rcond=None)[0][:, 0]
    return cp.clip(r_traj + cp.tensordot(al, fkk, 1), 0, 1)


nonparametrics.npneq = npneq_lag
X = cp.asarray(h.cv_matrix(d['X'], h.VARIANTS['full']['cvs']))
kw = dict(gamma=0.1, ny=6, print_step=a.step, metrics_print=('iter', 'max_z_zq_stopped', 'delta_x'),
          save_min_metric='max_z_zq_stopped')
rows = []


def run(stage, r0, envelope, n_iter):
    np.random.seed(0); cp.random.seed(0)
    q = orc.CommittorNE(boundary0=d['A'], boundary1=d['B'], i_traj=d['i_traj'], t_traj=d['t_traj'],
                        path_weights=d['path_weights'], group_traj=d['group_traj'], seed_r=r0)
    q.set_fixed_traj_length_trap(d['A'] | d['B'], 2 ** 16)
    orc.envelope_scale = envelope
    done, t_start = 0, time.time()
    while done < n_iter:
        n = min(a.step, n_iter - done)
        q.fit_transform(lambda: X[:, np.random.randint(X.shape[1])], max_iter=n - 1 if done == 0 else n, **kw)
        done += n
        r = cp.asnumpy(q.r_traj).astype(float)
        sc = g.score(d, r, qp, same, plateau)
        row = dict(stage=stage, lag=a.lag, iter=done, zqs_z=float(q.metrics_history['max_z_zq_stopped'][-1]),
                   minutes=(time.time() - t_start) / 60, **sc)
        rows.append(row)
        print('CHECK ' + json.dumps(row), flush=True)
        json.dump(rows, open(os.path.join(HERE, f'lag{a.lag}.json'), 'w'), indent=1)
    np.save(os.path.join(HERE, f'q_{stage}_lag{a.lag}.npy'), r.astype(np.float32))
    del q
    cp.get_default_memory_pool().free_all_blocks()


if 'half' in a.stages:
    run('half', None, 1e6, a.iters_half)
if 'cluster' in a.stages:
    r0 = np.load(os.path.join(os.path.dirname(HERE), 'galerkin', f'q_kmeans10000_lag{a.lag}.npy')).astype(float)
    sc0 = g.score(d, r0, qp, same, plateau)
    print('START cluster solve: ' + json.dumps(sc0), flush=True)
    run('cluster', r0, 0.01, a.iters_cluster)
print('DONE', flush=True)
