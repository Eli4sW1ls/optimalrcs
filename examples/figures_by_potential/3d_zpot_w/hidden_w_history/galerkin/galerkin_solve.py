"""Full-basis Galerkin solve of the NPNEq committor equation, as a check of the greedy NPNEq iteration.

NPNEq iterates on the projected equation
    sum_t w_t phi_k(x_t) [q(x_{t+lag}) - q(x_t)] = 0      (transitions inside pieces; stopped at A / B)
but each step only uses a polynomial in (current q, one CV), and on the gate (zw) staple data it ends in states
0.10-0.16 too low on the plateau (../drift_tests), in CuPy and in TensorFlow alike, while the exact committor is
a stable point of it. Here the same equation is solved once, over a fixed basis in all CVs together:

    q = 1_B + sum_j c_j phi_j,  phi_j = 0 in A and B   ->   M c = b,
    M_kj = sum_t w_t phi_k(x_t) [phi_j(x_t') - phi_j(x_t)],  b_k = -sum_t w_t phi_k(x_t) [1_B(x_t') - 1_B(x_t)]

with t' = t + lag inside the piece, or the piece's last frame if the piece ends in A or B before t + lag
(transitions running past the end of a piece that ends in neither are dropped). A linear system has one
solution, so there is no path to get stuck on. Bases:

    poly<d>      products of Legendre polynomials of the CVs scaled to [-1, 1], total degree <= d
                 (5 CVs: d = 6 -> 462 functions, d = 8 -> 1287)
    kmeans<n>    indicators of n k-means clusters (the committor of the Markov state model on those clusters)
    rbf<n>       Gaussians on n k-means centres, width = the median distance to the 5 nearest centres

CVs: y, x, w scaled to unit variance (u, v are linear in y, x and add nothing to a full basis). Scores against the
exact committor (validation only) and the stopped Z_q criterion of metrics_tis. A cluster basis at lag 1 needs clusters not much wider than a step
(500 clusters: plateau bias +0.18, the coarse-state model is not Markovian at that resolution). With --refine N, NPNEq (local
envelope 0.01, gamma 0.1) then continues from the solved committor for N iterations, to see whether the default
step stays near a good start.

Usage: python galerkin_solve.py --basis poly6 kmeans2000 rbf1000 [--lag 1] [--refine 500]
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hidden_w_history as h  # noqa: E402

CV_NAMES = ('y', 'x', 'w')


def transitions(d, lag):
    """(start index, target index) of the transitions inside pieces, stopped at A / B."""
    it = d['i_traj']
    n = len(it)
    starts = np.flatnonzero(np.r_[True, it[1:] != it[:-1]])
    ends = np.r_[starts[1:], n] - 1                         # last frame of each piece
    piece_of = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n]))
    last = ends[piece_of]
    AB = d['A'] | d['B']
    ends_in_state = AB[last]
    t = np.arange(n)
    tgt = t + lag
    inside = ~AB                                            # phi(x_t) = 0 on A / B frames anyway
    over = tgt > last
    keep = inside & (~over | ends_in_state) & (t < last)
    tgt = np.where(over, last, tgt)
    return t[keep], tgt[keep]


def scaled_cvs(d):
    X = d['X'][:, [h.CV_COLS[c] for c in CV_NAMES]].astype(np.float64)
    inner = ~d['A'] & ~d['B']
    lo, hi = np.quantile(X[inner], 0.001, axis=0), np.quantile(X[inner], 0.999, axis=0)
    return X, lo, hi


def legendre_features(Z, deg, xp):
    """Products of Legendre polynomials P_i(z_1) P_j(z_2) ... with total degree <= deg; Z in [-1, 1]."""
    n, m = Z.shape
    P = [[xp.ones(n), Z[:, k]] for k in range(m)]
    for k in range(m):
        for i in range(2, deg + 1):
            P[k].append(((2 * i - 1) * Z[:, k] * P[k][i - 1] - (i - 1) * P[k][i - 2]) / i)
    exps = [e for e in itertools.product(range(deg + 1), repeat=m) if sum(e) <= deg]
    cols = []
    for e in exps:
        c = P[0][e[0]]
        for k in range(1, m):
            c = c * P[k][e[k]]
        cols.append(c)
    return xp.stack(cols, axis=1)


def solve_dense(d, feat, t0, t1, chunk=None):
    """M c = b with dense features feat(idx) -> (len(idx), n_basis) on the GPU; phi = 0 on A / B frames."""
    import cupy as cp
    if chunk is None:                                       # about 0.6 GB per feature block
        chunk = max(10_000, int(8e7 / int(feat(cp.arange(2)).shape[1])))
    AB = cp.asarray(d['A'] | d['B'])
    Bf = cp.asarray(d['B'].astype(np.float64))
    w = cp.asarray(d['path_weights'])
    M = b = None
    for s in range(0, len(t0), chunk):
        a = cp.asarray(t0[s:s + chunk]); c = cp.asarray(t1[s:s + chunk])
        Fa = feat(a) * (~AB[a])[:, None]
        Fc = feat(c) * (~AB[c])[:, None]
        wa = w[a][:, None]
        Mi = (Fa * wa).T @ (Fc - Fa)
        bi = -(Fa * wa).T @ (Bf[c] - Bf[a])
        M = Mi if M is None else M + Mi
        b = bi if b is None else b + bi
    reg = 1e-10 * float(cp.trace(cp.abs(M))) / M.shape[0]
    coef = cp.linalg.solve(M - reg * cp.eye(M.shape[0]), b)
    n = len(d['A'])
    q = cp.zeros(n)
    for s in range(0, n, chunk):
        idx = cp.arange(s, min(s + chunk, n))
        q[s:s + chunk] = feat(idx) @ coef * (~AB[idx])
    q = q + Bf
    return cp.asnumpy(q), float(np.linalg.cond(cp.asnumpy(M))) if M.shape[0] <= 3000 else None


def solve_kmeans(d, lab, n_cl, t0, t1):
    """Indicator basis: the committor of the clusters' transition matrix (sparse, CPU)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    w = d['path_weights']
    A, B = d['A'], d['B']
    la = lab[t0]
    to_state = np.where(A[t1], n_cl, np.where(B[t1], n_cl + 1, lab[t1]))
    C = coo_matrix((w[t0], (la, to_state)), shape=(n_cl, n_cl + 2)).tocsr()
    rows = np.asarray(C.sum(1)).ravel()
    ok = rows > 0
    T = C[:, :n_cl].multiply(1 / np.where(ok, rows, 1)[:, None]).tocsr()
    tB = np.asarray(C[:, n_cl + 1].todense()).ravel() / np.where(ok, rows, 1)
    from scipy.sparse import identity
    Msys = (identity(n_cl) - T).tocsc()
    Msys = Msys + identity(n_cl) * 1e-12
    c = spsolve(Msys, tB)
    c = np.where(ok, c, np.nan)
    q = np.where(A, 0.0, np.where(B, 1.0, c[lab]))
    q = np.where(np.isfinite(q), q, 0.5)
    return q, int((~ok).sum())


def stopped_zq(d, q):
    import cupy as cp
    orc = h._optimalrcs_cupy()
    import metrics_tis
    rc = orc.CommittorNE(boundary0=d['A'], boundary1=d['B'], i_traj=d['i_traj'], t_traj=d['t_traj'],
                         path_weights=d['path_weights'], group_traj=d['group_traj'], seed_r=np.asarray(q, float))
    rc.set_fixed_traj_length_trap(d['A'] | d['B'], 2 ** 16)
    z = float(metrics_tis.max_z_zq_stopped(rc))
    del rc
    cp.get_default_memory_pool().free_all_blocks()
    return z


def score(d, q, qp, same, plateau):
    sc = h.scores(d, q, qp)
    return dict(rms=sc['rms'], r2_w=sc['r2_w'],
                plateau_bias=float(np.average((q - d['q_exact'])[plateau], weights=d['path_weights'][plateau])),
                dirichlet=float(np.sum(d['path_weights'][:-1] * np.where(same, np.diff(q), 0) ** 2)),
                q_min=float(q.min()), q_max=float(q.max()))


def refine(d, q0, n_iter, step=250):
    """NPNEq from the solved committor (local envelope 0.01, gamma 0.1, variant `full` CVs)."""
    import cupy as cp
    orc = h._optimalrcs_cupy()
    np.random.seed(0); cp.random.seed(0)
    rc = orc.CommittorNE(boundary0=d['A'], boundary1=d['B'], i_traj=d['i_traj'], t_traj=d['t_traj'],
                         path_weights=d['path_weights'], group_traj=d['group_traj'],
                         seed_r=np.clip(np.asarray(q0, float), 0, 1))
    rc.set_fixed_traj_length_trap(d['A'] | d['B'], 2 ** 16)
    X = cp.asarray(h.cv_matrix(d['X'], h.VARIANTS['full']['cvs']))
    orc.envelope_scale = 0.01
    out, done = [], 0
    while done < n_iter:
        n = min(step, n_iter - done)
        rc.fit_transform(lambda: X[:, np.random.randint(X.shape[1])], gamma=0.1, ny=6, max_iter=n - 1 if done == 0 else n,
                         print_step=step, metrics_print=('iter', 'max_z_zq_stopped'), save_min_metric='max_z_zq_stopped')
        done += n
        out.append((done, cp.asnumpy(rc.r_traj).astype(float), float(rc.metrics_history['max_z_zq_stopped'][-1])))
    del rc, X
    cp.get_default_memory_pool().free_all_blocks()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--basis', nargs='+', default=['poly6', 'kmeans2000', 'rbf1000'])
    ap.add_argument('--lag', type=int, default=1)
    ap.add_argument('--refine', type=int, default=0)
    ap.add_argument('--refine-step', type=int, default=250, help='iterations between refinement checks')
    a = ap.parse_args()
    import cupy as cp
    d = h.load_data()
    qp = h.projection(d)
    same = d['i_traj'][1:] == d['i_traj'][:-1]
    inner = ~d['A'] & ~d['B']
    plateau = inner & (d['q_exact'] > 0.2) & (d['q_exact'] < 0.4)
    t0, t1 = transitions(d, a.lag)
    X, lo, hi = scaled_cvs(d)
    print(f'{len(t0)} transitions at lag {a.lag}; exact committor: zqs_z {stopped_zq(d, d["q_exact"]):.2f}', flush=True)
    results = []
    for basis in a.basis:
        tt = time.time()
        info = {}
        if basis.startswith('poly'):
            deg = int(basis[4:])
            Z = cp.asarray(np.clip(2 * (X - lo) / (hi - lo) - 1, -1, 1))
            feat = lambda idx, Z=Z, deg=deg: legendre_features(Z[idx], deg, cp)
            q, cond = solve_dense(d, feat, t0, t1)
            info['n_basis'] = int(feat(cp.arange(2)).shape[1])
            info['cond'] = cond
            del Z
        else:
            from sklearn.cluster import MiniBatchKMeans
            n_cl = int(basis[6:] if basis.startswith('kmeans') else basis[3:])
            S = X / X[inner].std(0)
            rng = np.random.default_rng(0)
            sub = rng.choice(np.flatnonzero(inner), size=min(400_000, int(inner.sum())), replace=False)
            km = MiniBatchKMeans(n_clusters=n_cl, random_state=0, batch_size=8192, n_init=3).fit(S[sub])
            info['n_basis'] = n_cl
            if basis.startswith('kmeans'):
                lab = km.predict(S)
                q, n_empty = solve_kmeans(d, lab, n_cl, t0, t1)
                info['clusters_without_transitions'] = n_empty
            else:
                from sklearn.neighbors import NearestNeighbors
                cen = km.cluster_centers_
                dist = NearestNeighbors(n_neighbors=6).fit(cen).kneighbors(cen)[0][:, 1:]
                sig = float(np.median(dist))
                Sg, Cg = cp.asarray(S), cp.asarray(cen)

                def feat(idx, Sg=Sg, Cg=Cg, sig=sig):
                    x = Sg[idx]
                    d2 = (x * x).sum(1)[:, None] - 2 * x @ Cg.T + (Cg * Cg).sum(1)[None, :]
                    return cp.exp(-cp.maximum(d2, 0) / (2 * sig ** 2))
                q, cond = solve_dense(d, feat, t0, t1)
                info.update(width=sig, cond=cond)
                del Sg, Cg
        cp.get_default_memory_pool().free_all_blocks()
        qc = np.clip(q, 0, 1)
        row = dict(basis=basis, lag=a.lag, minutes=(time.time() - tt) / 60, **info,
                   outside_01=float(np.mean((q < -1e-6) | (q > 1 + 1e-6))), **score(d, qc, qp, same, plateau),
                   zqs_z=stopped_zq(d, qc))
        print('SOLVE ' + json.dumps(row), flush=True)
        np.save(os.path.join(HERE, f'q_{basis}_lag{a.lag}.npy'), qc.astype(np.float32))
        if a.refine:
            row['refine'] = []
            for it_done, r, zq in refine(d, qc, a.refine, a.refine_step):
                rr = dict(iter=it_done, zqs_z=zq, **score(d, r, qp, same, plateau))
                row['refine'].append(rr)
                print(f'REFINE {basis} ' + json.dumps(rr), flush=True)
        results.append(row)
        json.dump(results, open(os.path.join(HERE, f'results_lag{a.lag}.json'), 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
