"""Exact committor and rate of the M3 gate system (ase_systems.zw_br, toytis zpot_w.ZChannelsW) on a 3D grid:
the method of toy2d.exact_committor_brownian with a third axis. Finite volumes for the overdamped generator
div(exp(-V/kT) grad q) = 0 on cells of side h (the same h on every axis), q = 0 for y <= lambda_A - s and 1 for
y >= lambda_B + s (s = 0.5826 sigma, the mean overshoot of a check every step), and the TPT rate
k_AB = D sum_edges w (dq)^2 / h^2 / sum rho (1 - q). V is the effective potential (gravity added back), the one
whose gradient is the engine's force.

The linear system is symmetric: Jacobi-preconditioned CG, on the GPU (cupy) when --gpu, warm-started from the
solution of a coarser grid (--init) or from a ramp in y.

    python -m loop.exact_zw --h 0.008                     # CPU, ~0.3M cells: a few minutes
    python -m loop.exact_zw --h 0.004 --gpu --init <h0.008 file>   # ~2M cells: GPU (workstation)

Output: figures_by_potential/3d_zpot_w/exact/q_tpt_h<h>.npz (y, x, w, q[iy, ix, iw], k_ab, steps_per_ab, h, ...).
Check: the reference rate of the GPU MD scan (ase_systems.zw_br md_rate, 2.10e-5 per time unit +- 2.4%).
"""
import argparse
import os
import time

import numpy as np

from loop import env, ase_systems

BOX = dict(y=(-0.03, 0.93), x=(-0.03, 0.33), w=(-0.21, 0.21))


def grid_potential(pot, ys, xs, ws):
    V = np.empty((len(ys), len(xs), len(ws)))
    vel = np.zeros(3)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            for k, w in enumerate(ws):
                V[i, j, k] = pot.effective_potential((np.array([y, x, w]), vel))
    return V


def solve(system, h, gpu=False, init=None, tol=1e-10, max_iter=200000, log=print):
    import scipy.sparse as sp
    from loop.ase_calcs import load_toytis_potential
    c = dict(system['calculator'])
    pot = load_toytis_potential(c['spec'], c['params'])
    ys = np.arange(BOX['y'][0] + h / 2, BOX['y'][1], h)
    xs = np.arange(BOX['x'][0] + h / 2, BOX['x'][1], h)
    ws = np.arange(BOX['w'][0] + h / 2, BOX['w'][1], h)
    t0 = time.time()
    V = grid_potential(pot, ys, xs, ws)
    log(f'grid {V.shape} = {V.size} cells, potential in {time.time() - t0:.0f} s')
    kT, gamma, dt = system['kT'], system['gamma'], system['dt']
    bv = (V - V.min()) / kT
    sigma = np.sqrt(2 * kT * dt / gamma)
    shift = 0.5826 * sigma
    shape = V.shape
    idx = np.arange(V.size).reshape(shape)
    fixed0 = np.broadcast_to((ys <= system['lambda_a'] - shift)[:, None, None], shape)
    fixed1 = np.broadcast_to((ys >= system['lambda_b'] + shift)[:, None, None], shape)
    rows, cols, vals, wts = [], [], [], []
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax], b[ax] = slice(None, -1), slice(1, None)
        we = np.exp(-(bv[tuple(a)] + bv[tuple(b)]) / 2)
        wts.append(we)
        rows += [idx[tuple(a)].ravel(), idx[tuple(b)].ravel()]
        cols += [idx[tuple(b)].ravel(), idx[tuple(a)].ravel()]
        vals += [we.ravel(), we.ravel()]
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V.size, V.size)).tocsr()
    L = L - sp.diags(np.asarray(L.sum(axis=1)).ravel())
    qfix = fixed1.ravel().astype(float)
    free = ~(fixed0 | fixed1).ravel()
    F, C = np.flatnonzero(free), np.flatnonzero(~free)
    A = (-L[F][:, F]).tocsr()                                  # symmetric positive definite
    rhs = L[F][:, C] @ qfix[C]
    if init is not None:
        from scipy.interpolate import RegularGridInterpolator
        z = np.load(init)
        f = RegularGridInterpolator((z['y'], z['x'], z['w']), z['q'], bounds_error=False, fill_value=None)
        Y, X, W = np.meshgrid(ys, xs, ws, indexing='ij')
        x0 = np.clip(f(np.c_[Y.ravel(), X.ravel(), W.ravel()]), 0, 1)[F]
    else:
        Y = np.broadcast_to(ys[:, None, None], shape).ravel()
        x0 = np.clip((Y[F] - system['lambda_a']) / (system['lambda_b'] - system['lambda_a']), 0, 1)
    d = A.diagonal()
    t0 = time.time()
    if gpu:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as cspl
        Ag = csp.csr_matrix(A)
        Mi = csp.diags(cp.asarray(1.0 / d))
        sol, info = cspl.cg(Ag, cp.asarray(rhs), x0=cp.asarray(x0), tol=tol, maxiter=max_iter, M=Mi)
        sol = cp.asnumpy(sol)
    else:
        import scipy.sparse.linalg as spla
        Mi = sp.diags(1.0 / d)
        sol, info = spla.cg(A, rhs, x0=x0, rtol=tol, maxiter=max_iter, M=Mi)
    res = np.linalg.norm(A @ sol - rhs) / np.linalg.norm(rhs)
    log(f'CG ({"GPU" if gpu else "CPU"}): info {info}, relative residual {res:.2e}, {time.time() - t0:.0f} s')
    q = qfix.copy()
    q[F] = sol
    Q = q.reshape(shape)
    D = kT / gamma
    flux = D * sum((we * np.diff(Q, axis=ax) ** 2).sum() for ax, we in enumerate(wts)) / h ** 2
    rho = np.exp(-bv)
    k_ab = flux / (rho * (1 - Q)).sum()
    return dict(y=ys, x=xs, w=ws, q=Q.astype(np.float32), k_ab=k_ab, steps_per_ab=1 / (k_ab * dt), shift=shift, h=h,
                lambda_a=system['lambda_a'], lambda_b=system['lambda_b'], cg_info=info, residual=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=float, default=0.008)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--init', default=None, help='a coarser solution (npz) as the CG start')
    ap.add_argument('--tol', type=float, default=1e-10)
    a = ap.parse_args()
    s = ase_systems.get('zw_br')
    r = solve(s, a.h, gpu=a.gpu, init=a.init, tol=a.tol)
    out = os.path.join(env.EXAMPLES, 'figures_by_potential', '3d_zpot_w', 'exact', f'q_tpt_h{a.h:g}.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **r)
    ref = s['md_rate']
    print(f'h {a.h}: k_AB {r["k_ab"]:.5g} per time unit ({r["steps_per_ab"]:.4g} steps per transition); MD reference '
          f'{ref["rate"]:.4g} +- {100 * ref["rel_err"]:.1f}% -> {out}', flush=True)


if __name__ == '__main__':
    main()
