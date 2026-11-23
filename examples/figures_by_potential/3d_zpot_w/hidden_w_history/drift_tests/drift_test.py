"""Why does the NPNEq fit drift away from the exact committor on the gate (zw) staple data?

From q = 1/2 with the global envelope (1e6), every seed is close on the plateau after 500 iterations (bias about
-0.01) and drifts to a bias of about -0.12 by 1000 (seed_select.log), while the exact committor is a stable fixed
point (from_exact). One change per test, same start, seed 0, variant `full` (y, x, u, v, w), checks every 250
iterations against the exact committor (validation only):

    baseline     gamma 0.1, ny 6, clipping to [0, 1]          (CuPy)
    gamma0       gamma 0
    stable       stable=True with stable_negative=True (q(t) <- projected E[q(t+1) | basis]; the default sign of
                 stable moves q by -E[dq] and is flagged POSSIBLE_BUG in nonparametrics._npneq_moments)
    ny10         polynomial degree 10
    noclip       no clipping of q to [0, 1] after each update (nonparametrics._npneq_update patched here)
    tf           the TensorFlow version (optimalrcs/), baseline settings

Usage: python drift_test.py <test> [--iters 1500] [--step 250]
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hidden_w_history as h  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('test', choices=['baseline', 'gamma0', 'stable', 'ny10', 'noclip', 'tf'])
ap.add_argument('--iters', type=int, default=1500)
ap.add_argument('--step', type=int, default=250)
a = ap.parse_args()

d = h.load_data()
qp = h.projection(d)
same = d['i_traj'][1:] == d['i_traj'][:-1]
inner = ~d['A'] & ~d['B']
plateau = inner & (d['q_exact'] > 0.2) & (d['q_exact'] < 0.4)
Xn = h.cv_matrix(d['X'], h.VARIANTS['full']['cvs'])
kw = dict(gamma=0.0 if a.test == 'gamma0' else 0.1, ny=10 if a.test == 'ny10' else 6, print_step=a.step,
          metrics_print=('iter', 'max_z_zq_stopped', 'delta_x'), save_min_metric='max_z_zq_stopped')
np.random.seed(0)

if a.test == 'tf':
    sys.path.insert(0, h.REPO)
    import optimalrcs as orc_tf                     # the TensorFlow package
    orc = orc_tf.optimalrcs
    q = orc.CommittorNE(boundary0=d['A'], boundary1=d['B'], i_traj=d['i_traj'], t_traj=d['t_traj'],
                        path_weights=d['path_weights'], group_traj=d['group_traj'])
    if hasattr(q, 'set_fixed_traj_length_trap'):
        q.set_fixed_traj_length_trap(d['A'] | d['B'], 2 ** 16)
    X = Xn
    comp_y = lambda: X[:, np.random.randint(X.shape[1])]
    to_np = lambda r: np.asarray(r)
    kw['metrics_print'] = ('max_z_zq_stopped',)          # the TF metrics have no 'iter' / 'delta_x'
else:
    import cupy as cp
    cp.random.seed(0)
    orc = h._optimalrcs_cupy()
    import nonparametrics
    if a.test == 'noclip':
        def _update_noclip(r_traj, fk, al_j, step):
            if not callable(fk):
                return r_traj + cp.tensordot(al_j, fk, 1)
            rn = r_traj.copy()
            for s in range(0, r_traj.shape[0], step):
                e = min(s + step, r_traj.shape[0])
                rn[s:e] += cp.tensordot(al_j, fk(s, e), 1)
            return rn
        nonparametrics._npneq_update = _update_noclip
    q = orc.CommittorNE(boundary0=d['A'], boundary1=d['B'], i_traj=d['i_traj'], t_traj=d['t_traj'],
                        path_weights=d['path_weights'], group_traj=d['group_traj'])
    q.set_fixed_traj_length_trap(d['A'] | d['B'], 2 ** 16)
    X = cp.asarray(Xn)
    comp_y = lambda: X[:, np.random.randint(X.shape[1])]
    to_np = lambda r: cp.asnumpy(r)
    if a.test == 'stable':
        kw.update(stable=True, stable_negative=True)
orc.envelope_scale = 1e6

rows, t0, done = [], time.time(), 0
while done < a.iters:
    n = min(a.step, a.iters - done)
    q.fit_transform(comp_y, max_iter=n - 1 if done == 0 else n, **kw)
    done += n
    r = to_np(q.r_traj).astype(float).ravel()
    sc = h.scores(d, r, qp)
    zq = q.metrics_history.get('max_z_zq_stopped', [np.nan])[-1]
    row = dict(test=a.test, iter=done, zqs_z=float(zq), rms=sc['rms'],
               plateau_bias=float(np.average((r - d['q_exact'])[plateau], weights=d['path_weights'][plateau])),
               dirichlet=float(np.sum(d['path_weights'][:-1] * np.where(same, np.diff(r), 0) ** 2)),
               r_min=float(r.min()), r_max=float(r.max()), minutes=(time.time() - t0) / 60)
    rows.append(row)
    print('CHECK ' + json.dumps(row), flush=True)
json.dump(rows, open(os.path.join(HERE, f'{a.test}.json'), 'w'), indent=1)
print('DONE', a.test, flush=True)
