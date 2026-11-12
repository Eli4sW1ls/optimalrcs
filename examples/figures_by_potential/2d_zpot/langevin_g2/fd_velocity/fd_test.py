"""2D Langevin Z staple run (sim_istarz_2603), positions only: finite-difference velocities of y and x
(change over one frame, computed on the whole path before the cut) as extra CVs, no history.
A: standard lag-1 NPNEq fit. B: lag-4 fit (stopped windows: sum_t f_k(t)[q(min(t+4, tau_t)) - q(t)] = 0),
as in examples/figures_by_potential/1d_cos_well/langevin_staple/checks/well_test.py. Same data and settings as base_documented_2d_langevin.ipynb
otherwise (1000 paths/ensemble, no mirrored pieces, gamma 0.2, envelope 0.001, ny 6)."""
import argparse, io, contextlib, os, sys, time
import numpy as np, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, f'{REPO}/optimalrcs-cupy/'); sys.path.insert(0, f'{REPO}/examples/')
import optimalrcs, nonparametrics, metrics_tis as M, tis_data
plt.show = lambda *a, **k: None
OUT = os.path.dirname(os.path.abspath(__file__))
MAX_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
args = p.parse_args(['/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_2603',
                     '--paths-per-ensemble', '1000', '--seed', '0', '--fd-velocity']
                    + (['--reversed'] if os.environ.get('REVERSED') == '1' else []))   # REVERSED=1: mirrored pieces
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(args)
print(f"{len(data.lam)} frames, CVs: y, x, dy, dx per frame -> shape {data.X.shape}", flush=True)
DIAG = os.environ.get('DIAG') == '1'
if DIAG:
    # coordinate and velocity normal to the 45-degree ridge (y = -x + 0.6): y + x, dy + dx
    data.X = np.c_[data.X, data.X[:, 0] + data.X[:, 1], data.X[:, 2] + data.X[:, 3]]
    print(f"with ridge-normal CVs y + x, dy + dx -> shape {data.X.shape}", flush=True)
TAG = ('rev_' if os.environ.get('REVERSED') == '1' else '') + ('diag_' if DIAG else '')
X = cp.asarray(data.X)
b = data.boundary0 | data.boundary1
tau_np = cp.asnumpy(M.stop_index(data.i_traj, b))
LAGS = (1, 2, 4, 8, 16, 64, 256, 1024)

def zrow(r):
    out = []
    for dt in LAGS:
        _, z = M.comp_zq_stopped_zscore(cp.asarray(r), b, data.i_traj, data.path_weights, dt,
                                        tau=cp.asarray(tau_np), g_traj=data.group_traj)
        out.append(float(cp.max(cp.abs(z[5:]))))
    return out

class LazyRY:
    """basis_poly_ry evaluated on chunks of frames, normalized over all frames as basis_poly_ry does."""
    def __init__(self, r, y, n, fenv):
        self.r, self.y, self.n = r / cp.max(cp.abs(r)), y / cp.max(cp.abs(y)), n
        self.fenv = cp.ones_like(r) if fenv is None else fenv
    def cols(self, idx):
        return optimalrcs._poly_ry(self.r[idx], self.y[idx], self.n, self.fenv[idx])

W = cp.asarray(data.path_weights)
n = len(data.lam)
original_npneq = nonparametrics.npneq
def make_npneq_lag(lag):
    u = cp.asarray(np.minimum(np.arange(n) + lag, tau_np))
    starts = cp.asarray(np.flatnonzero(~b & (np.minimum(np.arange(n) + lag, tau_np) > np.arange(n))))
    def npneq_lag(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk=None, w_traj=None, **kw):
        akj, bb = 0, 0
        for s0 in range(0, starts.size, 400_000):
            st = starts[s0:s0 + 400_000]
            fa, fb, wt = fk.cols(st), fk.cols(u[st]), W[st]
            akj = akj + fa @ (fb * wt - fa * (wt + gamma)).T
            bb = bb + fa @ (-(r_traj[u[st]] - r_traj[st]) * wt)
        al = cp.linalg.lstsq(akj, bb, rcond=None)[0]
        new = r_traj.copy()
        for s0 in range(0, n, 1_000_000):
            idx = cp.arange(s0, min(s0 + 1_000_000, n))
            new[idx] = r_traj[idx] + al @ fk.cols(idx)
        return cp.clip(new, 0, 1)
    return npneq_lag

for name, lag in (("lag1", 1), ("lag4", 4)):
    optimalrcs.envelope_scale = 0.001
    np.random.seed(0); cp.random.seed(0)
    q = optimalrcs.CommittorNE(boundary0=data.boundary0, boundary1=data.boundary1, i_traj=data.i_traj,
                               t_traj=data.t_traj, path_weights=data.path_weights, group_traj=data.group_traj)
    q.set_fixed_traj_length_trap(b, 2 ** 16)
    kw = dict(lazy_basis=True)
    if lag != 1:
        optimalrcs.nonparametrics.npneq = make_npneq_lag(lag)
        kw = dict(basis_functions=lambda r_, y_, n_, f_: LazyRY(r_, y_, n_, f_))
    else:
        optimalrcs.nonparametrics.npneq = original_npneq
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        q.fit_transform(lambda: X[:, np.random.randint(X.shape[1])], gamma=0.2, ny=6, max_iter=MAX_ITER,
                        print_step=250, metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'delta_x'),
                        save_min_metric='max_z_zq_stopped', **kw)
    h = q.metrics_history
    r = cp.asnumpy(q.r_traj)
    np.save(f"{OUT}/q_{TAG}{name}.npy", r)
    z = zrow(r)
    print(f"{name}: {time.time() - t0:.0f} s | zqs_z by print step {np.round(h.get('max_z_zq_stopped', []), 1).tolist()} | "
          f"final |dx| {(h.get('delta_x') or [np.nan])[-1]:.1e}", flush=True)
    print(f"{name}: max|z| at lags {LAGS}: {' '.join(f'{v:5.1f}' for v in z)}", flush=True)
    q.plots_tis(); plt.gcf().savefig(f"{OUT}/tis_{TAG}{name}.png", dpi=110); plt.close("all")
    del q; cp.get_default_memory_pool().free_all_blocks()
if not DIAG and not TAG: np.savez(f"{OUT}/frames.npz", lam=data.lam, x=data.X_all[:, 1], vy=data.X[:, 2], vx=data.X[:, 3],
         b0=data.boundary0, b1=data.boundary1, i_traj=data.i_traj)
print("ALL DONE", flush=True)
