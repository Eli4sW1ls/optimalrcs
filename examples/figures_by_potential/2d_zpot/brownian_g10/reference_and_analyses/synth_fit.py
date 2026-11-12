"""NPNEq on synthetic walker data with a known committor (synth_shoot.py).
1. Calibration: the stopped Z_q z-score of the exact committor on these unbiased data.
2. Fit with the staple settings (gamma 0.1, envelope ENV, ny 6, lazy basis, A/B traps, lag-1 NPNEq), from
   scratch or from the exact committor; error against the exact committor every CHUNK iterations.
    python synth_fit.py DATA.npz MODE(scratch|exact) N_ITER [ENV] [TAG]
"""
import sys, time, numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/')
import optimalrcs, metrics_tis as M
HERE = REPO + '/examples/figures/z_brownian/'
DATA, MODE, N_ITER = sys.argv[1], sys.argv[2], int(sys.argv[3])
ENV = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
TAG = sys.argv[5] if len(sys.argv) > 5 else f'{MODE}_env{ENV}'
LAG = int(sys.argv[6]) if len(sys.argv) > 6 else 1
CHUNK = 2500

d = np.load(DATA)
y, x, it, tt = d['y'], d['x'], d['i_traj'], d['t_traj'].astype(float)
A, B = y <= float(d['lambda_a']), y >= float(d['lambda_b'])
b = A | B; free = ~b
ref = np.load(HERE + 'committor_reference_h0.001.npz')
q_ex = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)
q_ex[A] = 0; q_ex[B] = 1
print(f'{TAG}: {len(y)} frames, {np.unique(it).size} walkers, {A.sum()} in A, {B.sum()} in B', flush=True)

ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
nb = np.histogram2d(y[free], x[free], bins=(ey, ex))[0]
bm = lambda v: np.where(nb >= 20, np.histogram2d(y[free], x[free], bins=(ey, ex), weights=v[free])[0] / np.maximum(nb, 1), np.nan)
QX = bm(q_ex); trans = (QX > 0.05) & (QX < 0.95); plateau = free & (q_ex > 0.55) & (q_ex < 0.85)
PTS = [(0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]
near = [free & ((y - a) ** 2 + (x - c) ** 2 < 0.015 ** 2) for a, c in PTS]
def report(r, label):
    print(f'{TAG} {label}: RMS per bin {np.sqrt(np.nanmean((bm(r) - QX)[trans] ** 2)):.4f} | plateau (fit - exact) '
          f'{np.mean(r[plateau] - q_ex[plateau]):+.4f} | at the points ' + ' '.join(f'{r[m].mean():.3f}' for m in near)
          + ' (exact ' + ' '.join(f'{q_ex[m].mean():.3f}' for m in near) + ')', flush=True)

if MODE == 'calib':
    Q, Bc, I = cp.asarray(q_ex), cp.asarray(b), cp.asarray(it)
    tau = M.stop_index(I, Bc)
    for name, r in (('exact committor', q_ex),):
        out = []
        for dt in (1, 4, 16, 64, 256):
            e, z = M.comp_zq_stopped_zscore(cp.asarray(r), Bc, I, None, dt, tau=tau)
            z = cp.asnumpy(z)[5:]; e = cp.asnumpy(e)[5:5 + len(z)]; k = np.argmax(np.abs(z)); out.append(f'{z[k]:+5.1f} (q {e[k]:.2f})')
        print(f'{TAG} calibration, {name}: worst z at lags 1 4 16 64 256: ' + '  '.join(out), flush=True)
    sys.exit()

optimalrcs.envelope_scale = ENV
Y = cp.asarray(np.c_[y, x])
KW = dict(lazy_basis=True)
if LAG > 1:
    # stopped lag-LAG windows: [t, min(t + LAG, tau_t)], tau_t = first A/B frame or the walker's last frame
    n = len(y); ends = np.r_[np.flatnonzero(it[1:] != it[:-1]), n - 1]; sts = np.r_[0, ends[:-1] + 1]
    nxt = np.minimum.accumulate(np.where(b, np.arange(n), n)[::-1])[::-1]
    u_np = np.minimum(np.arange(n) + LAG, np.minimum(nxt, np.repeat(ends, ends - sts + 1)))
    U = cp.asarray(u_np); STARTS = cp.asarray(np.flatnonzero(free & (u_np > np.arange(n))))
    class LazyRY:
        def __init__(self, r, yy, k, fenv):
            self.r, self.y, self.n = r / cp.max(cp.abs(r)), yy / cp.max(cp.abs(yy)), k
            self.fenv = cp.ones_like(r) if fenv is None else fenv
        def cols(self, idx):
            return optimalrcs._poly_ry(self.r[idx], self.y[idx], self.n, self.fenv[idx])
    def npneq_lag(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk=None, w_traj=None, **kw):
        akj, bb = 0, 0
        for s0 in range(0, STARTS.size, 400_000):
            st = STARTS[s0:s0 + 400_000]
            fa, fb = fk.cols(st), fk.cols(U[st])
            akj = akj + fa @ (fb - fa * (1 + gamma)).T
            bb = bb + fa @ (-(r_traj[U[st]] - r_traj[st]))
        al = cp.linalg.lstsq(akj, bb, rcond=None)[0]
        new = r_traj.copy()
        for s0 in range(0, r_traj.size, 1_000_000):
            idx = cp.arange(s0, min(s0 + 1_000_000, r_traj.size)); new[idx] = r_traj[idx] + al @ fk.cols(idx)
        return cp.clip(new, 0, 1)
    optimalrcs.nonparametrics.npneq = npneq_lag
    KW = dict(basis_functions=lambda r_, y_, k_, f_: LazyRY(r_, y_, k_, f_))
r_prev = q_ex.copy() if MODE == 'exact' else None
if r_prev is not None:
    report(r_prev, 'start (exact committor)')
t0 = time.time()
for k in range(N_ITER // CHUNK):
    np.random.seed(k); cp.random.seed(k)
    q = optimalrcs.CommittorNE(boundary0=A, boundary1=B, i_traj=it, t_traj=tt, seed_r=r_prev)
    q.set_fixed_traj_length_trap(b, 2 ** 16)
    q.fit_transform(lambda: Y[:, np.random.randint(2)], gamma=0.1, ny=6, max_iter=CHUNK, print_step=CHUNK,
                    metrics_print=('iter', 'max_z_zq_stopped', 'delta_x'), save_min_metric='max_z_zq_stopped', **KW)
    r_prev = cp.asnumpy(q.r_traj); h = q.metrics_history
    report(r_prev, f'{CHUNK * (k + 1)} it. ({time.time() - t0:.0f} s, zqs_z {h["max_z_zq_stopped"][-1]:.2f}, |dx| {h["delta_x"][-1]:.1e})')
    del q; cp.get_default_memory_pool().free_all_blocks()
np.save(HERE + f'synth/r_{TAG}.npy', r_prev)
print(f'{TAG} DONE', flush=True)
