"""2D Brownian Z long MD, seeds 1-8 (2.4M kept rows): does a lag-Delta NPNEq fit remove the
plateau overshoot? Lag in saved rows (1 row = 20 MD steps); windows stopped at the first A/B
frame (or the end of the excursion). CVs (y, x); gamma 0.1, envelope ENV, ny 6.
Compared with the exact committor for A: y <= 0.1, B: y >= 0.85, absorption every 20 steps.
    python lag_md.py LAG [ENV] [ITER] [SEED_R.npy] [TAG]
"""
import sys, time, numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/optimalrcs-cupy/')
import optimalrcs
HERE = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures/z_brownian/'
LAG = int(sys.argv[1]); ENV = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
N_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
SEED_R = (sys.argv[4] or None) if len(sys.argv) > 4 else None
TAG = sys.argv[5] if len(sys.argv) > 5 else f'lag{LAG}_env{ENV}'

d = np.load(HERE + 'npq_md/npq_md.npz')
import os
keep = (d['seed'] > 8) if os.environ.get('HALF') == '2' else (d['seed'] <= 8)
y, x, it = d['y'][keep], d['x'][keep], d['i_traj'][keep]
LA, LB = float(d['lambda_a']), float(d['lambda_b'])
A, B = y <= LA, y >= LB
free = ~(A | B)
n = len(y)
ref = np.load(HERE + 'committor_reference_h0.001_A0.1_B0.85_s20.npz')
q_ref = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)

# tau: first A/B row at or after t within its excursion, else the excursion's last row
seg_end = np.r_[np.flatnonzero(it[1:] != it[:-1]), n - 1]
end_of = np.repeat(seg_end, np.diff(np.r_[-1, seg_end]))
marker = np.where(A | B, np.arange(n), n)
nxt = np.minimum.accumulate(marker[::-1])[::-1]
tau = np.minimum(nxt, end_of)
u_np = np.minimum(np.arange(n) + LAG, tau)
U = cp.asarray(u_np)
STARTS = cp.asarray(np.flatnonzero(free & (u_np > np.arange(n))))


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
    for s0 in range(0, n, 1_000_000):
        idx = cp.arange(s0, min(s0 + 1_000_000, n))
        new[idx] = r_traj[idx] + al @ fk.cols(idx)
    return cp.clip(new, 0, 1)
optimalrcs.nonparametrics.npneq = npneq_lag

ey, ex = np.linspace(LA, LB, 76), np.linspace(-0.01, 0.31, 33)
nb = np.histogram2d(y[free], x[free], bins=(ey, ex))[0]
def bmean(v):
    return np.where(nb >= 20, np.histogram2d(y[free], x[free], bins=(ey, ex), weights=v[free])[0] / np.maximum(nb, 1), np.nan)
QR = bmean(q_ref); trans = (QR > 0.05) & (QR < 0.95)
PTS = [(0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]
near = [free & ((y - a) ** 2 + (x - b) ** 2 < 0.02 ** 2) for a, b in PTS]
def report(r, label):
    rms = np.sqrt(np.nanmean((bmean(r) - QR)[trans] ** 2))
    pts = ' '.join(f'{r[m].mean():.3f}/{q_ref[m].mean():.3f}' for m in near)
    plateau = free & (q_ref > 0.55) & (q_ref < 0.85)
    print(f'{TAG} {label}: RMS per bin {rms:.4f} | mean(fit - ref) on plateau {np.mean(r[plateau] - q_ref[plateau]):+.4f} '
          f'| fit/ref at points {pts}', flush=True)

Y = cp.asarray(np.c_[y, x])
optimalrcs.envelope_scale = ENV
r_prev = None if SEED_R is None else np.load(HERE + SEED_R)
t0 = time.time()
for k in range(N_ITER // 1000):
    np.random.seed(k); cp.random.seed(k)
    q = optimalrcs.CommittorNE(boundary0=A, boundary1=B, i_traj=it, t_traj=np.arange(n, dtype=float), seed_r=r_prev)
    q.fit_transform(lambda: Y[:, np.random.randint(2)], basis_functions=lambda r_, y_, k_, f_: LazyRY(r_, y_, k_, f_),
                    gamma=0.1, ny=6, max_iter=1000, print_step=1000, metrics_print=('iter', 'delta_x'),
                    save_min_delta_zq=False)
    r_prev = cp.asnumpy(q.r_traj)
    report(r_prev, f'{1000 * (k + 1)} it ({time.time() - t0:.0f} s, |dx| {q.metrics_history["delta_x"][-1]:.1e})')
    del q; cp.get_default_memory_pool().free_all_blocks()
np.save(HERE + f'lag_md_{TAG}.npy', r_prev)
