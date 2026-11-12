"""How exact is the "exact" (finite-volume PDE) committor of the 2D Brownian runs?

For the Z potential (sim_istarz_br_g10) and the curved channel (test_cc_br):
1. grid convergence: the PDE committor at the path frames for two grid spacings;
2. shooting to absorption with the engine's own update (Euler-Maruyama, x += dt/gamma F + N(0, sigma),
   absorption checked every step) from 200 path frames, 20 in each of 10 bins of the PDE committor,
   N_SHOTS trajectories each; residual PDE - shooting with its binomial error.
    python check_reference.py z|cc [N_SHOTS]
"""
import sys, time, glob, numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import chi2
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/examples')
import toy2d
OUT = REPO + '/examples/figures/reference_check/'
CASE = sys.argv[1]
N_SHOTS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rng = np.random.default_rng(0)

if CASE == 'z':
    sys.path.insert(0, toy2d.TOYTIS)
    from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
    pot = RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)
    KT, GAMMA, DT, LA, LB = 0.05, 10.0, 0.002, 0.05, 0.85
    d = np.load(REPO + '/examples/figures/istarz_br_g10/committor.npz')
    grids = {h: np.load(REPO + f'/examples/figures/z_brownian/committor_reference_h{h}.npz') for h in ('0.002', '0.001')}
    def interp(g, y, x):
        return np.clip(RegularGridInterpolator((g['y'], g['x']), g['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)

    def force(y, x):                     # vectorized, term by term as in potential_and_force
        p = pot
        fx = cp.where(x < 0, -p.k * x, cp.where(x > p.Lx, -p.k * (x - p.Lx), 0.0))
        fy = cp.where(y < 0, -p.k * y, cp.where(y > p.Ly, -p.k * (y - p.Ly), 0.0))
        norm = np.sqrt(p.r ** 2 + 1)
        D = cp.abs(p.r * x - y + p.b) / norm
        dV = cp.where(D < p.L / 4, -2 * np.pi * p.A / p.L * cp.sin(2 * np.pi * D / p.L), 0.0)
        sgn = cp.sign(y - p.r * x - p.b)
        fx = fx + sgn * dV * p.r / norm
        fy = fy - sgn * dV / norm
        for yc, amp, ell in zip(p.state_y, p.state_A, p.state_L):
            Ds = cp.abs(y - yc)
            fy = fy + cp.where(Ds < ell / 4, amp * 2 * np.pi / ell * cp.sin(2 * np.pi * Ds / ell) * cp.sign(y - yc), 0.0)
        span = p.slope_end_y - p.slope_start_y
        fy = fy + cp.where((y >= p.slope_start_y) & (y <= p.slope_end_y), -p.slope_strength_y / span, 0.0) - p.gravity
        return fy, fx
    chk = rng.uniform([-0.02, -0.02], [0.92, 0.32], size=(300, 2))
    fv = np.column_stack([a.get() for a in force(cp.asarray(chk[:, 0]), cp.asarray(chk[:, 1]))])
    assert np.allclose(fv, [pot.potential_and_force((p_, np.zeros(2)))[1] for p_ in chk], atol=1e-9)
else:
    run = toy2d.read_full_settings(f'{toy2d.TOYTIS}/simulations/test_cc_br')
    pot = toy2d.potential_from_settings(run)
    KT, GAMMA, DT = run['temperature'], run['friction'], run['dt']
    LA, LB = run['interfaces'][0], run['interfaces'][-1]
    d = np.load(glob.glob(REPO + '/examples/figures/nb_toy2d_staple/test_cc_br/committor_*.npz')[0])
    yy, xx = d['X_all'][:, 0], d['X_all'][:, 1]
    yr = (np.percentile(yy, 0.1) - 0.05, np.percentile(yy, 99.9) + 0.05)
    xr = (np.percentile(xx, 0.1) - 0.1, np.percentile(xx, 99.9) + 0.1)
    grids = {h: toy2d.exact_committor_brownian(pot, KT, GAMMA, DT, LA, LB, h=float(h), y_range=yr, x_range=xr)
             for h in ('0.004', '0.002')}
    interp = lambda g, y, x: toy2d.interp_grid(g, y, x)
    def force(y, x):
        _, fy, fx = pot.potential_and_force_xy(y, x, cp)
        return fy, fx

X_all = d['X_all']
y_all, x_all = X_all[:, 0].astype(float), X_all[:, 1].astype(float)
free = ~(d['boundary0'] | d['boundary1'])
h_coarse, h_fine = grids
sub = np.flatnonzero(free)[::50]
qc, qf = interp(grids[h_coarse], y_all[sub], x_all[sub]), interp(grids[h_fine], y_all[sub], x_all[sub])
dq = qf - qc
print(f'{CASE}: grid h {h_coarse} vs {h_fine} at {len(sub)} path frames: RMS {np.sqrt(np.mean(dq ** 2)):.4f}, '
      f'max |diff| {np.abs(dq).max():.4f} (at q {qf[np.argmax(np.abs(dq))]:.2f})', flush=True)

# start points: 20 path frames in each of 10 bins of the (fine) PDE committor
q_all = interp(grids[h_fine], y_all[free], x_all[free])
idx_free = np.flatnonzero(free)
k = np.clip((q_all * 10).astype(int), 0, 9)
pts = np.concatenate([rng.choice(idx_free[k == j], size=min(20, np.sum(k == j)), replace=False) for j in range(10)])
y0, x0 = y_all[pts], x_all[pts]
n = len(pts) * N_SHOTS
Y = cp.asarray(np.repeat(y0, N_SHOTS)); X = cp.asarray(np.repeat(x0, N_SHOTS))
out = cp.full(n, -1.0); alive = cp.arange(n)
sigma = np.sqrt(2 * KT * DT / GAMMA)
cp.random.seed(1)
t0, step = time.time(), 0
while alive.size:
    ya, xa = Y[alive], X[alive]
    for _ in range(200):                                  # 200 steps between compactions
        fy, fx = force(ya, xa)
        ya = ya + DT / GAMMA * fy + sigma * cp.random.standard_normal(ya.size)
        xa = xa + DT / GAMMA * fx + sigma * cp.random.standard_normal(xa.size)
        hitA, hitB = ya <= LA, ya >= LB
        first = (out[alive] < 0) & (hitA | hitB)
        out[alive[first]] = cp.where(hitB[first], 1.0, 0.0)
    step += 200
    Y[alive], X[alive] = ya, xa
    alive = alive[out[alive] < 0]
    if step % 200000 == 0:
        print(f'  step {step}: {alive.size} running ({time.time() - t0:.0f} s)', flush=True)
res = out.get().reshape(len(pts), N_SHOTS)
q_shoot = res.mean(1); se = np.sqrt(np.maximum(q_shoot * (1 - q_shoot), 1 / N_SHOTS) / N_SHOTS)
print(f'shooting: {n} trajectories in {time.time() - t0:.0f} s ({step} steps)')
summary = {}
for h, g in grids.items():
    qp = interp(g, y0, x0)
    z = (qp - q_shoot) / se
    c2 = np.sum(z ** 2)
    summary[h] = (qp, z)
    print(f'PDE h {h} - shooting at {len(pts)} frames: mean {np.mean(qp - q_shoot):+.4f}, RMS {np.sqrt(np.mean((qp - q_shoot) ** 2)):.4f}, '
          f'chi2/dof {c2 / len(pts):.2f} (p {chi2.sf(c2, len(pts)):.3g}), |z| > 3 at {np.sum(np.abs(z) > 3)} frames; '
          f'mean shooting s.e. {se.mean():.4f}')
    worst = np.argsort(-np.abs(z))[:5]
    for i in worst:
        print(f'    (y, x) = ({y0[i]:.3f}, {x0[i]:.3f}): PDE {qp[i]:.3f}, shooting {q_shoot[i]:.3f} +- {se[i]:.3f} (z {z[i]:+.1f})')
np.savez(OUT + f'{CASE}_check.npz', y0=y0, x0=x0, q_shoot=q_shoot, se=se, **{f'q_pde_h{h}': v[0] for h, v in summary.items()})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for h, (qp, z) in summary.items():
    axes[0].errorbar(q_shoot, qp - q_shoot, yerr=se, fmt='.', ms=4, alpha=0.6, label=f'PDE h {h}')
    axes[1].scatter(y0, x0, c=np.clip(z, -5, 5), cmap='PuOr', vmin=-5, vmax=5, s=18) if h == h_fine else None
axes[0].axhline(0, color='k', lw=0.8)
axes[0].set(xlabel='shooting committor', ylabel='PDE $-$ shooting', title=f'{CASE}: {len(pts)} path frames x {N_SHOTS} trajectories')
axes[0].legend()
g = grids[h_fine]
axes[1].contour(g['y'], g['x'], g['q'].T, levels=[0.1, 0.25, 0.5, 0.75, 0.9], colors='k', linewidths=0.7)
axes[1].set(xlabel='y', ylabel='x', title=f'(PDE h {h_fine} $-$ shooting) / s.e. at the start frames (colour, clipped at ±5)')
fig.tight_layout()
fig.savefig(OUT + f'{CASE}_check.png', dpi=110)
print('saved', OUT)
