"""Exact committor q(y, x) of the 2D Brownian curved channel (toytis curved_channel.CurvedChannelDoubleWell, defaults;
gamma 10, kT 0.05, dt 0.005), staple definition (sim_istar_cc_br): A: y <= 0.15, B: y >= 0.85, checked every step.
The force is smooth (no jump lines), so only the absorption at A and B is split in the noise integral.
1. Grid solve (exact_tools.solve_brownian_2d) at several spacings; error estimate = change between the finest two.
2. GPU shooting with N_SHOOT trajectories at POINTS (binomial standard errors).
3. Comparison with the old reference (toy2d.exact_committor_brownian: continuous generator, finite volumes, h 0.004,
   A/B shifted by 0.5826 sigma), as used by toy2d_md.ipynb / toy2d_staple_analysis.ipynb.
4. Figures in cc/.
"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.dirname(os.path.dirname(HERE))                    # examples/figures_by_potential
sys.path.insert(0, f'{FIGROOT}/_shared/exact_committors')           # exact_tools
sys.path.insert(0, os.path.dirname(FIGROOT))                        # examples/ (toy2d)
import exact_tools as E
import toy2d
OUT = f'{HERE}/'; os.makedirs(OUT, exist_ok=True)
KT, GAMMA, DT = 0.05, 10.0, 0.005
A, B = 0.15, 0.85
LEVELS = [float(h) for h in os.environ.get('CC_LEVELS', '0.004,0.002,0.0015').split(',')]
X_RANGE = (-0.45, 0.6)                    # channel centre -0.15 .. 0.2; V > 13 kT beyond 0.3 from it
N_SHOOT = 1_000_000
pot = toy2d.load_potential('curved_channel:CurvedChannelDoubleWell')


def cc_force(y, x, xp=np):
    _, fy, fx = pot.potential_and_force_xy(y, x, xp)
    return fy, fx


POINTS = [(y, float(pot.centre_line(y))) for y in (0.17, 0.30, 0.45, 0.50, 0.55, 0.70, 0.83)]
POINTS += [(0.50, float(pot.centre_line(0.5)) - 0.08), (0.50, float(pot.centre_line(0.5)) + 0.08)]
log = open(OUT + 'cc_exact.log', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
rms = lambda a: np.sqrt(np.nanmean(np.asarray(a) ** 2))
ev = lambda sol, y, x: E.eval_q_brownian_2d(cc_force, sol, np.asarray(y, float), np.asarray(x, float))

say(f'===== curved channel, Brownian: gamma {GAMMA}, kT {KT}, dt {DT}, sigma {np.sqrt(2 * KT * DT / GAMMA):.5f}; '
    f'A: y <= {A}, B: y >= {B}, absorption every step =====')
sols = {}
for h in LEVELS:
    t0 = time.time()
    cache = OUT + f'q_exact_h{h}.npz'
    if os.path.exists(cache):
        c_ = np.load(cache)
        sols[h] = {k: (c_[k] if c_[k].ndim else c_[k].item()) for k in ('ys', 'xs', 'w', 'A', 'B', 'mu', 'stride', 'h')}
        say(f'  h {h}: loaded {cache}; q at POINTS ' + ' '.join(f'{v:.4f}' for v in ev(sols[h], *zip(*POINTS))))
        continue
    sols[h], r = E.solve_brownian_2d(cc_force, DT, GAMMA, KT, A, B, h, x_range=X_RANGE, stride=1, verbose=False)
    say(f'  h {h}: grid {sols[h]["w"].shape[0]} x {sols[h]["w"].shape[1]}, residual {r:.1e}, {time.time() - t0:.0f} s; '
        f'q at POINTS ' + ' '.join(f'{v:.4f}' for v in ev(sols[h], *zip(*POINTS))))
    np.savez(cache, **sols[h], kT=KT, gamma=GAMMA, dt=DT,
             note='q(s) = w(s + mu F(s)); evaluate with exact_tools.eval_q_brownian_2d(cc_force, ...) (see cc_exact.py)')

# changes on a fine evaluation grid; "channel" = V - V_min < 10 kT (where the data are)
gy, gx = np.linspace(A, B, 701), np.linspace(-0.4, 0.55, 476)
GY, GX = np.meshgrid(gy, gx, indexing='ij')
Vg = (pot.potential_and_force_xy(GY, GX)[0] - pot.stationary_points()['A'][2]) / KT
chan = Vg < 10
vals = {h: ev(sols[h], GY.ravel(), GX.ravel()).reshape(GY.shape) for h in LEVELS}
for h1, h2 in zip(LEVELS[:-1], LEVELS[1:]):
    dd = vals[h2] - vals[h1]
    say(f'  change h {h1} -> {h2}: channel (V < 10 kT) max {np.abs(dd[chan]).max():.1e}, RMS {rms(dd[chan]):.1e}; '
        f'whole box max {np.abs(dd).max():.1e}')
hs = LEVELS
Q = vals[hs[-1]]
err = np.abs(vals[hs[-1]] - vals[hs[-2]]) if len(hs) > 1 else np.full(Q.shape, np.nan)
sol = sols[hs[-1]]

# shooting
t0 = time.time()
p, se = E.shoot_brownian_2d(cc_force, DT, GAMMA, KT, A, B, POINTS, N_SHOOT, stride=1, seed=7)
qg = ev(sol, *zip(*POINTS))
say(f'  shooting, {N_SHOOT} trajectories per point ({time.time() - t0:.0f} s):')
for pt, pp, ss, qq in zip(POINTS, p, se, qg):
    say(f'    (y, x) = ({pt[0]:.2f}, {pt[1]:+.3f}): shooting {pp:.5f} +- {ss:.5f} | grid {qq:.5f} | diff {qq - pp:+.5f} ({(qq - pp) / ss:+.1f} s.e.)')
say(f'  chi^2 = {np.sum(((qg - p) / se) ** 2):.1f} for {len(POINTS)} points')

# old reference (as in the notebooks)
t0 = time.time()
old = toy2d.exact_committor_brownian(pot, KT, GAMMA, DT, A, B, stride=1, h=0.004)
f_old = RegularGridInterpolator((old['y'], old['x']), old['q'], bounds_error=False, fill_value=None)
Qold = np.clip(f_old(np.c_[GY.ravel(), GX.ravel()]), 0, 1).reshape(GY.shape)
do = Qold - Q
say(f'  old reference (toy2d.exact_committor_brownian, h 0.004, {time.time() - t0:.0f} s) - exact: '
    f'channel max {np.abs(do[chan]).max():.4f}, RMS {rms(do[chan]):.4f}; whole box max {np.abs(do).max():.4f}; '
    f'at POINTS ' + ' '.join(f'{v:+.4f}' for v in np.clip(f_old(np.array(POINTS)), 0, 1) - qg))
wB = np.exp(-Vg) * chan
say(f'  Boltzmann-weighted RMS (old - exact) over the channel: {np.sqrt((wB * do ** 2).sum() / wB.sum()):.4f}; '
    f'transition-region weighted (q(1-q) Boltzmann): {np.sqrt((wB * Q * (1 - Q) * do ** 2).sum() / (wB * Q * (1 - Q)).sum()):.4f}')
np.savez(OUT + 'q_exact_eval_grid.npz', y=gy, x=gx, q=Q, err=err, q_old=Qold, V_kT=Vg)

# ---------------------------------------------------------------- figures
vy, vx = np.linspace(0.0, 1.0, 401), np.linspace(-0.45, 0.6, 211)
VY, VX = np.meshgrid(vy, vx, indexing='ij')
V = pot.potential_and_force_xy(VY, VX)[0] / KT; V -= V.min()
lev = np.arange(0, 12.5, 1.0)
Fy = -np.log(np.exp(-V).sum(1) * (vx[1] - vx[0]))
fig, axes = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={'width_ratios': [1.6, 1]})
im = axes[0].pcolormesh(vy, vx, np.minimum(V, 12).T, cmap='viridis', shading='auto'); fig.colorbar(im, ax=axes[0], label='V / kT (clipped at 12)')
axes[0].contour(vy, vx, V.T, levels=lev, colors='0.85', linewidths=0.4)
axes[0].plot(vy, pot.centre_line(vy), 'w--', lw=0.8)
for b in (A, B): axes[0].axvline(b, color='w', ls='--', lw=0.8)
axes[0].plot(*zip(*POINTS), 'r+', ms=9)
axes[0].set(xlabel='order parameter y', ylabel='x', title='curved channel V/kT; white dashed: centre line, A, B; +: shooting points')
axes[1].plot(vy, Fy - Fy.min(), 'k'); axes[1].set(xlabel='y', ylabel='F(y) / kT', title='equilibrium free energy along y', ylim=(0, 8))
for b in (A, B): axes[1].axvline(b, color='C3', ls='--', lw=0.8)
fig.tight_layout(); fig.savefig(OUT + 'landscape.png', dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(21, 4.5))
im = axes[0].pcolormesh(gy, gx, Q.T, cmap='coolwarm', vmin=0, vmax=1, shading='auto'); fig.colorbar(im, ax=axes[0], label='q')
axes[0].contour(vy, vx, V.T, levels=lev, colors='0.4', linewidths=0.5)
axes[0].contour(gy, gx, Q.T, levels=[0.1, 0.5, 0.9], colors='k', linewidths=1, linestyles='--')
axes[0].set(title=f'exact committor (A: y <= {A}, B: y >= {B}, every step); dashed: q = 0.1, 0.5, 0.9', xlabel='y', ylabel='x')
im = axes[1].pcolormesh(gy, gx, np.log10(np.maximum(err, 1e-9)).T, cmap='magma', vmin=-8, vmax=-3, shading='auto')
fig.colorbar(im, ax=axes[1], label='log10 error estimate'); axes[1].contour(vy, vx, V.T, levels=[10], colors='w', linewidths=0.8)
axes[1].set(title=f'error estimate (h {hs[-2]} vs {hs[-1]}): channel max {np.nanmax(err[chan]):.0e}, RMS {rms(err[chan]):.0e}', xlabel='y', ylabel='x')
im = axes[2].pcolormesh(gy, gx, do.T, cmap='PuOr', vmin=-0.02, vmax=0.02, shading='auto'); fig.colorbar(im, ax=axes[2])
axes[2].contour(vy, vx, V.T, levels=lev, colors='0.4', linewidths=0.5)
axes[2].set(title=f'old reference (toy2d, h 0.004) - exact: channel RMS {rms(do[chan]):.4f}, max {np.abs(do[chan]).max():.4f}', xlabel='y', ylabel='x')
fig.tight_layout(); fig.savefig(OUT + 'committor_errors_old_reference.png', dpi=100); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.errorbar(np.arange(len(POINTS)), (qg - p) * 1e3, yerr=2e3 * se, fmt='o', capsize=3, label='grid - shooting (±2 s.e.)')
qo = np.clip(f_old(np.array(POINTS)), 0, 1)
ax.plot(np.arange(len(POINTS)), (qo - p) * 1e3, 'C1s', label='old reference - shooting')
ax.axhline(0, color='k', lw=0.8); ax.set_xticks(range(len(POINTS)))
ax.set_xticklabels([f'({a:.2f}, {b:+.2f})\nq={c:.3f}' for (a, b), c in zip(POINTS, qg)], fontsize=8)
ax.set(ylabel='×10⁻³', xlabel='(y, x)', title=f'check against shooting ({N_SHOOT // 10**6}M trajectories per point)'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + 'shooting_check.png', dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
hst, e = np.histogram(Q[chan], bins=100, range=(0, 1), weights=np.exp(-Vg[chan]))
with np.errstate(divide='ignore'):
    axes[0].plot((e[1:] + e[:-1]) / 2, -np.log(hst / hst.sum()), 'k')
axes[0].set(xlabel='exact q', ylabel='F(q) / kT (+ const)', title='equilibrium free energy along the exact committor (Boltzmann on the grid)')
for yc in (0.3, 0.45, 0.5, 0.55, 0.7):
    j = np.argmin(np.abs(gy - yc)); m = chan[j]
    axes[1].plot(gx[m], Q[j, m], label=f'y = {yc}')
axes[1].set(xlabel='x', ylabel='q', title='exact q across the channel (V < 10 kT) at fixed y: the committor depends on x')
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + 'fq_and_cuts.png', dpi=110); plt.close(fig)
say('ALL DONE')
