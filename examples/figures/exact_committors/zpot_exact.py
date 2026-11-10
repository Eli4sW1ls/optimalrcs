"""Exact committor q(y, x) of the 2D Brownian Z potential (toytis sjoelbak_angle, Lx 0.3, 45 deg; gamma 10,
kT 0.05, dt 0.002), for the two definitions used in the tests:
  * staple   : A: y <= 0.05, B: y >= 0.85, checked every step (sim_istarz_br_g10)
  * MD       : A: y <= 0.10, B: y >= 0.85, checked every 20 steps (md_zbr_g10_s*, one row per 20 steps)
1. Grid solve (exact_tools.solve_brownian_2d) at several spacings; error estimate = change between the finest two.
2. GPU shooting with N_SHOOT trajectories at POINTS (binomial standard errors).
3. Comparison with the old PDE references (continuous generator, boundary shift) and with the fits:
   staple NPNEq (istarz_br_g10/committor.npz), MD NPq (z_brownian/npq_md/npq_md.npz) and the MD outcomes.
4. Figures in zpot/.
"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exact_tools as E
OUT = f'{HERE}/zpot/'; os.makedirs(OUT, exist_ok=True)
FIG = os.path.dirname(HERE)
KT, GAMMA, DT = 0.05, 10.0, 0.002
DEFS = {'staple': dict(A=0.05, B=0.85, stride=1, levels=(0.004, 0.002, 0.001),
                       old=FIG + '/z_brownian/committor_reference_h0.001.npz')}
POINTS = [(0.30, 0.15), (0.40, 0.05), (0.45, 0.15), (0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]
N_SHOOT = 1_000_000
log = open(OUT + 'zpot_exact.log', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
rms = lambda a: np.sqrt(np.nanmean(np.asarray(a) ** 2))
ev = lambda sol, y, x: E.eval_q_brownian_2d(E.zpot_force, sol, np.asarray(y, float), np.asarray(x, float))

res = {}
for name, d in DEFS.items():
    say(f'===== {name}: A {d["A"]}, B {d["B"]}, absorption every {d["stride"]} steps =====')
    sols = {}
    for h in d['levels']:
        t0 = time.time()
        cache = OUT + f'q_exact_{name}_h{h}.npz'
        if os.path.exists(cache):                          # solved before: reuse
            c_ = np.load(cache)
            sols[h] = {k: (c_[k] if c_[k].ndim else c_[k].item()) for k in ('ys', 'xs', 'w', 'A', 'B', 'mu', 'stride', 'h')}
            say(f'  h {h}: loaded {cache}; q at POINTS ' + ' '.join(f'{v:.4f}' for v in ev(sols[h], *zip(*POINTS))))
            continue
        sols[h], r = E.solve_brownian_2d(E.zpot_force, DT, GAMMA, KT, d['A'], d['B'], h, stride=d['stride'], verbose=False,
                                         y_jumps=E.ZPOT_Y_JUMPS, ridge_jumps=E.ZPOT_RIDGE_JUMPS)
        say(f'  h {h}: grid {sols[h]["w"].shape[0]} x {sols[h]["w"].shape[1]}, residual {r:.1e}, {time.time() - t0:.0f} s; '
            f'q at POINTS ' + ' '.join(f'{v:.4f}' for v in ev(sols[h], *zip(*POINTS))))
        np.savez(OUT + f'q_exact_{name}_h{h}.npz', **sols[h], kT=KT, gamma=GAMMA, dt=DT,
                 note='q(s) = w(s + mu F(s)); evaluate with exact_tools.eval_q_brownian_2d(exact_tools.zpot_force, ...)')
    # changes on a fine evaluation grid inside the box (0 <= x <= 0.3)
    gy, gx = np.linspace(d['A'], d['B'], 801), np.linspace(0.0, 0.3, 301)
    GY, GX = np.meshgrid(gy, gx, indexing='ij')
    hs = d['levels']
    vals = {h: ev(sols[h], GY.ravel(), GX.ravel()).reshape(GY.shape) for h in hs}
    for h1, h2 in zip(hs[:-1], hs[1:]):
        dd = vals[h2] - vals[h1]
        say(f'  change h {h1} -> {h2}: max {np.abs(dd).max():.1e}, RMS {rms(dd):.1e}, 99th percentile {np.percentile(np.abs(dd), 99):.1e}')
    err = np.abs(vals[hs[-1]] - vals[hs[-2]])
    res[name] = dict(sol=sols[hs[-1]], err=err, gy=gy, gx=gx, Q=vals[hs[-1]], d=d)
    # shooting
    t0 = time.time()
    p, se = E.shoot_brownian_2d(E.zpot_force, DT, GAMMA, KT, d['A'], d['B'], POINTS, N_SHOOT, stride=d['stride'], seed=5)
    qg = ev(sols[hs[-1]], *zip(*POINTS))
    say(f'  shooting, {N_SHOOT} trajectories per point ({time.time() - t0:.0f} s):')
    for pt, pp, ss, qq in zip(POINTS, p, se, qg):
        say(f'    (y, x) = ({pt[0]:.2f}, {pt[1]:.2f}): shooting {pp:.5f} +- {ss:.5f} | grid {qq:.5f} | diff {qq - pp:+.5f} ({(qq - pp) / ss:+.1f} s.e.)')
    say(f'  chi^2 = {np.sum(((qg - p) / se) ** 2):.1f} for {len(POINTS)} points')
    res[name].update(p=p, se=se, qg=qg)
    # old reference
    old = np.load(d['old'])
    f_old = RegularGridInterpolator((old['y'], old['x']), old['q'], bounds_error=False, fill_value=None)
    Qold = np.clip(f_old(np.c_[GY.ravel(), GX.ravel()]), 0, 1).reshape(GY.shape)
    res[name]['Qold'] = Qold
    say(f'  old PDE reference - exact: max {np.abs(Qold - res[name]["Q"]).max():.4f}, RMS {rms(Qold - res[name]["Q"]):.4f}; '
        f'at POINTS ' + ' '.join(f'{v:+.4f}' for v in np.clip(f_old(np.array(POINTS)), 0, 1) - qg))

# ---------------------------------------------------------------- the fits against the exact committor
ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
def bin_mean(y, x, val, m, min_n=20):
    n = np.histogram2d(y[m], x[m], bins=(ey, ex))[0]; s = np.histogram2d(y[m], x[m], bins=(ey, ex), weights=val[m])[0]
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan)
fits = {}
d = np.load(FIG + '/istarz_br_g10/committor.npz', mmap_mode='r')
sub = slice(None, None, 4)                                     # every 4th frame (memory)
ys_, xs_ = np.array(d['X_all'][sub, 0]), np.array(d['X_all'][sub, 1]); r_ = np.array(d['r'][sub])
free_ = ~(np.array(d['boundary0'][sub]) | np.array(d['boundary1'][sub]))
qe_ = ev(res['staple']['sol'], ys_, xs_)
fits['staple NPNEq (10000 it.)'] = (bin_mean(ys_, xs_, r_, free_), bin_mean(ys_, xs_, qe_, free_), ys_, xs_, r_, qe_, free_)
SHOOT_PTS = [(0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]
for lab, (F, X, y_, x_, r__, qe__, m_) in fits.items():
    tr = (X > 0.05) & (X < 0.95)
    near = [m_ & ((y_ - a) ** 2 + (x_ - b) ** 2 < 0.01 ** 2) for a, b in SHOOT_PTS]
    say(f'{lab}: per bin RMS(fit - exact) {rms((F - X)[tr]):.4f}, mean {np.nanmean((F - X)[tr]):+.4f}; '
        f'fit/exact at plateau points ' + ' '.join(f'{r__[n_].mean():.3f}/{qe__[n_].mean():.3f}' for n_ in near))

# ---------------------------------------------------------------- figures
import importlib
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis')
from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
pot = RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)
vy, vx = np.linspace(-0.01, 0.91, 369), np.linspace(-0.01, 0.31, 129)
Veff = np.array([[pot.effective_potential((np.array([a, b]), np.zeros(2))) for b in vx] for a in vy]) / KT
Veff -= Veff.min()
lev = np.linspace(np.percentile(Veff, 1), np.percentile(Veff, 92), 14)
dyv = vy[1] - vy[0]
Fy = -np.log(np.exp(-Veff).sum(1) * (vx[1] - vx[0]))
fig, axes = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={'width_ratios': [1.6, 1]})
im = axes[0].pcolormesh(vy, vx, Veff.T, cmap='viridis', vmax=12, shading='auto'); fig.colorbar(im, ax=axes[0], label='V_eff / kT')
for b in (0.05, 0.10, 0.85): axes[0].axvline(b, color='w', ls='--', lw=0.8)
axes[0].set(xlabel='order parameter y', ylabel='x', title='effective potential (incl. the constant force -0.2 along y); dashed: A (0.05 staple, 0.10 MD), B')
axes[1].plot(vy, Fy - Fy.min(), 'k'); axes[1].set(xlabel='y', ylabel='F(y) / kT', title='equilibrium free energy along y', ylim=(0, 14))
for b in (0.05, 0.10, 0.85): axes[1].axvline(b, color='C3', ls='--', lw=0.8)
fig.tight_layout(); fig.savefig(OUT + 'landscape.png', dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(21, 4.5))
for row, name in zip([axes], ('staple',)):
    r = res[name]
    im = row[0].pcolormesh(r['gy'], r['gx'], r['Q'].T, cmap='coolwarm', vmin=0, vmax=1, shading='auto'); fig.colorbar(im, ax=row[0], label='q')
    row[0].contour(vy, vx, Veff.T, levels=lev, colors='0.4', linewidths=0.5)
    row[0].contour(r['gy'], r['gx'], r['Q'].T, levels=[0.1, 0.5, 0.9], colors='k', linewidths=1, linestyles='--')
    row[0].set(title=f'exact committor, {name} (A: y <= {r["d"]["A"]}, every {r["d"]["stride"]} step(s))', xlabel='y', ylabel='x')
    im = row[1].pcolormesh(r['gy'], r['gx'], np.log10(np.maximum(r['err'], 1e-9)).T, cmap='magma', vmin=-7, vmax=-2, shading='auto')
    fig.colorbar(im, ax=row[1], label='log10 error estimate')
    row[1].set(title=f'error estimate (finest - previous grid): max {r["err"].max():.0e}, RMS {rms(r["err"]):.0e}', xlabel='y', ylabel='x')
    im = row[2].pcolormesh(r['gy'], r['gx'], (r['Qold'] - r['Q']).T, cmap='PuOr', vmin=-0.05, vmax=0.05, shading='auto'); fig.colorbar(im, ax=row[2])
    row[2].contour(vy, vx, Veff.T, levels=lev, colors='0.4', linewidths=0.5)
    row[2].set(title=f'old PDE reference - exact (RMS {rms(r["Qold"] - r["Q"]):.4f})', xlabel='y', ylabel='x')
fig.tight_layout(); fig.savefig(OUT + 'committor_errors_old_reference.png', dpi=100); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
for k, name in enumerate(('staple',)):
    r = res[name]
    ax.errorbar(np.arange(len(POINTS)) + 0.15 * k, (r['qg'] - r['p']) * 1e3, yerr=2e3 * r['se'], fmt='os'[k], capsize=3, label=f'{name}: grid - shooting (±2 s.e.)')
ax.axhline(0, color='k', lw=0.8); ax.set_xticks(range(len(POINTS))); ax.set_xticklabels([f'({p[0]:.2f}, {p[1]:.2f})' for p in POINTS], fontsize=8)
ax.set(ylabel='×10⁻³', xlabel='(y, x)', title=f'check against shooting ({N_SHOOT // 10**6}M trajectories per point)'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + 'shooting_check.png', dpi=110); plt.close(fig)

fig, axes = plt.subplots(len(fits), 2, figsize=(16, 4 * len(fits)), squeeze=False)
for row, (lab, (F, X, *_)) in zip(axes, fits.items()):
    im = row[0].imshow(F.T, origin='lower', extent=[ey[0], ey[-1], ex[0], ex[-1]], aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    row[0].contour(vy, vx, Veff.T, levels=lev, colors='0.4', linewidths=0.5); fig.colorbar(im, ax=row[0])
    row[0].set(title=lab, xlabel='y', ylabel='x', xlim=(0.05, 0.85), ylim=(-0.01, 0.31))
    tr = (X > 0.05) & (X < 0.95)
    im = row[1].imshow((F - X).T, origin='lower', extent=[ey[0], ey[-1], ex[0], ex[-1]], aspect='auto', cmap='PuOr', vmin=-0.15, vmax=0.15)
    row[1].contour(vy, vx, Veff.T, levels=lev, colors='0.4', linewidths=0.5); fig.colorbar(im, ax=row[1])
    row[1].set(title=f'{lab} - exact (RMS {rms((F - X)[tr]):.3f}, mean {np.nanmean((F - X)[tr]):+.3f})', xlabel='y', ylabel='x', xlim=(0.05, 0.85), ylim=(-0.01, 0.31))
fig.tight_layout(); fig.savefig(OUT + 'fits_vs_exact.png', dpi=100); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
for name in ('staple',):
    r = res[name]
    wB = np.exp(-RegularGridInterpolator((vy, vx), Veff, bounds_error=False, fill_value=50)(np.c_[np.meshgrid(r['gy'], r['gx'], indexing='ij')[0].ravel(), np.meshgrid(r['gy'], r['gx'], indexing='ij')[1].ravel()]))
    hst, e = np.histogram(r['Q'].ravel(), bins=100, range=(0, 1), weights=wB)
    with np.errstate(divide='ignore'):
        ax.plot((e[1:] + e[:-1]) / 2, -np.log(hst / hst.sum()), label=name)
ax.set(xlabel='exact q', ylabel='F(q) / kT (+ const)', title='equilibrium free energy along the exact committor (Boltzmann on the grid)')
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT + 'fq_exact.png', dpi=110); plt.close(fig)
say('ALL DONE')
