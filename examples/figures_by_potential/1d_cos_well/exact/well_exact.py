"""Exact committor q(x, v) of the 1D underdamped cosine well (toytis CosWellWalls, gamma 1, kT 1, m 1, dt 0.005,
A: x <= -0.35, B: x >= 0.35, checked every step) for the toytis BBK integrator and for Euler-Maruyama.

1. Grid solve (exact_tools.solve_underdamped_1d) at four resolutions, two quadrature orders and two v ranges;
   error estimate = change between the two finest grids, on MD frames.
2. Independent check: GPU shooting, N_SHOOT trajectories at each of POINTS (binomial standard errors).
3. Comparison with the old reference (figures_by_potential/1d_cos_well/langevin_staple/checks/reference.npz, 600 trajectories per grid point) and with
   the model-free MD-outcome committor of the BBK and EM MDs.
4. Figures in well/.
"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.dirname(os.path.dirname(HERE))                    # examples/figures_by_potential
sys.path.insert(0, f'{FIGROOT}/_shared/exact_committors')           # exact_tools
import exact_tools as E
OUT = f'{HERE}/'; os.makedirs(OUT, exist_ok=True)
FIG = FIGROOT
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/'
A, B, DT, GAMMA = -0.35, 0.35, 0.005, 1.0
POINTS = [(-0.349, -0.2), (-0.30, 0.0), (-0.28, 0.6), (-0.20, -0.5), (-0.20, 0.3), (-0.10, 0.8), (0.0, 0.0),
          (0.18, 0.7), (0.30, -0.3), (0.345, 0.3)]
N_SHOOT = 2_000_000
LEVELS = [(351, 251), (701, 501), (1401, 1001), (2801, 1001)]
log = open(OUT + 'well_exact.log', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

md = {'BBK': np.load(SIM + 'md_istarwell2_100M/md_path.npy', mmap_mode='r')[:6_000_000],
      'EM': np.load(FIG + '/1d_cos_well/langevin_md/fits/euler_md_wel_lgv_3M.npy', mmap_mode='r')[:3_000_000]}
frames = {k: (np.array(a[:, 0]), np.array(a[:, 1])) for k, a in md.items()}
schemes = {'BBK': (E.bbk_step, E.bbk_coefficients(DT, GAMMA)), 'EM': (E.em_step, E.em_coefficients(DT, GAMMA))}
rms = lambda a: np.sqrt(np.nanmean(np.asarray(a) ** 2))
results = {}
for name, (step, coef) in schemes.items():
    say(f'===== {name} =====')
    qev = lambda sol, x, v: E.eval_q_underdamped(step, coef, E.coswell_force, sol, x, v)
    x, v = frames[name]
    free = (x > A) & (x < B)
    xe, ve = x[free][::10], v[free][::10]
    near = np.minimum(xe - A, B - xe) < 0.02
    sols, vals = {}, {}
    for lv in LEVELS:
        t0 = time.time()
        sols[lv], res = E.solve_underdamped_1d(step, coef, E.coswell_force, A, B, *lv, verbose=False)
        vals[lv] = qev(sols[lv], xe, ve)
        say(f'  grid {lv[0]:5d} x {lv[1]:5d}: residual {res:.1e}, {time.time() - t0:.0f} s')
    extra = {'quadrature 12x7 -> 16x9': dict(n1=16, n2=9), 'quadrature 12x7 -> 8x5': dict(n1=8, n2=5),
             'vmax 5 -> 6': dict(vmax=6.0)}
    say('  changes on MD frames: max over all | max for frames > 0.02 from A/B | RMS | 99.9th percentile')
    for l1, l2 in zip(LEVELS[:-1], LEVELS[1:]):
        d = vals[l2] - vals[l1]
        say(f'    grid {l1} -> {l2}: {np.abs(d).max():.1e} | {np.abs(d[~near]).max():.1e} | {rms(d):.1e} | {np.percentile(np.abs(d), 99.9):.1e}')
    for lab, kw in extra.items():
        sol_x = E.solve_underdamped_1d(step, coef, E.coswell_force, A, B, *LEVELS[1], verbose=False, **kw)[0]
        d = qev(sol_x, xe, ve) - vals[LEVELS[1]]
        say(f'    {lab} at {LEVELS[1]}: {np.abs(d).max():.1e} | {np.abs(d[~near]).max():.1e} | {rms(d):.1e} | {np.percentile(np.abs(d), 99.9):.1e}')
    best = sols[LEVELS[-1]]
    err = np.abs(vals[LEVELS[-1]] - vals[LEVELS[-2]])           # error estimate of the finest grid
    say(f'  error estimate of the finest grid (change from the previous level): max {err.max():.1e}, '
        f'max > 0.02 from A/B {err[~near].max():.1e}, RMS {rms(err):.1e}')
    sym = qev(best, xe, ve) + qev(best, -xe, -ve) - 1
    say(f'  symmetry |q(x,v) + q(-x,-v) - 1| on MD frames: max {np.abs(sym).max():.1e}')
    np.savez(OUT + f'q_exact_{name}.npz', **{k: v_ for k, v_ in best.items()}, dt=DT, gamma=GAMMA,
             m_previous=sols[LEVELS[-2]]['m'], xs_previous=sols[LEVELS[-2]]['xs'], vs_previous=sols[LEVELS[-2]]['vs'],
             note=f'{name}: q = pB + (1 - pA - pB) m; evaluate with exact_tools.eval_q_underdamped')
    t0 = time.time()
    p, se, nsteps = E.shoot_underdamped_1d(step, coef, E.coswell_force, A, B, POINTS, N_SHOOT, seed=11)
    qg = qev(best, [pt[0] for pt in POINTS], [pt[1] for pt in POINTS])
    say(f'  shooting, {N_SHOOT} trajectories per point ({time.time() - t0:.0f} s, mean {nsteps:.0f} steps):')
    for pt, pp, ss, qq in zip(POINTS, p, se, qg):
        say(f'    (x, v) = ({pt[0]:+.3f}, {pt[1]:+.2f}): shooting {pp:.5f} +- {ss:.5f} | grid {qq:.5f} | diff {qq - pp:+.5f} ({(qq - pp) / ss:+.1f} s.e.)')
    chi2 = np.sum(((qg - p) / se) ** 2)
    say(f'  chi^2 = {chi2:.1f} for {len(POINTS)} points')
    results[name] = dict(sol=best, err=err, xe=xe, ve=ve, p=p, se=se, qg=qg, qev=qev)

# EM vs BBK exact on the BBK MD frames
xb, vb = frames['BBK']; fb = (xb > A) & (xb < B); sub = np.flatnonzero(fb)[::10]
d_em = results['EM']['qev'](results['EM']['sol'], xb[sub], vb[sub]) - results['BBK']['qev'](results['BBK']['sol'], xb[sub], vb[sub])
say(f'EM exact - BBK exact on BBK MD frames: max {np.abs(d_em).max():.4f}, RMS {rms(d_em):.4f}, mean {d_em.mean():+.5f}')

# old reference and MD outcome vs new exact
ref = np.load(FIG + '/1d_cos_well/langevin_staple/checks/reference.npz')
f_old = RegularGridInterpolator((ref['xs'], ref['vs']), ref['q'], bounds_error=False, fill_value=None)
el, evb = np.linspace(A, B, 36), np.linspace(-3, 3, 31)
def bmap(x, v, val, m, min_n=200):
    n = np.histogram2d(x[m], v[m], [el, evb])[0]; s = np.histogram2d(x[m], v[m], [el, evb], weights=val[m])[0]
    return np.where(n > min_n, s / np.maximum(n, 1), np.nan)
maps = {}
for name in ('BBK', 'EM'):
    x, v = frames[name]; n = len(x)
    Am, Bm = x <= A, x >= B
    idx = np.where(Am | Bm, np.arange(n), n); nxt = np.minimum.accumulate(idx[::-1])[::-1]
    out = np.where(nxt < n, Bm[np.minimum(nxt, n - 1)].astype(float), np.nan)
    ok = ~Am & ~Bm & np.isfinite(out)
    qn = np.full(n, np.nan); qn[ok] = results[name]['qev'](results[name]['sol'], x[ok], v[ok])
    qo = np.full(n, np.nan)
    qo[ok] = np.clip(f_old(np.c_[np.clip(x[ok], ref['xs'][0], ref['xs'][-1]), np.clip(v[ok], ref['vs'][0], ref['vs'][-1])]), 0, 1)
    h = np.arange(n) < n // 2
    Mo, Qn, Qo = bmap(x, v, out, ok), bmap(x, v, qn, ok), bmap(x, v, qo, ok)
    noise = rms(bmap(x, v, out, ok & h, 100) - bmap(x, v, out, ok & ~h, 100)) / 2
    say(f'{name} MD ({n} rows): per bin RMS(MD outcome - new exact) {rms(Mo - Qn):.4f} (noise {noise:.4f}), '
        f'RMS(MD outcome - old reference) {rms(Mo - Qo):.4f}, RMS(old - new) {rms(Qo - Qn):.4f}; '
        f'per frame RMS(old - new) {rms(qo - qn):.4f}, max {np.nanmax(np.abs(qo - qn)):.3f}')
    np.save(OUT + f'q_exact_{name}_on_md_frames.npy', qn.astype(np.float32))
    maps[name] = dict(Mo=Mo, Qn=Qn, Qo=Qo, noise=noise, qn=qn, ok=ok)

# ---------------------------------------------------------------- figures
sol = results['BBK']['sol']
xg = np.linspace(-0.6, 0.6, 1201)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(xg, E.coswell_potential(xg), 'k'); axes[0].set(xlabel='x', ylabel='V(x) / kT', title='potential (CosWellWalls)', ylim=(-1.2, 1.5))
for b in (A, B): axes[0].axvline(b, color='C3', ls='--')
X, V = np.meshgrid(np.linspace(-0.55, 0.55, 300), np.linspace(-4, 4, 300), indexing='ij')
Fxv = E.coswell_potential(X) + V ** 2 / 2
im = axes[1].pcolormesh(X, V, Fxv - Fxv.min(), cmap='viridis', vmax=8); fig.colorbar(im, ax=axes[1], label='F(x, v) / kT = V + v²/2')
axes[1].contour(X, V, Fxv, levels=[0.15, 0.5, 1, 2, 3], colors='w', linewidths=0.6)
for b in (A, B): axes[1].axvline(b, color='C3', ls='--')
axes[1].set(xlabel='x', ylabel='v', title='equilibrium free energy (white: E = 0.15 = bump tops, 0.5, 1, 2, 3)')
fig.tight_layout(); fig.savefig(OUT + 'landscape.png', dpi=110); plt.close(fig)

gx, gv = np.linspace(A + 1e-6, B - 1e-6, 701), np.linspace(-4, 4, 401)
GX, GV = np.meshgrid(gx, gv, indexing='ij')
Qg = results['BBK']['qev'](sol, GX.ravel(), GV.ravel()).reshape(GX.shape)
fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
im = axes[0].pcolormesh(gx, gv, Qg.T, cmap='coolwarm', vmin=0, vmax=1, shading='auto'); fig.colorbar(im, ax=axes[0], label='q')
axes[0].contour(gx, gv, Qg.T, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='k', linewidths=0.7)
axes[0].set(xlabel='x', ylabel='v', title='exact committor, BBK integrator')
r = results['BBK']
sc = axes[1].scatter(r['xe'][::10], r['ve'][::10], c=np.log10(np.maximum(r['err'][::10], 1e-8)), s=2, cmap='magma', vmin=-7, vmax=-2)
fig.colorbar(sc, ax=axes[1], label='log10 error estimate')
axes[1].set(xlabel='x', ylabel='v', ylim=(-4, 4),
            title=f'error estimate on MD frames (finest - previous grid): max {r["err"].max():.0e}, RMS {rms(r["err"]):.0e}')
for k, (name, mk) in enumerate((('BBK', 'o'), ('EM', 's'))):
    rr = results[name]
    axes[2].errorbar(np.arange(len(POINTS)) + 0.15 * k, (rr['qg'] - rr['p']) * 1e4, yerr=2e4 * rr['se'], fmt=mk, capsize=3,
                     label=f'{name}: grid - shooting (±2 s.e.)')
axes[2].axhline(0, color='k', lw=0.8)
axes[2].set_xticks(range(len(POINTS))); axes[2].set_xticklabels([f'({p[0]:+.3f},{p[1]:+.1f})' for p in POINTS], rotation=40, fontsize=8)
axes[2].set(ylabel='grid - shooting (×10⁻⁴)', title=f'check against shooting ({N_SHOOT // 10**6}M trajectories per point)')
axes[2].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(OUT + 'committor_and_errors.png', dpi=110); plt.close(fig)

m = maps['BBK']
fig, axes = plt.subplots(2, 3, figsize=(19, 9))
pan = [(m['Qn'], 'new exact (BBK), mean per bin', 'coolwarm', 1), (m['Qo'], 'old reference (600 traj./point, 29x29 grid)', 'coolwarm', 1),
       (m['Mo'], 'MD outcome (no fit), BBK MD 6M rows', 'coolwarm', 1),
       (m['Qo'] - m['Qn'], f'old reference - new exact (RMS {rms(m["Qo"] - m["Qn"]):.3f})', 'PuOr', 0.05),
       (m['Mo'] - m['Qn'], f'MD outcome - new exact (RMS {rms(m["Mo"] - m["Qn"]):.3f}; noise {m["noise"]:.3f})', 'PuOr', 0.05),
       (m['Mo'] - m['Qo'], f'MD outcome - old reference (RMS {rms(m["Mo"] - m["Qo"]):.3f})', 'PuOr', 0.05)]
for ax, (M_, t, cm, lim) in zip(axes.ravel(), pan):
    im = ax.pcolormesh(el, evb, M_.T, cmap=cm, vmin=0 if lim == 1 else -lim, vmax=lim); ax.set(title=t, xlabel='x', ylabel='v'); fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(OUT + 'old_reference_and_md_vs_exact.png', dpi=100); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for name in ('BBK', 'EM'):
    qn = maps[name]['qn'][maps[name]['ok']]
    hst, e = np.histogram(qn, bins=100, range=(0, 1)); c_ = (e[1:] + e[:-1]) / 2
    with np.errstate(divide='ignore'):
        axes[0].plot(c_, -np.log(hst / hst.sum()), label=f'{name} MD frames')
axes[0].set(xlabel='exact q', ylabel='F(q) / kT (+ const)', title='free energy along the exact committor (MD frames between A and B)')
axes[0].legend(frameon=False)
sc = axes[1].scatter(xb[sub][::3], vb[sub][::3], c=d_em[::3], s=2, cmap='PuOr', vmin=-0.02, vmax=0.02)
fig.colorbar(sc, ax=axes[1], label='q_EM - q_BBK')
axes[1].set(xlabel='x', ylabel='v', ylim=(-4, 4), title=f'exact EM - exact BBK (RMS {rms(d_em):.4f}, max {np.abs(d_em).max():.3f})')
fig.tight_layout(); fig.savefig(OUT + 'fq_and_em_vs_bbk.png', dpi=110); plt.close(fig)
say('ALL DONE')
