"""Figures for README.md (integrator comparison): EM and BBK from run_fits.py, ASE from infretis_langevin_md/."""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
WELL = os.path.dirname(HERE)
ASE = f'{WELL}/infretis_langevin_md/langevin_md_ase_long'
DATA = {'EM': (f'{WELL}/langevin_md/fits/euler_md_wel_lgv_3M.npy', f'{WELL}/exact/q_exact_EM_on_md_frames.npy'),
        'BBK': ('/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/md_istarwell2_100M/md_path.npy',
                f'{WELL}/exact/q_exact_BBK_on_md_frames.npy')}
N, LA, LB = 2_000_000, -0.35, 0.35
rows = json.load(open(f'{HERE}/results.json'))
last = lambda pattern: sorted([f for f in os.listdir(os.path.dirname(pattern)) if f.endswith(os.path.basename(pattern))],
                              key=lambda f: int(''.join(c for c in f.split('_')[0] if c.isdigit()) or 0))[-1]


def zq_panel(png):
    im = mpimg.imread(png)
    return im[:, int(im.shape[1] * 0.655):]


def done(folder):
    """finished run: q_traj.npy is written at the end (ASE folders: always finished)"""
    return not folder.startswith(HERE) or os.path.exists(f'{folder}/q_traj.npy')


def pending(a, text='running'):
    a.text(0.5, 0.5, text, ha='center', va='center', fontsize=14, color='0.4', transform=a.transAxes); a.axis('off')


def final_png(folder, kind):
    fs = [f for f in os.listdir(folder) if f.endswith(f'_{kind}.png') or (f.startswith(kind) and f.endswith('.png'))]
    num = lambda f: int(''.join(c for c in f if c.isdigit()) or 0)
    return os.path.join(folder, max(fs, key=num))


# 1. Z_q, x_v vs history [0, 1], for the three integrators (seed 1; ASE history_x: seed 0, the only one)
cases = [('EM', f'{HERE}/EM/x_v/seed1', f'{HERE}/EM/history_x/seed1'),
         ('BBK', f'{HERE}/BBK/x_v/seed1', f'{HERE}/BBK/history_x/seed1'),
         ('ASE', f'{ASE}/x_v/seed1', f'{ASE}/history_x/seed0')]
fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
for (name, f_xv, f_h), ax in zip(cases, axes):
    for a, f, lab in ((ax[0], f_xv, 'x, v (recorded)'), (ax[1], f_h, 'x + history [0, 1]')):
        a.set_title(f'{name}: {lab}', fontsize=11)
        if done(f): a.imshow(zq_panel(final_png(f, 'feps'))); a.axis('off')
        else: pending(a)
fig.suptitle('$Z_q$ cut profiles at several lags (flat at $N_{AB}$ for a committor)', fontsize=12)
fig.tight_layout(); fig.savefig(f'{HERE}/fig_zq.png', dpi=110); plt.close(fig)

# 2. q(x, v) maps and error maps, EM and BBK (seed 1), and q - exact along x for every seed
fig, axes = plt.subplots(2, 5, figsize=(24, 8))
prof = {}
for integ, ax in zip(('EM', 'BBK'), axes):
    d = np.load(DATA[integ][0], mmap_mode='r')[:N]
    lam, v = np.array(d[:, 0]), np.array(d[:, 1])
    qe = np.array(np.load(DATA[integ][1], mmap_mode='r')[:N], float)
    free = (lam > LA) & (lam < LB)
    el, ev = np.linspace(LA, LB, 71), np.linspace(-2.6, 2.6, 53)
    n = np.histogram2d(lam[free], v[free], [el, ev])[0]
    mean = lambda r: np.where(n > 20, np.histogram2d(lam[free], v[free], [el, ev], weights=r[free])[0] / np.maximum(n, 1), np.nan).T
    qx = {k: (np.load(f'{HERE}/{integ}/{k}/seed1/q_traj.npy').astype(float)
              if os.path.exists(f'{HERE}/{integ}/{k}/seed1/q_traj.npy') else None) for k in ('x_v', 'history_x')}
    mm = lambda r: None if r is None else mean(r)
    panels = [('exact', mean(qe), 'coolwarm', (0, 1)), ('x, v', mm(qx['x_v']), 'coolwarm', (0, 1)),
              ('x + history [0, 1]', mm(qx['history_x']), 'coolwarm', (0, 1)),
              ('x, v $-$ exact', None if qx['x_v'] is None else mean(qx['x_v'] - qe), 'RdBu_r', (-0.1, 0.1)),
              ('history $-$ exact', None if qx['history_x'] is None else mean(qx['history_x'] - qe), 'RdBu_r', (-0.1, 0.1))]
    for a, (t, m, cm, (lo, hi)) in zip(ax, panels):
        if m is None:
            a.set_title(f'{integ}: {t}'); pending(a); continue
        im = a.pcolormesh(el, ev, m, cmap=cm, vmin=lo, vmax=hi); fig.colorbar(im, ax=a)
        a.set(title=f'{integ}: {t}', xlabel='$x$', ylabel='$v$')
    xb = np.clip(np.digitize(lam, el) - 1, 0, 69)
    cnt = np.bincount(xb[free], minlength=70)
    pm = lambda r: np.bincount(xb[free], r[free], 70) / np.maximum(cnt, 1)
    prof[integ] = {}
    for variant in ('x_v', 'history_x', 'sub20_hist_fine'):
        for s in (1, 2, 3):
            p = f'{HERE}/{integ}/{variant}/seed{s}/q_traj.npy'
            if os.path.exists(p):
                prof[integ][f'{variant}, seed {s}'] = pm(np.load(p).astype(float) - qe)
fig.tight_layout(); fig.savefig(f'{HERE}/fig_maps.png', dpi=90); plt.close(fig)

mid = (np.linspace(LA, LB, 71)[1:] + np.linspace(LA, LB, 71)[:-1]) / 2
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for a, integ in zip(axes, ('EM', 'BBK')):
    for lab, p in prof[integ].items():
        a.plot(mid, p, label=lab, ls='--' if 'sub20' in lab else ('-' if 'x_v' in lab else ':'))
    a.axhline(0, color='k', lw=0.8); a.set(title=integ, xlabel='$x$', ylabel='mean of fit $-$ exact per $x$ bin'); a.grid(alpha=0.3)
[a.legend(fontsize=8) for a in axes if a.lines]
fig.tight_layout(); fig.savefig(f'{HERE}/fig_profiles.png', dpi=110); plt.close(fig)

# 3. convergence: RMS per frame vs iteration
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for a, integ in zip(axes, ('EM', 'BBK')):
    for r in rows:
        if r['integrator'] != integ: continue
        pr = json.load(open(f"{HERE}/{integ}/{r['variant']}/seed{r['seed']}/metrics.json"))['progress']
        a.plot([p['iter'] for p in pr], [p['rms_frame'] for p in pr], 'o-',
               ls='-' if r['variant'] == 'x_v' else (':' if r['variant'] == 'history_x' else '--'),
               label=f"{r['variant']}, seed {r['seed']}")
    a.set(title=integ, xlabel='iteration', ylabel='RMS(fit $-$ exact) per frame', yscale='log'); a.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{HERE}/fig_convergence.png', dpi=110); plt.close(fig)

# 4. stride: Z_q of the lag-20 fits (EM, BBK) and the ASE stride runs' q(x, v) maps
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for a, integ in zip(axes, ('EM', 'BBK')):
    f = f'{HERE}/{integ}/sub20_hist_fine/seed1'
    if done(f): a.imshow(zq_panel(final_png(f, 'feps'))); a.axis('off')
    else: pending(a)
    a.set_title(f'{integ}: x + history [0, 1], fit lag/stride 20', fontsize=10)
fig.tight_layout(); fig.savefig(f'{HERE}/fig_stride_zq.png', dpi=110); plt.close(fig)
runs = ['sub2_x_v/seed111', 'sub2_hist_fine/seed111', 'sub2_hist_coarse/seed111', 'sub20_hist_fine/seed111', 'sub50_hist_fine/seed111']
fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4))
for a, r in zip(axes, runs):
    a.imshow(mpimg.imread(f'{ASE}/{r}/q_x_v.png')); a.axis('off')
fig.tight_layout(); fig.savefig(f'{HERE}/fig_stride_ase_maps.png', dpi=110); plt.close(fig)
print('figures written')

# 5. q(x, v) and fit - exact for every fit and seed (EM, BBK); ASE: the notebook's q(x, v) images
FITS_ALL = [('x_v', 'x, v'), ('history_x', 'x + history [0, 1]'), ('sub20_hist_fine', 'x + history [0, 1], lag/stride 20')]
for integ in ('EM', 'BBK'):
    d = np.load(DATA[integ][0], mmap_mode='r')[:N]
    lam, v = np.array(d[:, 0]), np.array(d[:, 1])
    qe = np.array(np.load(DATA[integ][1], mmap_mode='r')[:N], float)
    free = (lam > LA) & (lam < LB)
    el, ev = np.linspace(LA, LB, 71), np.linspace(-2.6, 2.6, 53)
    n = np.histogram2d(lam[free], v[free], [el, ev])[0]
    mean = lambda r: np.where(n > 20, np.histogram2d(lam[free], v[free], [el, ev], weights=r[free])[0] / np.maximum(n, 1), np.nan).T
    fig, axes = plt.subplots(len(FITS_ALL), 6, figsize=(30, 4.6 * len(FITS_ALL)))
    for (variant, label), ax in zip(FITS_ALL, axes):
        for k, s in enumerate((1, 2, 3)):
            aq, ae = ax[2 * k], ax[2 * k + 1]
            f = f'{HERE}/{integ}/{variant}/seed{s}'
            title = f'{integ}, {label}, seed {s}'
            if not os.path.exists(f'{f}/q_traj.npy'):
                aq.set_title(title, fontsize=10); ae.set_title('fit $-$ exact', fontsize=10)
                txt = 'not run' if (variant == 'sub20_hist_fine' and s != 1) else 'running'
                pending(aq, txt); pending(ae, txt); continue
            r = np.load(f'{f}/q_traj.npy').astype(float)
            rms = np.sqrt(np.mean((r[free] - qe[free]) ** 2))
            im = aq.pcolormesh(el, ev, mean(r), cmap='coolwarm', vmin=0, vmax=1); fig.colorbar(im, ax=aq)
            aq.set(title=title, xlabel='$x$', ylabel='$v$')
            im = ae.pcolormesh(el, ev, mean(r - qe), cmap='RdBu_r', vmin=-0.1, vmax=0.1); fig.colorbar(im, ax=ae)
            ae.set(title=f'fit $-$ exact (RMS/frame {rms:.4f})', xlabel='$x$')
    fig.tight_layout(); fig.savefig(f'{HERE}/fig_maps_all_{integ}.png', dpi=70); plt.close(fig)

ase_cases = [('x, v', [f'x_v/seed{s}/it11000_q_x_v.png' for s in (1, 2, 3)]),
             ('x + history [0, 1]', ['history_x/seed0/it11000_q_x_v.png']),
             ('x + history [0, 1, 2, 4]', [f'history_x5/seed{s}/it11000_q_x_v.png' for s in (1, 2, 3)]),
             ('stride runs (seed 111)', ['sub2_x_v/seed111/q_x_v.png', 'sub2_hist_fine/seed111/q_x_v.png',
                                         'sub2_hist_coarse/seed111/q_x_v.png', 'sub20_hist_fine/seed111/q_x_v.png',
                                         'sub50_hist_fine/seed111/q_x_v.png'])]
ncol = max(len(c[1]) for c in ase_cases)
fig, axes = plt.subplots(len(ase_cases), ncol, figsize=(4.3 * ncol, 3.6 * len(ase_cases)))
for (label, files), ax in zip(ase_cases, axes):
    for a in ax: a.axis('off')
    for a, f in zip(ax, files):
        a.imshow(mpimg.imread(f'{ASE}/{f}')); a.set_title(f'ASE {label}: {f.split("/")[0]}/{f.split("/")[1]}', fontsize=8)
ax_ref = axes[0][-1]
ax_ref.imshow(mpimg.imread(f'{ASE}/q_x_v.png')[:, -int(mpimg.imread(f'{ASE}/q_x_v.png').shape[1] / 7):]); ax_ref.set_title('ASE reference', fontsize=8)
fig.tight_layout(); fig.savefig(f'{HERE}/fig_maps_all_ASE.png', dpi=80); plt.close(fig)
print('all-case maps written')
