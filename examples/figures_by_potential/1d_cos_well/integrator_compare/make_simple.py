"""Figures for SIMPLE.md: Z_q overview and, per integrator, the best seed of every setting (F(q)/Z_q plot,
q(x, v) map, q - exact map). ASE maps need langevin_md_ase_long/traj.npy (x, v per frame) next to the copied q's."""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
WELL = os.path.dirname(HERE)
ASE = f'{WELL}/infretis_langevin_md/langevin_md_ase_long'
ASE_TRAJ = f'{WELL}/langevin_md_ase_long/traj.npy'
N, LA, LB = 2_000_000, -0.35, 0.35
rows = json.load(open(f'{HERE}/results.json'))


def best(integ, variant):
    rr = [r for r in rows if r['integrator'] == integ and r['variant'] == variant]
    return min(rr, key=lambda r: r['rms_frame'])['seed']


# (label, feps png, q file) per integrator and setting; EM/BBK: best seed by RMS per frame
CASES = {}
for integ in ('EM', 'BBK'):
    CASES[integ] = []
    for variant, label in (('x_v', 'x, v'), ('history_x', 'x + history [0, 1]'), ('history_x5', 'x + history [0, 1, 2, 4]')):
        if not any(r['integrator'] == integ and r['variant'] == variant for r in rows):
            continue
        s = best(integ, variant)
        f = f'{HERE}/{integ}/{variant}/seed{s}'
        CASES[integ].append((f'{label} (seed {s})', f'{f}/it11000_feps.png', f'{f}/q_traj.npy'))
CASES['ASE'] = [('x, v (seed 1)', f'{ASE}/x_v/seed1/it11000_feps.png', f'{ASE}/x_v/seed1/q_trajxv.npy'),
                ('x + history [0, 1] (seed 0)', f'{ASE}/history_x/seed0/it11000_feps.png',
                 f'{ASE}/history_x/seed0/q_trajhist.npy'),
                ('x + history [0, 1, 2, 4] (seed 1)', f'{ASE}/history_x5/seed1/it11000_feps.png',
                 f'{ASE}/history_x5/q_trajhist5.npy')]


def load_md(integ):
    if integ == 'EM':
        d = np.load(f'{WELL}/langevin_md/fits/euler_md_wel_lgv_3M.npy', mmap_mode='r')[:N]
        qe = np.load(f'{WELL}/exact/q_exact_EM_on_md_frames.npy', mmap_mode='r')[:N]
    elif integ == 'BBK':
        d = np.load('/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/md_istarwell2_100M/md_path.npy',
                    mmap_mode='r')[:N]
        qe = np.load(f'{WELL}/exact/q_exact_BBK_on_md_frames.npy', mmap_mode='r')[:N]
    else:
        if not os.path.exists(ASE_TRAJ):
            return None
        d = np.load(ASE_TRAJ, mmap_mode='r')[:N]
        d = d[0] if d.ndim == 3 else d
        ref = np.load(f'{WELL}/exact/q_exact_ASE.npz')
        f = RegularGridInterpolator((ref['xs'], ref['vs']), ref['m'], bounds_error=False, fill_value=None)
        x, v = np.array(d[:, 0], float), np.array(d[:, 1], float)
        qe = np.clip(f(np.c_[np.clip(x, ref['xs'][0], ref['xs'][-1]), np.clip(v, ref['vs'][0], ref['vs'][-1])]), 0, 1)
    return np.array(d[:, 0], float), np.array(d[:, 1], float), np.array(qe, float)


# 1. Z_q overview: rows EM, BBK, ASE; columns x, v | x + history [0, 1] | x + history [0, 1, 2, 4]
zq = lambda png: (lambda im: im[:, int(im.shape[1] * 0.655):])(mpimg.imread(png))
fig, axes = plt.subplots(3, 3, figsize=(16, 10.5))
for ax, integ in zip(axes, ('EM', 'BBK', 'ASE')):
    for a, case in zip(ax, CASES[integ] + [None] * (3 - len(CASES[integ]))):
        a.axis('off')
        if case is None:
            a.set_title(f'{integ}: x + history [0, 1, 2, 4]', fontsize=11)
            a.text(0.5, 0.5, 'not run', ha='center', va='center', color='0.5', fontsize=13, transform=a.transAxes)
            continue
        a.imshow(zq(case[1])); a.set_title(f'{integ}: {case[0]}', fontsize=11)
fig.suptitle('$Z_q$ at several lags (flat at $N_{AB}$ for a committor); best seed per setting', fontsize=12)
fig.tight_layout(); fig.savefig(f'{HERE}/simple_zq.png', dpi=100); plt.close(fig)

# 2. per integrator: one row per setting: full F(q)/Z_q plot | q(x, v) | q - exact
summary = {}
for integ, cases in CASES.items():
    md = load_md(integ)
    fig = plt.figure(figsize=(26, 4.6 * len(cases)))
    gs = fig.add_gridspec(len(cases), 4, width_ratios=[2.6, 1, 1, 1])
    if md is not None:
        x, v, qe = md
        free = (x > LA) & (x < LB)
        el, ev = np.linspace(LA, LB, 71), np.linspace(-2.6, 2.6, 53)
        n = np.histogram2d(x[free], v[free], [el, ev])[0]
        mean = lambda r: np.where(n > 20, np.histogram2d(x[free], v[free], [el, ev], weights=r[free])[0]
                                  / np.maximum(n, 1), np.nan).T
    for i, (label, feps, qf) in enumerate(cases):
        a = fig.add_subplot(gs[i, 0]); a.imshow(mpimg.imread(feps)); a.axis('off')
        a.set_title(f'{integ}, {label}: $F(q)$, $F(q)$ in natural coordinate, $Z_q$', fontsize=11)
        aq, ad, aref = (fig.add_subplot(gs[i, k]) for k in (1, 2, 3))
        if md is None:
            for b in (aq, ad, aref):
                b.axis('off'); b.text(0.5, 0.5, 'needs traj.npy', ha='center', va='center', color='0.5',
                                      transform=b.transAxes)
            continue
        r = np.load(qf).astype(float)[:len(x)]
        if not ((r[x <= LA] == 0).all() and (r[x >= LB] == 1).all()):   # q is set to 0/1 on A/B frames by the fit
            raise SystemExit(f'{qf}: q is not 0 on A / 1 on B frames: trajectory and q are from different runs')
        rms = float(np.sqrt(np.mean((r[free] - qe[free]) ** 2)))
        rms_b = float(np.sqrt(np.nanmean((mean(r) - mean(qe)) ** 2)))
        summary[f'{integ}, {label}'] = dict(rms_frame=rms, rms_bin=rms_b)
        im = aq.pcolormesh(el, ev, mean(r), cmap='coolwarm', vmin=0, vmax=1); fig.colorbar(im, ax=aq)
        aq.set(title='fit $q(x, v)$', xlabel='$x$', ylabel='$v$')
        im = ad.pcolormesh(el, ev, mean(r - qe), cmap='RdBu_r', vmin=-0.1, vmax=0.1); fig.colorbar(im, ax=ad)
        ad.set(title=f'fit $-$ exact (RMS {rms:.4f} / frame, {rms_b:.4f} / bin)', xlabel='$x$')
        im = aref.pcolormesh(el, ev, mean(qe), cmap='coolwarm', vmin=0, vmax=1); fig.colorbar(im, ax=aref)
        aref.set(title=f'exact ({integ})', xlabel='$x$')
    fig.tight_layout(); fig.savefig(f'{HERE}/simple_best_{integ}.png', dpi=80); plt.close(fig)
json.dump(summary, open(f'{HERE}/simple_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))

# 3. larger fit lag: EM and BBK, x + history [0, 1], fit lag/stride 20 (seed 1), same layout as above
fig = plt.figure(figsize=(26, 9.2))
gs = fig.add_gridspec(2, 4, width_ratios=[2.6, 1, 1, 1])
stride = {}
for i, integ in enumerate(('EM', 'BBK')):
    f = f'{HERE}/{integ}/sub20_hist_fine/seed1'
    x, v, qe = load_md(integ)
    free = (x > LA) & (x < LB)
    el, ev = np.linspace(LA, LB, 71), np.linspace(-2.6, 2.6, 53)
    n = np.histogram2d(x[free], v[free], [el, ev])[0]
    mean = lambda r: np.where(n > 20, np.histogram2d(x[free], v[free], [el, ev], weights=r[free])[0] / np.maximum(n, 1), np.nan).T
    a = fig.add_subplot(gs[i, 0]); a.imshow(mpimg.imread(f'{f}/it16000_feps.png')); a.axis('off')
    a.set_title(f'{integ}, x + history [0, 1], fit lag/stride 20 (seed 1): $F(q)$, $F(q)$ natural, $Z_q$', fontsize=11)
    r = np.load(f'{f}/q_traj.npy').astype(float)
    rms, rms_b = float(np.sqrt(np.mean((r[free] - qe[free]) ** 2))), float(np.sqrt(np.nanmean((mean(r) - mean(qe)) ** 2)))
    stride[integ] = dict(rms_frame=rms, rms_bin=rms_b)
    for k, (m, cm, lo, hi, t) in enumerate([(mean(r), 'coolwarm', 0, 1, 'fit $q(x, v)$'),
                                            (mean(r - qe), 'RdBu_r', -0.1, 0.1, f'fit $-$ exact (RMS {rms:.4f} / frame, {rms_b:.4f} / bin)'),
                                            (mean(qe), 'coolwarm', 0, 1, f'exact ({integ})')]):
        b = fig.add_subplot(gs[i, k + 1]); im = b.pcolormesh(el, ev, m, cmap=cm, vmin=lo, vmax=hi)
        fig.colorbar(im, ax=b); b.set(title=t, xlabel='$x$', ylabel='$v$' if k == 0 else None)
fig.tight_layout(); fig.savefig(f'{HERE}/simple_stride.png', dpi=80); plt.close(fig)
print('stride', json.dumps(stride))

# 4. one bad x, v seed as an example (BBK seed 2), same layout as the best-seed rows
BAD = ('BBK', 2)
integ, s = BAD
f = f'{HERE}/{integ}/x_v/seed{s}'
x, v, qe = load_md(integ)
free = (x > LA) & (x < LB)
el, ev = np.linspace(LA, LB, 71), np.linspace(-2.6, 2.6, 53)
n = np.histogram2d(x[free], v[free], [el, ev])[0]
mean = lambda r: np.where(n > 20, np.histogram2d(x[free], v[free], [el, ev], weights=r[free])[0] / np.maximum(n, 1), np.nan).T
fig = plt.figure(figsize=(26, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[2.6, 1, 1, 1])
a = fig.add_subplot(gs[0, 0]); a.imshow(mpimg.imread(f'{f}/it11000_feps.png')); a.axis('off')
a.set_title(f'{integ}, x, v (seed {s}, bad seed): $F(q)$, $F(q)$ natural, $Z_q$', fontsize=11)
r = np.load(f'{f}/q_traj.npy').astype(float)
rms, rms_b = float(np.sqrt(np.mean((r[free] - qe[free]) ** 2))), float(np.sqrt(np.nanmean((mean(r) - mean(qe)) ** 2)))
for k, (m, cm, lo, hi, t) in enumerate([(mean(r), 'coolwarm', 0, 1, 'fit $q(x, v)$'),
                                        (mean(r - qe), 'RdBu_r', -0.1, 0.1, f'fit $-$ exact (RMS {rms:.4f} / frame, {rms_b:.4f} / bin)'),
                                        (mean(qe), 'coolwarm', 0, 1, f'exact ({integ})')]):
    b = fig.add_subplot(gs[0, k + 1]); im = b.pcolormesh(el, ev, m, cmap=cm, vmin=lo, vmax=hi)
    fig.colorbar(im, ax=b); b.set(title=t, xlabel='$x$', ylabel='$v$' if k == 0 else None)
fig.tight_layout(); fig.savefig(f'{HERE}/simple_bad.png', dpi=80); plt.close(fig)
print('bad', integ, s, rms, rms_b)
