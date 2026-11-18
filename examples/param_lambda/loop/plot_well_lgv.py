#!/usr/bin/env python3
"""M5 `well_lgv` (1D cos well, ASE Langevin): the exact committor q(x, v) and, if the MD run saved walkers, the
check of q against the long MD (fraction of frames whose next state is B, binned in (x, v) and in q_exact).

    python -m loop.plot_well_lgv [--md DIR ...] [--stride 10] [--out FILE]

--md: md_well_lgv output directories (rate.json, optionally traj.npy); default langevin_md_ase and
langevin_md_ase_long if present. The empirical committor uses the walkers of the first directory with a traj.npy.
Default out: figures_by_potential/1d_cos_well/exact/well_lgv_exact_md.png.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, ase_systems  # noqa: E402
from loop.plots import plt, SEQ, DIV, BLUE, RED, INK, INK2, MUTED, _logit  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

sys.path.insert(0, os.path.join(env.EXAMPLES, 'figures_by_potential', '_shared', 'exact_committors'))
import exact_tools as E  # noqa: E402

WELL = os.path.join(env.EXAMPLES, 'figures_by_potential', '1d_cos_well')
S = ase_systems.get('well_lgv')
A, B = S['lambda_a'], S['lambda_b']


def next_state(x):
    """0 / 1 for the state (A / B) each frame reaches next (itself if inside one), -1 after the last visit."""
    s = np.full(len(x), -1, np.int8)
    s[x <= A], s[x >= B] = 0, 1
    hit = np.flatnonzero(s >= 0)
    j = np.searchsorted(hit, np.arange(len(x)))
    out = np.full(len(x), -1, np.int8)
    ok = j < len(hit)
    out[ok] = s[hit[j[ok]]]
    return out


def md_committor(traj_file, stride, xe, ve, qe, qfn, max_frames=4_000_000):
    """Counts per (x, v) bin and per q_exact bin of frames inside (A, B) and of those that reach B next."""
    tr = np.load(traj_file, mmap_mode='r')
    nw, nt = tr.shape[:2]
    stride = max(stride, int(np.ceil(nw * nt / max_frames)))
    n2, h2 = np.zeros((len(xe) - 1, len(ve) - 1)), np.zeros((len(xe) - 1, len(ve) - 1))
    nq, hq, sq = (np.zeros(len(qe) - 1) for _ in range(3))
    xs_all = []
    for w in range(nw):
        x = np.asarray(tr[w, :, 0], float)
        ns = next_state(x)
        sel = np.arange(0, nt, stride)
        sel = sel[(ns[sel] >= 0) & (x[sel] > A) & (x[sel] < B)]
        xv = np.asarray(tr[w, sel], float)
        y = ns[sel].astype(float)
        n2 += np.histogram2d(xv[:, 0], xv[:, 1], [xe, ve])[0]
        h2 += np.histogram2d(xv[:, 0], xv[:, 1], [xe, ve], weights=y)[0]
        q = qfn(xv)
        b = np.clip(np.searchsorted(qe, _logit(q)) - 1, 0, len(qe) - 2)
        nq += np.bincount(b, minlength=len(qe) - 1)
        hq += np.bincount(b, y, minlength=len(qe) - 1)
        sq += np.bincount(b, q, minlength=len(qe) - 1)
        xs_all.append(x[::stride])
    return dict(n2=n2, h2=h2, nq=nq, hq=hq, sq=sq, x=np.concatenate(xs_all), stride=stride, nw=nw, nt=nt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--md', nargs='+', default=[d for d in (os.path.join(WELL, 'langevin_md_ase'),
                                                           os.path.join(WELL, 'langevin_md_ase_long')) if os.path.isdir(d)])
    ap.add_argument('--stride', type=int, default=10, help='every n-th frame of the saved walkers')
    ap.add_argument('--out', default=os.path.join(WELL, 'exact', 'well_lgv_exact_md.png'))
    a = ap.parse_args()
    qfn = ase_systems.exact_committor(S)
    z = np.load(os.path.join(env.REPO, S['exact']['file']))
    xg = np.linspace(A + 1e-4, B - 1e-4, 281)
    vg = np.linspace(-3.5, 3.5, 241)
    X, Vv = np.meshgrid(xg, vg, indexing='ij')
    Q = qfn(np.c_[X.ravel(), Vv.ravel()]).reshape(X.shape)
    H = E.coswell_potential(X) + 0.5 * Vv ** 2

    rates = []
    for d in a.md:
        p = os.path.join(d, 'rate.json')
        if os.path.isfile(p):
            with open(p) as f:
                rates.append((os.path.basename(d), json.load(f)))
    traj = next((os.path.join(d, 'traj.npy') for d in a.md if os.path.isfile(os.path.join(d, 'traj.npy'))), None)
    xe, ve = np.linspace(A, B, 36), np.linspace(-3.5, 3.5, 36)
    qe = np.linspace(-8, 8, 33)
    mc = md_committor(traj, a.stride, xe, ve, qe, qfn) if traj else None

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), layout='constrained')
    # (a) potential and the MD density in x
    ax = axs[0, 0]
    xx = np.linspace(-0.62, 0.62, 600)
    ax.plot(xx, E.coswell_potential(xx), color=INK, lw=1.6, label='V(x) [kT]')
    for v in (A, B):
        ax.axvline(v, color=INK2, lw=0.8, ls='--')
    ax.text(A, 0.25, ' A', color=INK2, ha='right', va='bottom'); ax.text(B, 0.25, 'B ', color=INK2, ha='left', va='bottom')
    ax.set(ylim=(float(E.coswell_potential(xx).min()) - 0.1, 1.6), xlabel='x', ylabel='V(x) [kT]', title='potential (black) and MD density of x (blue)')
    if mc is not None:
        ax2 = ax.twinx()
        ax2.hist(mc['x'], bins=200, range=(-0.62, 0.62), density=True, color=BLUE, alpha=0.35)
        bz = np.exp(-E.coswell_potential(xx))
        ax2.plot(xx, bz / (bz.sum() * (xx[1] - xx[0])), color=BLUE, lw=1, ls='--', label='exp(-V) normalized')
        ax2.set_ylabel('density', color=BLUE)
        ax2.legend(fontsize=7, loc='upper center')
    # (b) exact committor
    ax = axs[0, 1]
    im = ax.pcolormesh(xg, vg, Q.T, cmap=SEQ, vmin=0, vmax=1, shading='auto')
    cs = ax.contour(xg, vg, Q.T, levels=[0.05, 0.25, 0.5, 0.75, 0.95], colors='w', linewidths=0.8)
    ax.clabel(cs, fontsize=6, fmt='%g')
    ax.contour(xg, vg, H.T, levels=np.arange(0.5, 7, 0.5), colors='0.55', linewidths=0.35)
    ax.set(xlabel='x', ylabel='v', title='exact q(x, v) from the grid solve\n(white: isocommittors; grey: H = 0.5, 1, ... kT)')
    fig.colorbar(im, ax=ax, label='q')
    # (c) committor along v at fixed x
    ax = axs[0, 2]
    for x0, c in zip((-0.3, -0.15, 0.0, 0.15, 0.3), ('#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b')):
        ax.plot(vg, qfn(np.c_[np.full(len(vg), x0), vg]), color=c, lw=1.6, label=f'x = {x0:+.2f}')
    ax.set(xlabel='v', ylabel='q', title='exact q along v at fixed x')
    ax.grid(color='#e6e5e1', lw=0.5)
    ax.legend(fontsize=7)
    # (d, e) MD empirical committor
    if mc is not None:
        qm = np.where(mc['n2'] >= 30, mc['h2'] / np.maximum(mc['n2'], 1), np.nan)
        ax = axs[1, 0]
        im = ax.pcolormesh(xe, ve, np.ma.masked_invalid(qm).T, cmap=SEQ, vmin=0, vmax=1, shading='flat')
        ax.contour(xg, vg, Q.T, levels=[0.05, 0.25, 0.5, 0.75, 0.95], colors='w', linewidths=0.8)
        ax.set(xlabel='x', ylabel='v', title='MD: fraction reaching B next\n(bins >= 30 frames; white: exact isocommittors)')
        fig.colorbar(im, ax=ax, label='q_MD')
        ax = axs[1, 1]
        qc = 0.5 * (xe[1:] + xe[:-1]); vc = 0.5 * (ve[1:] + ve[:-1])
        QX = qfn(np.c_[np.repeat(qc, len(vc)), np.tile(vc, len(qc))]).reshape(len(qc), len(vc))
        se = np.sqrt(np.maximum(QX * (1 - QX), 1e-12) / np.maximum(mc['n2'], 1))
        zsc = np.where(mc['n2'] >= 30, (qm - QX) / se, np.nan)
        im = ax.pcolormesh(xe, ve, np.ma.masked_invalid(zsc).T, cmap=DIV, norm=TwoSlopeNorm(0, -4, 4), shading='flat')
        fin = np.isfinite(zsc)
        ax.set(xlabel='x', ylabel='v', title=f'(q_MD - q_exact) / binomial s.e. at the bin centre; RMS {np.sqrt(np.mean(zsc[fin] ** 2)):.2f}\n'
                                             '(frames along a walker are correlated: look for structure, not single bins)')
        fig.colorbar(im, ax=ax, label='z')
        ax = axs[1, 2]
        ok = mc['nq'] >= 30
        pm = mc['hq'][ok] / mc['nq'][ok]
        qmean = mc['sq'][ok] / mc['nq'][ok]
        err = np.sqrt(np.maximum(pm * (1 - pm), 1e-12) / mc['nq'][ok])
        ax.errorbar(qmean, pm, yerr=err, fmt='o', ms=4, color=BLUE, label='MD frames, binned in logit q_exact')
        ax.plot([0, 1], [0, 1], color=MUTED, ls='--', lw=1, label='exact')
        ax.set(xlabel='mean q_exact in the bin', ylabel='fraction reaching B next',
               title=f'calibration: {mc["nw"]} walkers x {mc["nt"]:,} steps, every {mc["stride"]}th frame')
        ax.grid(color='#e6e5e1', lw=0.5)
        ax.legend(fontsize=7)
        ins = ax.inset_axes([0.62, 0.1, 0.35, 0.3])
        ins.bar(0.5 * (qe[1:] + qe[:-1]), mc['nq'], width=qe[1] - qe[0], color=BLUE, alpha=0.5)
        ins.set_yscale('log'); ins.set_xlabel('logit q_exact', fontsize=6); ins.tick_params(labelsize=5)
    else:
        for ax in axs[1]:
            ax.set_axis_off()
        axs[1, 1].text(0.5, 0.5, 'no traj.npy: run loop.md_well_lgv with --save-walkers', ha='center', color=MUTED,
                       transform=axs[1, 1].transAxes)
    rtxt = '; '.join(f"{n}: k_AB {r['k_ab']:.5f} +- {r['k_ab_se']:.5f}, k_BA {r['k_ba']:.5f} +- {r['k_ba_se']:.5f} "
                     f"({r['walkers']} x {r['steps_per_walker']:,} steps)" for n, r in rates)
    fig.suptitle(f'well_lgv: 1D cos well, ASE Langevin (kT 1, gamma 1, dt 0.005), A x <= {A}, B x >= {B}\nMD rate {rtxt}',
                 x=0.01, ha='left', fontsize=10)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == '__main__':
    main()
