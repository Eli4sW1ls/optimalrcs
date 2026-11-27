#!/usr/bin/env python3
"""Where does the stopped Z_q of a staple-TIS committor fit come from? Diagnostics from the saved fit arrays only
(numpy; no GPU, no tensorflow, no TIS data directory needed).

    python trypsin_zq_diag.py <fit folder or committor.npz> [--out DIR] [--lags 1 64 512 4096 32768]
                              [--subsample 0.3] [--seed 0]

Reads <fit folder>/committor.npz as written by infretis_staple_trypsin.ipynb (r, lam, i_traj, path_weights, group_traj,
boundary0, boundary1; X_all is not loaded). Prints one text report, writes <out>/zq_diag.json and <out>/zq_diag.png
(out defaults to the fit folder). Send the text report (and the png).

What it reports (z = stopped Z_q / standard error, as in the TIS plot; lag in frames):
 1  data support: per q bin and per lambda bin, frames, weight share, distinct paths, n_eff of the path-aggregated
    weights, and the weighted share of long-lag windows that stop at a turn (not in A/B);
 2  z at q = 0.05 ... 1.0 and max |z| (q > 0.85 and everywhere) for variants of the same data:
      path        independence group = path number (the notebook's)           <- reproduces the plot
      piece       group = piece
      block10/100/1000   groups = blocks of consecutive path numbers (pnr grows with the MC time; paths close in
                  the chain share frames, so single-path groups understate the error)
      first/second half  only windows of the first / second half of the paths (rank of pnr)
      pos<=1000 / pos>1000  only window starts within / beyond 1000 frames of their piece start (conditions on the
                  past only, so a committor stays at z ~ 0; selecting pieces by their total length would not)
      drop        windows that run past a piece end at a turn are dropped (standard Z_q) instead of stopped
      unweighted  all weights 1
 3  z against lambda (not q) for the path and block100 groupings;
 4  the paths that carry the long-lag Z of q > 0.85: top contributors and their share of the variance.
--subsample f keeps a random fraction f of the paths (weights x 1/f) if the arrays do not fit in memory.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_metrics():
    # numpy-only module, loaded by path: importing the optimalrcs package may start tensorflow / cupy
    path = os.path.join(HERE, '..', 'optimalrcs', 'metrics_tis.py')
    if os.path.isfile(path):
        spec = importlib.util.spec_from_file_location('metrics_tis_local', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    from optimalrcs import metrics_tis as ms
    return ms


ms = load_metrics()

ap = argparse.ArgumentParser()
ap.add_argument('fit')
ap.add_argument('--out')
ap.add_argument('--lags', type=int, nargs='+', default=[1, 64, 512, 4096, 32768])
ap.add_argument('--subsample', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
npz = a.fit if a.fit.endswith('.npz') else os.path.join(a.fit, 'committor.npz')
out = a.out or os.path.dirname(os.path.abspath(npz))
os.makedirs(out, exist_ok=True)
T0 = time.time()
say = lambda *s: print(*s, flush=True)

z = np.load(npz, allow_pickle=True)
r = np.asarray(z['r'], float)
lam = np.asarray(z['lam'], float)
i_traj = np.asarray(z['i_traj'])
w = np.asarray(z['path_weights'], float)
grp_raw = np.asarray(z['group_traj'])
b = np.asarray(z['boundary0'], bool) | np.asarray(z['boundary1'], bool)
if a.subsample < 1.0:                                   # whole paths, inverse-probability weights
    rng = np.random.default_rng(a.seed)
    up, inv0 = np.unique(grp_raw, return_inverse=True)
    keep = rng.random(len(up)) < a.subsample
    m = keep[inv0.reshape(-1)]
    r, lam, i_traj, w, grp_raw, b = r[m], lam[m], i_traj[m], w[m] / a.subsample, grp_raw[m], b[m]
n = len(r)
seg = ms._segments(i_traj, n)
n_pieces = int(seg[-1]) + 1
upaths, rank = np.unique(grp_raw, return_inverse=True)
rank = rank.reshape(-1)
n_paths = len(upaths)
idx = np.arange(n)
piece_last = np.r_[np.flatnonzero(seg[1:] != seg[:-1]), n - 1]
seg_end = piece_last[seg]                               # last frame of the frame's piece
piece_len = np.bincount(seg)[seg]
piece_first = np.r_[0, piece_last[:-1] + 1][seg]
pos = idx - piece_first                                 # frames since the piece start
tau = ms.stop_index(i_traj, b)
free = ~b
say(f'# {os.path.abspath(npz)}')
say(f'# {n} frames, {n_pieces} pieces, {n_paths} paths, {int(b.sum())} frames in A/B; weights: mean {w.mean():.3g}, '
    f'max {w.max():.3g}, Kish n_eff {w.sum() ** 2 / (w ** 2).sum():.0f}; pieces: median {np.median(np.bincount(seg)):.0f}, '
    f'max {np.bincount(seg).max()} frames; pnr {upaths.min()}..{upaths.max()}')
say(f'# lags {a.lags}; q-bins 200 (as the plot); first 5 bins skipped for max|z|; subsample {a.subsample}')

NQ = 200
q_edges = np.linspace(0, 1, NQ + 1)
q_check = [0.05, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
q_chk_bin = [min(int(np.searchsorted(q_edges, v, side='right')) - 1, NQ - 1) for v in q_check]
hi_bin = int(np.searchsorted(q_edges, 0.85))
rep = {'n_frames': int(n), 'n_pieces': n_pieces, 'n_paths': n_paths}


def contributions(dt, weights, drop=False):
    u = np.minimum(idx + dt, tau)
    c = (r[u] - r) * weights / dt
    ok = free.copy()
    if drop:                                            # window ends past a turn-ended piece: dropped
        ok &= b[tau] | (idx + dt <= tau)
    return c, ok


def zprofile(binvar, edges, c, ok, grp):
    nb = len(edges) - 1
    s = np.flatnonzero(ok)
    bins = np.clip(np.searchsorted(edges, binvar[s], side='right') - 1, 0, nb - 1)
    zq = np.cumsum(np.bincount(bins, weights=c[s], minlength=nb))
    sig = ms._sigma_zq(grp[s], bins, c[s], nb)
    return np.divide(zq, sig, out=np.zeros(nb), where=sig > 0)


g_path = rank
g_piece = seg
variants = [('path', dict(grp=g_path)), ('piece', dict(grp=g_piece))]
for K in (10, 100, 1000):
    if n_paths // K >= 20:
        variants.append((f'block{K}', dict(grp=rank // K)))
variants += [('first half', dict(grp=g_path, mask=rank < n_paths // 2)),
             ('second half', dict(grp=g_path, mask=rank >= n_paths // 2)),
             ('pos<=1000', dict(grp=g_path, mask=pos <= 1000)),
             ('pos>1000', dict(grp=g_path, mask=pos > 1000)),
             ('drop', dict(grp=g_path, drop=True)),
             ('unweighted', dict(grp=g_path, unw=True))]
ones = np.ones(n)

# ---- 2: z variants ---------------------------------------------------------------------------------------------
say('\n== 2. z = stopped Z_q / std. error, by variant and lag')
say('   cols: max|z| for q>0.85 (at q) | max|z| all q (at q) | z at q = ' + ' '.join(f'{v:g}' for v in q_check))
zq_store = {}
for name, kw in variants:
    say(f'-- {name}')
    for dt in a.lags:
        c, ok = contributions(dt, ones if kw.get('unw') else w, drop=kw.get('drop', False))
        if 'mask' in kw:
            ok &= kw['mask']
        zz = zprofile(r, q_edges, c, ok, kw['grp'])
        zq_store[(name, dt)] = zz
        hi = np.abs(zz[hi_bin:])
        al = np.abs(zz[5:])
        say(f'   lag {dt:6d}: {hi.max():7.1f} (q {q_edges[hi_bin + int(hi.argmax())]:.3f}) | {al.max():7.1f} '
            f'(q {q_edges[5 + int(al.argmax())]:.3f}) | ' + ' '.join(f'{zz[k]:7.1f}' for k in q_chk_bin))
        rep.setdefault('z_variants', {}).setdefault(name, {})[str(dt)] = dict(
            max_hi=float(hi.max()), max_all=float(al.max()), at=[float(zz[k]) for k in q_chk_bin])

# ---- 1: data support ----------------------------------------------------------------------------------------------
say('\n== 1. data support (non-boundary frames; n_eff of the path-aggregated weight in the bin)')
dt_stop = min(a.lags, key=lambda d: abs(np.log(d) - np.log(4096)))
c_ds, _ = contributions(dt_stop, w)
u_ds = np.minimum(idx + dt_stop, tau)
turn_stop = free & (tau < idx + dt_stop) & ~b[tau]
bqe = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0000001])
lam_lo, lam_hi = np.quantile(lam[free], [0.001, 0.999])
bl_edges = np.linspace(lam_lo, lam_hi, 13)


def support(binvar, edges, label):
    say(f'-- by {label}: bin | frames | weight % | paths | n_eff paths | turn-stop share @lag {dt_stop} (weighted)')
    rows = []
    wt = w * free
    for k in range(len(edges) - 1):
        s = free & (binvar >= edges[k]) & (binvar < edges[k + 1])
        if not s.any():
            say(f'   {edges[k]:8.3f}-{edges[k + 1]:8.3f} | 0'); continue
        gw = np.bincount(rank[s], weights=w[s], minlength=n_paths)
        gw = gw[gw > 0]
        neff = gw.sum() ** 2 / (gw ** 2).sum()
        ts = np.sum(w[s & turn_stop]) / np.sum(w[s])
        say(f'   {edges[k]:8.3f}-{edges[k + 1]:8.3f} | {int(s.sum()):9d} | {100 * wt[s].sum() / wt.sum():7.3f} | {len(gw):6d} | '
            f'{neff:8.0f} | {ts:.3f}')
        rows.append(dict(lo=float(edges[k]), hi=float(edges[k + 1]), frames=int(s.sum()), paths=int(len(gw)),
                         neff=float(neff), turn_stop=float(ts), wshare=float(wt[s].sum() / wt.sum())))
    return rows


rep['support_q'] = support(r, bqe, 'q')
rep['support_lam'] = support(lam, bl_edges, 'lambda')

# ---- 3: z against lambda ---------------------------------------------------------------------------------------
say('\n== 3. z against lambda (cumulative over lambda bins; lag, grouping): z at lambda = bin upper edges')
NL = 60
lam_e = np.linspace(lam_lo, lam_hi, NL + 1)
chk = np.linspace(0.1, 1.0, 10)
chk_bins = (chk * (NL - 1)).astype(int)
say('   lambda: ' + ' '.join(f'{lam_e[k + 1]:7.2f}' for k in chk_bins))
lam_store = {}
for name, grp in (('path', g_path), ('block100', rank // 100 if n_paths // 100 >= 20 else g_path)):
    for dt in a.lags:
        c, ok = contributions(dt, w)
        zz = zprofile(lam, lam_e, c, ok & (lam >= lam_lo) & (lam <= lam_hi), grp)
        lam_store[(name, dt)] = zz
        say(f'   {name:9s} lag {dt:6d}: ' + ' '.join(f'{zz[k]:7.1f}' for k in chk_bins) + f' | max|z| {np.abs(zz).max():.1f} at '
            f'lambda {lam_e[1 + int(np.abs(zz).argmax())]:.2f}')

# ---- 4: contributors of q > 0.85 -------------------------------------------------------------------------------------
say('\n== 4. paths carrying the Z of q > 0.85 (window starts with q >= 0.85; grouping = path)')
for dt in [d for d in a.lags if d >= 64][-2:]:
    c, ok = contributions(dt, w)
    s = ok & (r >= 0.85)
    sg = np.bincount(rank[s], weights=c[s], minlength=n_paths)
    tot = sg.sum()
    sq = np.sum(sg ** 2)
    pr = sq ** 2 / np.sum(sg ** 4) if np.sum(sg ** 4) > 0 else np.nan
    say(f'-- lag {dt}: Z(window) {tot:.4g}, sigma(path) {np.sqrt(sq):.4g} (z {tot / np.sqrt(sq):.1f}), participation ratio '
        f'of the squared contributions {pr:.1f} of {int((sg != 0).sum())} paths')
    top = np.argsort(-np.abs(sg))[:6]
    for t in top:
        m = s & (rank == t)
        say(f'   pnr {upaths[t]}: contribution {sg[t]:.4g} ({100 * sg[t] / tot:.1f}% of Z, {100 * sg[t] ** 2 / sq:.1f}% of the '
            f'variance), {int(m.sum())} window starts, mean weight {w[m].mean():.3g}, longest piece '
            f'{int(piece_len[m].max())} frames, lambda {lam[m].min():.2f}..{lam[m].max():.2f}')
    rep.setdefault('top_paths', {})[str(dt)] = dict(Z=float(tot), sigma=float(np.sqrt(sq)), participation=float(pr))

# ---- png ------------------------------------------------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dts = [d for d in a.lags if d >= 64] or a.lags
    dtp = dts[-2] if len(dts) > 1 else dts[-1]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    qm = 0.5 * (q_edges[1:] + q_edges[:-1])
    for name in ('path', 'piece', 'block10', 'block100', 'block1000', 'drop', 'pos>1000'):
        if (name, dtp) in zq_store:
            ax[0].plot(qm, zq_store[(name, dtp)], label=name, lw=1.2)
    ax[0].axhspan(-2, 2, color='0.9', zorder=0); ax[0].legend(fontsize=8)
    ax[0].set(xlabel='$q$', ylabel='stopped $Z_q$ / std. error', title=f'lag {dtp}: groupings and variants')
    lm = 0.5 * (lam_e[1:] + lam_e[:-1])
    for (name, dt), zz in lam_store.items():
        if dt == dtp:
            ax[1].plot(lm, zz, label=name)
    ax[1].axhspan(-2, 2, color='0.9', zorder=0); ax[1].legend()
    ax[1].set(xlabel='$\\lambda$', ylabel='stopped $Z_q$ / std. error', title=f'lag {dtp}: cumulated over lambda')
    sq_ = rep['support_q']
    xs = [0.5 * (x['lo'] + min(x['hi'], 1)) for x in sq_]
    ax[2].semilogy(xs, [x['neff'] for x in sq_], 'o-', label='n_eff paths')
    ax[2].semilogy(xs, [x['paths'] for x in sq_], 's--', label='distinct paths')
    ax2 = ax[2].twinx(); ax2.plot(xs, [x['turn_stop'] for x in sq_], 'r^-', label='turn-stop share'); ax2.set_ylim(0, 1)
    ax[2].legend(loc='lower left'); ax2.legend(loc='lower right')
    ax[2].set(xlabel='$q$ (bin centre)', title='data support per q bin')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'zq_diag.png'), dpi=110)
except Exception as e:                                  # noqa: BLE001
    say(f'(png not written: {type(e).__name__}: {e})')
json.dump(rep, open(os.path.join(out, 'zq_diag.json'), 'w'), indent=1)
say(f'\n# done in {time.time() - T0:.0f} s; wrote {out}/zq_diag.json and zq_diag.png')
