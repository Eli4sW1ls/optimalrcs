"""Martingale test of path data with the exact committor: for every piece, D = q_exact(stop) - q_exact(first
frame with q_exact in BAND), stop = first A/B frame or the piece's last frame. For unbiased data E[D] = 0
(optional stopping). One number per piece; weighted mean with a standard error over independent paths.
    python martingale_test.py synth | staple
"""
import sys, argparse, io, contextlib, numpy as np
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import tis_data
HERE = REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/'
ref = np.load(HERE + 'committor_reference_h0.001.npz')
fref = RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)
BANDS = [(0.05, 0.3), (0.3, 0.55), (0.55, 0.7), (0.7, 0.85)]

def piece_D(y, x, b0, b1, it, band):
    """Per piece: (piece id, D) for pieces with a frame in the band before they stop."""
    q = np.clip(fref(np.c_[y, x]), 0, 1); q[b0] = 0; q[b1] = 1
    n = len(q); b = b0 | b1
    ends = np.r_[np.flatnonzero(it[1:] != it[:-1]), n - 1]; starts = np.r_[0, ends[:-1] + 1]
    marker = np.where(b, np.arange(n), n); nxt = np.minimum.accumulate(marker[::-1])[::-1]
    end_of = np.repeat(ends, ends - starts + 1)
    stop = np.minimum(nxt, end_of)
    inb = (~b) & (q > band[0]) & (q < band[1])
    idx = np.where(inb, np.arange(n), n)
    # first in-band frame of each piece
    first = np.minimum.reduceat(idx, starts)
    ok = first < n
    f = first[ok]
    return it[starts[ok]], q[stop[f]] - q[f], q[f]

def summarize(label, pid, D, w, grp):
    W = w[pid]; G = grp[pid]
    m = np.sum(W * D) / np.sum(W)
    # standard error over independent groups (paths): sum per group of W*(D - m)
    gs = np.bincount(np.unique(G, return_inverse=True)[1], weights=W * (D - m))
    se = np.sqrt(np.sum(gs ** 2)) / np.sum(W)
    return f'{label:44s} n={len(D):6d}  E[D] = {m:+.4f} +- {se:.4f}  ({m / se:+.1f} sigma)'

mode = sys.argv[1]
if mode == 'synth':
    d = np.load(HERE + 'synth/synth_30k_s50.npz')
    y, x, it = d['y'], d['x'], d['i_traj']; b0, b1 = y <= 0.05, y >= 0.85
    w = np.ones(it.max() + 1); grp = np.arange(it.max() + 1)
    for band in BANDS:
        pid, D, _ = piece_D(y, x, b0, b1, it, band)
        print(summarize(f'synthetic walkers, band {band}', pid, D, w, grp), flush=True)
    sys.exit()

SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10'
for weights in ('rejections', 'staple'):
    p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
    with contextlib.redirect_stdout(io.StringIO()):
        d = tis_data.prepare_data(p.parse_args([SIM, '--reversed', '--weights', weights, '--paths-per-ensemble', '1500', '--seed', '0']))
    y, x, it = d.X_all[:, 0], d.X_all[:, 1], d.i_traj
    seg_start = np.r_[True, it[1:] != it[:-1]]
    w = np.zeros(it.max() + 1); w[it[seg_start]] = d.path_weights[seg_start]
    grp = np.zeros(it.max() + 1, int); grp[it[seg_start]] = d.group_traj[seg_start]
    # forward = first piece of its path (group), mirrored = second
    seen = set(); fwd = np.zeros(it.max() + 1, bool)
    for pc, g in zip(it[seg_start], d.group_traj[seg_start]):
        if g not in seen: seen.add(g); fwd[pc] = True
    counts = [2983] + [3000] * 9
    ens = np.repeat(np.arange(1, 11), counts)                    # ensemble of each piece id
    # rank within ensemble (pieces are in cycle order): early half vs late half of the run
    rank = np.zeros(it.max() + 1); off = 0
    for c in counts:
        rank[off:off + c] = np.arange(c) / c; off += c
    for band in BANDS:
        pid, D, _ = piece_D(y, x, d.boundary0, d.boundary1, it, band)
        print(f'--- {weights} weights, band {band}', flush=True)
        for lab, sel in (('all pieces', np.ones(len(pid), bool)), ('forward pieces', fwd[pid]), ('mirrored pieces', ~fwd[pid]),
                         ('early half of the run', rank[pid] < 0.5), ('late half of the run', rank[pid] >= 0.5)):
            print(summarize(lab, pid[sel], D[sel], w, grp), flush=True)
        if weights == 'rejections' and band == (0.55, 0.7):
            for k in range(1, 11):
                sel = ens[pid] == k
                if sel.sum() > 20:
                    print(summarize(f'  ensemble {k:03d}', pid[sel], D[sel], w, grp), flush=True)
    del d
