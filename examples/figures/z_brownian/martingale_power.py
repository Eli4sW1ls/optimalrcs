"""Power of the first-plateau-entry martingale test: fraction of pieces that reach A or B after entering the
plateau, and E[D] for the exact committor vs the fitted q (mirrored-pieces staple fit, +0.031 on the plateau)."""
import sys, argparse, io, contextlib, numpy as np
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import tis_data
F = REPO + '/examples/figures/'
ref = np.load(F + 'z_brownian/committor_reference_h0.001.npz')
fref = RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    d = tis_data.prepare_data(p.parse_args(['/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10',
                                             '--reversed', '--paths-per-ensemble', '1500', '--seed', '0']))
y, x, it = d.X_all[:, 0], d.X_all[:, 1], d.i_traj
b0, b1 = d.boundary0, d.boundary1; b = b0 | b1; n = len(y)
qe = np.clip(fref(np.c_[y, x]), 0, 1); qe[b0] = 0; qe[b1] = 1
qf = np.load(F + 'istarz_br_g10_reversed/r_continued.npy')
ends = np.r_[np.flatnonzero(it[1:] != it[:-1]), n - 1]; starts = np.r_[0, ends[:-1] + 1]
nxt = np.minimum.accumulate(np.where(b, np.arange(n), n)[::-1])[::-1]
stop = np.minimum(nxt, np.repeat(ends, ends - starts + 1))
seg_start = np.r_[True, it[1:] != it[:-1]]
w = np.zeros(it.max() + 1); w[it[seg_start]] = d.path_weights[seg_start]
grp = np.zeros(it.max() + 1, int); grp[it[seg_start]] = d.group_traj[seg_start]
for band in ((0.55, 0.7), (0.7, 0.85)):
    inb = (~b) & (qe > band[0]) & (qe < band[1])
    first = np.minimum.reduceat(np.where(inb, np.arange(n), n), starts); ok = first < n; f = first[ok]; pid = it[starts[ok]]
    s = stop[f]; absorbed = b[s]; toB = b1[s]
    W = w[pid]
    print(f'band {band}: {ok.sum()} pieces; weighted fraction reaching A or B {np.sum(W * absorbed) / W.sum():.3f} '
          f'(B {np.sum(W * toB) / W.sum():.3f})', flush=True)
    for name, q in (('exact committor', qe), ('fitted q (+0.031)', qf)):
        D = q[s] - q[f]
        m = np.sum(W * D) / W.sum()
        gs = np.bincount(np.unique(grp[pid], return_inverse=True)[1], weights=W * (D - m))
        se = np.sqrt(np.sum(gs ** 2)) / W.sum()
        print(f'   {name:20s}: E[D] = {m:+.4f} +- {se:.4f}  (mean q at entry {np.sum(W * q[f]) / W.sum():.3f})', flush=True)
