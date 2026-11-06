"""Stopped Z_q z-score of the EXACT committor per staple ensemble and per quarter of the MC run (cycle order)."""
import sys, numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/optimalrcs-cupy')
import metrics_tis
F = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures'
d = np.load(f'{F}/istarz_br_g10/committor.npz')
X = d['X_all']; i_traj = d['i_traj']; w = d['path_weights']; b = d['boundary0'] | d['boundary1']; b1 = d['boundary1']
ref = np.load(f'{F}/z_brownian/committor_reference_h0.001.npz')
q = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(X), 0, 1)
del X
q[b] = b1[b].astype(float)
SIZES = [2976, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 2999, 3000]   # paths per ensemble 001..010, file (= cycle) order
seg_start = np.r_[True, i_traj[1:] != i_traj[:-1]]
rank = np.cumsum(seg_start) - 1                                          # path rank 0..29974
ens_of_rank = np.repeat(np.arange(1, 11), SIZES)
pos_of_rank = np.concatenate([np.arange(n) / n for n in SIZES])         # position within its ensemble's cycle order
ens, pos = ens_of_rank[rank], pos_of_rank[rank]
QS, LAGS, NQ = (0.3, 0.63, 0.75), (16, 256, 4096), 4

def zvals(m):
    r, bb, ii, ww = cp.asarray(q[m]), cp.asarray(b[m]), cp.asarray(i_traj[m]), cp.asarray(w[m])
    tau = metrics_tis.stop_index(ii, bb)
    out = []
    for lag in LAGS:
        e, z = metrics_tis.comp_zq_stopped_zscore(r, bb, ii, ww, lag, tau=tau)
        e, z = cp.asnumpy(e), cp.asnumpy(z)
        out.append([z[min(np.searchsorted(e, x) - 1, len(z) - 1)] for x in QS])
    return np.array(out)    # [lag, q]

print(f'z of the exact committor at q = {QS}; lags {LAGS}; quarters of the MC cycles (Q1 earliest)')
for k in [None] + list(range(1, 11)):
    base = np.ones(len(q), bool) if k is None else (ens == k)
    rows = [zvals(base & (pos >= j / NQ) & (pos < (j + 1) / NQ)) for j in range(NQ)]
    full = zvals(base)
    name = 'all' if k is None else f'{k:03d}'
    for li, lag in enumerate(LAGS):
        txt = ' | '.join(' '.join(f'{v:+5.1f}' for v in r_[li]) for r_ in rows)
        print(f'{name} lag {lag:4d}: quarters [{txt}]   whole: {" ".join(f"{v:+5.1f}" for v in full[li])}', flush=True)
    cp.get_default_memory_pool().free_all_blocks()
