"""Stopped Z_q z-score of the exact committor on the staple data, split by ensemble and by odd/even path."""
import sys, numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/optimalrcs-cupy/')
import metrics_tis as M
HERE = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures_by_potential/'
d = np.load(HERE + '2d_zpot/brownian_g10/staple/original/committor.npz')
y, x = d['X_all'][:, 0], d['X_all'][:, 1]
b0, b1, it, w = d['boundary0'], d['boundary1'], d['i_traj'], d['path_weights']
b = b0 | b1
ref = np.load(HERE + '2d_zpot/brownian_g10/reference_and_analyses/committor_reference_h0.001.npz')
q = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)
q[b0] = 0; q[b1] = 1
# ensemble of each path, from the loading order (paths per ensemble 001..010, from run.log)
counts = [2976, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 2999, 3000]
ens_of_path = np.repeat(np.arange(1, 11), counts)
paths = np.unique(it); assert len(paths) == sum(counts)
ens = ens_of_path[np.searchsorted(paths, it)]
tau = M.stop_index(cp.asarray(it), cp.asarray(b))
LAGS = (16, 256, 4096)
def worst(mask):
    out = []
    for dt in LAGS:
        e, z = M.comp_zq_stopped_zscore(cp.asarray(q), cp.asarray(b), cp.asarray(it), cp.asarray(w * mask), dt,
                                        tau=tau, g_traj=cp.asarray(it))
        z = cp.asnumpy(z)[5:]; e = cp.asnumpy(e)[5:5 + len(z)]; k = np.argmax(np.abs(z))
        out.append(f'{z[k]:+5.1f} (q {e[k]:.2f})')
    return '  '.join(out)
print(f'worst z of the exact committor at lags {LAGS}', flush=True)
for k in range(1, 11):
    m = (ens == k).astype(float)
    print(f'ensemble {k:03d} ({int(m.sum()):8d} frames): {worst(m)}', flush=True)
odd = (np.searchsorted(paths, it) % 2).astype(float)
print(f'even paths: {worst(1 - odd)}', flush=True)
print(f'odd paths : {worst(odd)}', flush=True)
