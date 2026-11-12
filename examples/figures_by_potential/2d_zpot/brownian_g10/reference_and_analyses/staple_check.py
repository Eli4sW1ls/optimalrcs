"""Is the staple data set (sim_istarz_br_g10, 3000 paths/ensemble, cut at M, rejection weights)
consistent with the exact committor? Stopped Z_q / standard error per lag for: the exact
committor (reference, A: y <= 0.05, B: y >= 0.85, absorption every step), the original fit,
and the exact committor with all path weights 1."""
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
q_ex = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)
q_ex[b0] = 0; q_ex[b1] = 1
tau = M.stop_index(cp.asarray(it), cp.asarray(b))
LAGS = (1, 4, 16, 64, 256, 1024, 4096, 16384)
print(f'{len(y)} frames, {len(np.unique(it))} paths; lags {LAGS}', flush=True)
for name, r, ww in (('exact q, rejection weights', q_ex, w), ('original fit, rejection weights', d['r'], w),
                    ('exact q, all weights 1', q_ex, np.ones_like(w))):
    zs, worst_q = [], []
    for dt in LAGS:
        edges, z = M.comp_zq_stopped_zscore(cp.asarray(r), cp.asarray(b), cp.asarray(it), cp.asarray(ww), dt,
                                            tau=tau, g_traj=cp.asarray(it))
        z = cp.asnumpy(z)[5:]; e = cp.asnumpy(edges)[5:5 + len(z)]
        k = np.argmax(np.abs(z)); zs.append(z[k]); worst_q.append(e[k])
        cp.get_default_memory_pool().free_all_blocks()
    print(f'{name:34s}: worst z per lag ' + ' '.join(f'{v:+5.1f}' for v in zs)
          + ' | at q ' + ' '.join(f'{v:.2f}' for v in worst_q), flush=True)
