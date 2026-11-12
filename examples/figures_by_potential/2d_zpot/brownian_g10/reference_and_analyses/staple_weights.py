"""Exact committor vs the staple data with the 'staple' weighting (rejection repeats x high-acceptance factor)."""
import sys, argparse, io, contextlib, numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import metrics_tis as M, tis_data
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args(['/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10', '--weights', 'staple']))
y, x = data.X_all[:, 0], data.X_all[:, 1]; b = data.boundary0 | data.boundary1
old = np.load(REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/staple/original/committor.npz')
assert np.array_equal(old['lam'], data.lam), 'different frames'
print('same frames as the fit; weights differ in', int((np.abs(old['path_weights'] - data.path_weights) > 1e-9).sum()), 'frames', flush=True)
ref = np.load(REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/committor_reference_h0.001.npz')
q = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)
q[data.boundary0] = 0; q[data.boundary1] = 1
tau = M.stop_index(cp.asarray(data.i_traj), cp.asarray(b))
out = []
for dt in (1, 16, 256, 4096, 16384):
    e, z = M.comp_zq_stopped_zscore(cp.asarray(q), cp.asarray(b), cp.asarray(data.i_traj), cp.asarray(data.path_weights), dt,
                                    tau=tau, g_traj=cp.asarray(data.i_traj))
    z = cp.asnumpy(z)[5:]; e = cp.asnumpy(e)[5:5 + len(z)]; k = np.argmax(np.abs(z)); out.append(f'{z[k]:+5.1f} (q {e[k]:.2f})')
print('exact q, staple weights, worst z at lags 1 16 256 4096 16384:', '  '.join(out), flush=True)
