"""Exact committor vs the staple data WITH mirrored pieces (rejection weights, 1500 paths per ensemble,
seed 0): stopped Z_q z-score split by ensemble and by odd/even path."""
import sys, argparse, io, contextlib, numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import metrics_tis as M, tis_data
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    d = tis_data.prepare_data(p.parse_args(['/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10',
                                             '--reversed', '--paths-per-ensemble', '1500', '--seed', '0']))
b = d.boundary0 | d.boundary1
ref = np.load(REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/committor_reference_h0.001.npz')
q = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[d.X_all[:, 0], d.X_all[:, 1]]), 0, 1)
q[d.boundary0] = 0; q[d.boundary1] = 1
counts = [2983] + [3000] * 9                                   # pieces per ensemble 001..010 (run.log)
ens = np.repeat(np.arange(1, 11), counts)[d.i_traj]
Q, B, I, G = cp.asarray(q), cp.asarray(b), cp.asarray(d.i_traj), cp.asarray(d.group_traj)
tau = M.stop_index(I, B)
LAGS = (16, 256, 4096)
def worst(mask):
    out = []
    for dt in LAGS:
        e, z = M.comp_zq_stopped_zscore(Q, B, I, cp.asarray(d.path_weights * mask), dt, tau=tau, g_traj=G)
        z = cp.asnumpy(z)[5:]; e = cp.asnumpy(e)[5:5 + len(z)]; k = np.argmax(np.abs(z)); out.append(f'{z[k]:+5.1f} (q {e[k]:.2f})')
        sp = (e > 0.55) & (e < 0.85); out[-1] += f' pl {z[sp][np.argmax(np.abs(z[sp]))]:+4.1f}'
    return '   '.join(out)
print(f'exact committor, with mirrored pieces; worst z (and worst on the plateau, "pl") at lags {LAGS}', flush=True)
print(f'all           : {worst(np.ones(len(q)))}', flush=True)
for k in range(1, 11):
    print(f'ensemble {k:03d}  : {worst((ens == k).astype(float))}', flush=True)
odd = (d.group_traj % 2).astype(float)
print(f'even paths    : {worst(1 - odd)}', flush=True)
print(f'odd paths     : {worst(odd)}', flush=True)
