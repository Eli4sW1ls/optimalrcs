"""Exact committor, mirrored-pieces staple data: stopped Z_q z-score per ensemble for the forward pieces
alone, the mirrored pieces alone, and both (lag 256)."""
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
counts = [2983] + [3000] * 9
ens = np.repeat(np.arange(1, 11), counts)[d.i_traj]
# within a path (group) the first piece is the forward one, a second piece the mirrored one
first_piece_of_group = {}
piece_is_fwd = np.zeros(d.i_traj.max() + 1, bool)
seg_start = np.r_[True, d.i_traj[1:] != d.i_traj[:-1]]
for piece, grp in zip(d.i_traj[seg_start], d.group_traj[seg_start]):
    if grp not in first_piece_of_group:
        first_piece_of_group[grp] = piece; piece_is_fwd[piece] = True
fwd = piece_is_fwd[d.i_traj]
print(f'forward pieces {np.unique(d.i_traj[fwd]).size}, mirrored pieces {np.unique(d.i_traj[~fwd]).size}', flush=True)
Q, B, I, G = cp.asarray(q), cp.asarray(b), cp.asarray(d.i_traj), cp.asarray(d.group_traj)
tau = M.stop_index(I, B)
def zline(mask, dt=256):
    e, z = M.comp_zq_stopped_zscore(Q, B, I, cp.asarray(d.path_weights * mask), dt, tau=tau, g_traj=G)
    z = cp.asnumpy(z); e = cp.asnumpy(e)[:len(z)]
    return ' '.join(f'{z[np.searchsorted(e, x)]:+6.1f}' for x in (0.15, 0.4, 0.62, 0.7, 0.8))
print('z of the exact committor at lag 256, at q = 0.15 0.4 0.62 0.7 0.8', flush=True)
for k in list(range(1, 11)) + ['all']:
    m = np.ones(len(q), bool) if k == 'all' else (ens == k)
    print(f'ensemble {k!s:>3}: forward {zline((m & fwd).astype(float))} | mirrored {zline((m & ~fwd).astype(float))} '
          f'| both {zline(m.astype(float))}', flush=True)
