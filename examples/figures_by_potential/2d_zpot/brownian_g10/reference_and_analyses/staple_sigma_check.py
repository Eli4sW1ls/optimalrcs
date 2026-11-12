"""Additivity and outlier check of the stopped Z_q of the exact committor, mirrored-pieces staple data."""
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
ref = np.load(REPO + '/examples/figures/z_brownian/committor_reference_h0.001.npz')
q = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[d.X_all[:, 0], d.X_all[:, 1]]), 0, 1)
q[d.boundary0] = 0; q[d.boundary1] = 1
Q, B, I, G = cp.asarray(q), cp.asarray(b), cp.asarray(d.i_traj), cp.asarray(d.group_traj)
tau = M.stop_index(I, B)
dt, nb = 256, 200
lx, bins, contrib, starts = M._stopped_contributions(Q, B, cp.asarray(d.path_weights), dt, nb, tau)
odd = (G % 2) == 1
def zs(mask):
    m = starts & mask
    Z = cp.cumsum(cp.bincount(bins[m], weights=contrib[m], minlength=nb))
    S = M._sigma_zq(G[m], bins[m], contrib[m], nb)
    return cp.asnumpy(Z), cp.asnumpy(S)
Za, Sa = zs(cp.ones_like(starts)); Ze, Se = zs(~odd); Zo, So = zs(odd)
lxn = cp.asnumpy(lx)
for qq in (0.15, 0.3, 0.5, 0.65, 0.75):
    k = np.searchsorted(lxn, qq)
    print(f'q {qq}: Z all {Za[k]:+9.3f} (even+odd {Ze[k] + Zo[k]:+9.3f}) | sigma all {Sa[k]:8.3f} '
          f'(sqrt(e^2+o^2) {np.hypot(Se[k], So[k]):8.3f}) | even {Ze[k]:+8.3f}/{Se[k]:.3f} odd {Zo[k]:+8.3f}/{So[k]:.3f}', flush=True)
# per-path contribution up to q = 0.65: the largest |S_g|
m = cp.asnumpy(starts & (Q <= 0.65))
g = d.group_traj[m]; c = cp.asnumpy(contrib)[m]
Sg = np.bincount(g, weights=c)
order = np.argsort(-np.abs(Sg))
print(f'paths with frames below q 0.65: {np.count_nonzero(Sg)}; sum of S_g^2 {np.sum(Sg ** 2):.3f}; '
      f'top 5 |S_g| {np.round(Sg[order[:5]], 3).tolist()} carry {np.sum(Sg[order[:5]] ** 2) / np.sum(Sg ** 2):.2f} of the variance', flush=True)
w_of_g = np.bincount(d.group_traj, weights=d.path_weights) / np.maximum(np.bincount(d.group_traj), 1)
n_of_g = np.bincount(d.group_traj)
print('their weights', np.round(w_of_g[order[:5]], 2).tolist(), 'frames', n_of_g[order[:5]].tolist(), flush=True)
