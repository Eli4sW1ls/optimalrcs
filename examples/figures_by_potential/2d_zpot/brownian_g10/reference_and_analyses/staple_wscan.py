"""Scan the high-acceptance factor f for shot LML/RMR staple paths: exact committor vs the staple data."""
import sys, argparse, io, contextlib, numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import metrics_tis as M, tis_data
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    ds = tis_data.prepare_data(p.parse_args(['/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10', '--weights', 'staple']))
old = np.load(REPO + '/examples/figures/istarz_br_g10/committor.npz')
assert np.array_equal(old['lam'], ds.lam)
w_rej, it = old['path_weights'], old['i_traj']
ratio = ds.path_weights / w_rej
lo = ratio.min(); ha = ratio > 1.5 * lo                       # frames of paths that got the factor 2
print(f'ratio values: {np.unique(np.round(ratio / lo, 3))[:6]}; HA frames {ha.mean():.3f}', flush=True)
counts = [2976, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 2999, 3000]
paths = np.unique(it); ens = np.repeat(np.arange(1, 11), counts)[np.searchsorted(paths, it)]
first = np.r_[True, it[1:] != it[:-1]]
for k in range(1, 11):
    m = first & (ens == k)
    print(f'  ensemble {k:03d}: {ha[m].mean():.2f} of paths HA, their share of the rejection weight {np.sum(w_rej[m & ha]) / np.sum(w_rej[m]):.2f}', flush=True)
y, x = old['X_all'][:, 0], old['X_all'][:, 1]; b0, b1 = old['boundary0'], old['boundary1']; b = b0 | b1
ref = np.load(REPO + '/examples/figures/z_brownian/committor_reference_h0.001.npz')
q = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)
q[b0] = 0; q[b1] = 1
tau = M.stop_index(cp.asarray(it), cp.asarray(b)); Q, B, I = cp.asarray(q), cp.asarray(b), cp.asarray(it)
def test(w, label):
    near, plat = [], []
    for dt in (1, 16, 256, 4096):
        e, z = M.comp_zq_stopped_zscore(Q, B, I, cp.asarray(w), dt, tau=tau, g_traj=I)
        z = cp.asnumpy(z); e = cp.asnumpy(e)[:len(z)]
        sel_n = ((e > 0.02) & (e < 0.1)) | (e > 0.9); sel_p = (e > 0.55) & (e < 0.85)
        near.append(z[sel_n][np.argmax(np.abs(z[sel_n]))]); plat.append(z[sel_p][np.argmax(np.abs(z[sel_p]))])
    print(f'{label:34s}: plateau worst z ' + ' '.join(f'{v:+5.1f}' for v in plat)
          + ' | near A/B ' + ' '.join(f'{v:+5.1f}' for v in near), flush=True)
print('lags 1 16 256 4096', flush=True)
for f in (0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0):
    test(w_rej * np.where(ha, f, 1.0), f'HA factor {f}')
test(w_rej * np.where(ha & (ens > 1), 2.0, 1.0), 'HA factor 2, not in ensemble 001')
test(w_rej * np.where(ha & (ens == 1), 2.0, 1.0), 'HA factor 2 only in ensemble 001')
