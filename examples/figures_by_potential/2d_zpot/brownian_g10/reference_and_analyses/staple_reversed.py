"""Exact committor vs the staple data, forward pieces only vs forward + mirrored pieces, with
rejection and staple weights. 1500 paths per ensemble (seed 0), so all variants use the same paths."""
import sys, argparse, io, contextlib, numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import metrics_tis as M, tis_data
ref = np.load(REPO + '/examples/figures/z_brownian/committor_reference_h0.001.npz')
f = RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10'
for weights in ('rejections', 'staple'):
    for rev in (False, True):
        p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
        argv = [SIM, '--weights', weights, '--paths-per-ensemble', '1500', '--seed', '0'] + (['--reversed'] if rev else [])
        with contextlib.redirect_stdout(io.StringIO()):
            d = tis_data.prepare_data(p.parse_args(argv))
        b = d.boundary0 | d.boundary1
        q = np.clip(f(np.c_[d.X_all[:, 0], d.X_all[:, 1]]), 0, 1); q[d.boundary0] = 0; q[d.boundary1] = 1
        Q, B, I, W, G = (cp.asarray(a) for a in (q, b, d.i_traj, d.path_weights, d.group_traj))
        tau = M.stop_index(I, B)
        plat, near = [], []
        for dt in (1, 16, 256, 4096):
            e, z = M.comp_zq_stopped_zscore(Q, B, I, W, dt, tau=tau, g_traj=G)
            z = cp.asnumpy(z); e = cp.asnumpy(e)[:len(z)]
            sn = ((e > 0.02) & (e < 0.1)) | (e > 0.9); sp = (e > 0.55) & (e < 0.85)
            near.append(z[sn][np.argmax(np.abs(z[sn]))]); plat.append(z[sp][np.argmax(np.abs(z[sp]))])
        print(f'{weights:10s} {"forward + mirrored" if rev else "forward only      "} ({len(q) / 1e6:5.1f}M frames): plateau '
              + ' '.join(f'{v:+5.1f}' for v in plat) + ' | near A/B ' + ' '.join(f'{v:+5.1f}' for v in near), flush=True)
        del Q, B, I, W, G, tau, d; cp.get_default_memory_pool().free_all_blocks()
