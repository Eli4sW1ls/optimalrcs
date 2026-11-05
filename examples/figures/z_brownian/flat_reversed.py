"""1D flat overdamped staple run (sim_istar0br10intf): exact committor vs forward / forward + mirrored
pieces, rejection and staple weights. 6000 paths per ensemble (seed 0)."""
import sys, argparse, io, contextlib, numpy as np, cupy as cp
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import metrics_tis as M, tis_data
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istar0br10intf'
la, lb, s = -0.1, 0.8, 0.5826 * np.sqrt(2 * 1.0 * 2e-4 / 1.0)
for weights in ('rejections', 'staple'):
    for rev in (False, True):
        p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
        argv = [SIM, '--weights', weights, '--paths-per-ensemble', '6000', '--seed', '0'] + (['--reversed'] if rev else [])
        with contextlib.redirect_stdout(io.StringIO()):
            d = tis_data.prepare_data(p.parse_args(argv))
        b = d.boundary0 | d.boundary1
        q = np.clip((d.lam - la + s) / (lb - la + 2 * s), 0, 1); q[d.boundary0] = 0; q[d.boundary1] = 1
        Q, B, I, W, G = (cp.asarray(a) for a in (q, b, d.i_traj, d.path_weights, d.group_traj))
        tau = M.stop_index(I, B); zs = []
        for dt in (1, 16, 256, 4096):
            e, z = M.comp_zq_stopped_zscore(Q, B, I, W, dt, tau=tau, g_traj=G)
            z = cp.asnumpy(z)[5:]; e = cp.asnumpy(e)[5:5 + len(z)]; k = np.argmax(np.abs(z)); zs.append(f'{z[k]:+5.1f} (q {e[k]:.2f})')
        print(f'{weights:10s} {"forward + mirrored" if rev else "forward only      "} ({len(q) / 1e6:4.1f}M frames): ' + '  '.join(zs), flush=True)
        del Q, B, I, W, G, tau, d; cp.get_default_memory_pool().free_all_blocks()
