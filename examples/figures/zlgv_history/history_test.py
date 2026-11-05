"""Coarser histories on the 2D Langevin Z staple run (sim_istarz_2603): same settings as
base_documented_2d_langevin.ipynb (1000 paths/ensemble, no mirrored pieces, gamma 0.2,
envelope 0.001, ny 6, shift r(t)), only the history lags change."""
import argparse, io, contextlib, os, sys, time
import numpy as np, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, f'{REPO}/optimalrcs-cupy/'); sys.path.insert(0, f'{REPO}/examples/')
import optimalrcs, tis_data
plt.show = lambda *a, **k: None
OUT = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
args = p.parse_args(['/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_2603',
                     '--paths-per-ensemble', '1000', '--seed', '0', '--n-cvs', '2'])
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(args)
X = cp.asarray(data.X)
for history in ([0, 4], [0, 8], [0, 16], [0, 1, 4, 16]):
    optimalrcs.envelope_scale = 0.001
    np.random.seed(0); cp.random.seed(0)
    q = optimalrcs.CommittorNE(boundary0=data.boundary0, boundary1=data.boundary1, i_traj=data.i_traj,
                               t_traj=data.t_traj, path_weights=data.path_weights, group_traj=data.group_traj)
    q.set_fixed_traj_length_trap(data.boundary0 | data.boundary1, 2 ** 16)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        q.fit_transform(lambda: X[:, np.random.randint(2)], history_delta_t=history, history_type=['y(t-d),y(t)'],
                        history_shift_type='r(t)', gamma=0.2, ny=6, max_iter=2000, print_step=250,
                        metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped', 'delta_x'),
                        save_min_metric='max_z_zq_stopped', lazy_basis=True)
    h = q.metrics_history
    tag = "_".join(map(str, history))
    print(f"history {history}: zqs_z by print step {np.round(h['max_z_zq_stopped'], 1).tolist()} | "
          f"final |dx| {h['delta_x'][-1]:.1e} | {time.time() - t0:.0f} s", flush=True)
    q.plots_tis(); plt.gcf().savefig(f"{OUT}/tis_h{tag}.png", dpi=110); plt.close("all")
    np.save(f"{OUT}/q_h{tag}.npy", cp.asnumpy(q.r_traj))
    del q; cp.get_default_memory_pool().free_all_blocks()
print("ALL DONE", flush=True)
