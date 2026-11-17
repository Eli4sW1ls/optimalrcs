"""The notebook's fit (A/B traps + stopped ends, reversed pieces) on several runs; RMS and profile of fit - exact.
    python fit_runs.py OUT RUN:WEIGHTING [RUN:WEIGHTING ...]
"""
import argparse, contextlib, io, os, sys, time
import numpy as np
import cupy as cp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

EX = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples'
sys.path.insert(0, f'{EX}/../optimalrcs-cupy/'); sys.path.insert(0, EX)
import optimalrcs, tis_data

A, B, SIG = -0.1, 0.3, 0.00894
shift = 0.5826 * SIG
qx = lambda x: np.clip((x - A + shift) / (B - A + 2 * shift), 0, 1)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
edges = np.linspace(A, B, 41); mid = (edges[1:] + edges[:-1]) / 2
prof = {}
for spec in sys.argv[2:]:
    run, w = spec.split(':')
    p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
    with contextlib.redirect_stdout(io.StringIO()):
        d = tis_data.prepare_data(p.parse_args([run, '--burn-in-fraction', '0.3', '--weights', w, '--lambda-a', str(A),
                                                 '--lambda-b', str(B), '--paths-per-ensemble', '5000', '--reversed']))
    b_all = d.boundary0 | d.boundary1
    piece_end = np.r_[np.flatnonzero(np.diff(d.i_traj)), len(d.i_traj) - 1]
    stop_end = np.zeros(len(b_all), bool); stop_end[piece_end] = ~b_all[piece_end]
    longest = int(np.bincount(d.i_traj - d.i_traj.min()).max())
    np.random.seed(0); cp.random.seed(0)
    q = optimalrcs.CommittorNE(boundary0=d.boundary0, boundary1=d.boundary1, i_traj=d.i_traj, t_traj=d.t_traj,
                               path_weights=d.path_weights, group_traj=d.group_traj)
    q.set_fixed_traj_length_trap(b_all, max(2 ** 16, longest + 2 ** 15 + 1))
    q.set_stopped_traj_end(stop_end)
    X = cp.asarray(d.X)
    kw = dict(ny=6, print_step=1000, lazy_basis=True, save_min_metric='max_z_zq_stopped',
              metrics_print=('iter', 'max_z_zq_stopped'))
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        optimalrcs.envelope_scale = 1
        q.fit_transform(lambda: X[:, np.random.randint(X.shape[1])], gamma=0.1, max_iter=1000, **kw)
        optimalrcs.envelope_scale = 0.01
        q.fit_transform(lambda: X[:, np.random.randint(X.shape[1])], gamma=0.1, max_iter=1000, **kw)
    r = cp.asnumpy(q.r_traj)
    zq = float(q.metrics_history['max_z_zq_stopped'][-1])
    del q, X; cp.get_default_memory_pool().free_all_blocks()
    qe = qx(d.lam); fr = ~b_all
    bi = np.clip(np.digitize(d.lam, edges) - 1, 0, len(mid) - 1)
    W = d.path_weights
    num = np.bincount(bi[fr], (W * (r - qe))[fr], len(mid)); den = np.bincount(bi[fr], W[fr], len(mid))
    prof[spec] = num / den
    name = os.path.basename(run)
    print(f'{name:8s} {w:10s}: {len(np.unique(d.group_traj))} paths, {len(r)} frames, zqs_z {zq:.2f}, '
          f'max(fit-exact) per bin {prof[spec].max():+.4f}, RMS {np.sqrt(np.mean(prof[spec] ** 2)):.4f} '
          f'({time.time() - t0:.0f} s)', flush=True)
    np.savez(f'{OUT}/prof.npz', mid=mid, **{k.replace('/', '_'): v for k, v in prof.items()})
fig, ax = plt.subplots(figsize=(8, 4.5))
for k, v in prof.items():
    run, w = k.split(':')
    ax.plot(mid, v, label=f'{os.path.basename(run)} {w}')
ax.axhline(0, color='k', lw=0.8); ax.set(xlabel='lambda', ylabel='fit - exact (per bin)'); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/profiles.png', dpi=110)
print('ALL DONE')
