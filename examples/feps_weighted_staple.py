"""Regenerate the weighted feps plots of the staple_compare fits from their saved q_traj.npy.

    python feps_weighted_staple.py figures/staple_compare/sim_istar_cc_br_p2000_s0 [name ...]

q_traj.npy holds only q, so the frames are re-read with the OLD tis_data (random subset, seed 0),
taken from git; the current tis_data (burn-in + thinning) selects other paths and the length check fails.
Writes <out>/<name>/seed0/feps_weighted.png.
"""
import argparse, importlib.util, os, subprocess, sys
import cupy as cp
import matplotlib; matplotlib.use('Agg')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'optimalrcs-cupy')); sys.path.insert(0, HERE)
import optimalrcs, toy2d
src = subprocess.check_output(['git', 'show', 'f18c794:examples/tis_data.py'], cwd=HERE).decode()
open(f'{HERE}/tis_data_old.py', 'w').write(src)
import tis_data_old as tis_data

OUT, *names = sys.argv[1:] or sys.exit(__doc__)
TIS = f'{toy2d.TOYTIS}/simulations/sim_istar_cc_br'          # 2d_cc preset of the notebook
CFG = {'fwd_rej': '', 'rev_rej': '--reversed', 'fwd_staple': '--weights staple', 'rev_staple': '--weights staple --reversed',
       'fwd_none': '--weights none', 'rev_none': '--weights none --reversed', 'nocut_rej': '--no-cut',
       'rev_rej_trunc500': '--reversed --max-length 500 --long-pieces truncate',
       'rev_rej_disc500': '--reversed --max-length 500 --long-pieces discard'}
for name in names or CFG:
    p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
    a = p.parse_args([TIS, '--lambda-a', '0.15', '--lambda-b', '0.85', '--paths-per-ensemble', '2000', '--seed', '0'] + CFG[name].split())
    d = tis_data.prepare_data(a)
    r = np.load(f'{OUT}/{name}/seed0/q_traj.npy')
    assert len(r) == len(d.i_traj), f'{name}: {len(r)} saved vs {len(d.i_traj)} loaded frames: wrong tis_data/paths'
    q = optimalrcs.CommittorNE(boundary0=d.boundary0, boundary1=d.boundary1, i_traj=d.i_traj, t_traj=d.t_traj,
                               path_weights=d.path_weights, group_traj=d.group_traj)
    q.set_fixed_traj_length_trap(d.boundary0 | d.boundary1, 2 ** 16)
    q.r_traj = cp.asarray(r, dtype=cp.float64)
    q.plots_feps(); import matplotlib.pyplot as plt
    plt.gcf().savefig(f'{OUT}/{name}/seed0/feps_weighted.png', dpi=100); plt.close('all'); print('saved', name, flush=True)
    del q, d; cp.get_default_memory_pool().free_all_blocks()
