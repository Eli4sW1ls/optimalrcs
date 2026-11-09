"""Redraw the feps panels of saved path-data committors without and with the path weights.

Row 1: F(q), F(natural q) and Z_q (lags 1..2^15) without weights, as plots_feps drew them before
       it passed the path weights (the cupy plot_zq also never handed w_traj to comp_zq).
Row 2: the same panels with w_traj = path_weights, as plots_feps draws them now.
A/B ends are traps (2**16), as in the fits. Long lags stay biased either way: comp_zq drops the
windows that run past a piece end; the stopped Z_q (plots_tis) is the check for those.

Input: committor.npz from tis_data.plot_committor (key 'r') or from toy2d_staple_analysis.ipynb
(key 'q_fit'); both hold boundary0/1, i_traj, t_traj and path_weights. Saves feps_weighted.png
(or feps_weighted_<npz name>.png) next to each npz.

    python feps_weighted.py figures/istarz_br_g10_reversed/committor.npz [more.npz ...]

Run one file per process for the large runs (20M frames need ~10 GB host memory), memory-capped:
    systemd-run --user --wait --pipe -p MemoryMax=16G -p MemorySwapMax=0 python feps_weighted.py <npz>
"""
import os
import sys

import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'optimalrcs-cupy'))
import cut_profiles
import optimalrcs
import plots


def feps_weighted(npz):
    d = np.load(npz)
    r = d['r'] if 'r' in d.files else d['q_fit']
    q = optimalrcs.CommittorNE(boundary0=d['boundary0'], boundary1=d['boundary1'], i_traj=d['i_traj'],
                               t_traj=d['t_traj'], path_weights=d['path_weights'])
    q.set_fixed_traj_length_trap(d['boundary0'] | d['boundary1'], 2 ** 16)
    q.r_traj = cp.asarray(r)
    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    for row, w in enumerate((None, q.path_weights)):
        plots.plot_fep(ax[row, 0], q.r_traj, i_traj=q.i_traj, t_traj=q.t_traj, w_traj=w)
        plots.plot_fep(ax[row, 1], q.r_traj, i_traj=q.i_traj, t_traj=q.t_traj, w_traj=w, natural=True)
        for dt in plots.ldt0:
            lx, ly = cut_profiles.comp_zq(q.r_traj, q.b_traj, q.i_traj, q.future_boundary, q.past_boundary,
                                          w_traj=w, dt=dt)
            ax[row, 2].plot(lx.get()[:-2], ly.get()[:-1], lw=1, label=f'{dt}')
        ax[row, 2].set(ylabel='$Z_q$', xlabel='$q$')
        ax[row, 2].grid()
        tag = 'no weights (old plots_feps)' if w is None else 'with path weights'
        for a, name in zip(ax[row], ('F(q)', 'F(natural q)', 'Z_q, lags 1..2^15')):
            a.set_title(f'{name}: {tag}', fontsize=10)
    ax[1, 2].legend(fontsize=6, ncol=2, title='lag', title_fontsize=7)
    lo = min(ax[0, 2].get_ylim()[0], ax[1, 2].get_ylim()[0])
    hi = max(ax[0, 2].get_ylim()[1], ax[1, 2].get_ylim()[1])
    for a in ax[:, 2]:
        a.set_ylim(lo, hi)                      # same Z_q scale in both rows
    fig.suptitle(os.path.abspath(npz), fontsize=9)
    fig.tight_layout()
    stem = os.path.splitext(os.path.basename(npz))[0]
    name = 'feps_weighted' if stem == 'committor' else f'feps_weighted_{stem}'
    out = os.path.join(os.path.dirname(os.path.abspath(npz)), name + '.png')
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print('saved', out, flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for npz in sys.argv[1:]:
        feps_weighted(npz)
