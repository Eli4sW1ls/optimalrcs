"""Plots of the stopped-segment validation metrics in `metrics_tis`."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

from . import cut_profiles, metrics_tis


def plot_zq_stopped(ax, r_traj, b_traj, i_traj, w_traj=None, ldt=None, zscore=True, nbins=200,
                    xlabel='$q$', colorbar=True):
    """
    Stopped cut profiles for all lag times, one line per lag (light = short, dark = long).

    With `zscore` (default) each profile is divided by its standard error and a
    +-2 band is shaded: for an RC consistent with the committor all lines stay
    near that band at every q.
    """
    if ldt is None:
        ldt = metrics_tis.ldt0
    tau = metrics_tis.stop_index(i_traj, b_traj)
    cmap = plt.get_cmap('Blues')
    log_max = np.log2(max(ldt[-1], 2))
    if zscore:
        ax.axhspan(-2, 2, color='0.9', lw=0, zorder=0)
    for dt in ldt:
        if zscore:
            lx, ly = metrics_tis.comp_zq_stopped_zscore(r_traj, b_traj, i_traj, w_traj, dt, nbins, tau)
        else:
            lx, ly = metrics_tis.comp_zq_stopped(r_traj, b_traj, i_traj, w_traj, dt, nbins, tau)
        ax.plot(lx, ly, color=cmap(0.3 + 0.7 * np.log2(dt) / log_max), lw=1.1)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set(xlabel=xlabel, ylabel='stopped $Z_q$ / std. error' if zscore else 'stopped $Z_q$')
    ax.grid()
    if colorbar:
        cb = plt.colorbar(ScalarMappable(norm=LogNorm(ldt[0], max(ldt[-1], 2)), cmap=cmap), ax=ax)
        cb.set_label(r'lag $\Delta t$ (frames)')


def plot_zq_short_lags(ax, r_traj, b_traj, i_traj, future_boundary, past_boundary, w_traj=None,
                       ldt=(1, 2, 4, 8), xlabel='$q$'):
    """
    The standard (not stopped) Z_q of `cut_profiles.comp_zq` at a few short lag times.

    It drops windows that run past the end of their segment, which biases it for
    path segments that end at a stopping time; the bias grows with the fraction
    of dropped windows, shown for every lag in the legend. Use it while that
    fraction is a few percent; `plot_zq_stopped` is valid at every lag.
    """
    import tensorflow as tf
    r_np = r_traj.numpy() if tf.is_tensor(r_traj) else np.asarray(r_traj)
    cmap = plt.get_cmap('Blues')
    for k, dt in enumerate(ldt):
        lx, ly = cut_profiles.comp_zq(r_np, b_traj, i_traj, future_boundary, past_boundary,
                                      dt=tf.constant(dt), w_traj=w_traj)
        lx, ly = ((lx[1:] + lx[:-1]) / 2)[:-1].numpy(), ly[:-1].numpy()
        dropped = 100 * metrics_tis.dropped_fraction(i_traj, b_traj, dt, w_traj, future_boundary)
        ax.plot(lx, ly, color=cmap(0.4 + 0.6 * k / max(len(ldt) - 1, 1)), lw=1.2,
                label=rf'$\Delta t$={dt} ({dropped:.1f}% of windows dropped)')
    ax.set(xlabel=xlabel, ylabel='$Z_q$ (windows past a segment end dropped)')
    ax.legend(fontsize='small')
    ax.grid()


def plot_obs_pred_stopped(ax, r_traj, b_traj, i_traj, w_traj=None, nbins=50, halves=True, seed=0):
    """
    Mean outcome where each window stops (A -> 0, B -> 1, else r at the segment end)
    against the predicted r(t). With `halves`, also for two random halves of the
    segments, to show the noise.
    """
    tau = metrics_tis.stop_index(i_traj, b_traj)
    ax.plot((0, 1), (0, 1), '--k', label='obs = pred')
    pred, obs, _ = metrics_tis.comp_obs_pred_stopped(r_traj, b_traj, i_traj, w_traj, nbins, tau=tau)
    ax.plot(pred, obs, '-r', label='obs vs pred')
    if halves:
        seg = metrics_tis._segments(i_traj, len(tau))
        half = np.random.default_rng(seed).random(seg.max() + 1)[seg] < 0.5
        for sel, lt, label in ((half, ':g', 'random half 1'), (~half, ':b', 'random half 2')):
            pred, obs, _ = metrics_tis.comp_obs_pred_stopped(r_traj, b_traj, i_traj, w_traj, nbins,
                                                             select=sel, tau=tau)
            ax.plot(pred, obs, lt, label=label)
    ax.set(xlabel='$q$ predicted', ylabel='$q$ observed where the window stops')
    ax.legend()
    ax.grid()
