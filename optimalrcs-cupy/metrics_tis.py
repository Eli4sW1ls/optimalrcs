"""Committor validation metrics for path-sampling segments that end at a stopping time (CuPy).

CuPy counterpart of optimalrcs/metrics_tis.py; see there for the reasoning.
(RE)TIS, REPPTIS and staple segments, cut at their first M crossing, end where
the sampler stopped them (A/B, L/R exit, completed turn). The standard Z_q drops
windows that run past a segment end, which biases it; here each window
[t, t+dt] is stopped at tau_t, the first frame >= t in A/B or at the end of the
segment, instead. For the committor the stopped increments have zero mean, so
Z(x, dt) = sum_{t: r(t) <= x} w_t [r(min(t+dt, tau_t)) - r(t)] / dt is flat at
zero, and Z / standard error stays within a few units of zero.
"""

import cupy as cp
import numpy as np

ldt0 = [2 ** i for i in range(16)]


def _cp(x):
    return None if x is None else cp.asarray(x)


def _segments(i_traj, n):
    """Segment number 0, 1, ... of every frame (one segment if `i_traj` is None)."""
    if i_traj is None:
        return cp.zeros(n, dtype=cp.int64)
    i_traj = cp.asarray(i_traj)
    return cp.concatenate([cp.zeros(1, dtype=cp.int64),
                           cp.cumsum(i_traj[1:] != i_traj[:-1], dtype=cp.int64)])


def _groups(g_traj, i_traj, n):
    """Independence group 0, 1, ... of every frame: `g_traj` if given, else the segments."""
    if g_traj is None:
        return _segments(i_traj, n)
    return cp.asarray(np.unique(cp.asnumpy(cp.asarray(g_traj)), return_inverse=True)[1].reshape(-1))


def stop_index(i_traj, b_traj):
    """
    Index of the frame where each frame's window stops.

    tau[t] is the first frame >= t of the same segment that is in A/B (b_traj != 0)
    or is the segment's last frame. `i_traj` None means a single segment. Computed
    once on the CPU (a reverse running minimum) and returned as a CuPy array.
    """
    b_traj = cp.asnumpy(cp.asarray(b_traj))
    n = len(b_traj)
    seg = cp.asnumpy(_segments(i_traj, n))
    idx = np.arange(n)
    seg_end = np.r_[seg[1:] != seg[:-1], True]
    marker = np.where((b_traj != 0) | seg_end, idx, n)
    # A reverse running minimum never passes a segment's last frame, since that frame is marked.
    return cp.asarray(np.minimum.accumulate(marker[::-1])[::-1])


def _stopped_contributions(r_traj, b_traj, w_traj, dt, nbins, tau):
    """Bin of each window start, its weighted stopped increment / dt, and which frames start windows."""
    n = r_traj.shape[0]
    starts = b_traj == 0
    u = cp.minimum(cp.arange(n) + dt, tau)
    edges = cp.linspace(0, 1, nbins + 1)
    bins = cp.clip(cp.searchsorted(edges, r_traj, side='right') - 1, 0, nbins - 1)
    contrib = (r_traj[u] - r_traj) / dt
    if w_traj is not None:
        contrib = contrib * w_traj
    return (edges[1:] + edges[:-1]) / 2, bins, contrib, starts


def _sigma_zq(seg, bins, contrib, nbins):
    """
    Standard error of Z(x) = cumsum over bins of the contributions.

    Var Z(x) = sum over segments of S_seg(x)^2, with S_seg(x) the segment's own
    running sum up to bin x. Sorted by (segment, bin), the running sum after
    entry k holds from its bin up to the bin of the segment's next entry, so the
    squares are added over those bin ranges; memory stays O(frames).
    """
    order = cp.argsort(seg * nbins + bins)
    seg_s, bin_s, c_s = seg[order], bins[order], contrib[order]
    cs = cp.cumsum(c_s)
    first = cp.concatenate([cp.ones(1, dtype=bool), seg_s[1:] != seg_s[:-1]])
    before_seg = (cs - c_s)[first]                      # running total before each segment
    sq = (cs - before_seg[cp.cumsum(first) - 1]) ** 2
    same_next = cp.concatenate([seg_s[1:] == seg_s[:-1], cp.zeros(1, dtype=bool)])
    bin_next = cp.where(same_next, cp.concatenate([bin_s[1:], cp.zeros(1, dtype=bin_s.dtype)]), nbins)
    diff = (cp.bincount(bin_s, weights=sq, minlength=nbins + 1)
            - cp.bincount(bin_next, weights=sq, minlength=nbins + 1))
    return cp.sqrt(cp.maximum(cp.cumsum(diff)[:nbins], 0))


def comp_zq_stopped(r_traj, b_traj, i_traj, w_traj=None, dt=1, nbins=200, tau=None):
    """
    Stopped cut profile Z(x, dt), flat at zero for the committor at every dt.

    Parameters
    ----------
    r_traj : array_like
        Reaction coordinate time-series.
    b_traj : array_like
        Boundary indicator, nonzero in A or B.
    i_traj : array_like or None
        Segment index of every frame; segments are contiguous. None means one segment.
    w_traj : array_like, optional
        Per-frame weight (the Monte Carlo weight of the frame's path).
    dt : int, optional
        Lag time in frames (default: 1).
    nbins : int, optional
        Number of bins of r on [0, 1] (default: 200).
    tau : array_like, optional
        Precomputed `stop_index(i_traj, b_traj)`.

    Returns
    -------
    lx, zq : cp.ndarray
        Bin centers and Z at the bin centers.
    """
    r_traj, b_traj, w_traj = _cp(r_traj), _cp(b_traj), _cp(w_traj)
    tau = stop_index(i_traj, b_traj) if tau is None else _cp(tau)
    lx, bins, contrib, starts = _stopped_contributions(r_traj, b_traj, w_traj, dt, nbins, tau)
    return lx, cp.cumsum(cp.bincount(bins[starts], weights=contrib[starts], minlength=nbins))


def comp_zq_stopped_zscore(r_traj, b_traj, i_traj, w_traj=None, dt=1, nbins=200, tau=None, g_traj=None):
    """
    Stopped cut profile Z(x, dt) divided by its standard error.

    For the committor the result stays within a few units of zero at every x and
    dt; parameters as in `comp_zq_stopped`, and `g_traj` (optional): the
    independence group of every frame for the standard error (default: the
    segments); pieces of one path that share frames, like its forward and
    time-reversed piece, must share a group.

    Returns
    -------
    lx, zscore : cp.ndarray
        Bin centers and Z / standard error (0 where there is no data yet).
    """
    r_traj, b_traj, w_traj = _cp(r_traj), _cp(b_traj), _cp(w_traj)
    tau = stop_index(i_traj, b_traj) if tau is None else _cp(tau)
    lx, bins, contrib, starts = _stopped_contributions(r_traj, b_traj, w_traj, dt, nbins, tau)
    grp = _groups(g_traj, i_traj, r_traj.shape[0])
    zq = cp.cumsum(cp.bincount(bins[starts], weights=contrib[starts], minlength=nbins))
    sigma = _sigma_zq(grp[starts], bins[starts], contrib[starts], nbins)
    return lx, cp.where(sigma > 0, zq / cp.where(sigma > 0, sigma, 1), 0)


def _comp_max_zq_stopped(r_traj, b_traj, i_traj, w_traj=None, ldt=None, nbins=200, tau=None, skip_bins=5,
                         g_traj=None):
    """
    Worst deviation of the stopped cut profile over all lag times.

    The first `skip_bins` bins next to A are left out: there a few frames carry
    the whole profile and the standard error is unreliable.

    Returns
    -------
    tuple of float
        (max |z|, its dt, max standard deviation of Z over x, its dt).
    """
    if ldt is None:
        ldt = ldt0
    tau = stop_index(i_traj, b_traj) if tau is None else tau
    max_z = max_sd = (0, 0)
    for dt in ldt:
        _, z = comp_zq_stopped_zscore(r_traj, b_traj, i_traj, w_traj, dt, nbins, tau, g_traj)
        _, zq = comp_zq_stopped(r_traj, b_traj, i_traj, w_traj, dt, nbins, tau)
        z_dt, sd_dt = float(cp.max(cp.abs(z[skip_bins:]))), float(cp.std(zq))
        if z_dt > max_z[0]:
            max_z = z_dt, dt
        if sd_dt > max_sd[0]:
            max_sd = sd_dt, dt
    return max_z[0], max_z[1], max_sd[0], max_sd[1]


def dropped_fraction(i_traj, b_traj, dt, w_traj=None, future_boundary=None):
    """
    Weighted fraction of the windows the standard Z_q drops at lag `dt`.

    `cut_profiles.comp_zq` leaves out every non-boundary frame t whose window
    [t, t+dt] runs past the end of its segment, as given by
    `future_boundary.delta_i_to_end` (also when it reaches A or B first, since
    segments end at their first A/B frame, unless those ends are made traps with
    `set_fixed_traj_length_trap`). Without `future_boundary` the plain distance
    to the segment's last frame is used. For segments that end at a stopping
    time the dropped windows are those of frames about to exit, so the standard
    Z_q is only trustworthy while this fraction is a few percent.
    """
    b_traj = cp.asarray(b_traj)
    n = b_traj.shape[0]
    if future_boundary is not None:
        to_end = cp.asarray(future_boundary.delta_i_to_end)
    else:
        seg = cp.asnumpy(_segments(i_traj, n))
        idx = np.arange(n)
        to_end = cp.asarray(np.minimum.accumulate(np.where(np.r_[seg[1:] != seg[:-1], True], idx, n)[::-1])[::-1]
                            - idx)
    starts = b_traj == 0
    w = cp.ones(n) if w_traj is None else cp.asarray(w_traj)
    return float(cp.sum(w[starts & (to_end < dt)]) / cp.sum(w[starts]))


def comp_obs_pred_stopped(r_traj, b_traj, i_traj, w_traj=None, nbins=50, select=None, tau=None):
    """
    Observed vs predicted committor, with the outcome read where each frame's window stops.

    The outcome of frame t is r(tau_t): 0 or 1 if the segment reaches A or B,
    else r at the segment end. `select` (bool, optional) restricts the frames;
    other parameters as in `comp_zq_stopped`.

    Returns
    -------
    pred, obs, weight : cp.ndarray
        Weighted mean r(t), weighted mean r(tau_t) and total weight, per non-empty bin.
    """
    r_traj, b_traj, w_traj = _cp(r_traj), _cp(b_traj), _cp(w_traj)
    tau = stop_index(i_traj, b_traj) if tau is None else _cp(tau)
    w = cp.ones_like(r_traj) if w_traj is None else w_traj
    keep = b_traj == 0
    if select is not None:
        keep &= cp.asarray(select)
    edges = cp.linspace(0, 1, nbins + 1)
    bins = cp.clip(cp.searchsorted(edges, r_traj, side='right') - 1, 0, nbins - 1)[keep]
    sw = cp.bincount(bins, weights=w[keep], minlength=nbins)
    pred = cp.bincount(bins, weights=(w * r_traj)[keep], minlength=nbins)
    obs = cp.bincount(bins, weights=(w * r_traj[tau])[keep], minlength=nbins)
    ok = sw > 0
    return pred[ok] / sw[ok], obs[ok] / sw[ok], sw[ok]


def _cached_max_zq_stopped(rc):
    """`_comp_max_zq_stopped` for the RC's current r_traj, computed once per r_traj."""
    if getattr(rc, '_tau_stopped', None) is None:
        rc._tau_stopped = stop_index(rc.i_traj, rc.b_traj)
    cache = getattr(rc, '_zq_stopped_cache', None)
    if cache is None or cache[0] is not rc.r_traj:
        stats = _comp_max_zq_stopped(rc.r_traj, rc.b_traj, rc.i_traj, getattr(rc, 'path_weights', None),
                                     tau=rc._tau_stopped, g_traj=getattr(rc, 'group_traj', None))
        rc._zq_stopped_cache = (rc.r_traj, stats)
    return rc._zq_stopped_cache[1]


def max_z_zq_stopped(rc):
    """
    Worst |Z / standard error| of the stopped cut profile over x and lag times.

    The convergence criterion for segment data: about 2 or less for an RC that
    is consistent with the committor.
    """
    return _cached_max_zq_stopped(rc)[0]


def max_sd_zq_stopped(rc):
    """Worst (over lag times) standard deviation across x of the stopped cut profile."""
    return _cached_max_zq_stopped(rc)[2]


metric2function = {'max_z_zq_stopped': max_z_zq_stopped, 'max_sd_zq_stopped': max_sd_zq_stopped}

metrics_short_name = {'max_z_zq_stopped': 'zqs_z', 'max_sd_zq_stopped': 'sdzqs'}
