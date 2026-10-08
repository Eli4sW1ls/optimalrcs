"""Committor validation metrics for path-sampling segments that end at a stopping time.

(RE)TIS, REPPTIS and staple paths, cut at their first crossing of the ensemble's
interface M, end where the sampler stopped them: in A or B, at the first exit
through L or R, or when a staple turn completes. The standard Z_q drops every
window [t, t+dt] that runs past the end of its segment. Because the end is set
by the dynamics, the dropped windows are exactly those of frames about to exit,
which biases Z_q at every dt > 1, even for the exact committor.

Here a window is stopped instead of dropped: [t, t+dt] becomes
[t, min(t+dt, tau_t)], with tau_t the first frame >= t in A/B or at the end of
the segment. The change over the window splits into r(tau) - r(t), which is in
the data, and r(t+dt) - r(tau), which is not. For the committor the second part
has zero mean given the path up to tau (it is a fair bet from any point on), so
leaving it out does not bias the result. This needs only that the segment end
is decided by the path up to that point, and that the moves up to it are
unbiased (frames before the M crossing cut, Monte Carlo path weights applied).
The library already treats A and B this way; L/R and turn ends get r(tau)
instead of 0 or 1. When every segment ends in A or B, the stopped
observed-vs-predicted plot is the usual one.

The zero-mean property holds per frame, so the cut profile
Z(x, dt) = sum_{t: r(t) <= x} w_t [r(min(t+dt, tau_t)) - r(t)] / dt
is flat at zero for the committor at every dt. Segments are independent, so its
standard error is sqrt(sum over segments of (segment contribution)^2), and
Z / standard error stays within a few units of zero for the committor.
"""

import numpy as np

ldt0 = [2 ** i for i in range(16)]


def _np(x):
    return None if x is None else np.asarray(x)


def _segments(i_traj, n):
    """Segment number 0, 1, ... of every frame (one segment if `i_traj` is None)."""
    if i_traj is None:
        return np.zeros(n, dtype=np.int64)
    i_traj = _np(i_traj)
    return np.r_[0, np.cumsum(i_traj[1:] != i_traj[:-1])]


def stop_index(i_traj, b_traj):
    """
    Index of the frame where each frame's window stops.

    Parameters
    ----------
    i_traj : array_like or None
        Trajectory (segment) index of every frame; segments are contiguous.
        None means a single segment.
    b_traj : array_like
        Boundary indicator, nonzero in A or B.

    Returns
    -------
    np.ndarray
        tau[t]: the first frame >= t of the same segment that is in A/B or is
        the segment's last frame.
    """
    b_traj = _np(b_traj)
    n = len(b_traj)
    seg = _segments(i_traj, n)
    idx = np.arange(n)
    seg_end = np.r_[seg[1:] != seg[:-1], True]
    marker = np.where((b_traj != 0) | seg_end, idx, n)
    # A reverse running minimum never passes a segment's last frame, since that frame is marked.
    return np.minimum.accumulate(marker[::-1])[::-1]


def _stopped_contributions(r_traj, b_traj, w_traj, dt, nbins, tau):
    """Bin of each window start, its weighted stopped increment / dt, and which frames start windows."""
    n = len(r_traj)
    starts = b_traj == 0
    u = np.minimum(np.arange(n) + dt, tau)
    edges = np.linspace(0, 1, nbins + 1)
    bins = np.clip(np.searchsorted(edges, r_traj, side="right") - 1, 0, nbins - 1)
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
    order = np.argsort(seg * nbins + bins, kind='stable')
    seg_s, bin_s, c_s = seg[order], bins[order], contrib[order]
    cs = np.cumsum(c_s)
    first = np.r_[True, seg_s[1:] != seg_s[:-1]]
    before_seg = (cs - c_s)[first]                      # running total before each segment
    sq = (cs - before_seg[np.cumsum(first) - 1]) ** 2
    same_next = np.r_[seg_s[1:] == seg_s[:-1], False]
    bin_next = np.where(same_next, np.r_[bin_s[1:], 0], nbins)
    diff = (np.bincount(bin_s, weights=sq, minlength=nbins + 1)
            - np.bincount(bin_next, weights=sq, minlength=nbins + 1))
    return np.sqrt(np.maximum(np.cumsum(diff)[:nbins], 0))


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
    lx : np.ndarray
        Bin centers.
    zq : np.ndarray
        Z at the bin centers.
    """
    r_traj, b_traj, w_traj = _np(r_traj), _np(b_traj), _np(w_traj)
    tau = stop_index(i_traj, b_traj) if tau is None else _np(tau)
    lx, bins, contrib, starts = _stopped_contributions(r_traj, b_traj, w_traj, dt, nbins, tau)
    return lx, np.cumsum(np.bincount(bins[starts], weights=contrib[starts], minlength=nbins))


def comp_zq_stopped_zscore(r_traj, b_traj, i_traj, w_traj=None, dt=1, nbins=200, tau=None):
    """
    Stopped cut profile Z(x, dt) divided by its standard error.

    Segments are independent, so Var Z(x) is the sum over segments of the squared
    contribution of each segment to Z(x). For the committor the result stays
    within a few units of zero at every x and dt; parameters as in `comp_zq_stopped`.

    Returns
    -------
    lx : np.ndarray
        Bin centers.
    zscore : np.ndarray
        Z / standard error at the bin centers (0 where there is no data yet).
    """
    r_traj, b_traj, w_traj = _np(r_traj), _np(b_traj), _np(w_traj)
    tau = stop_index(i_traj, b_traj) if tau is None else _np(tau)
    lx, bins, contrib, starts = _stopped_contributions(r_traj, b_traj, w_traj, dt, nbins, tau)
    seg = _segments(i_traj, len(r_traj))
    zq = np.cumsum(np.bincount(bins[starts], weights=contrib[starts], minlength=nbins))
    sigma = _sigma_zq(seg[starts], bins[starts], contrib[starts], nbins)
    return lx, np.divide(zq, sigma, out=np.zeros(nbins), where=sigma > 0)


def _comp_max_zq_stopped(r_traj, b_traj, i_traj, w_traj=None, ldt=None, nbins=200, tau=None, skip_bins=5):
    """
    Worst deviation of the stopped cut profile over all lag times.

    The first `skip_bins` bins next to A are left out: there a few frames carry
    the whole profile and the standard error is unreliable.

    Returns
    -------
    tuple
        (max |z|, its dt, max standard deviation of Z over x, its dt).
    """
    if ldt is None:
        ldt = ldt0
    tau = stop_index(i_traj, b_traj) if tau is None else tau
    max_z = max_sd = (0, 0)
    for dt in ldt:
        _, z = comp_zq_stopped_zscore(r_traj, b_traj, i_traj, w_traj, dt, nbins, tau)
        _, zq = comp_zq_stopped(r_traj, b_traj, i_traj, w_traj, dt, nbins, tau)
        z_dt, sd_dt = np.max(np.abs(z[skip_bins:])), np.std(zq)
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
    b_traj, w_traj = _np(b_traj), _np(w_traj)
    n = len(b_traj)
    if future_boundary is not None:
        to_end = np.asarray(future_boundary.delta_i_to_end)
    else:
        seg = _segments(i_traj, n)
        idx = np.arange(n)
        to_end = np.minimum.accumulate(np.where(np.r_[seg[1:] != seg[:-1], True], idx, n)[::-1])[::-1] - idx
    starts = b_traj == 0
    w = np.ones(n) if w_traj is None else w_traj
    return float(np.sum(w[starts & (to_end < dt)]) / np.sum(w[starts]))


def comp_obs_pred_stopped(r_traj, b_traj, i_traj, w_traj=None, nbins=50, select=None, tau=None):
    """
    Observed vs predicted committor, with the outcome read where each frame's window stops.

    The outcome of frame t is r(tau_t): 0 or 1 if the segment reaches A or B,
    else r at the segment end. For the committor the binned mean outcome equals
    the binned mean prediction.

    Parameters
    ----------
    select : array_like of bool, optional
        Frames to include (e.g. a random half of the paths).
    Other parameters as in `comp_zq_stopped`.

    Returns
    -------
    pred, obs, weight : np.ndarray
        Weighted mean r(t), weighted mean r(tau_t) and total weight, per non-empty bin.
    """
    r_traj, b_traj, w_traj = _np(r_traj), _np(b_traj), _np(w_traj)
    tau = stop_index(i_traj, b_traj) if tau is None else _np(tau)
    w = np.ones_like(r_traj) if w_traj is None else w_traj
    keep = b_traj == 0
    if select is not None:
        keep &= _np(select)
    edges = np.linspace(0, 1, nbins + 1)
    bins = np.clip(np.searchsorted(edges, r_traj, side="right") - 1, 0, nbins - 1)[keep]
    sw = np.bincount(bins, weights=w[keep], minlength=nbins)
    pred = np.bincount(bins, weights=(w * r_traj)[keep], minlength=nbins)
    obs = np.bincount(bins, weights=(w * r_traj[tau])[keep], minlength=nbins)
    ok = sw > 0
    return pred[ok] / sw[ok], obs[ok] / sw[ok], sw[ok]


def _cached_max_zq_stopped(rc):
    """`_comp_max_zq_stopped` for the RC's current r_traj, computed once per r_traj."""
    if getattr(rc, "_tau_stopped", None) is None:
        rc._tau_stopped = stop_index(rc.i_traj, rc.b_traj)
    cache = getattr(rc, "_zq_stopped_cache", None)
    if cache is None or cache[0] is not rc.r_traj:
        stats = _comp_max_zq_stopped(rc.r_traj, rc.b_traj, rc.i_traj,
                                     getattr(rc, "path_weights", None), tau=rc._tau_stopped)
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
