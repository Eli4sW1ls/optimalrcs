"""Step 4: metrics of a candidate order parameter on whole held-out paths.

A candidate is given by its raw score z (any strictly increasing function of lambda gives the same crossings
and turns) and its calibrated committor m = g(z) (`calibrate`). Every metric uses the MC path weights of the
frames; the stopped Z_q follows the notebooks (toy2d.z_per_lag): w_traj = path weights, g_traj = path,
A/B frames 0/1, first 5 bins skipped.

- isosurface spread: planned interfaces at q_k equally spaced in ln q; slab k holds the frames with
  |logit m - logit q_k| < slab_half_logit; spread_k = weighted variance of logit q_ref in slab k (a lambda whose
  isosurfaces are isocommittors scores about slab_half_logit^2 / 3).
- smoothness: step ratio = sum w (d logit m)^2 / sum w (d logit q_np)^2 over lag-1 pairs; increment RMSE of
  d logit m - d logit q_np per lag; sign-change ratio = rate of sign changes of consecutive lag-1 increments of
  z, over that of logit q_np; and a replay of the sampler's turn rule (pgmoves.turn_detected) with the planned
  interfaces: turns and interface crossings per 1000 frames.
- dynamical: stopped Z_q of m (max |Z/s.e.| and SD over q, per lag) and the stopped Delta r^2 (lags 1-8).
- static: Spearman correlation with q_np, binned logit RMSE (bins in CV space) against q_np and exact q.
"""
import bisect

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

import common
from lambda_runtime import pchip_eval

# ---------------------------------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------------------------------


def calibrate(z, q, w, n_bins=300, min_count=30):
    """Monotone map g with q ~ g(z): isotonic regression of q on z over bins of equal width in z (between the
    0.05% and 99.95% quantiles, plus the two tails; bins with < min_count frames dropped), then a PCHIP through
    the centres of the isotonic blocks (strictly increasing, continuous). Equal widths in z (a logit-like score)
    keep the low- and high-q ends resolved, where few frames are. Returns (zk, qk, dk)."""
    lo, hi = np.quantile(z, [0.0005, 0.9995])
    edges = np.r_[-np.inf, np.linspace(lo, hi, n_bins - 1), np.inf]
    bi = np.clip(np.searchsorted(edges, z, side='right') - 1, 0, n_bins - 1)
    cnt = np.bincount(bi, minlength=n_bins)
    sw = np.bincount(bi, w, minlength=n_bins)
    ok = (cnt >= min_count) & (sw > 0)
    zb = np.bincount(bi, w * z, minlength=n_bins)[ok] / sw[ok]
    qb = np.bincount(bi, w * q, minlength=n_bins)[ok] / sw[ok]
    wb = sw[ok]
    iso = IsotonicRegression(increasing=True, y_min=common.Q_CLIP, y_max=1 - common.Q_CLIP).fit(zb, qb, sample_weight=wb)
    qi = iso.predict(zb)
    # centres of the blocks of equal isotonic value
    starts = np.r_[0, np.flatnonzero(np.diff(qi) > 0) + 1]
    ends = np.r_[starts[1:], len(qi)]
    zk = np.array([np.average(zb[s:e], weights=wb[s:e]) for s, e in zip(starts, ends)])
    qk = qi[starts]
    if len(zk) < 2:                                   # constant: no information in z
        zk, qk = np.array([zb[0], zb[-1] + 1e-9]), np.array([qk[0], qk[0] + 1e-9])
    keep = np.r_[True, np.diff(zk) > 0]
    zk, qk = zk[keep], qk[keep]
    dk = PchipInterpolator(zk, qk).derivative()(zk)
    return zk, qk, dk


def apply_cal(cal, z):
    return pchip_eval(*cal, z)


# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------


def planned_levels(cfg):
    e = cfg['eval']
    return np.exp(np.linspace(np.log(e['q_lo']), np.log(e['q_hi']), e['n_interfaces']))


def _wvar(x, w):
    if w.sum() <= 0:
        return np.nan
    m = np.average(x, weights=w)
    return np.average((x - m) ** 2, weights=w)


def iso_spread(m, zref, w, levels, half_logit, min_n=20):
    """Weighted variance of zref (logit of the reference q) in the slab |logit m - logit q_k| < half_logit of each
    planned level: the same width at every level, so a lambda that follows the isocommittors scores ~half^2/3."""
    lm, lq = common.logit(m), common.logit(np.asarray(levels))
    var, n = np.full(len(levels), np.nan), np.zeros(len(levels), int)
    for k, l in enumerate(lq):
        sel = np.abs(lm - l) < half_logit
        n[k] = sel.sum()
        if n[k] >= min_n:
            var[k] = _wvar(zref[sel], w[sel])
    return var, n


def spread_levels(ref, half_logit, min_ref=50, max_ratio=10.0):
    """Planned levels used for the mean spread: the reference row (the exact committor as lambda, else q_np) has
    at least min_ref frames in the slab, and - with the exact committor - logit q_np varies there by less than
    max_ratio x the slab-width baseline (half^2/3): below some q the q_np labels are noise, not a committor."""
    use = np.asarray(ref['spread_n']) >= min_ref
    if 'spread_exact' in ref:
        v = np.asarray(ref['spread'], float)
        use &= np.isfinite(v) & (v < max_ratio * half_logit ** 2 / 3)
    return use


def spread_summary(res, use):
    """Mean spread over the levels `use` (the same for every candidate); levels a candidate cannot reach (fewer than
    min_n frames in its slab) count as missing."""
    out = {'spread_levels_used': int(np.sum(use))}
    for key in ('spread', 'spread_exact'):
        if key in res:
            v = np.asarray(res[key], float)[use]
            out[key + '_mean'] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            out[key + '_missing'] = int(np.sum(~np.isfinite(v)))
    return out


def _pairs_in(piece, valid, lag):
    i = np.flatnonzero(valid[:-lag])
    i = i[(piece[i + lag] == piece[i]) & valid[i + lag]]
    return i, i + lag


def sign_change_rate(x, piece, valid, w):
    """Weighted fraction of consecutive lag-1 increments of x (inside one piece) whose sign differs."""
    i = np.flatnonzero(valid[:-2])
    i = i[(piece[i + 2] == piece[i]) & valid[i + 1] & valid[i + 2]]
    d1, d2 = x[i + 1] - x[i], x[i + 2] - x[i + 1]
    ok = (d1 != 0) & (d2 != 0)
    return np.sum(w[i][ok] * (np.sign(d1[ok]) != np.sign(d2[ok]))) / np.sum(w[i][ok])


def turn_replay(x, piece, levels, w):
    """Turns and interface crossings per 1000 frames under the sampler's turn rule, interfaces at `levels`.

    pgmoves.turn_detected: going up, after the maximum, a turn is when x drops below the interface just below
    the highest interface under the maximum (and mirror image going down). Here each piece is scanned in its
    own time order; after a turn the excursion restarts at that frame. x and levels on the same scale.
    """
    lv = np.sort(np.asarray(levels, float)).tolist()
    n_lv = len(lv)
    starts = np.r_[0, np.flatnonzero(np.diff(piece)) + 1]
    ends = np.r_[starts[1:], len(piece)]
    turns = cross = frames = 0.0
    for s, e in zip(starts, ends):
        xs = x[s:e]
        wt = float(w[s])
        side = np.searchsorted(lv, xs)
        cross += wt * np.sum(side[1:] != side[:-1])
        frames += wt * len(xs)
        mx = mn = float(xs[0])
        for xv in xs[1:].tolist():
            if xv > mx:
                mx = xv
            if xv < mn:
                mn = xv
            # up (lr = +1): elig = interfaces <= max, cond = elig[-2]; turn if x < cond after the max
            k_up = bisect.bisect_right(lv, mx)
            # down (lr = -1): elig = interfaces >= min, cond = elig[1]; turn if x > cond after the min
            j_dn = bisect.bisect_left(lv, mn)
            if (k_up >= 2 and xv < lv[k_up - 2]) or (j_dn + 1 < n_lv and xv > lv[j_dn + 1]):
                turns += wt
                mx = mn = xv                             # restart the excursion here
    return 1000 * turns / frames, 1000 * cross / frames


def stopped_dr2(r, b, piece, w, lag, tau):
    """Weighted stopped Delta r^2 / lag: windows [t, min(t + lag, tau_t)] of the non-A/B frames."""
    n = len(r)
    t = np.flatnonzero(b == 0)
    u = np.minimum(t + lag, tau[t])
    return np.sum(w[t] * (r[u] - r[t]) ** 2) / lag


def zq_stats(r, b, piece, w, path, lags, zscore=True, skip_bins=5):
    """Per lag: max |stopped Z_q / s.e.| over q (skipping the first bins, as z_per_lag) and SD of Z_q over q."""
    mt = common.metrics_tis()
    tau = mt.stop_index(piece, b)
    zmax, sd = [], []
    for lag in lags:
        _, zq = mt.comp_zq_stopped(r, b, piece, w, lag, tau=tau)
        sd.append(float(np.std(zq)))
        if zscore:
            _, z = mt.comp_zq_stopped_zscore(r, b, piece, w, lag, tau=tau, g_traj=path)
            zmax.append(float(np.max(np.abs(z[skip_bins:]))))
    return np.array(zmax), np.array(sd), tau


def binned_logit_rmse(cv, zm, zref, w, valid, bins=40):
    """RMSE between the bin means of logit m and of logit q_ref, bins on a grid in CV space (weighted by bin weight)."""
    c = cv[valid].astype(float)
    idx = np.zeros(len(c), int)
    for j in range(c.shape[1]):
        e = np.quantile(c[:, j], np.linspace(0, 1, bins + 1))
        idx = idx * bins + np.clip(np.searchsorted(e, c[:, j], side='right') - 1, 0, bins - 1)
    ww = w[valid]
    sw = np.bincount(idx, ww)
    ok = sw > 0
    a = np.bincount(idx, ww * zm[valid])[ok] / sw[ok]
    r = np.bincount(idx, ww * zref[valid])[ok] / sw[ok]
    return float(np.sqrt(np.average((a - r) ** 2, weights=sw[ok])))


# ---------------------------------------------------------------------------------------------------
# all metrics of one candidate on one split
# ---------------------------------------------------------------------------------------------------


def reference_turns(frames, mask, cfg):
    """Turn and crossing rates of q_np and of the exact committor used as order parameters (same interfaces)."""
    idx = np.flatnonzero(mask)
    piece, w, levels = frames.piece[idx], frames.w[idx].astype(float), planned_levels(cfg)
    out = dict(zip(('turns_qnp', 'crossings_qnp'), turn_replay(frames.q[idx], piece, levels, w)))
    if frames.has_exact:
        out.update(zip(('turns_exact', 'crossings_exact'),
                       turn_replay(frames.q_exact[idx].astype(float), piece, levels, w)))
    return out


def evaluate(frames, mask, z, cal, cfg, zq_lags=None, full=True, ref_turns=None):
    """Metrics of a candidate on the frames in `mask` (whole pieces): z = its raw score on those frames
    (z[k] belongs to the k-th frame of the mask), cal = its calibration.

    With full=False only the selection metric (SD of the stopped Z_q over `eval.select_lags`) and the
    isosurface spread are computed. `ref_turns` (from `reference_turns`) adds the turn-rate ratios.
    """
    e = cfg['eval']
    idx = np.flatnonzero(mask)
    piece, path, w = frames.piece[idx], frames.path[idx], frames.w[idx].astype(float)
    A, B, valid = frames.A[idx], frames.B[idx], frames.valid[idx]
    b = (A | B).astype(float)
    zc = np.asarray(z, float)
    assert len(zc) == len(idx)
    m = apply_cal(cal, zc)
    r = m.copy()
    r[A], r[B] = 0.0, 1.0
    q, zq_np = frames.q[idx], frames.z[idx]
    levels = planned_levels(cfg)
    out = {}
    spread, n_slab = iso_spread(m[valid], zq_np[valid], w[valid], levels, e['slab_half_logit'])
    out['spread'], out['spread_n'] = spread, n_slab
    out['spread_mean'] = float(np.nanmean(spread))          # provisional; cmd_evaluate uses spread_summary
    if frames.has_exact:
        zx = common.logit(frames.q_exact[idx].astype(float))
        sx, _ = iso_spread(m[valid], zx[valid], w[valid], levels, e['slab_half_logit'])
        out['spread_exact'], out['spread_exact_mean'] = sx, float(np.nanmean(sx))
    lags = e['select_lags'] if not full else (zq_lags or e['zq_lags'])
    zmax, sd, tau = zq_stats(r, b, piece, w, path, lags, zscore=full)
    out['zq_lags'], out['zq_sd'] = np.array(lags), sd
    out['sdzqs'] = float(sd.max())
    if not full:
        return out
    out['zq_z'], out['zqs_z'] = zmax, float(zmax.max())

    # smoothness
    zm = common.logit(m)
    out['inc_rmse'] = {}
    for lag in cfg['train']['lags']:
        i0, i1 = _pairs_in(piece, valid, lag)
        d_m, d_q = zm[i1] - zm[i0], zq_np[i1] - zq_np[i0]
        ww = w[i0]
        out['inc_rmse'][lag] = float(np.sqrt(np.average((d_m - d_q) ** 2, weights=ww)))
        if lag == 1:
            out['step_ratio'] = float(np.sum(ww * d_m ** 2) / np.sum(ww * d_q ** 2))
    out['sign_ratio'] = float(sign_change_rate(zc, piece, valid, w) / sign_change_rate(zq_np, piece, valid, w))
    out['turns'], out['crossings'] = turn_replay(m, piece, levels, w)
    if ref_turns is not None:
        out['turn_ratio_qnp'] = out['turns'] / ref_turns['turns_qnp']
        out['cross_ratio_qnp'] = out['crossings'] / ref_turns['crossings_qnp']
        if 'turns_exact' in ref_turns:
            out['turn_ratio_exact'] = out['turns'] / ref_turns['turns_exact']
            out['cross_ratio_exact'] = out['crossings'] / ref_turns['crossings_exact']

    # dynamical: stopped Delta r^2, relative to q_np on the same windows
    rq = np.clip(q, 0, 1).copy()
    rq[A], rq[B] = 0.0, 1.0
    out['dr2_ratio'] = {lag: float(stopped_dr2(r, b, piece, w, lag, tau) / stopped_dr2(rq, b, piece, w, lag, tau))
                        for lag in (1, 2, 4, 8)}

    # static
    sub = np.flatnonzero(valid)
    sub = sub[np.random.default_rng(0).choice(len(sub), min(len(sub), 200_000), replace=False)]
    out['spearman'] = float(spearmanr(zc[sub], q[sub]).statistic)
    cv = frames.cv[idx]
    out['logit_rmse'] = binned_logit_rmse(cv, zm, zq_np, w, valid)
    if frames.has_exact:
        out['logit_rmse_exact'] = binned_logit_rmse(cv, zm, zx, w, valid)
        mid = valid & (frames.q_exact[idx] > 0.05) & (frames.q_exact[idx] < 0.95)
        out['rmse_q_exact_mid'] = float(np.sqrt(np.average((m[mid] - frames.q_exact[idx][mid]) ** 2, weights=w[mid])))
    return out
