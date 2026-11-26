"""Interface placement: a plan in the natural coordinate, a cost check and the spacing gate (constraint 9).

    plan   = plan_natural(model, view, cfg)                 # levels equally spaced in u(h), ratio cap, length budget
    placed = place(model, paths, plan, cfg, mode)           # greedy thinning until every check passes, then refill
    table  = check(paths, interfaces, cfg, mode)            # all checks on a set of sampled paths

`paths` are whole sampled paths at the sampler's resolution (one lambda value per saved frame), each with a
weight: (lam, w) pairs from `path_arrays`. Checks (defaults eps = 1%, c = 1, n_A = 3, n_sh = 3):

- skip, both directions: P(a step crossing lambda_k also crosses the next interface) <= eps, where lambda_k has
  at least 100 crossings in that direction;
- gap (waived where the skip rate is measured, >= 100 crossings; `gap_if_unmeasured = false` keeps it everywhere):
  lambda_{k+1} - lambda_k >= c delta(lambda_k), delta = 95% quantile of |d lambda| over the steps crossing
  lambda_k (the only test where crossings are rare; delta borrowed from the nearest sampled level, and the
  interface marked unverified, where a level has no crossings);
- distance from the states: the 5% quantile of s - lambda_A on frames crossing the first interface >= n_A RMS
  steps of s; the same for the last interface against B;
- shooting-region occupancy per ensemble: an excursion of a path through the ensemble's window (L, R) that
  crosses M must hold frames strictly inside: P(no frame inside) <= eps and the 5% quantile of frames inside
  >= n_sh, the latter not for [0+], whose short A -> A excursions are legitimate (for RETIS the window is (lambda_0, lambda_B) with M = lambda_i, for staples the sampler's own
  (lambda_{i-1}, lambda_i, lambda_{i+1}));
- single-step turns: the fraction of turns (the staple turn rule replayed on lambda) completed within one step of
  the extremum <= eps.

Placement thins with limits stricter than the gate by `place_margin` (hysteresis). Thinning can leave a pair of
neighbours far apart in crossing probability, so `refill` then re-inserts, into every pair whose ratio exceeds
`max_ratio` (end pairs measured from the effective state levels of `state_levels`), the best-splitting candidate
that passes the same margined limits as the thinning. Config keys under
[interfaces]: refill (default true), refill_candidates (40). Where the dynamics cross a committor stretch in a few
saved frames no candidate passes: that limit is set by the sampler resolution, not by lambda.

Reference start (`[interfaces] start = "reference"`, for passes 1..`reference_passes`, 0 = every pass): instead of
thinning a fresh natural plan, start from the interface set that was just sampled, carried over to the new lambda
at the same committor levels: pass 1 from infinit's last set (`seed_reference_levels`: the mean committor at the
first crossings of each seed interface), pass k > 1 from pass k-1's set (h = lambda_{k-1}'s levels). If the carried
set is thinned and refilled as usual, and so is the natural plan; the reference result is kept unless it leaves a
pair of neighbours more than max_ratio apart (`worst_ratio`) that the plan's result closes better, or fails the gate.
The set then changes by a few interfaces per pass instead of being re-placed whole. (loop_zu_b_ref pass 1: infinit's
crowded levels near the barrier landed on the ridge, thinning removed them all and the set ended at q 0.02: worst
ratio ~50, one ensemble from q 0.02 to B, against 13 interfaces from the plan.)
"""
import os

import numpy as np

import common
import natural


# ---------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------
def path_arrays(pool, model, passes=None, state_cv=0, max_paths=5000, lineage='include'):
    """[(lam, s, w), ...] of the stored whole paths of the given passes under `model` (None = the recorded lambda)."""
    passes = pool.passes() if passes is None else passes
    out = []
    for k in sorted(passes, reverse=True):
        z, m = pool.load(k)
        starts = np.r_[0, np.cumsum(z['raw_n'])[:-1]]
        W = z['raw_W']
        for i in range(len(z['raw_n'])):
            if lineage == 'exclude' and z['raw_tainted'][i]:
                continue
            cv = z['raw_cv'][starts[i]:starts[i] + z['raw_n'][i]].astype(float)
            lam = cv[:, 0] if model is None else model(cv[:, 1:])
            w = float(W[i].max()) if W.shape[1] else 1.0
            if w <= 0 or len(lam) < 2:
                continue
            out.append((lam, cv[:, 1 + state_cv], w))
            if len(out) >= max_paths:
                return out
    return out


def _steps(paths):
    """Concatenated lag-1 steps (a, b, s_b, w) over all paths."""
    a, b, sb, w = [], [], [], []
    for lam, s, wt in paths:
        a.append(lam[:-1]); b.append(lam[1:]); sb.append(s[1:]); w.append(np.full(len(lam) - 1, wt))
    return np.concatenate(a), np.concatenate(b), np.concatenate(sb), np.concatenate(w)


def rms_step_s(paths):
    d, w = [], []
    for lam, s, wt in paths:
        d.append(np.diff(s)); w.append(np.full(len(s) - 1, wt))
    d, w = np.concatenate(d), np.concatenate(w)
    return float(np.sqrt(np.average(d * d, weights=w)))


# ---------------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------------
def crossing_stats(steps, level, nxt_up=None, nxt_dn=None, min_cross=100):
    """Up and down crossings of `level`: counts, skip rates to the next interface in each direction, and the
    95% quantile of |d lambda| over the crossing steps."""
    a, b, sb, w = steps
    up = (a < level) & (b >= level)
    dn = (a >= level) & (b < level)
    out = dict(n_up=int(up.sum()), n_dn=int(dn.sum()))
    wu, wd = w[up], w[dn]
    out['skip_up'] = float(np.sum(wu[b[up] >= nxt_up]) / wu.sum()) if nxt_up is not None and up.any() else np.nan
    out['skip_dn'] = float(np.sum(wd[b[dn] < nxt_dn]) / wd.sum()) if nxt_dn is not None and dn.any() else np.nan
    out['skip_up_ok'] = out['n_up'] < min_cross or out['skip_up'] <= 0 or np.isnan(out['skip_up'])
    out['skip_dn_ok'] = out['n_dn'] < min_cross or out['skip_dn'] <= 0 or np.isnan(out['skip_dn'])
    cross = up | dn
    out['delta'] = float(np.quantile(np.abs(b[cross] - a[cross]), 0.95)) if cross.any() else np.nan
    out['first_frames_s'] = sb[up]
    return out


def turn_stats(paths, interfaces, max_paths=2000):
    """Turns under the staple turn rule replayed on lambda: fraction completed within one step of the extremum."""
    lv = np.sort(np.asarray(interfaces, float))
    n_lv = len(lv)
    one, total = 0.0, 0.0
    for lam, s, wt in paths[:max_paths]:
        mx = mn = lam[0]
        i_mx = i_mn = 0
        for i in range(1, len(lam)):
            x = lam[i]
            if x > mx:
                mx, i_mx = x, i
            if x < mn:
                mn, i_mn = x, i
            k_up = np.searchsorted(lv, mx, side='right')
            j_dn = np.searchsorted(lv, mn, side='left')
            turn_up = k_up >= 2 and x < lv[k_up - 2]
            turn_dn = j_dn + 1 < n_lv and x > lv[j_dn + 1]
            if turn_up or turn_dn:
                ext = i_mx if turn_up else i_mn
                total += wt
                if i - ext <= 1:
                    one += wt
                mx = mn = x
                i_mx = i_mn = i
    return dict(n_turns=float(total), single_step=float(one / total) if total > 0 else np.nan)


def window_stats(paths, L, M, R, min_frames=1):
    """Excursions through (L, R) that cross M: fraction without a frame strictly inside and the 5% quantile of
    the frames inside (weighted)."""
    inside_counts, w_all, empty = [], [], []
    for lam, s, wt in paths:
        inside = (lam > L) & (lam < R)
        edges = np.flatnonzero(np.diff(np.r_[False, inside, False]))
        for a, b in zip(edges[::2], edges[1::2] - 1):
            lo, hi = max(a - 1, 0), min(b + 1, len(lam) - 1)
            seg = lam[lo:hi + 1]
            if seg.min() < M <= seg.max():
                inside_counts.append(b - a + 1)
                w_all.append(wt)
        # excursions that jump the whole window in one step: a step from <= L to >= R or back
        jump = ((lam[:-1] <= L) & (lam[1:] >= R)) | ((lam[:-1] >= R) & (lam[1:] <= L))
        for _ in range(int(jump.sum())):
            inside_counts.append(0)
            w_all.append(wt)
    if not inside_counts:
        return dict(n_excursions=0, p_empty=np.nan, q05_frames=np.nan)
    c, w = np.array(inside_counts, float), np.array(w_all, float)
    o = np.argsort(c)
    cw = np.cumsum(w[o]) / w.sum()
    return dict(n_excursions=len(c), p_empty=float(np.sum(w[c == 0]) / w.sum()),
                q05_frames=float(c[o][np.searchsorted(cw, 0.05)]))


def ensembles_of(interfaces, mode):
    """(e, L, M, R) windows of the ensembles (infretis column e >= 1)."""
    lv = list(map(float, interfaces))
    out = [(1, lv[0], lv[0], lv[1])]
    for e in range(2, len(lv)):
        if mode == 'retis':
            out.append((e, lv[0], lv[e - 1], lv[-1]))
        else:
            out.append((e, lv[e - 2], lv[e - 1], lv[e]))
    return out


def check(paths, interfaces, cfg, mode, verified=None):
    """Every spacing check of an interface set on the sampled paths. Returns a dict with per-interface rows,
    per-ensemble rows, the failures and `ok`."""
    eps, c, n_a, n_sh = float(cfg.get('eps', 0.01)), float(cfg.get('c', 1.0)), float(cfg.get('n_a', 3)), float(cfg.get('n_sh', 3))
    max_empty = float(cfg.get('max_empty', eps))          # allowed fraction of excursions with an empty shooting region
    gap_waived = bool(cfg.get('gap_if_unmeasured', True))      # the gap rule is the proxy for the skip rate: waived where measured
    lv = [float(v) for v in interfaces]
    steps = _steps(paths)
    sigma_s = rms_step_s(paths)
    rows, fails = [], []
    stats = [crossing_stats(steps, lv[k], nxt_up=lv[k + 1] if k + 1 < len(lv) else None,
                            nxt_dn=lv[k - 1] if k > 0 else None) for k in range(len(lv))]
    deltas = np.array([st['delta'] for st in stats])
    for k in range(1, len(lv) - 1):
        st = stats[k]
        row = dict(k=k, lam=lv[k], n_up=st['n_up'], n_dn=st['n_dn'], skip_up=st['skip_up'], skip_dn=st['skip_dn'],
                   delta=st['delta'], unverified=not np.isfinite(st['delta']))
        delta = st['delta']
        if not np.isfinite(delta):
            have = np.flatnonzero(np.isfinite(deltas))
            delta = float(deltas[have[np.argmin(np.abs(have - k))]]) if len(have) else np.nan
            row['delta_borrowed'] = delta
        row['gap_up'] = (lv[k + 1] - lv[k]) / delta if delta and np.isfinite(delta) else np.nan
        row['gap_dn'] = (lv[k] - lv[k - 1]) / delta if delta and np.isfinite(delta) else np.nan
        if st['n_up'] >= 100 and np.isfinite(st['skip_up']) and st['skip_up'] > eps:
            fails.append(('skip_up', k, st['skip_up']))
        if st['n_dn'] >= 100 and np.isfinite(st['skip_dn']) and st['skip_dn'] > eps:
            fails.append(('skip_dn', k, st['skip_dn']))
        if np.isfinite(row['gap_up']) and row['gap_up'] < c and not (gap_waived and st['n_up'] >= 100):
            fails.append(('gap', k, row['gap_up']))
        if k == 1:
            d = st['first_frames_s'] - lv[0]
            row['dist_a'] = float(np.quantile(d, 0.05) / sigma_s) if len(d) else np.nan
            if np.isfinite(row['dist_a']) and row['dist_a'] < n_a:
                fails.append(('dist_a', k, row['dist_a']))
        if k == len(lv) - 2:
            a, b, sb, w = steps
            dn = (a >= lv[k]) & (b < lv[k])
            d = lv[-1] - sb[dn]
            row['dist_b'] = float(np.quantile(d, 0.05) / sigma_s) if dn.any() else np.nan
            if np.isfinite(row['dist_b']) and row['dist_b'] < n_a:
                fails.append(('dist_b', k, row['dist_b']))
        rows.append(row)
    ens_rows = []
    for e, L, M, R in ensembles_of(lv, mode):
        ws = window_stats(paths, L, M, R)
        ws.update(e=e, L=L, M=M, R=R)
        if ws['n_excursions'] >= 20:
            if ws['p_empty'] > max_empty:
                fails.append(('empty_region', e, ws['p_empty']))
            # [0+] (L = M = lambda_0): short A -> A excursions with one interior frame are legitimate paths, so
            # only the empty-region test applies there
            if e >= 2 and ws['q05_frames'] < n_sh:
                fails.append(('region_frames', e, ws['q05_frames']))
        ens_rows.append(ws)
    ts = turn_stats(paths, lv)
    if ts['n_turns'] >= 20 and np.isfinite(ts['single_step']) and ts['single_step'] > eps:
        fails.append(('single_step_turns', -1, ts['single_step']))
    return dict(interfaces=lv, rows=rows, ensembles=ens_rows, turns=ts, sigma_s=sigma_s, fails=fails,
                ok=not fails, n_paths=len(paths), unverified=[r['k'] for r in rows if r['unverified']])


# ---------------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------------
def plan_natural(model, view, cfg, log=None):
    """Interior levels equally spaced in the natural coordinate u(h) of the model on the view's increments,
    with the ratio cap and the path-length budget. Returns dict(h, lam, n_requested, predicted, ...)."""
    n = int(cfg.get('n_interfaces', 10))
    n_max = int(cfg.get('n_max', 40))
    R = float(cfg.get('max_ratio', 10.0))
    budget = float(cfg.get('length_budget', np.inf))
    inc = natural.Increments(view, model)
    hk, uk, dk = inc.natural_table()
    hh = np.linspace(0, 1, 20001) ** 3
    uu = np.interp(hh, hk, uk)

    def levels(n_req):
        u = np.linspace(0, 1, n_req + 2)[1:-1]
        h = np.interp(u, uu, hh)
        h = _cap_ratio(h, R)
        return h
    pred_longest, med = None, []
    if cfg.get('no_growth'):                      # fallback plan only: n levels, no length prediction / budget growth
        h = levels(n)
        lam = model.scale(np.asarray(h, float))
        return dict(h=list(map(float, h)), lam=[model.lambda_a] + [float(v) for v in lam] + [model.lambda_b],
                    n_requested=n, predicted_median=[], predicted_longest=None, budget=budget)
    while True:
        h = levels(n)
        pred = natural.predict_lengths(inc, np.r_[0.0, h, 1.0], n_walk=1000)
        med = [float(np.median(pred[k])) for k in sorted(pred) if np.isfinite(np.median(pred[k]))]
        pred_longest = max(med) if med else np.nan
        if not np.isfinite(pred_longest) or 1.3 * pred_longest <= budget or n >= n_max:
            break
        n += 2
    lam = model.scale(np.asarray(h, float))
    out = dict(h=list(map(float, h)), lam=[model.lambda_a] + [float(v) for v in lam] + [model.lambda_b],
               n_requested=n, predicted_median=med, predicted_longest=pred_longest, budget=budget)
    if log:
        common.say(f'plan: {len(h)} interior levels (natural coordinate, ratio <= {R:g}); predicted longest median '
                   f'{pred_longest:.0f} frames (budget {budget:g}); levels q = {np.round(h, 4).tolist()}', log=log)
    return out


def seed_reference_levels(pool, model, interfaces, passes, state_cv, lambda_a, min_cross=20, max_paths=5000):
    """Committor levels of a seed interface set (infinit's, in the recorded lambda of the given pool passes) under
    a new model: for each interface lambda_i, the weighted mean of the model's committor m over the first
    crossings of lambda_i by the A-started excursions of the stored paths. That mean is P(B | crossed lambda_i)
    (P_cross(lambda_B) = P_cross(lambda_i) <m>_first crossings), so the new lambda's isosurface at it has the
    seed interface's place in crossing probability. Levels with fewer than `min_cross` crossings are dropped.
    Returns dict(levels (seed lambda), h, n_cross)."""
    inner = np.asarray([float(v) for v in interfaces[1:-1]])
    frames, owner, weight = [], [], []
    n_paths = 0
    for k in sorted(passes, reverse=True):
        z, _ = pool.load(k)
        starts = np.r_[0, np.cumsum(z['raw_n'])[:-1]]
        W = z['raw_W']
        for i in range(len(z['raw_n'])):
            w = float(W[i].max()) if W.shape[1] else 1.0
            if w <= 0 or z['raw_n'][i] < 2:
                continue
            cv = z['raw_cv'][starts[i]:starts[i] + z['raw_n'][i]].astype(float)
            lam, s = cv[:, 0], cv[:, 1 + state_cv]
            in_a = np.flatnonzero(s <= lambda_a)
            if not len(in_a):
                continue
            # excursions: the frames after an A frame up to the next A frame
            for a, b in zip(in_a, np.r_[in_a[1:], len(s)]):
                if b - a < 2:
                    continue
                run = np.maximum.accumulate(lam[a + 1:b])
                j = np.searchsorted(run, inner, side='left')       # first frame with lam >= lambda_i
                hit = j < len(run)
                for li in np.flatnonzero(hit):
                    frames.append(cv[a + 1 + j[li], 1:])
                    owner.append(li)
                    weight.append(w)
            n_paths += 1
            if n_paths >= max_paths:
                break
        if n_paths >= max_paths:
            break
    if not frames:
        return dict(levels=inner.tolist(), h=[], n_cross=[0] * len(inner))
    m = np.clip(model.q(np.asarray(frames)), 0.0, 1.0)
    owner, weight = np.asarray(owner), np.asarray(weight)
    h, n_cross, kept = [], [], []
    for li in range(len(inner)):
        sel = owner == li
        n_cross.append(int(sel.sum()))
        if sel.sum() >= min_cross:
            h.append(float(np.average(m[sel], weights=weight[sel])))
            kept.append(float(inner[li]))
    return dict(levels=kept, h=h, n_cross=n_cross)


def reference_set(model, h, interfaces_ab, min_rel_step=1e-3):
    """[lambda_A] + the model's isosurfaces at the levels h (strictly increasing, inside the states) + [lambda_B]."""
    la, lb = float(interfaces_ab[0]), float(interfaces_ab[-1])
    h = np.clip(np.asarray(h, float), 1e-9, 1 - 1e-9)
    lam = np.asarray(model.scale(h), float) if len(h) else np.zeros(0)
    out = [la]
    for x in lam:
        if la < x < lb and x - out[-1] > min_rel_step * (lb - la):
            out.append(float(x))
    if lb - out[-1] <= min_rel_step * (lb - la) and len(out) > 1:
        out.pop()
    return out + [lb]


def _cap_ratio(h, R):
    h = [float(v) for v in h]
    changed = True
    while changed and len(h) < 200:
        changed = False
        for k in range(len(h) - 1):
            a, b = h[k], h[k + 1]
            if b <= 0.5 and b / a > R:
                h.insert(k + 1, float(np.sqrt(a * b))); changed = True; break
            if a >= 0.5 and (1 - a) / (1 - b) > R:
                h.insert(k + 1, float(1 - np.sqrt((1 - a) * (1 - b)))); changed = True; break
            if a < 0.5 < b and (b / a > R or (1 - a) / (1 - b) > R):        # a pair that straddles q = 0.5
                h.insert(k + 1, 0.5); changed = True; break
    return np.array(h)


# ---------------------------------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------------------------------
def place(model, paths, plan_lam, cfg, mode, log=None, max_rounds=50):
    """Thin the planned interface set until every check passes on `paths`. Greedy forward pass (skip, gap,
    distance) as in the spacing note, then the ensemble and turn checks with one interface removed per round
    (the upper interface of the failing window). Then, if `model` is given and cfg `refill` (default true), `refill`
    re-inserts interfaces into pairs more than max_ratio apart that the thinning left behind, under the same limits. Returns dict(interfaces, check, removed, rounds, refilled)."""
    # placement uses stricter limits than the gate (`check`), so that an interface set that passes on the stored
    # paths does not fail on the next pass's own paths, which sample the new interfaces better (hysteresis)
    margin = float(cfg.get('place_margin', 1.25))
    cfg = dict(cfg, eps=float(cfg.get('eps', 0.01)) / margin, c=float(cfg.get('c', 1.0)) * margin,
               n_a=float(cfg.get('n_a', 3)) * margin, n_sh=float(cfg.get('n_sh', 3)) * margin, max_empty=float(cfg.get('max_empty', cfg.get('eps', 0.01))) / margin)
    eps, c, n_a = cfg['eps'], cfg['c'], cfg['n_a']
    gap_waived = bool(cfg.get('gap_if_unmeasured', True))
    la, lb = float(plan_lam[0]), float(plan_lam[-1])
    inner = [float(v) for v in plan_lam[1:-1]]
    steps = _steps(paths)
    sigma_s = rms_step_s(paths)
    removed = []
    # forward pass
    placed = [la]
    for lv in inner:
        st_prev = crossing_stats(steps, placed[-1], nxt_up=lv)
        st = crossing_stats(steps, lv, nxt_dn=placed[-1])
        reasons = []
        if st_prev['n_up'] >= 100 and placed[-1] > la and np.isfinite(st_prev['skip_up']) and st_prev['skip_up'] > eps:
            reasons.append(f'skip_up {st_prev["skip_up"]:.3f}')
        if st['n_dn'] >= 100 and np.isfinite(st['skip_dn']) and st['skip_dn'] > eps:
            reasons.append(f'skip_dn {st["skip_dn"]:.3f}')
        delta = st_prev['delta'] if placed[-1] > la else st['delta']
        measured = gap_waived and placed[-1] > la and st_prev['n_up'] >= 100
        if np.isfinite(delta) and lv - placed[-1] < c * delta and not measured:
            reasons.append(f'gap {(lv - placed[-1]) / delta:.2f}')
        if len(placed) == 1:
            d = st['first_frames_s'] - la
            if len(d) >= 20 and np.quantile(d, 0.05) / sigma_s < n_a:
                reasons.append(f'dist_a {np.quantile(d, 0.05) / sigma_s:.1f}')
        if reasons:
            removed.append((lv, '; '.join(reasons)))
        else:
            placed.append(lv)
    # the last interface against B
    while len(placed) > 1:
        st = crossing_stats(steps, placed[-1], nxt_up=lb)
        a, b, sb, w = steps
        dn = (a >= placed[-1]) & (b < placed[-1])
        d = lb - sb[dn]
        bad = (np.isfinite(st['delta']) and lb - placed[-1] < c * st['delta']
               and not (gap_waived and st['n_up'] >= 100)) or \
              (st['n_up'] >= 100 and np.isfinite(st['skip_up']) and st['skip_up'] > eps) or \
              (dn.sum() >= 20 and np.quantile(d, 0.05) / sigma_s < n_a)
        if not bad:
            break
        removed.append((placed[-1], 'last interface too close to B'))
        placed.pop()
    placed.append(lb)
    # ensemble and turn checks
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        res = check(paths, placed, cfg, mode)
        if res['ok'] or len(placed) <= 3:
            break
        kind, where, value = res['fails'][0]
        if kind in ('empty_region', 'region_frames'):
            k = min(where, len(placed) - 2)                     # the upper interface of the window (never lambda_B)
        elif kind in ('skip_up', 'gap'):
            k = min(where + 1, len(placed) - 2)
        elif kind in ('skip_dn', 'dist_a'):
            k = max(where, 1)
        elif kind == 'dist_b':
            k = len(placed) - 2
        else:                                                   # single-step turns: drop the closest pair's upper member
            gaps = np.diff(placed)
            k = int(np.argmin(gaps)) + 1 if len(gaps) > 2 else 1
            k = min(max(k, 1), len(placed) - 2)
        removed.append((placed[k], f'{kind} {value:.3g} (ensemble/interface {where})'))
        placed.pop(k)
    res = check(paths, placed, cfg, mode)
    if log:
        common.say(f'placement: {len(placed)} interfaces kept of {len(plan_lam)} planned; removed '
                   + (', '.join(f'{v:.4f} ({r})' for v, r in removed) if removed else 'none')
                   + ('' if res['ok'] else f'; still failing: {res["fails"]}'), log=log)
    refilled = []
    if model is not None and res['ok'] and cfg.get('refill', True):
        # the same (margined) limits as the thinning: with the bare gate, refill re-added what thinning had just
        # removed (Z: 0.8455 'too close to B' in every round) and its interfaces failed the next run's re-check
        placed, refilled = refill(model, paths, placed, cfg, mode, R=float(cfg.get('max_ratio', 10.0)),
                                  n_cand=int(cfg.get('refill_candidates', 40)), steps=steps)
        if refilled:
            res = check(paths, placed, cfg, mode)
            if log:
                common.say('placement refill (placement limits): added '
                           + ', '.join(f'{r["lam"]:.4f} (q {r["q"]:.3g}; ratio {r["ratio_before"]:.0f} -> '
                                       f'{r["ratio_after"]:.1f})' for r in refilled), log=log)
    return dict(interfaces=placed, check=res, removed=removed, rounds=rounds, refilled=refilled)


# ---------------------------------------------------------------------------------------------------
# measured committor levels ([interfaces] levels = "measured")
# ---------------------------------------------------------------------------------------------------
def _lg(q):
    q = np.clip(np.asarray(q, float), 1e-12, 1 - 1e-12)
    return np.log(q / (1 - q))


def _sg(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def measured_levels(run_dir, model, burn_in=0.1, max_paths=4000, min_cross=20, include_a=False):
    """Committor levels of a finished run's interfaces, measured, and the model's levels of the same surfaces.

    Measured: inftools iSTAR (`rates.staple_rate`) gives P_A(lambda_k), the probability that a path that left A
    reaches interface k. For interfaces that are isocommittor surfaces, the committor is a martingale along the
    dynamics, so P_A(lambda_k) = q_1st / q_k with q_1st the mean committor of the first frames past lambda_A, and
    q_k = P_A(lambda_B) / P_A(lambda_k) (gambler's ruin, exact for any q_k). This needs no model.

    Model: the run's interfaces are surfaces of the lambda the run used, which need not be `model`'s lambda (the
    loop refits lambda every pass). The model's level of surface k is the median of logit `model.q` over the frames
    just past an up-crossing of lambda_k in the run's own paths (the recorded lambda, column 0 of the order), of
    model.h_of_lambda(model(cv)): the level a placement with `model` assigns to those frames.

    Returns dict(run, interfaces, pcross, q_meas, h_model, n_cross) for the interior interfaces with at least
    `min_cross` up-crossings and a positive P_A. With `include_a` also lambda_A itself (P_A = 1, so q_meas =
    P_A(lambda_B), the mean committor of the first frames past lambda_A): anchors the levels between lambda_A and the
    first interface, which are otherwise extrapolated (loop_z_dp: first pair p 0.17 from the model's levels against
    0.022 measured, 2026-11-25)."""
    from loop import rates
    from loop.backends import infretis_backend as be
    r = rates.staple_rate(run_dir, burn_in=burn_in)
    lv = [float(v) for v in r['interfaces']]
    P = np.asarray(r['pcross'], float)
    raw = be.raw_paths(run_dir, max_paths=max_paths)
    a_l, b_cv = [], []
    for o in raw['ops']:
        a_l.append(o[:-1, 0]); b_cv.append(o[1:])
    lam0 = np.concatenate(a_l)
    nxt = np.concatenate(b_cv)
    out = dict(run=run_dir, interfaces=[], pcross=[], q_meas=[], h_model=[], n_cross=[])
    for k in range(0 if include_a else 1, len(lv) - 1):
        up = (lam0 < lv[k]) & (nxt[:, 0] >= lv[k])
        if up.sum() < min_cross or not P[k] > 0:
            continue
        hm = float(_sg(np.median(_lg(model.h_of_lambda(model(nxt[up, 1:]))))))   # the level placement sees
        out['interfaces'].append(lv[k]); out['pcross'].append(float(P[k]))
        out['q_meas'].append(float(P[-1] / P[k])); out['h_model'].append(hm); out['n_cross'].append(int(up.sum()))
    out['pcross_global'] = float(P[-1])
    return out


class LevelMap:
    """The model with its committor levels re-mapped to measured ones (`measured_levels`): logit q_meas against
    logit h_model, piecewise linear through the measured points (both made non-decreasing), and outside them the
    model's logit shifted by the offset of the nearest point. The surfaces (lambda) are the model's; only the
    level a placement assigns to each surface changes. Provides what placement uses: h_of_lambda, scale,
    lambda_a, lambda_b (anything else is the model's)."""

    def __init__(self, model, meas):
        self.model, self.meas = model, meas
        x, y = _lg(meas['h_model']), _lg(meas['q_meas'])
        o = np.argsort(x)
        x, y = x[o], np.maximum.accumulate(y[o])
        keep = np.r_[True, np.diff(x) > 1e-9]
        self.x, self.y = x[keep], y[keep]

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, cv):
        return self.model(cv)

    def _fwd(self, z):
        z = np.asarray(z, float)
        if len(self.x) == 1:
            return z + (self.y[0] - self.x[0])
        out = np.interp(z, self.x, self.y)
        out = np.where(z < self.x[0], z + (self.y[0] - self.x[0]), out)
        return np.where(z > self.x[-1], z + (self.y[-1] - self.x[-1]), out)

    def _inv(self, w):
        w = np.asarray(w, float)
        if len(self.x) == 1:
            return w - (self.y[0] - self.x[0])
        y = self.y + 1e-9 * np.arange(len(self.y))               # strictly increasing for the inverse
        out = np.interp(w, y, self.x)
        out = np.where(w < y[0], w - (self.y[0] - self.x[0]), out)
        return np.where(w > y[-1], w - (self.y[-1] - self.x[-1]), out)

    def h_of_lambda(self, lam):
        h = np.asarray(self.model.h_of_lambda(lam), float)
        return np.where((h <= 0) | (h >= 1), h, _sg(self._fwd(_lg(h))))

    def scale(self, h):
        return self.model.scale(_sg(self._inv(_lg(h))))

    def table(self):
        m = self.meas
        return [dict(lam=l_, h_model=float(hm), q_meas=float(qm), n_cross=n_) for l_, hm, qm, n_ in
                zip(m['interfaces'], m['h_model'], m['q_meas'], m['n_cross'])]


def level_model(model, run_dir, cfg, log=None):
    """`model`, or with [interfaces] levels = "measured" its LevelMap from the finished run in `run_dir`. Falls back
    to `model` (logged) when the run cannot be analysed or has fewer than 2 usable interfaces."""
    if cfg.get('levels', 'model') != 'measured' or not run_dir:
        return model
    try:
        meas = measured_levels(run_dir, model, burn_in=float(cfg.get('levels_burn_in', 0.1)))
        if len(meas['interfaces']) < 2:
            raise ValueError(f'{len(meas["interfaces"])} usable interfaces')
        lm = LevelMap(model, meas)
        if log:
            common.say(f'levels (measured, {os.path.relpath(run_dir, os.path.dirname(os.path.dirname(run_dir)))}): '
                       + ', '.join(f'{r["lam"]:.4f}: model q {r["h_model"]:.3g} -> measured {r["q_meas"]:.3g}'
                                   for r in lm.table()), log=log)
        return lm
    except Exception as e:                                       # noqa: BLE001
        if log:
            common.say(f'levels: measured levels from {run_dir} failed ({type(e).__name__}: {e}); model levels', log=log)
        return model


# ---------------------------------------------------------------------------------------------------
# refill
# ---------------------------------------------------------------------------------------------------
def q_ratio(qa, qb):
    """The crossing-probability ratio of a pair of committor levels qa < qb, as in `_cap_ratio`: qb / qa below
    q = 1/2, (1 - qa) / (1 - qb) above it, the larger of the two for a pair that straddles 1/2."""
    qa, qb = max(float(qa), 1e-12), min(float(qb), 1 - 1e-12)
    lo, hi = qb / qa, (1 - qa) / (1 - qb)
    if qb <= 0.5:
        return lo
    if qa >= 0.5:
        return hi
    return max(lo, hi)


def _local_check(paths, steps, lv, k, gate, mode):
    """The gate restricted to what inserting interface k changes: skip/gap of k - 1, k, k + 1 and the
    shooting-region tests of the ensembles whose middle interface is one of them (turns: see `refill`)."""
    eps, c, n_sh = gate['eps'], gate['c'], gate['n_sh']
    max_empty = float(gate.get('max_empty', eps))
    gap_waived = bool(gate.get('gap_if_unmeasured', True))
    for j in (k - 1, k, k + 1):
        if j <= 0 or j >= len(lv) - 1:
            continue
        st = crossing_stats(steps, lv[j], nxt_up=lv[j + 1], nxt_dn=lv[j - 1])
        if st['n_up'] >= 100 and np.isfinite(st['skip_up']) and st['skip_up'] > eps:
            return False, f'skip_up at {lv[j]:.4f}'
        if st['n_dn'] >= 100 and np.isfinite(st['skip_dn']) and st['skip_dn'] > eps:
            return False, f'skip_dn at {lv[j]:.4f}'
        gap_bad = ((lv[j + 1] - lv[j] < c * st['delta'] and not (gap_waived and st['n_up'] >= 100))
                   or (lv[j] - lv[j - 1] < c * st['delta'] and not (gap_waived and st['n_dn'] >= 100)))
        if np.isfinite(st['delta']) and gap_bad:
            return False, f'gap at {lv[j]:.4f}'
        if not np.isfinite(st['delta']):
            return False, f'no crossings of {lv[j]:.4f}'           # refill only where the paths verify the level
        if gate.get('end_dist') and (j == 1 or j == len(lv) - 2):   # first / last interface against the states
            n_a, sig = gate.get('n_a', 3), gate.get('sigma_s') or rms_step_s(paths)
            if j == 1 and len(st['first_frames_s']) and np.quantile(st['first_frames_s'] - lv[0], 0.05) / sig < n_a:
                return False, f'dist_a at {lv[j]:.4f}'
            if j == len(lv) - 2:
                a_, b_, sb_, w_ = steps
                dn_ = (a_ >= lv[j]) & (b_ < lv[j])
                if dn_.any() and np.quantile(lv[-1] - sb_[dn_], 0.05) / sig < n_a:
                    return False, f'dist_b at {lv[j]:.4f}'
    mids = {lv[j] for j in (k - 1, k, k + 1) if 0 < j < len(lv) - 1}
    for e, L, M, R in ensembles_of(lv, mode):
        if M not in mids:
            continue
        ws = window_stats(paths, L, M, R)
        if ws['n_excursions'] >= 20 and (ws['p_empty'] > max_empty or (e >= 2 and ws['q05_frames'] < n_sh)):
            return False, f'region of ensemble {e}'
    return True, ''


def state_levels(model, steps, la, lb):
    """Effective committor levels of the states for the end pairs: h_A = the weighted mean of h at the first frames
    past lambda_A (up-crossings), 1 - h_B the same of 1 - h at the first frames below lambda_B (down-crossings).
    With committor-level interfaces, P(reach lambda_1 before A | crossed lambda_A) = E[q(first frame)] / q(lambda_1)
    (optional stopping), so q_ratio(h_A, h_1) is the end pair's crossing-probability ratio; h(lambda_A) itself is 0.
    Without crossings of a state its level falls back to 0 (A) or 1 (B), which leaves that end pair alone.

    Caveat: the export's ramps near the states hold h above the committor there (Z, frozen lambda: first frames past
    A have model h 3.3e-5 vs exact 1.8e-6; first frames below B 0.990 vs 0.998), so the end pairs' ratios are
    underestimated (Z: 11 and 13 vs about 100 and 64 from the exact committor and the production run) and refill
    adds fewer interfaces there than the cap would ask for, never more."""
    a, b, _, w = steps
    up = (a < la) & (b >= la)
    dn = (a > lb) & (b <= lb)
    h_a = float(np.average(np.clip(model.h_of_lambda(b[up]), 0, 1), weights=w[up])) if up.any() else 1e-9
    h_b = float(np.average(np.clip(model.h_of_lambda(b[dn]), 0, 1), weights=w[dn])) if dn.any() else 1 - 1e-9
    return float(np.clip(h_a, 1e-9, 0.5)), float(np.clip(h_b, 0.5, 1 - 1e-9))


def _pair_p(h_i, h_j, first, last, h_a_known, h_b_known):
    """Local crossing probability of a pair of committor levels (1 / q_ratio), the end pairs as in `worst_ratio`."""
    if first and not h_a_known:
        return max(1.0 - h_j, 1e-12)
    if last and not h_b_known:
        return max(h_i, 1e-12)
    return 1.0 / q_ratio(h_i, h_j)


def place_dp(model, paths, cfg, mode, n_grid=120, max_rounds=25, log=None):
    """Interfaces chosen at once by dynamic programming over a grid of committor levels (instead of greedy
    thinning and refill).

    Hard limits, the frame checks of the gate on the stored paths with the placement margin: skip <= eps both ways,
    gap >= c delta, first / last interface >= n_A RMS steps from the states; the window checks (empty shooting
    region, frames in the window) on the solved set, a failing window excluded and the problem solved again.
    Objective: sum over neighbour pairs of (ln(p / target_p))^2, p = 1 / q_ratio of the model's levels (end pairs
    from the effective state levels, or the forward measure where the paths do not cross a state boundary);
    target_p (default 0.3) is the crossing probability aimed at from one interface to the next. Path lengths are
    evened out afterwards by `even_regions` (the program runs over pairs (previous, current) so that a failing
    window (previous, next) can be excluded).
    """
    margin = float(cfg.get('place_margin', 1.25))
    eps, c = float(cfg.get('eps', 0.01)) / margin, float(cfg.get('c', 1.0)) * margin
    n_a = float(cfg.get('n_a', 3)) * margin
    gate = dict(cfg, eps=eps, c=c, n_a=n_a, n_sh=float(cfg.get('n_sh', 3)) * margin, max_empty=float(cfg.get('max_empty', cfg.get('eps', 0.01))) / margin)
    gap_waived = bool(cfg.get('gap_if_unmeasured', True))
    p_star = float(cfg.get('target_p', 0.3))
    steps = _steps(paths)
    a, b, sb, w = steps
    sigma_s = rms_step_s(paths)
    la, lb = float(model.lambda_a), float(model.lambda_b)
    h_a, h_b = state_levels(model, steps, la, lb)
    h_a_known, h_b_known = h_a > 1e-8, h_b < 1 - 1e-8
    lo, hi = max(h_a, 1e-8), min(h_b, 1 - 1e-6)
    lg = np.linspace(np.log(lo / (1 - lo)), np.log(hi / (1 - hi)), n_grid + 2)[1:-1]
    hg = 1 / (1 + np.exp(-lg))
    lam_g = np.asarray(model.scale(hg), float)
    keep = np.r_[True, np.diff(lam_g) > 1e-6 * (lb - la)] & (lam_g > la) & (lam_g < lb)
    lam = np.r_[la, lam_g[keep], lb]
    h = np.r_[h_a, hg[keep], h_b]
    N = len(lam)
    # per node: crossings (weighted and counted), skip rates to every other node, step size, distance to the states
    up_n, dn_n, delta, cross_w = np.zeros(N, int), np.zeros(N, int), np.full(N, np.nan), np.zeros(N)
    skip_up, skip_dn = np.zeros((N, N)), np.zeros((N, N))
    ok_first, ok_last = np.zeros(N, bool), np.zeros(N, bool)
    for i in range(N):
        L = lam[i]
        up, dn = (a < L) & (b >= L), (a >= L) & (b < L)
        up_n[i], dn_n[i] = int(up.sum()), int(dn.sum())
        cr = up | dn
        cross_w[i] = float(w[cr].sum())
        if cr.any():
            delta[i] = np.quantile(np.abs(b[cr] - a[cr]), 0.95)
        if up.any():
            o = np.argsort(b[up]); bu, wu = b[up][o], w[up][o]
            cw = np.r_[0, np.cumsum(wu)]
            skip_up[i] = (cw[-1] - cw[np.searchsorted(bu, lam, side='left')]) / cw[-1]      # b >= lam_j
            ok_first[i] = np.quantile(sb[up] - la, 0.05) / sigma_s >= n_a
        if dn.any():
            o = np.argsort(b[dn]); bd, wd = b[dn][o], w[dn][o]
            cw = np.r_[0, np.cumsum(wd)]
            skip_dn[:, i] = cw[np.searchsorted(bd, lam, side='left')] / cw[-1]          # b < lam_k (k < i)
            ok_last[i] = np.quantile(lb - sb[dn], 0.05) / sigma_s >= n_a
    # pair feasibility and cost
    pcost = np.full((N, N), np.inf)
    for i in range(N - 1):
        for j in range(i + 1, N):
            if i == 0 and j < N - 1 and not ok_first[j]:
                continue
            if j == N - 1 and i > 0 and not ok_last[i]:
                continue
            if 0 < i and np.isfinite(delta[i]) and lam[j] - lam[i] < c * delta[i] and not (gap_waived and up_n[i] >= 100):
                continue
            if 0 < i and up_n[i] >= 100 and skip_up[i, j] > eps:
                continue
            if j < N - 1 and dn_n[j] >= 100 and skip_dn[i, j] > eps:
                continue
            p = _pair_p(h[i], h[j], i == 0, j == N - 1, h_a_known, h_b_known)
            pcost[i, j] = np.log(p / p_star) ** 2

    def solve(wcost, pc):
        """best[prev, cur]: cheapest chain A -> ... -> prev -> cur; the ensemble at cur is charged when its next
        interface is chosen (window (prev, next))."""
        best = np.full((N, N), np.inf)
        argp = np.zeros((N, N), int)
        best[0, 1:] = pc[0, 1:]
        for cur in range(1, N - 1):
            col = best[:cur, cur]
            if not np.isfinite(col).any():
                continue
            tot = col[:, None] + wcost[:cur, :]                       # (prev, next)
            ip = np.argmin(tot, axis=0)
            cand = tot[ip, np.arange(N)] + pc[cur]
            cand[:cur + 1] = np.inf
            better = cand < best[cur]
            best[cur, better] = cand[better]
            argp[cur, better] = ip[better]
        prev = int(np.argmin(best[:, N - 1]))
        if not np.isfinite(best[prev, N - 1]):
            return None, np.inf
        chain = [N - 1, prev]
        cur, nxt = prev, N - 1
        while cur != 0:
            p_ = int(argp[cur, nxt]); chain.append(p_); nxt, cur = cur, p_
        return chain[::-1], float(best[chain[-2], N - 1])

    def windows(chain):
        return [(chain[k - 1], chain[k + 1]) for k in range(1, len(chain) - 1)]

    banned = np.zeros((N, N), bool)                                   # windows (previous, next)
    pc = pcost.copy()                                                 # pairs; [0+] failures ban (A, first)
    rounds, chain, obj = [], None, np.inf
    for r in range(max_rounds):
        wcost = np.where(banned, np.inf, 0.0)
        chain, obj = solve(wcost, pc)
        if chain is None:
            break
        lv = [float(lam[k]) for k in chain]
        chk = check(paths, lv, gate, mode)
        rounds.append(dict(n=len(lv), objective=obj, fails=chk['fails'][:6]))
        win = [f for f in chk['fails'] if f[0] in ('empty_region', 'region_frames') and f[1] >= 1]
        if not win:
            break
        for kind, e, _ in win:
            if e == 1:                                                # [0+]: window (lambda_A, first interface)
                pc[0, chain[1]] = np.inf
            else:                                                     # ensemble e: window (lv[e-2], lv[e])
                banned[chain[e - 2], chain[min(e, len(chain) - 1)]] = True
    if chain is None:
        return None
    lv = [float(lam[k]) for k in chain]
    hs = h[chain]
    p = [_pair_p(hs[i], hs[i + 1], i == 0, i + 1 == len(lv) - 1, h_a_known, h_b_known) for i in range(len(lv) - 1)]
    out = dict(interfaces=lv, check=check(paths, lv, cfg, mode), removed=[], refilled=[], rounds=rounds,
               objective=obj, p=[float(x) for x in p], h=[float(x) for x in hs], target_p=p_star, n_grid=N)
    if log:
        common.say(f'placement (dp): {len(lv)} interfaces, local p {[round(x, 2) for x in p]} (target {p_star:g}), '
                   f'{len(rounds)} round(s); q = {[float(f"{x:.2g}") for x in hs[1:-1]]}', log=log)
    return out


def window_frames(paths, L, M, R, stat='mean', near=None):
    """(length, number of staple paths) of the shooting region of the staple ensemble with window (L, M, R).

    Every stored path is cut into the staple paths it would contain under the trial interfaces: from each run of
    frames inside (L, R) that crosses M (seed.window_cores), follow the path outward on both sides until the
    sampler's turn rule fires or A / B is reached (seed.extend_turn, checked against infretis' turn_detected); a
    path that ends before a turn is cut at its end. The region of one staple path is its number of frames strictly
    inside (L, R); the result is the unweighted mean (`stat` = 'mean') or median over the distinct staple paths.

    `near` = (lambda_A, interface below L or None, interface above R or None, lambda_B): the interfaces the turn rule
    sees. One interface beyond each side of the window covers turns of paths that go up to two interfaces past it;
    further ones change little and would make every window depend on the whole set.

    Why this measure (loop_z_dp, 2026-11-24):
    - Unweighted, per staple path: that is what the sampler's paths are. The former estimate (path_f-weighted median
      per run inside the window) was up to 100x short on the plateau, where many paths cross a window quickly and a
      few wander for thousands of frames.
    - Cut at the trial set's turns: an extra interface ends staple paths sooner. A sum over the whole stored path
      cannot see that, so no split ever looked useful.
    - Mean: the plateau distribution is bimodal (~10 or ~5000 frames), and a median from a short run flips."""
    vals = window_lengths(paths, L, M, R, near=near)
    if not vals:
        return (np.nan, 0)
    return (float(np.mean(vals) if stat == 'mean' else np.median(vals)), len(vals))


def window_lengths(paths, L, M, R, near=None, ext=None):
    """The shooting-region length of every distinct staple path of the window (L, M, R) cut from `paths` (see
    `window_frames`). `ext` = (min, max) of lambda per path, to skip paths that never enter the window."""
    from loop import seed as seed_mod
    la, below, above, lb = near if near is not None else (-np.inf, None, None, np.inf)
    intf = [v for v in (la, below, L, M, R, above, lb) if v is not None and np.isfinite(v)]
    intf = sorted(set(intf))
    m_idx = intf.index(M)
    vals = []
    idx = range(len(paths)) if ext is None else np.flatnonzero((ext[0] < M) & (ext[1] > M))
    for i in idx:
        lam = paths[i][0]
        seen = set()
        for a_, b_ in seed_mod.window_cores(lam, L, M, R):
            if seen and any(s_ <= a_ <= t_ for s_, t_ in seen):
                continue                                         # this run belongs to a staple path already counted
            t = seed_mod.extend_turn(lam, intf, m_idx, b_, +1)
            s = seed_mod.extend_turn(lam, intf, m_idx, a_, -1)
            s = 0 if s is None else s
            t = len(lam) - 1 if t is None else t
            seen.add((s, t))
            seg = lam[s:t + 1]
            vals.append(int(np.count_nonzero((seg > L) & (seg < R))))
    return vals


def window_profile(paths, qs, L, M, R, qL, qM, qR, near=None, ext=None, with_index=False):
    """Per distinct staple path of the window (L, M, R) cut from `paths` (as `window_lengths`): the region length,
    the whole staple path length, and the predicted NCR probability of a shot from it. `qs` = committor levels of
    every frame of every path (same shapes as the lambda arrays); qL, qM, qR the levels of the interfaces.

    NCR: a shot from frame x reaches M before the window edge with phi = (q(x) - qL) / (qM - qL) left of M,
    (qR - q(x)) / (qR - qM) right of it (optional stopping at isocommittor surfaces); the two halves are independent
    given x, so P(no crossing of M | x) = (1 - phi)^2, averaged over the region frames. Validated on loop_z_dp
    (production_measured, its repeat, offline_noswap, 2026-11-25): correlation 0.90 with the observed NCR share of
    the moves not rejected as too long (mean error 0.07); with `length_acceptance`, A_len x (1 - NCR) against the
    observed acceptance: correlation 0.85, mean error 0.07 (predicted high, most in the plateau ensembles).
    Returns (region lengths, path lengths, NCR per path) as arrays, with `with_index` also the index in `paths` of
    the stored path each staple path was cut from."""
    from loop import seed as seed_mod
    la, below, above, lb = near if near is not None else (-np.inf, None, None, np.inf)
    intf = sorted(set(v for v in (la, below, L, M, R, above, lb) if v is not None and np.isfinite(v)))
    m_idx = intf.index(M)
    reg, ln, ncr, src = [], [], [], []
    idx = range(len(paths)) if ext is None else np.flatnonzero((ext[0] < M) & (ext[1] > M))
    for i in idx:
        lam, q = paths[i][0], qs[i]
        seen = set()
        for a_, b_ in seed_mod.window_cores(lam, L, M, R):
            if seen and any(s_ <= a_ <= t_ for s_, t_ in seen):
                continue
            t = seed_mod.extend_turn(lam, intf, m_idx, b_, +1)
            s = seed_mod.extend_turn(lam, intf, m_idx, a_, -1)
            s = 0 if s is None else s
            t = len(lam) - 1 if t is None else t
            seen.add((s, t))
            seg, qseg = lam[s:t + 1], q[s:t + 1]
            inside = (seg > L) & (seg < R)
            if not inside.any():
                continue
            x, qx = seg[inside], qseg[inside]
            phi = np.where(x < M, (qx - qL) / max(qM - qL, 1e-300), (qR - qx) / max(qR - qM, 1e-300))
            reg.append(int(inside.sum()))
            ln.append(t - s + 1)
            ncr.append(float(np.mean((1.0 - np.clip(phi, 0.0, 1.0)) ** 2)))
            src.append(int(i))
    out = (np.asarray(reg, float), np.asarray(ln, float), np.asarray(ncr, float))
    return out + (np.asarray(src, int),) if with_index else out


def length_acceptance_w(n, w, m=3000, seed=0):
    """`length_acceptance` of region lengths `n` with weights `w` (a weighted resample of `m`)."""
    n, w = np.asarray(n, float), np.asarray(w, float)
    ok = (n > 0) & (w > 0)
    if ok.sum() < 5:
        return np.nan
    n, w = n[ok], w[ok] / w[ok].sum()
    return length_acceptance(n[np.random.default_rng(seed).choice(len(n), m, p=w)], m=m, seed=seed)


def length_acceptance(n, m=3000, seed=0):
    """Expected acceptance of the flexible-length rule, A = E[min(1, n_i / n_j)] with n_i, n_j drawn independently
    from the shooting-region lengths `n` of one ensemble (subsampled to `m`). Scale-free: only the spread of the
    lengths within the ensemble matters, not their size. Validated on loop_z_dp (production_measured, offline runs,
    2026-11-24): against the observed share of moves not rejected as too long, correlation 0.84 with the logged
    regions and 0.75 with regions cut from the stored paths (mean error 0.05-0.06); it ranks the ensembles right but
    compresses the low end (predicted 0.62-0.67 where 0.47-0.55 is observed, ensembles with 30-800x spread)."""
    n = np.asarray(n, float)
    n = n[n > 0]
    if len(n) < 5:
        return np.nan
    if len(n) > m:
        n = np.random.default_rng(seed).choice(n, m, replace=False)
    return float(np.minimum(n[:, None] / n[None, :], 1.0).mean())


def region_frames(paths, interfaces, stat='mean'):
    """Shooting-region length per staple ensemble e = 2 .. n-1 of an interface set (`window_frames` of the window
    (lambda_{e-2}, lambda_e) around lambda_{e-1}, turns with one interface beyond each side). Returns
    {e: (length, n staple paths)}. Placement passes the newest run's own paths (`region_paths`), not the whole pool."""
    lv = [float(v) for v in interfaces]
    return {e: window_frames(paths, lv[e - 2], lv[e - 1], lv[e], stat, near=_near(lv, e)) for e in range(2, len(lv))}


def _near(lv, e):
    """(lambda_A, interface below the window, interface above it, lambda_B) of ensemble e (window lv[e-2], lv[e])."""
    return (lv[0], lv[e - 3] if e - 3 >= 0 else None, lv[e + 1] if e + 1 < len(lv) else None, lv[-1])


def region_paths(pool, model, state_cv=0, n_passes=1, max_paths=5000):
    """The paths the shooting-region lengths are measured on: the whole accepted paths of the newest `n_passes`
    pool passes (the last run: a learning pass or a verification run), replayed under `model`."""
    return path_arrays(pool, model, passes=pool.passes()[-int(n_passes):], state_cv=state_cv, max_paths=max_paths)


def balance_regions(model, paths, placed, cfg, mode, log=None, sweeps=3, n_cand=14, min_gain=0.01, rpaths=None):
    """Make the shooting-region lengths uniform: move interior interfaces, one at a time, to the position between
    their neighbours that minimises the squared spread of ln(region length) over the ensembles (`region_frames` on
    `rpaths`, the last run's own paths; default `paths`) plus
    `balance_p_weight` (default 0.3) x the sum of (ln(p / target_p))^2 over all pairs, the end pairs included, keeping
    only moves that pass the local gate (skip rates to the neighbours above all: a path that crosses an interface
    must not reach the next one in one step) and leave both adjacent pairs with a crossing probability of at least
    target_p / 3 and at most p_max (default 0.9, unless the pair was already above it). Sweeps over the interfaces until no move improves the spread by `min_gain` (relative)."""
    p_floor = float(cfg.get('balance_p_min', cfg.get('merge_p_min', float(cfg.get('target_p', 0.3)) / 3)))
    p_max = float(cfg.get('p_max', 0.9))
    margin = float(cfg.get('place_margin', 1.25))
    gate = dict(cfg, eps=float(cfg.get('eps', 0.01)) / margin, c=float(cfg.get('c', 1.0)) * margin,
                n_a=float(cfg.get('n_a', 3)) * margin, n_sh=float(cfg.get('n_sh', 3)) * margin, max_empty=float(cfg.get('max_empty', cfg.get('eps', 0.01))) / margin, end_dist=True,
                sigma_s=rms_step_s(paths))
    steps = _steps(paths)
    rp, stat = (paths if rpaths is None else rpaths), cfg.get('region_stat', 'mean')   # region lengths: own paths
    lv = [float(v) for v in placed['interfaces']]
    cache = {}
    target_p, w_p = float(cfg.get('target_p', 0.3)), float(cfg.get('balance_p_weight', 0.3))
    h_a, h_b = state_levels(model, steps, lv[0], lv[-1])
    h_a_known, h_b_known = h_a > 1e-8, h_b < 1 - 1e-8

    def win(v, e):
        nr = _near(v, e)
        key = tuple(round(x, 9) if x is not None else None for x in (v[e - 2], v[e - 1], v[e], nr[1], nr[2]))
        if key not in cache:
            cache[key] = window_frames(rp, v[e - 2], v[e - 1], v[e], stat, near=nr)[0]
        return cache[key]

    def spread(v):
        f = np.array([win(v, e) for e in range(2, len(v))])
        if not np.all(np.isfinite(f)) or np.any(f <= 0):
            return np.inf
        return float(np.sum((np.log(f) - np.mean(np.log(f))) ** 2)) + w_p * p_dev(v)

    def p_dev(v):
        # sum of (ln(p / target_p))^2 over all pairs, the end pairs too (levels of the states as in place_dp)
        h = np.r_[0.0, np.clip(model.h_of_lambda(np.asarray(v[1:-1])), 1e-9, 1 - 1e-9), 1.0]
        n = len(v)
        ps = [_pair_p(h[j], h[j + 1], j == 0, j + 1 == n - 1, h_a_known, h_b_known) if j > 0 or j + 1 < n - 1
              else 1.0 for j in range(n - 1)]
        return float(np.sum([np.log(max(x, 1e-12) / target_p) ** 2 for x in ps]))

    def pair_p(v, k):
        h = np.r_[0.0, np.clip(model.h_of_lambda(np.asarray(v[1:-1])), 1e-9, 1 - 1e-9), 1.0]
        return [1.0 / q_ratio(h[j], h[j + 1]) if 0 < j and j + 1 < len(v) - 1 else np.nan for j in (k - 1, k)]

    def p_ok(v, k, p_before):
        # both adjacent pairs keep p >= p_floor, and none is pushed above p_max (a pair crossed almost surely adds an
        # ensemble and no information) unless it already was that high before the move
        cap = max(p_max, np.nanmax(p_before) if np.any(np.isfinite(p_before)) else 0.0)
        return all(np.isnan(x) or p_floor <= x <= cap for x in pair_p(v, k))

    J = spread(lv)
    start, moves = J, []
    for _ in range(sweeps):
        moved = False
        for k in range(1, len(lv) - 1):
            best, best_J = None, J
            p_before = pair_p(lv, k)
            for lc in np.linspace(lv[k - 1], lv[k + 1], n_cand + 2)[1:-1]:
                if abs(lc - lv[k]) < 1e-9:
                    continue
                trial = lv[:k] + [float(lc)] + lv[k + 1:]
                Jt = spread(trial)
                if Jt < best_J * (1 - min_gain) and p_ok(trial, k, p_before) and \
                        _local_check(paths, steps, trial, k, gate, mode)[0]:
                    best, best_J = trial, Jt
            if best is not None:
                moves.append((lv[k], best[k]))
                lv, J, moved = best, best_J, True
        if not moved:
            break
    rf = region_frames(rp, lv, stat)
    if log:
        common.say(f'placement (balance): {len(moves)} moves, objective {start:.2f} -> {J:.2f}; '
                   f'{stat} region frames per window {[int(v[0]) if np.isfinite(v[0]) else None for v in rf.values()]}', log=log)
    return dict(placed, interfaces=lv, check=check(paths, lv, cfg, mode), balanced=moves,
                region_frames={int(e): v[0] for e, v in rf.items()})


# ---------------------------------------------------------------------------------------------------
# refinement on finished runs (acceptance: low crossing probability, long shooting regions)
# ---------------------------------------------------------------------------------------------------
def run_acceptance(run_dir):
    """Shooting moves per ensemble (infretis numbering) of a run, from the worker logs (sim.log is rewritten by a
    resume): {e: dict(moves, acc, ncr, too_long, sh_median)} with fractions of the moves and the median
    shooting-region length (frames) of the moves."""
    import glob
    import re
    from collections import Counter, defaultdict
    st, shn = defaultdict(Counter), defaultdict(list)
    for f in sorted(glob.glob(os.path.join(run_dir, 'worker*.log'))):
        ens = None
        with open(f, errors='ignore') as fh:
            for line in fh:
                m = re.search(r'starting st_sh in (\d+) ', line)
                if m:
                    ens = int(m.group(1))
                    continue
                m = re.search(r'staple_sh: ens=(\d+) .*sh_region=\((\d+), (\d+)\)', line)
                if m:
                    shn[int(m.group(1))].append(int(m.group(3)) - int(m.group(2)) + 1)
                    continue
                m = re.search(r'Move was \w+ with status (\w+)', line)
                if m and ens is not None:
                    st[ens][m.group(1)] += 1
                    ens = None
    out = {}
    for e, c in sorted(st.items()):
        n = sum(c.values())
        out[e] = dict(moves=n, acc=c['ACC'] / n, ncr=c['NCR'] / n,
                      too_long=(c['BTL'] + c['FTL'] + c['BTX'] + c['FTX']) / n,
                      sh_median=float(np.median(shn[e])) if shn[e] else np.nan)
    return out


def measured_levels_multi(run_dirs, model, burn_in=0.1, include_a=False):
    """`measured_levels` of several finished runs with the same interfaces, combined: per interface present in all
    of them the geometric mean of the measured levels q_meas (one run's iSTAR crossing probabilities scatter by up to
    1.5x per pair on the Z, 2026-11-24), the mean of the model levels, the summed crossing counts."""
    ms = [measured_levels(d, model, burn_in=burn_in, include_a=include_a) for d in run_dirs]
    if len(ms) == 1:
        return ms[0]
    common_l = [l_ for l_ in ms[0]['interfaces'] if all(np.any(np.isclose(m['interfaces'], l_, atol=1e-9)) for m in ms)]
    out = dict(run=list(run_dirs), interfaces=[], pcross=[], q_meas=[], h_model=[], n_cross=[])
    for l_ in common_l:
        idx = [int(np.flatnonzero(np.isclose(m['interfaces'], l_, atol=1e-9))[0]) for m in ms]
        out['interfaces'].append(float(l_))
        out['pcross'].append(float(np.exp(np.mean([np.log(m['pcross'][i]) for m, i in zip(ms, idx)]))))
        out['q_meas'].append(float(_sg(np.mean([_lg(m['q_meas'][i]) for m, i in zip(ms, idx)]))))
        out['h_model'].append(float(_sg(np.mean([_lg(m['h_model'][i]) for m, i in zip(ms, idx)]))))
        out['n_cross'].append(int(sum(m['n_cross'][i] for m, i in zip(ms, idx))))
    out['pcross_global'] = float(np.exp(np.mean([np.log(m['pcross_global']) for m in ms])))
    return out


def refine_measured(model, run_dirs, cfg, mode, state_cv=0, log=None, max_steps=12, n_cand=8, probes=()):
    """Refine the interface set of finished runs (same set and lambda) for convergence of the rate.

    `refine_objective` = "variance" (default): minimise the predicted variance of ln k per MD step,
        J = ln sum_e L_e + ln sum_j (1 - p_j) / (p_j acc_{j+1})
    with p_j = q_j / q_{j+1} the crossing probability of pair j from the measured levels (anchored at lambda_A:
    `measured_levels(include_a=True)`), acc_{j+1} the predicted acceptance of the ensemble whose middle is lambda_j
    (below; ensemble 001: observed), L_e its mean staple path length (MD cost of a move, equal moves per ensemble).
    Why: block variances of the pairs of three loop_z_dp productions (2026-11-25) follow (1 - p) / (p acc) per move
    (free log fit: exponents 1.12 and -0.99, R^2 0.82; the first pair is ~3x below it, iSTAR pools it: its term is
    scaled by `refine_first_pair_factor`, default 1/3), while the
    acceptance-balanced set (objective "acceptance", below) converged no faster per MD step: its five extra
    interfaces near B sat in pairs at p 0.9-0.97 that carried ~1 % of the variance. Insert, move and remove; no p
    floor or cap (the objective prices p).
    Window statistics are reweighted (`refine_reweight`, default true): every staple path cut from a stored path gets
    that path's path_f / path_w in the data ensembles whose middle lies inside the window (each ensemble normalised to
    1, pooled; else the ensemble with the nearest middle). Against the shooting regions logged by the sampler
    (loop_z_dp, three productions, every window scored on every run, 2026-11-25): correlation of the predicted path
    length with the observed MD per move 0.91 -> 0.96 in-sample, 0.64 -> 0.82 for narrowed and 0.68 -> 0.81 for
    widened windows; regions 1.34x the logged ones in-sample (1.56x unweighted). Unweighted, a window wider than the
    data's scored ~0.6x its own run's value (another set's runs hold its long excursions in the wrong proportion).
    Excursions the data runs hardly sampled stay short (a plateau window from runs whose interfaces end below it:
    0.2x, n_eff 27-37); every window reports n_eff and whether it is wider than any the runs sampled (`widened`),
    and a change is only taken if every widened window has n_eff >= `refine_min_neff_widened` (default 50).
    `refine_support_guard` (default false): only changes whose windows all lie inside windows the runs sampled.

    `refine_objective` = "acceptance" (2026-11-24/25): every ensemble sampled alike:
    predicted shooting acceptance high and even over the ensembles, and staple path lengths even over them.

    Per ensemble e of a trial set, from the runs' own paths cut into the staple paths of e (`window_profile`) and the
    committor levels measured on the runs (`measured_levels_multi`, LevelMap):
    - acc_e = A_len x (1 - NCR): A_len = `length_acceptance` of the region lengths (the flexible-length rule,
      min(1, n_old / n_new), only the spread of lengths inside the ensemble matters), NCR = the predicted share of
      shots whose segment does not cross the middle interface;
    - L_e = mean staple path length (the MD cost of a move in e).
    Objective (minimised), with smin / smax the soft minimum / maximum over the ensembles at temperature
    `refine_tau` (default 0.15 in ln units: smin(x) = -tau ln mean exp(-x / tau); at 0.05 only the single worst
    ensemble counted, and with two ensembles tied at the bottom no single change helped, so the greedy stopped):
        - smin(ln acc)                                         the worst acceptance up
        + `refine_w_acc_spread` (default 1)  x (smax - smin)(ln acc)   acceptance even
        + `refine_w_len_spread` (default 1)  x (smax - smin)(ln L)     path lengths even (log max / min)
        + `refine_w_len_mean` (default 0.25) x smax(ln L)              the longest paths short
    Worst-case terms, not means: a mean over the ensembles can be raised by adding easy ensembles (second try,
    2026-11-25: four interfaces near B with pairs at p 0.83-0.88, mean acceptance 0.48 -> 0.49, the worst ensemble
    unchanged at 0.26), and a spread of path lengths can be lowered by making every ensemble long (first try: 13 -> 8
    interfaces, one end pair at p 0.06). `refine_p_cap` (default p_max): no insertion or move creates a pair crossed
    with a higher p (a move may keep a pair that already was higher); 0.7 blocked the split of the plateau windows
    near B, where pairs at p ~0.9 cannot be avoided.
    Greedy steps over all candidate changes: insert an interface (n_cand positions in every pair) or move one
    (n_cand positions between its neighbours); removing one only with `refine_allow_remove` = true. Take the best
    if it lowers the objective by at least `refine_min_gain` (default 0.02; an extra ensemble has to earn its cost),
    at most `max_steps` steps. Every candidate passes the local gate and keeps every pair it changes, the end pairs
    included, within [min(balance_p_min, p before), max(p_max, p before)]. `refine_max_paths` (default 6000): own
    paths used (random subset), for speed.

    Why (loop_z_dp production_measured, 2026-11-24): 42 % acceptance overall but 29-31 % in ensemble 7 (47 % NCR:
    pair 0.238 -> 0.417 crossed with p 0.15) and 12-16 % in ensemble 9 (lengths spread 400-800x: 45 % too long);
    staple path lengths 250 to 5900 frames over the ensembles.

    Returns dict(interfaces, check, steps, before, after, acceptance, levels); before / after = per ensemble
    acc_pred, acc_len, ncr, path_len, frames (mean region), spread (q90 / q10 of region lengths), and p per pair."""
    from loop.backends import infretis_backend as be
    run_dirs = list(run_dirs)
    lv = [float(v) for v in be.read_toml(os.path.join(run_dirs[0], 'infretis.toml'))['simulation']['interfaces']]
    for d in run_dirs[1:]:
        other = [float(v) for v in be.read_toml(os.path.join(d, 'infretis.toml'))['simulation']['interfaces']]
        if len(other) != len(lv) or not np.allclose(other, lv, atol=1e-9):
            raise ValueError(f'{d} was run with other interfaces than {run_dirs[0]}')
    own, Wown = [], []
    for d in run_dirs:
        raw = be.raw_paths(d)
        for o, W in zip(raw['ops'], raw['W']):
            W = np.asarray(W, float)
            if W.max() > 0:
                own.append((o[:, 0], o[:, 1 + state_cv], float(W.max())))
                Wown.append(W / len(run_dirs))                        # runs pooled with equal weight
    n_max = int(cfg.get('refine_max_paths', 6000))
    if len(own) > n_max:
        keep = np.sort(np.random.default_rng(0).choice(len(own), n_max, replace=False))
        own, Wown = [own[i] for i in keep], [Wown[i] for i in keep]
    Wown = np.array(Wown)                                             # path_f / path_w per path and data ensemble
    reweight = bool(cfg.get('refine_reweight', True))
    ext = (np.array([float(p[0].min()) for p in own]), np.array([float(p[0].max()) for p in own]))
    variance = cfg.get('refine_objective', 'variance') == 'variance'
    meas = measured_levels_multi(run_dirs, model, burn_in=float(cfg.get('levels_burn_in', 0.1)), include_a=variance)
    lm = LevelMap(model, meas)
    qs = [np.clip(np.asarray(lm.h_of_lambda(p[0]), float), 1e-12, 1 - 1e-12) for p in own]
    steps = _steps(own)
    h_a, h_b = state_levels(lm, steps, lv[0], lv[-1])
    acc_obs = run_acceptance(run_dirs[0])
    acc_1 = float(np.mean([run_acceptance(d).get(1, {}).get('acc', np.nan) for d in run_dirs]))   # ensemble 001
    q_a = float(meas['pcross_global'])                     # measured level of lambda_A: P_A(lambda_B)
    first_fac = float(cfg.get('refine_first_pair_factor', 1 / 3))
    w_as, w_ls = float(cfg.get('refine_w_acc_spread', 1.0)), float(cfg.get('refine_w_len_spread', 1.0))
    w_lm = float(cfg.get('refine_w_len_mean', 0.25))
    tau = float(cfg.get('refine_tau', 0.15))
    p_cap = float(cfg.get('refine_p_cap', cfg.get('p_max', 0.9)))
    allow_remove = bool(cfg.get('refine_allow_remove', variance))
    min_gain = float(cfg.get('refine_min_gain', 0.02))
    p_max = float(cfg.get('p_max', 0.9))
    p_floor = float(cfg.get('balance_p_min', cfg.get('merge_p_min', 0.1)))
    margin = float(cfg.get('place_margin', 1.25))
    gate = dict(cfg, eps=float(cfg.get('eps', 0.01)) / margin, c=float(cfg.get('c', 1.0)) * margin,
                n_a=float(cfg.get('n_a', 3)) * margin, n_sh=float(cfg.get('n_sh', 3)) * margin,
                max_empty=float(cfg.get('max_empty', cfg.get('eps', 0.01))) / margin, end_dist=True,
                sigma_s=rms_step_s(own))
    cache = {}
    data_lv = list(lv)                                                # the set the runs sampled (W's ensembles)

    def weights(src, L, M, R):
        """Weight of every staple path (cut from stored path src[k]) in window (L, M, R): pooled over the data
        ensembles f whose middle lies inside (L, R) (else the one whose middle is nearest M), each with the paths'
        path_f / path_w in f normalised to total 1."""
        mids = {f: data_lv[f - 1] for f in range(2, len(data_lv))}
        pool = [f for f, m_ in mids.items() if L < m_ < R] or [min(mids, key=lambda f: abs(mids[f] - M))]
        w = np.zeros(len(src))
        for f in pool:
            wf = Wown[src, f]
            if wf.sum() > 0:
                w += wf / wf.sum()
        return w

    def win(v, e):
        nr = _near(v, e)
        key = tuple(round(x, 9) if x is not None else None for x in (v[e - 2], v[e - 1], v[e], nr[1], nr[2]))
        if key not in cache:
            L, M, R = v[e - 2], v[e - 1], v[e]
            qL, qM, qR = np.clip(lm.h_of_lambda(np.array([L, M, R])), 1e-12, 1 - 1e-12)
            reg, ln, ncr, src = window_profile(own, qs, L, M, R, qL, qM, qR, near=nr, ext=ext, with_index=True)
            w = weights(src, L, M, R) if reweight and len(src) else np.ones(len(reg))
            if len(reg) < 5 or w.sum() <= 0:
                cache[key] = None
            else:
                w = w / w.sum()
                al = length_acceptance_w(reg, w) if reweight else length_acceptance(reg)
                nc = float(w @ ncr)
                o_ = np.argsort(reg); cw = np.cumsum(w[o_])
                q10, q90 = reg[o_][np.searchsorted(cw, 0.1)], reg[o_][min(np.searchsorted(cw, 0.9), len(reg) - 1)]
                cache[key] = dict(acc_pred=al * (1 - nc), acc_len=al, ncr=nc, path_len=float(w @ ln),
                                  frames=float(w @ reg), spread=float(q90 / max(q10, 1.0)), n=len(reg),
                                  n_eff=float(1.0 / (w ** 2).sum()),
                                  widened=not any(a <= L + 1e-12 and R <= b + 1e-12 for a, b in
                                                  ((data_lv[f - 2], data_lv[f]) for f in range(2, len(data_lv)))))
        return cache[key]

    def pairs(v):
        if variance:          # p_j = q_j / q_{j+1} (martingale), q(lambda_A) measured (= P_A(lambda_B)), q(lambda_B) = 1
            # (not h_of_lambda(lambda_A): the model's level of the state surface itself is 0)
            q = np.r_[q_a, np.clip(lm.h_of_lambda(np.asarray(v[1:-1])), 1e-15, 1.0), 1.0]
            return [float(min(q[j] / q[j + 1], 1.0)) for j in range(len(v) - 1)]
        h = np.r_[h_a, np.clip(lm.h_of_lambda(np.asarray(v[1:-1])), 1e-9, 1 - 1e-9), h_b]
        return [1.0 / q_ratio(h[j], h[j + 1]) for j in range(len(v) - 1)]

    def var_terms(v):
        """(1 - p_j) / (p_j acc_{j+1}) per pair j (pair j is estimated by ensemble j + 1, middle lambda_j; ensemble
        001 is not modelled: its observed acceptance), and the summed staple path length of ensembles 002-."""
        w = [win(v, e) for e in range(2, len(v))]
        if any(x is None or not np.isfinite(x['acc_pred']) or x['acc_pred'] <= 0 for x in w):
            return None, None
        acc = [acc_1] + [x['acc_pred'] for x in w]
        pv = pairs(v)
        t = [(1 - p) / max(p, 1e-12) / a for p, a in zip(pv, acc)]
        t[0] *= first_fac
        return t, float(sum(x['path_len'] for x in w))

    def objective(v):
        if variance:        # ln(predicted variance of ln k x MD steps), up to a constant
            t, cost = var_terms(v)
            return np.inf if t is None else float(np.log(cost) + np.log(sum(t)))
        w = [win(v, e) for e in range(2, len(v))]
        if any(x is None or not np.isfinite(x['acc_pred']) or x['acc_pred'] <= 0 for x in w):
            return np.inf
        la_ = np.log([x['acc_pred'] for x in w])
        ll_ = np.log([x['path_len'] for x in w])
        smin = lambda x: float(-tau * np.log(np.mean(np.exp(-(x - x.min()) / tau))) + x.min())
        smax = lambda x: -smin(-x)
        return float(-smin(la_) + w_as * (smax(la_) - smin(la_)) + w_ls * (smax(ll_) - smin(ll_)) + w_lm * smax(ll_))

    def p_ok(v, js, before_p):
        if variance:                                  # the objective prices p itself
            return True
        pv = pairs(v)
        for j in js:
            if not (0 <= j < len(v) - 1):
                continue
            lo = min(p_floor, before_p) if before_p is not None else p_floor
            hi = min(p_cap, p_max)
            if before_p is not None:
                hi = max(hi, before_p)
            if not (lo <= pv[j] <= hi):
                return False
        return True

    def candidates(v):
        pv = pairs(v)
        for j in range(len(v) - 1):                                   # insert into pair j
            for lc in np.linspace(v[j], v[j + 1], n_cand + 2)[1:-1]:
                t = v[:j + 1] + [float(lc)] + v[j + 1:]
                yield ('insert', float(lc)), t, j + 1, (j, j + 1), None
        for k in range(1, len(v) - 1):                                # move interface k
            for lc in np.linspace(v[k - 1], v[k + 1], n_cand + 2)[1:-1]:
                if abs(lc - v[k]) < 1e-9:
                    continue
                t = v[:k] + [float(lc)] + v[k + 1:]
                yield ('move', v[k], float(lc)), t, k, (k - 1, k), min(pv[k - 1], pv[k])
        for k in range(1, len(v) - 1):                                # remove interface k
            if len(v) <= 4 or not allow_remove:
                break
            t = v[:k] + v[k + 1:]
            yield ('remove', v[k]), t, max(k - 1, 1), (k - 1,), pv[k - 1] * pv[k]

    def table(v):
        rows = {int(e): win(v, e) for e in range(2, len(v))}
        t, cost = var_terms(v)
        return dict(interfaces=list(v), p=pairs(v), ensembles=rows, objective=objective(v),
                    var_terms=t, cost=cost, var_x_cost=None if t is None else float(sum(t) * cost))

    guard = bool(cfg.get('refine_support_guard', False))
    data_win = [(lv[e - 2], lv[e]) for e in range(2, len(lv))]        # the windows the runs sampled

    def supported(v):
        """Every window of v lies inside a window the runs sampled: a wider one is scored on staple paths cut at the
        ends of the stored paths, too short (loop_z_dp, 2026-11-25: widened windows 0.59x their own run's length,
        up to 5x short on the plateau; narrowed ones unbiased, 1.08x)."""
        return all(any(a <= v[e - 2] + 1e-12 and v[e] <= b + 1e-12 for a, b in data_win) for e in range(2, len(v)))

    min_neff_w = float(cfg.get('refine_min_neff_widened', 50))

    def data_ok(v):
        """Every window wider than the runs sampled rests on at least `refine_min_neff_widened` effective paths
        (loop_z_dp, 2026-11-25: widened windows predicted 0.2-0.6x their logged regions had n_eff 27-37 (one 96),
        one with n_eff 68 came out 0.86x; unchecked, the refinement widened the costliest plateau window on n_eff 5
        and scored it shorter than the narrower window it contains)."""
        for e in range(2, len(v)):
            w = win(v, e)
            if w is not None and w.get('widened') and w.get('n_eff', 0) < min_neff_w:
                return False
        return True

    before = table(lv)
    J = before['objective']
    if log:
        common.say(f'refine: {len(own)} own paths of {len(run_dirs)} run(s); objective {J:.3f}', log=log)
    done = []
    for _ in range(max_steps):
        best = None
        for what, t, k, js, pb in candidates(lv):
            if not p_ok(t, js, pb) or (guard and not supported(t)):
                continue
            Jt = objective(t)
            if not np.isfinite(Jt) or Jt > J - min_gain or (best is not None and Jt >= best[0]):
                continue
            if not data_ok(t):
                continue
            if not _local_check(own, steps, t, k, gate, mode)[0]:
                continue
            best = (Jt, t, what)
        if best is None:
            break
        if log:
            common.say(f'refine: {best[2]}; objective {J:.3f} -> {best[0]:.3f}', log=log)
        done.append(dict(change=best[2], objective=best[0]))
        lv, J = best[1], best[0]
    probe_tables = []
    for pr in probes:          # trial sets to compare: interfaces added to the start set, or ('set', full set)
        if isinstance(pr, tuple) and pr[0] == 'set':
            probe_tables.append(dict(table(sorted(float(x) for x in pr[1])), added=None, full=True))
            continue
        t = sorted(set(before['interfaces']) | {float(x) for x in pr})
        probe_tables.append(dict(table(t), added=[float(x) for x in pr]))
    return dict(interfaces=lv, check=check(own, lv, cfg, mode), check_before=check(own, before['interfaces'], cfg, mode),
                steps=done, before=before, after=table(lv), probes=probe_tables,
                acceptance=acc_obs, levels=lm.table())


def even_regions(model, paths, placed, cfg, mode, log=None, max_moves=12, n_cand=20, min_gain=None, rpaths=None):
    """Even out the shooting-region lengths of a placed set (`region_frames` on `rpaths`, the last run's own paths;
    default `paths`; mean frames per staple path, turn-aware). While an ensemble's region exceeds long_ratio x the
    median over the ensembles: rank every candidate of its two pairs by the longest region of the windows the split
    creates or changes, and split at the best one that passes the gate if it shortens the region by at least
    split_gain. While one is below 1 / long_ratio of it: remove its middle interface if the merged pair keeps
    p >= merge_p_min and the gate passes. Too few frames for the sampler at all is the gate's region-frames check."""
    R = float(cfg.get('long_ratio', 2.0))
    if R <= 0:
        return placed
    min_gain = float(cfg.get('split_gain', 0.3)) if min_gain is None else min_gain
    p_max = float(cfg.get('p_max', 0.9))
    p_floor = float(cfg.get('merge_p_min', float(cfg.get('target_p', 0.3)) / 3))   # merged pair must keep p >= this
    margin = float(cfg.get('place_margin', 1.25))
    gate = dict(cfg, eps=float(cfg.get('eps', 0.01)) / margin, c=float(cfg.get('c', 1.0)) * margin,
                n_a=float(cfg.get('n_a', 3)) * margin, n_sh=float(cfg.get('n_sh', 3)) * margin, max_empty=float(cfg.get('max_empty', cfg.get('eps', 0.01))) / margin)
    steps = _steps(paths)
    rp, stat = (paths if rpaths is None else rpaths), cfg.get('region_stat', 'mean')   # region lengths: own paths
    lv = [float(v) for v in placed['interfaces']]
    lgt = lambda q: np.log(q / (1 - q))
    cl = lambda q: min(max(float(q), 1e-9), 1 - 1e-9)
    moves, tried = [], set()
    for _ in range(max_moves):
        rf = region_frames(rp, lv, stat)
        med = {e: v[0] for e, v in rf.items() if np.isfinite(v[0])}
        if len(med) < 2:
            break
        typical = float(np.median(list(med.values())))
        hs = np.r_[0.0, np.clip(model.h_of_lambda(np.asarray(lv[1:-1])), 1e-9, 1 - 1e-9), 1.0]
        long_ = sorted([e for e in med if med[e] > R * typical and ('split', lv[e - 1]) not in tried],
                       key=lambda e: -med[e])
        short = sorted([e for e in med if med[e] < typical / R and ('merge', lv[e - 1]) not in tried],
                       key=lambda e: med[e])
        done = False
        for e in long_[:1]:
            tried.add(('split', lv[e - 1]))
            k = e - 1                                              # middle interface of ensemble e
            # every candidate of both pairs of the ensemble, ranked by the longest shooting region of the windows the
            # split creates or changes (own paths, turn-aware: window_frames); the best one that passes the gate is
            # taken if it shortens the ensemble's region by split_gain. (Before 2026-11-24 only the first candidate
            # passing the gate, nearest the pair's logit midpoint, was judged, and the pair was given up if it did
            # not gain enough.)
            cands = []
            for j in (k - 1, k):
                ha, hb = max(hs[j], 1e-7), min(hs[j + 1], 1 - 1e-7)
                cand = 1 / (1 + np.exp(-np.linspace(lgt(ha), lgt(hb), n_cand + 2)[1:-1]))
                lcs = [float(v) for v in model.scale(cand)]
                if j == 0 or j + 1 == len(lv) - 1:                  # a pair touching a state: its logit midpoints sit
                    lcs = [float(v) for v in np.linspace(lv[j], lv[j + 1], n_cand + 2)[1:-1]]   # next to it: lambda
                for lc in lcs:
                    if not lv[j] < lc < lv[j + 1]:
                        continue
                    trial = lv[:j + 1] + [lc] + lv[j + 1:]
                    ht = np.r_[0.0, np.clip(model.h_of_lambda(np.asarray(trial[1:-1])), 1e-9, 1 - 1e-9), 1.0]
                    if any(0 < i_ and i_ + 1 < len(trial) - 1 and 1.0 / q_ratio(ht[i_], ht[i_ + 1]) > p_max
                           for i_ in (j, j + 1)):
                        continue                                    # a split must not make a pair crossed almost surely
                    aff = [e2 for e2 in (j + 1, j + 2, j + 3) if 2 <= e2 <= len(trial) - 1]
                    reg = [window_frames(rp, trial[e2 - 2], trial[e2 - 1], trial[e2], stat, near=_near(trial, e2))[0]
                           for e2 in aff]
                    new_long = max([r_ for r_ in reg if np.isfinite(r_)] or [np.inf])
                    cands.append((new_long, lc, j, trial))
            cands.sort(key=lambda c_: c_[0])
            for new_long, lc, j, trial in cands:
                if new_long > (1 - min_gain) * med[e]:
                    break                                           # ranked: no later candidate gains enough
                good, _ = _local_check(paths, steps, trial, j + 1, gate, mode)
                if good and check(paths, trial, gate, mode)['ok']:
                    x = float(np.clip(model.h_of_lambda(np.array([lc]))[0], 0, 1))
                    moves.append(dict(kind='split', lam=lc, q=x, ensemble=e, frames=med[e], after=new_long,
                                      typical=typical))
                    lv, done = trial, True
                    break
            if not done and cands:
                moves.append(dict(kind='skip', lam=cands[0][1], ensemble=e, frames=med[e], after=cands[0][0]))
                done = None
        if not done:
            for e in short[:1]:
                tried.add(('merge', lv[e - 1]))
                k = e - 1
                if not 1 <= k <= len(lv) - 2:
                    continue
                inner = 0 < k - 1 and k + 1 < len(lv) - 1
                p_m = 1.0 / q_ratio(cl(hs[k - 1]), cl(hs[k + 1])) if inner else 1.0
                trial = lv[:k] + lv[k + 1:]
                if p_m >= p_floor and check(paths, trial, gate, mode)['ok']:
                    moves.append(dict(kind='merge', lam=lv[k], ensemble=e, frames=med[e], typical=typical))
                    lv, done = trial, True
        if done is False:
            break
    rf = region_frames(rp, lv, stat)
    if log:
        common.say(f'placement (shooting regions): {[(m_["kind"], round(m_["lam"], 4)) for m_ in moves]}; {stat} '
                   f'frames per window {[int(v[0]) if np.isfinite(v[0]) else None for v in rf.values()]}', log=log)
    out = dict(placed, interfaces=lv, check=check(paths, lv, cfg, mode),
               region_frames={int(e): v[0] for e, v in rf.items()})
    out['refilled'] = list(placed.get('refilled', [])) + [m_ for m_ in moves if m_['kind'] == 'split']
    out['removed'] = list(placed.get('removed', [])) + [(m_['lam'], 'short shooting region') for m_ in moves
                                                        if m_['kind'] == 'merge']
    return out


def worst_ratio(model, paths, interfaces, steps=None):
    """The largest crossing-probability ratio (`q_ratio` of the model's h) between neighbouring interfaces of a set,
    the end pairs from the effective state levels (`state_levels`) as refill measures them, or, where the stored
    paths do not cross a state boundary, the forward ratio (B end: 1 / h of the last interface). Returns (ratio,
    pair)."""
    steps = _steps(paths) if steps is None else steps
    lv = [float(v) for v in interfaces]
    h = np.clip(model.h_of_lambda(np.asarray(lv)), 1e-9, 1 - 1e-9)
    h_a, h_b = state_levels(model, steps, lv[0], lv[-1])
    r = [q_ratio(h[i], h[i + 1]) for i in range(1, len(lv) - 2)]       # inner pairs
    # end pairs: from the effective state levels where the paths cross the state boundary; without such crossings
    # (state_levels falls back to 0 / 1, e.g. pass-0 paths never come back below lambda_B) the forward measure
    # instead: from the last interface B is reached with probability h, from A the first is crossed with ~1 - h
    if len(lv) > 2:
        ra = q_ratio(h_a, h[1]) if h_a > 1e-8 else 1.0 / max(1.0 - h[1], 1e-9)
        rb = q_ratio(h[-2], h_b) if h_b < 1 - 1e-8 else 1.0 / max(h[-2], 1e-9)
        r = [ra] + r + [rb]
    else:
        r = [1.0 / 1e-9]
    i = int(np.argmax(r))
    return float(r[i]), (lv[i], lv[i + 1])


def refill(model, paths, placed, gate, mode, R=10.0, n_cand=40, steps=None, max_add=10):
    """Re-insert interfaces where thinning left a pair of neighbours more than R apart in crossing probability
    (`q_ratio` of h = model.h_of_lambda; for the pairs touching a state the effective state level of
    `state_levels`, since h(lambda_A) = 0 would make every first pair look infinitely wide). For the worst such pair, `n_cand` candidates equally spaced in
    logit h between the two are tried, best split first (the smallest larger sub-ratio); the first that passes
    the given limits (`_local_check`, then the full `check` including the turns)
    is kept. Repeats until no pair over R can be split or `max_add` interfaces were added. `place` passes its
    margined limits, so a refilled interface has the same hysteresis protection as a kept one.

    Why: placement drops every planned level whose window holds too few frames, and with the margin it can empty
    a whole stretch (Z: one ensemble spanning q = 0.007 -> 0.645 after the ridge levels were removed). Where the
    dynamics cross the committor too fast for any interface (the ridge, ~10 MD steps for q 0.01 -> 0.2) no
    candidate passes and the pair stays as it is; the number of frames between two committor levels does not
    depend on how lambda labels them, so this is a sampler-resolution limit, not a lambda defect. Refilled
    interfaces meet the gate exactly, without hysteresis margin: the loop's re-check on the next pass's own
    paths (or freeze's verification rounds) decides whether they survive."""
    steps = _steps(paths) if steps is None else steps
    lv = [float(v) for v in placed]
    added, tried = [], set()
    h_a, h_b = state_levels(model, steps, lv[0], lv[-1])
    lg = lambda p: np.log(p / (1 - p))
    while len(added) < max_add:
        h = np.clip(model.h_of_lambda(np.asarray(lv)), 1e-9, 1 - 1e-9)
        h[0], h[-1] = h_a, h_b                                # effective levels of the states (see above)
        ratios = [(q_ratio(h[i], h[i + 1]), i) for i in range(len(lv) - 1) if (lv[i], lv[i + 1]) not in tried]
        ratios = [(r, i) for r, i in ratios if r > R]
        if not ratios:
            break
        r0, i = max(ratios)
        ha, hb = h[i], h[i + 1]
        hc = 1 / (1 + np.exp(-np.linspace(lg(ha), lg(hb), n_cand + 2)[1:-1]))
        cand = sorted(((max(q_ratio(ha, x), q_ratio(x, hb)), float(x)) for x in hc))
        lam_c = model.scale(np.array([x for _, x in cand]))
        ok = False
        for (sub, x), lc in zip(cand, lam_c):
            lc = float(lc)
            if not lv[i] < lc < lv[i + 1]:
                continue
            trial = lv[:i + 1] + [lc] + lv[i + 1:]
            good, _ = _local_check(paths, steps, trial, i + 1, gate, mode)
            if good and check(paths, trial, gate, mode)['ok']:
                added.append(dict(lam=lc, q=x, between=(lv[i], lv[i + 1]), ratio_before=r0, ratio_after=sub))
                lv, ok = trial, True
                break
        if not ok:
            tried.add((lv[i], lv[i + 1]))
    return lv, added


def health_table(res, pred_median=None):
    """Markdown table of the per-interface and per-ensemble checks."""
    lines = ['| k | lambda_k | up / down crossings | skip up | skip down | gap/delta | dist A | dist B | note |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in res['rows']:
        lines.append(f'| {r["k"]} | {r["lam"]:.4f} | {r["n_up"]} / {r["n_dn"]} | {_f(r["skip_up"])} | {_f(r["skip_dn"])} | '
                     f'{_f(r["gap_up"], 2)} | {_f(r.get("dist_a", np.nan), 1)} | {_f(r.get("dist_b", np.nan), 1)} | '
                     f'{"unverified" if r["unverified"] else ""} |')
    lines += ['', '| ensemble | window (L, M, R) | excursions | P(empty region) | 5% frames inside |',
              '|---|---|---|---|---|']
    for e in res['ensembles']:
        lines.append(f'| {e["e"]} | ({e["L"]:.4f}, {e["M"]:.4f}, {e["R"]:.4f}) | {e["n_excursions"]} | {_f(e["p_empty"])} | {_f(e["q05_frames"], 0)} |')
    lines.append(f'\nsingle-step turns: {_f(res["turns"]["single_step"])} of {res["turns"]["n_turns"]:.0f}; '
                 f'failures: {res["fails"] if res["fails"] else "none"}')
    return '\n'.join(lines)


def _f(x, nd=3):
    return '–' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def empty_shooting_regions(raw, interfaces, mode):
    """Per ensemble column: accepted paths (W > 0 there) and how many of them have no shooting region, by the
    sampler's own definition (infretis StaplePath.get_sh_region via infretis_data.staple_core for staples; the path
    interior for RETIS). Must be 0 in a production hand-off."""
    import infretis_data
    settings = {'interfaces': list(map(float, interfaces)), 'lambda_minus_one': None}
    W = np.asarray(raw['W'])
    out = {}
    for e in range(1, len(interfaces)):
        n = empty = 0
        for i, op in enumerate(raw['ops']):
            if W[i, e] <= 0:
                continue
            n += 1
            if mode == 'staple':
                mid = None
                for key in ('LMR', 'LML', 'RML', 'RMR'):
                    if key in raw['ptype'][i]:
                        mid = key
                gen = (int(raw['gen_ens'][i]), mid) if (mid and raw['gen_ens'][i] > -2) else None
                core = infretis_data.staple_core(np.asarray(op), settings, e, gen)
                if core is None or core[1] < core[0]:
                    empty += 1
            elif len(op) < 3:
                empty += 1
        out[e] = dict(n=n, empty=empty, fraction=empty / n if n else np.nan)
    return out
