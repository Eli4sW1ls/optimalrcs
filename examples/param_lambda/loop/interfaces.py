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
            if ws['p_empty'] > eps:
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
               n_a=float(cfg.get('n_a', 3)) * margin, n_sh=float(cfg.get('n_sh', 3)) * margin)
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
        if ws['n_excursions'] >= 20 and (ws['p_empty'] > eps or (e >= 2 and ws['q05_frames'] < n_sh)):
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
    gate = dict(cfg, eps=eps, c=c, n_a=n_a, n_sh=float(cfg.get('n_sh', 3)) * margin)
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


def window_frames(paths, L, M, R):
    """(weighted median frames, number of runs) of the shooting region of one window (L, M, R), measured on the
    stored paths: the runs inside (L, R) whose excursion crosses M (seed.window_cores)."""
    from loop import seed as seed_mod
    n_in, wts = [], []
    for lam, _, wt in paths:
        for a_, b_ in seed_mod.window_cores(lam, L, M, R):
            n_in.append(b_ - a_ + 1)
            wts.append(wt)
    if not n_in:
        return (np.nan, 0)
    o = np.argsort(n_in)
    c = np.cumsum(np.asarray(wts, float)[o]) / np.sum(wts)
    return (float(np.asarray(n_in)[o][np.searchsorted(c, 0.5)]), len(n_in))


def region_frames(paths, interfaces):
    """Shooting-region length per staple ensemble of an interface set, measured on the stored paths: the weighted
    median number of frames of the runs inside the window (lambda_{i-1}, lambda_{i+1}) that cross lambda_i
    (seed.window_cores, the runs the region-frames check counts). Returns {e: (median, n)} for infretis ensembles
    e = 2 .. n-1."""
    lv = [float(v) for v in interfaces]
    return {e: window_frames(paths, lv[e - 2], lv[e - 1], lv[e]) for e in range(2, len(lv))}


def balance_regions(model, paths, placed, cfg, mode, log=None, sweeps=3, n_cand=14, min_gain=0.01):
    """Make the shooting-region lengths uniform: move interior interfaces, one at a time, to the position between
    their neighbours that minimises the squared spread of ln(median frames) over the ensembles (`region_frames`) plus
    `balance_p_weight` (default 0.3) x the sum of (ln(p / target_p))^2 over all pairs, the end pairs included, keeping
    only moves that pass the local gate (skip rates to the neighbours above all: a path that crosses an interface
    must not reach the next one in one step) and leave both adjacent pairs with a crossing probability of at least
    target_p / 3 and at most p_max (default 0.9, unless the pair was already above it). Sweeps over the interfaces until no move improves the spread by `min_gain` (relative)."""
    p_floor = float(cfg.get('target_p', 0.3)) / 3
    p_max = float(cfg.get('p_max', 0.9))
    margin = float(cfg.get('place_margin', 1.25))
    gate = dict(cfg, eps=float(cfg.get('eps', 0.01)) / margin, c=float(cfg.get('c', 1.0)) * margin,
                n_a=float(cfg.get('n_a', 3)) * margin, n_sh=float(cfg.get('n_sh', 3)) * margin, end_dist=True,
                sigma_s=rms_step_s(paths))
    steps = _steps(paths)
    lv = [float(v) for v in placed['interfaces']]
    cache = {}
    target_p, w_p = float(cfg.get('target_p', 0.3)), float(cfg.get('balance_p_weight', 0.3))
    h_a, h_b = state_levels(model, steps, lv[0], lv[-1])
    h_a_known, h_b_known = h_a > 1e-8, h_b < 1 - 1e-8

    def win(L, M, R):
        key = (round(L, 9), round(M, 9), round(R, 9))
        if key not in cache:
            cache[key] = window_frames(paths, L, M, R)[0]
        return cache[key]

    def spread(v):
        f = np.array([win(v[e - 2], v[e - 1], v[e]) for e in range(2, len(v))])
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
    rf = region_frames(paths, lv)
    if log:
        common.say(f'placement (balance): {len(moves)} moves, objective {start:.2f} -> {J:.2f}; '
                   f'median frames per window {[int(v[0]) if np.isfinite(v[0]) else None for v in rf.values()]}', log=log)
    return dict(placed, interfaces=lv, check=check(paths, lv, cfg, mode), balanced=moves,
                region_frames={int(e): v[0] for e, v in rf.items()})


def even_regions(model, paths, placed, cfg, mode, log=None, max_moves=12, n_cand=20, min_gain=None):
    """Even out the shooting-region lengths of a placed set (`region_frames`): while an ensemble's median frames
    in its window exceed long_ratio x the median over the ensembles, split the wider (in logit q) of its two pairs
    at the candidate closest to the logit midpoint that passes the frame checks; while one is below 1 / long_ratio
    of it, remove its middle interface if the merged pair keeps a crossing probability >= target_p / 3 and the
    checks pass. Too few frames for the sampler at all is the gate's region-frames check (hard)."""
    R = float(cfg.get('long_ratio', 2.0))
    if R <= 0:
        return placed
    min_gain = float(cfg.get('split_gain', 0.3)) if min_gain is None else min_gain
    p_max = float(cfg.get('p_max', 0.9))
    p_floor = float(cfg.get('target_p', 0.3)) / 3
    margin = float(cfg.get('place_margin', 1.25))
    gate = dict(cfg, eps=float(cfg.get('eps', 0.01)) / margin, c=float(cfg.get('c', 1.0)) * margin,
                n_a=float(cfg.get('n_a', 3)) * margin, n_sh=float(cfg.get('n_sh', 3)) * margin)
    steps = _steps(paths)
    lv = [float(v) for v in placed['interfaces']]
    lgt = lambda q: np.log(q / (1 - q))
    cl = lambda q: min(max(float(q), 1e-9), 1 - 1e-9)
    moves, tried = [], set()
    for _ in range(max_moves):
        rf = region_frames(paths, lv)
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
            cw = lambda q: min(max(float(q), 1e-4), 1 - 1e-4)                # width ranking: a state end is not infinitely wide
            pairs = sorted([(abs(lgt(cw(hs[j + 1])) - lgt(cw(hs[j]))), j) for j in (k - 1, k)], reverse=True)
            for _, j in pairs:
                ha, hb = max(hs[j], 1e-7), min(hs[j + 1], 1 - 1e-7)
                mid = 0.5 * (lgt(ha) + lgt(hb))
                cand = 1 / (1 + np.exp(-np.linspace(lgt(ha), lgt(hb), n_cand + 2)[1:-1]))
                lcs = [float(v) for v in model.scale(cand[np.argsort(np.abs(lgt(cand) - mid))])]
                if j == 0 or j + 1 == len(lv) - 1:                  # a pair touching a state: its logit midpoint sits
                    mid_l = 0.5 * (lv[j] + lv[j + 1])               # next to the state, so try lambda midpoints
                    lcs = sorted(np.linspace(lv[j], lv[j + 1], n_cand + 2)[1:-1], key=lambda v: abs(v - mid_l))
                for lc in lcs:
                    x = float(np.clip(model.h_of_lambda(np.array([lc]))[0], 0, 1))
                    if not lv[j] < lc < lv[j + 1]:
                        continue
                    trial = lv[:j + 1] + [lc] + lv[j + 1:]
                    ht = np.r_[0.0, np.clip(model.h_of_lambda(np.asarray(trial[1:-1])), 1e-9, 1 - 1e-9), 1.0]
                    if any(0 < i_ and i_ + 1 < len(trial) - 1 and 1.0 / q_ratio(ht[i_], ht[i_ + 1]) > p_max
                           for i_ in (j, j + 1)):
                        continue                                    # a split must not make a pair crossed almost surely
                    good, _ = _local_check(paths, steps, trial, j + 1, gate, mode)
                    if good and check(paths, trial, gate, mode)['ok']:
                        # keep the split only if it shortens the longest of the new windows' regions clearly
                        # (flat lambda: a split barely shortens it, and more levels only add ensembles)
                        rf2 = region_frames(paths, trial)
                        new_long = max([rf2[e2][0] for e2 in (j + 1, j + 2, j + 3) if e2 in rf2
                                        and np.isfinite(rf2[e2][0])] or [np.inf])
                        if new_long > (1 - min_gain) * med[e]:
                            moves.append(dict(kind='skip', lam=lc, ensemble=e, frames=med[e], after=new_long))
                            done = None
                            break
                        moves.append(dict(kind='split', lam=lc, q=float(x), ensemble=e, frames=med[e], typical=typical))
                        lv, done = trial, True
                        break
                if done is not False:
                    break
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
    rf = region_frames(paths, lv)
    if log:
        common.say(f'placement (shooting regions): {[(m_["kind"], round(m_["lam"], 4)) for m_ in moves]}; median '
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
