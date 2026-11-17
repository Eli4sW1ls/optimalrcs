"""Interface placement: a plan in the natural coordinate, a cost check and the spacing gate (constraint 9).

    plan   = plan_natural(model, view, cfg)                 # levels equally spaced in u(h), ratio cap, length budget
    placed = place(model, paths, plan, cfg, mode)           # greedy thinning until every check passes, then refill
    table  = check(paths, interfaces, cfg, mode)            # all checks on a set of sampled paths

`paths` are whole sampled paths at the sampler's resolution (one lambda value per saved frame), each with a
weight: (lam, w) pairs from `path_arrays`. Checks (defaults eps = 1%, c = 1, n_A = 3, n_sh = 3):

- skip, both directions: P(a step crossing lambda_k also crosses the next interface) <= eps, where lambda_k has
  at least 100 crossings in that direction;
- gap: lambda_{k+1} - lambda_k >= c delta(lambda_k), delta = 95% quantile of |d lambda| over the steps crossing
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
        if np.isfinite(row['gap_up']) and row['gap_up'] < c:
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
    pred_longest = None
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
        if np.isfinite(delta) and lv - placed[-1] < c * delta:
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
        bad = (np.isfinite(st['delta']) and lb - placed[-1] < c * st['delta']) or \
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
    for j in (k - 1, k, k + 1):
        if j <= 0 or j >= len(lv) - 1:
            continue
        st = crossing_stats(steps, lv[j], nxt_up=lv[j + 1], nxt_dn=lv[j - 1])
        if st['n_up'] >= 100 and np.isfinite(st['skip_up']) and st['skip_up'] > eps:
            return False, f'skip_up at {lv[j]:.4f}'
        if st['n_dn'] >= 100 and np.isfinite(st['skip_dn']) and st['skip_dn'] > eps:
            return False, f'skip_dn at {lv[j]:.4f}'
        if np.isfinite(st['delta']) and (lv[j + 1] - lv[j] < c * st['delta'] or lv[j] - lv[j - 1] < c * st['delta']):
            return False, f'gap at {lv[j]:.4f}'
        if not np.isfinite(st['delta']):
            return False, f'no crossings of {lv[j]:.4f}'           # refill only where the paths verify the level
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
