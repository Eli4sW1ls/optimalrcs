"""Seed paths: one valid initial path per ensemble of a new (lambda, interfaces) set, cut from stored paths.

The new lambda is evaluated on the recorded CV pool of every stored path (constraint 2 keeps the states exact,
so start/end-in-state conditions survive the switch). Per ensemble type:

- RETIS [0-]: a stored path that lies in A except for its two end frames.
- RETIS [i+]: a segment of a stored path from a frame in A to the next frame in A or B whose maximum lambda
  exceeds lambda_i (whole stored A-started paths, or the A-excursions of any path).
- staple [i*] (infretis windows (L, M, R) = (lambda_{i-1}, lambda_i, lambda_{i+1}); [0*] = (lambda_0, lambda_0,
  lambda_1)): a window segment that crosses M as the core, followed outward on both sides along the stored path
  until the sampler's turn rule (infretis `turn_detected`) fires or A / B is reached. The candidate is validated
  with infretis' own StaplePath (check_turns, get_sh_region, get_pptype).

Candidates are taken from the most recent passes first, untainted paths before tainted ones (constraint 7), each
stored path used once per seed set when possible. If an ensemble gets no candidate, `SeedError` names it.

The seeds are written as infretis load directories: order.txt = [lambda_new, cv_0, ...] per frame, traj.txt rows
of the stored path's frames (the trajectory files hard-linked into accepted/, so lineage inodes carry over).
"""
import os
import shutil

import numpy as np

import infretis_data

from loop.backends import infretis_backend as be


class SeedError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------
# cutting rules on a lambda array
# ---------------------------------------------------------------------------------------------------
def retis_segments(lam, lambda_a, lambda_b):
    """(start, end) of every segment that starts with a frame in A and ends with the next frame in A or B."""
    in_a, in_b = lam <= lambda_a, lam >= lambda_b
    out = []
    starts = np.flatnonzero(in_a[:-1] & ~in_a[1:])          # an A frame followed by a frame outside A
    for s in starts:
        nxt = np.flatnonzero(in_a[s + 1:] | in_b[s + 1:])
        if len(nxt):
            out.append((int(s), int(s + 1 + nxt[0])))
    return out


def is_zero_minus(lam, lambda_a):
    """infretis' [0-] signature: all frames but at most two (the ends) in A, and frame 1 in A."""
    return len(lam) > 2 and np.count_nonzero(lam > lambda_a) <= 2 and lam[1] < lambda_a and lam[0] > lambda_a and lam[-1] > lambda_a


def window_cores(lam, L, M, R):
    """(start, end) of every maximal run of frames strictly inside (L, R) whose excursion (the run plus the
    frames just before and after it) crosses M."""
    inside = (lam > L) & (lam < R)
    if not inside.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[False, inside, False]))
    out = []
    for a, b in zip(edges[::2], edges[1::2] - 1):            # inclusive run [a, b]
        lo, hi = max(a - 1, 0), min(b + 1, len(lam) - 1)
        seg = lam[lo:hi + 1]
        if seg.min() < M <= seg.max() and a > 0 and b < len(lam) - 1:
            out.append((int(a), int(b)))
    return out


def extend_turn(lam, interfaces, m_idx, j, direction):
    """Follow the stored path from frame j outward (direction +1 forward, -1 backward) until infretis'
    turn_detected fires or A / B is reached. Returns the index of the last frame of the turn, or None.

    Vectorized form of the sampler's rule (infretis staple_path.turn_detected applied after every new frame, on the
    orders [lam[j], lam[j + d], ...]): with lr the side of the window the first new frame lies on and E_t the running
    extremum (max for lr = +1, min for lr = -1), the extension stops at the first t >= 2 where the new frame is in A
    or B, or lr * o_t <= lr * lambda_M, or o_t is not the extremum, at least two interfaces lie at or below (above)
    E_t, and lr * o_t <= lr * cond(E_t) with cond the second-highest (second-lowest) of them. (A point after the
    extremum that satisfies the condition fires at its own step, so the first such t is the stopping step; the
    first check, at three frames, also covers frames 0 and 1.) Checked against turn_detected on random walks.)"""
    intf = np.asarray(interfaces, float)
    n_i = len(intf)
    L, R = intf[max(m_idx - 1, 0)], intf[min(m_idx + 1, n_i - 1)]
    if m_idx == 0:
        L = intf[0]
    k = j + direction
    if not 0 <= k < len(lam):
        return None
    if lam[k] <= intf[0] or lam[k] >= intf[-1]:            # the sampler stops on entering A or B
        return int(k)
    lr = -1 if lam[k] < L else (1 if lam[k] > R else None)
    if lr is None:
        return None
    o = lam[j::direction] if direction > 0 else lam[j::-1]
    t = np.arange(len(o))
    E = np.maximum.accumulate(o) if lr == 1 else np.minimum.accumulate(o)
    is_ext = (o == E)
    if lr == 1:
        cnt = np.searchsorted(intf, E, side='right')          # interfaces <= E
        ok2 = cnt >= 2
        cond = intf[np.clip(cnt - 2, 0, n_i - 1)]
        fire_turn = ok2 & ~is_ext & (o <= cond)
    else:
        cnt = n_i - np.searchsorted(intf, E, side='left')      # interfaces >= E
        ok2 = cnt >= 2
        cond = intf[np.clip(n_i - cnt + 1, 0, n_i - 1)]
        fire_turn = ok2 & ~is_ext & (o >= cond)
    fire = (o <= intf[0]) | (o >= intf[-1]) | (lr * o <= lr * intf[m_idx]) | fire_turn
    fire &= t >= 2
    # the first check (3 frames) also sees frames 0 and 1 after the extremum, which had no check of their own
    if len(o) > 2 and ok2[2]:
        e2 = int(np.argmax(o[:3]) if lr == 1 else np.argmin(o[:3]))
        if e2 < 2 and np.any(lr * o[e2:3] <= lr * cond[2]):
            fire[2] = True
    hit = np.flatnonzero(fire)
    if not len(hit):
        return None
    return int(j + direction * hit[0])


def staple_candidates(lam, interfaces, e, max_candidates=5):
    """(start, end) candidates of a staple path for infretis ensemble column e (>= 1), cut from a stored path."""
    L, M, R = infretis_data.ensemble_window({'interfaces': list(interfaces), 'lambda_minus_one': None}, e)
    m_idx = e - 1 if e >= 2 else 0
    out = []
    for a, b in window_cores(lam, L, M, R):
        end = extend_turn(lam, interfaces, m_idx, b, +1)
        start = extend_turn(lam, interfaces, m_idx, a, -1)
        if start is not None and end is not None and end - start >= 3:
            out.append((start, end))
        if len(out) >= max_candidates:
            break
    return out


def validate_staple(op, interfaces, e):
    """infretis' own validity of a candidate staple path (order array (n, k), column 0 = lambda) in ensemble
    column e: valid turns, a shooting region, a path type; and the [1+] rule of REPEX_state_staple.load_paths.
    Returns (ok, pptype, sh_region)."""
    settings = {'interfaces': list(interfaces), 'lambda_minus_one': None}
    path = infretis_data._staple_path(op)
    st, end, valid = path.check_turns(list(interfaces))
    if not valid:
        return False, '***', (0, 0)
    window = infretis_data.ensemble_window(settings, e)
    try:
        pptype = path.get_pptype(list(interfaces), window)
        region = path.get_sh_region(list(interfaces), window)
    except Exception:
        return False, '***', (0, 0)
    if pptype == '***' or region == (0, 0) or region[0] > region[1] or region[0] <= 0 or region[1] >= len(op) - 1:
        return False, pptype, region
    if e == 2 and st[1] == end[1] == 0 and pptype != 'LML':
        return False, pptype, region
    if e == 1 and pptype == 'RMR':
        return False, pptype, region
    return True, pptype, region


# ---------------------------------------------------------------------------------------------------
# seed sets from the pool
# ---------------------------------------------------------------------------------------------------
def _stored_paths(pool, passes):
    """Stored whole paths, most recent pass first, untainted first within a pass (CVs only; lambda is evaluated
    lazily)."""
    out = []
    for k in sorted(passes, reverse=True):
        z, m = pool.load(k)
        starts = np.r_[0, np.cumsum(z['raw_n'])[:-1]]
        order = np.argsort(z['raw_tainted'], kind='stable')              # untainted first
        for i in order:
            out.append(dict(pass_index=k, workdir=m['workdir'], pnr=int(z['raw_pnr'][i]),
                            cv=z['raw_cv'][starts[i]:starts[i] + z['raw_n'][i]].astype(float),
                            tainted=bool(z['raw_tainted'][i])))
    return out


def build_seeds(pool, model, interfaces, mode, out_dir, passes=None, max_paths=3000, max_staple_len=None, log=None):
    """One seed path directory per ensemble under out_dir/<i>. Returns (dirs, tainted_inodes, report).

    Paths are scanned in preference order (most recent pass, untainted, not yet used) and the first valid candidate
    per ensemble is taken, so lambda and the validity checks run only on the paths actually inspected."""
    passes = pool.passes() if passes is None else passes
    paths = _stored_paths(pool, passes)[:max_paths]
    n = len(interfaces)
    la, lb = float(interfaces[0]), float(interfaces[-1])

    def lam_of(p):
        if 'lam' not in p:
            p['lam'] = model(p['cv'][:, 1:])                              # new lambda on the CV pool
        return p['lam']

    def candidates(e, p):
        lam = lam_of(p)
        if e == 0:
            return [(0, len(lam) - 1, {})] if is_zero_minus(lam, la) else []
        if mode == 'retis':
            lam_i = interfaces[max(e - 1, 0)]
            return [(s, t, dict(max_lambda=float(lam[s:t + 1].max()))) for s, t in retis_segments(lam, la, lb)
                    if lam[s:t + 1].max() > lam_i and t - s >= 2]
        out = []
        for s, t in staple_candidates(lam, interfaces, e):
            if max_staple_len and t - s + 1 > max_staple_len:
                continue
            op = np.column_stack([lam[s:t + 1], p['cv'][s:t + 1, 1:]])
            ok, pptype, region = validate_staple(op, interfaces, e)
            if ok:
                out.append((s, t, dict(pptype=pptype, sh_region=region)))
                break                                                     # one per path is enough
        return out

    chosen, report, used = {}, {}, set()
    for e in range(n):
        found = None
        for prefer_unused in (True, False):
            for p in paths:
                key = (p['pass_index'], p['pnr'])
                if prefer_unused and key in used:
                    continue
                if not os.path.isfile(os.path.join(p['workdir'], 'load', str(p['pnr']), 'traj.txt')):
                    continue                          # the source run's directory was cleaned up: not a usable seed
                c = candidates(e, p)
                if c:
                    found = (p, c[0])
                    break
            if found:
                break
        if not found:
            name = '[0-]' if e == 0 else ('[%d+]' % (e - 1) if mode == 'retis' else '[%d*]' % (e - 1))
            raise SeedError(f'ensemble {e} ({name}): no stored path yields a valid path for interfaces '
                            f'{np.round(interfaces, 4).tolist()}')
        p, (s, t, info) = found
        used.add((p['pass_index'], p['pnr']))
        chosen[e] = (p, s, t)
        report[e] = dict(pass_index=p['pass_index'], pnr=p['pnr'], frames=(s, t), tainted=p['tainted'], **info)
        if log:
            import common
            common.say(f'seed {e}: pass {p["pass_index"]} path {p["pnr"]} frames {s}-{t}', log=log)
    dirs, tainted = [], set()
    for e in range(n):
        p, s, t = chosen[e]
        dst = os.path.join(out_dir, str(e))
        write_seed(dst, os.path.join(p['workdir'], 'load', str(p['pnr'])), s, t, p['lam'][s:t + 1], p['cv'][s:t + 1, 1:],
                   note=f'seed from pass {p["pass_index"]} path {p["pnr"]} frames {s}-{t}')
        dirs.append(dst)
        if p['tainted']:
            tainted |= be.path_inodes(dst)
    return dirs, tainted, report


def write_seed(dst, src_dir, start, end, lam, cvs, note=''):
    """Write one load path directory from frames [start, end] of the stored path in src_dir."""
    rows = []
    with open(os.path.join(src_dir, 'traj.txt')) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            rows.append((parts[1], int(parts[2]), int(parts[3])))
    if end >= len(rows):
        raise ValueError(f'{src_dir}: traj.txt has {len(rows)} rows, order.txt frames up to {end}')
    os.makedirs(os.path.join(dst, 'accepted'))
    files = sorted({r[0] for r in rows[start:end + 1]})
    for fn in files:
        src = os.path.join(os.path.realpath(src_dir), 'accepted', fn)
        try:
            os.link(src, os.path.join(dst, 'accepted', fn))
        except OSError:
            shutil.copy2(src, os.path.join(dst, 'accepted', fn))
    with open(os.path.join(dst, 'traj.txt'), 'w') as f:
        f.write(f'# {note}\n#     Step              Filename       index    vel\n')
        for i, (fn, idx, vel) in enumerate(rows[start:end + 1]):
            f.write(f'{i:>10}  {fn:>20s}  {idx:>10}  {vel:>5}\n')
    with open(os.path.join(dst, 'order.txt'), 'w') as f:
        f.write(f'# {note}\n#     Time       Orderp\n')
        for i in range(len(lam)):
            f.write(f'{i:>10d}' + ''.join(f'{v:>12.6f}' for v in [lam[i]] + list(cvs[i])) + '\n')
    return dst
