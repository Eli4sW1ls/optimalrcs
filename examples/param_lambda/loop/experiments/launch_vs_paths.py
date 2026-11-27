#!/usr/bin/env python3
"""Launched short trajectories vs path sampling as teacher data: a controlled test on Z (loop_z_dp).

Common start (the loop after pass 0): the infinit paths of loop_z_dp (pool passes 0-4) and lambda_1 (its model
after pass 0, with the pass-1 interfaces). Each arm then spends the same extra MD budget B (MD steps):

    P   path sampling: a StapleTIS run with lambda_1 and the pass-1 interfaces (what the loop does next), seeds cut
        from the infinit paths only; pieces as the loop takes them (cut at M, path_f weights, mirrored, burn-in 0.2)
    L   launches: trajectories started from physical frames of the infinit paths, launch points stratified in
        equal numbers over bins of logit q under lambda_1; a share `abs_frac` of the budget runs to absorption
        (A or B, cap `lmax` steps), the rest as short trajectories of `ls` steps (stopped early in A or B);
        unweighted, not mirrored
    M   mix: the first half of the P run's rows + launches for the rest of the budget
    base  the infinit paths alone (no extra budget): the reference the arms improve on

Each arm is fitted twice: `+inf` (with a fixed subsample of the infinit pieces, the loop's situation: pass-2
teacher) and `alone` (the arm's own data: the quality of the source). Every data source is normalised to mean
weight 1 per frame before merging (the fair-bet equation holds for each source on its own, so any positive mix is
consistent; equal per-frame weight is the neutral choice). Then the teacher (NPNEq, warm start from lambda_1's q,
no provisional B) and the student (M3 + export, the loop_z_dp_u settings) are fitted exactly as in the loop.

Scored against the exact committor of the EM chain:
  - teacher labels on the arm's own frames, per band of exact q: mean and RMS of logit q_teacher - logit q_exact
  - the exported lambda on a fixed evaluation set (old loop_z_dp verification runs, pool passes 12-17, not used by
    any arm): bias and RMS of logit m - logit q_exact per band of exact q (small q, the plateau, near B), the
    isosurface spread at planned levels and the binned logit RMSE (diagnostics.exact_metrics)
  - where the data lie (frames per band) and how many paths / trajectories reach B

Stages (from examples/param_lambda; the experiment directory is <OUT_ROOT>/op_loop/<exp>, default launch_test):
    python -m loop.experiments.launch_vs_paths setup  [--seeds 1 2 3] [--cycles 3400]   P run dirs, not started
    python -m loop.experiments.launch_vs_paths run-p  <seed>                             the P run (long: systemd)
    python -m loop.experiments.launch_vs_paths ingest <seed>                             P and P/2 into the pool
    python -m loop.experiments.launch_vs_paths launch <seed>                             L (budget = P) and L/2
    python -m loop.experiments.launch_vs_paths fit    <seed> <arm> <base|alone|inf>      teacher + student + scores
    python -m loop.experiments.launch_vs_paths report                                    table over seeds
Options for every stage: --exp NAME. Launch options: --ls, --abs-frac, --lmax, --bins. Fit options: --iters,
--inf-frames.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

import common  # noqa: E402
from loop import env, controller, manifest, pool as pool_mod, seed as seed_mod, ase_systems  # noqa: E402,F401
from loop.backends import infretis_backend as be  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402

SRC_CFG = 'loop_z_dp'               # the source loop (infinit paths, lambda_1, verification runs for evaluation)
FIT_CFG = 'loop_z_dp_u'             # teacher / student settings of the fits
INF = [0, 1, 2, 3, 4]               # infinit pool passes of the source loop
EVAL = [12, 13, 14, 15, 16, 17]     # verification runs of the source loop: the evaluation set
BANDS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.7, 0.9, 0.99, 1 - 1e-4]   # band edges in exact q
LEVELS = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 0.8, 0.95]
QCLIP = 1e-9


def _src():
    cfg = controller.load_loop_config(SRC_CFG)
    S = json.load(open(os.path.join(cfg['loop_dir'], 'state.json')))
    rec = S['passes'][0]                                        # pass 1: lambda_1 and its interfaces
    return cfg, rec['model'], [float(v) for v in rec['interfaces']]


def _exp(name):
    d = os.path.join(common.OUT_ROOT, 'op_loop', name)
    os.makedirs(os.path.join(d, 'pool'), exist_ok=True)
    return d


def _log(exp):
    return os.path.join(exp, 'experiment.log')


def _arms(exp):
    f = os.path.join(exp, 'arms.json')
    return json.load(open(f)) if os.path.isfile(f) else {}


def _save_arms(exp, arms):
    json.dump(arms, open(os.path.join(exp, 'arms.json'), 'w'), indent=1)


def _pass_id(kind, seed):
    """Pool pass index of an arm's data: 100 + 10 seed + {P 0, Ph 1, L 2, Lh 3}."""
    return 100 + 10 * int(seed) + {'P': 0, 'Ph': 1, 'L': 2, 'Lh': 3}[kind]


def _lz(q):
    q = np.clip(np.asarray(q, float), QCLIP, 1 - QCLIP)
    return np.log(q) - np.log1p(-q)


# ------------------------------------------------------------------------------------------------------------------
# setup and the P runs
# ------------------------------------------------------------------------------------------------------------------
def setup(a):
    exp = _exp(a.exp)
    cfg, model_path, intfs = _src()
    src_pool = pool_mod.Pool(cfg['loop_dir'])
    pool = pool_mod.Pool(exp)
    for k in INF:                                               # the infinit passes, read-only links
        for f in (src_pool.path(k), src_pool.meta_path(k)):
            dst = os.path.join(pool.dir, os.path.basename(f))
            if not os.path.lexists(dst):
                os.symlink(f, dst)
    system, run = cfg['system_spec'], dict(cfg['run'])
    model = LambdaModel(model_path)
    for s in a.seeds:
        rd = os.path.join(exp, f'p_s{s}', 'run')
        if os.path.isdir(rd):
            print(f'{rd} exists, left as is'); continue
        dirs, tainted, rep = seed_mod.build_seeds(src_pool, model, intfs, 'staple', os.path.join(exp, f'p_s{s}', 'seeds'),
                                                  passes=INF)
        be.write_run(rd, system, run, intfs, be.learned_order_table(system, model_path), int(a.cycles), 'staple', dirs,
                     tainted_inodes=tainted, shooting=run.get('shooting', 'sh'), seed=7000 + int(s),
                     note=f'launch_vs_paths arm P, seed {s}: lambda_1 and the pass-1 interfaces of {SRC_CFG}')
        print(f'seed {s}: P run prepared in {rd} ({a.cycles} cycles, {len(intfs)} interfaces)')
    manifest.write(os.path.join(exp, 'setup.json'), source=SRC_CFG, fit_config=FIT_CFG, model=model_path,
                   interfaces=intfs, infinit_passes=INF, eval_passes=EVAL, cycles=int(a.cycles), seeds=list(a.seeds))


def run_p(a):
    exp = _exp(a.exp)
    rd = os.path.join(exp, f'p_s{a.seed}', 'run')
    steps = int(be.read_toml(os.path.join(rd, 'infretis.toml'))['simulation']['steps'])
    t0 = time.time()
    if not os.path.isfile(os.path.join(rd, 'restart.toml')):
        be.run(rd)
    elif be.cost(rd)['cycles'] < steps:
        be.resume(rd, steps)
    common.say(f'P run seed {a.seed}: {be.cost(rd)} ({time.time() - t0:.0f} s)', log=_log(exp))


def _store_run(pool, k, rd, data_file, meta_extra, system, cap):
    pieces = be.load_pieces(rd, burn_in_fraction=0.0, paths_per_ensemble=cap, weighting='rejections', data_file=data_file)
    raw = be.raw_paths(rd, data_file=data_file)
    pieces['q_exact'] = ase_systems.exact_committor(system)(pieces['cv'][:, 1:]).astype(np.float32)
    meta = dict(kind='pass', mode='staple', lambda_a=system['lambda_a'], lambda_b=system['lambda_b'],
                cv_names=system['cv_names'], state_cv=system['state_cv'], n_rows=len(raw['rows']), workdir=rd, **meta_extra)
    return pool.store(k, pieces, raw, meta)


def ingest(a):
    exp = _exp(a.exp)
    cfg, model_path, intfs = _src()
    system, cap = cfg['system_spec'], cfg['pool'].get('cap_paths_per_ensemble')
    pool = pool_mod.Pool(exp)
    rd = os.path.join(exp, f'p_s{a.seed}', 'run')
    cost = be.cost(rd)
    df = os.path.join(rd, 'infretis_data.txt')
    lines = open(df).read().splitlines()
    head = [ln for ln in lines if ln.startswith('#')]
    rows = [ln for ln in lines if ln.strip() and not ln.startswith('#')]
    half = os.path.join(rd, 'infretis_data_half.txt')
    open(half, 'w').write('\n'.join(head + rows[:len(rows) // 2]) + '\n')
    # MD cost of the first half of the rows: in proportion to the rows (the rows end in about equal numbers per cycle)
    md_half = int(round(cost['md_steps'] * (len(rows) // 2) / len(rows)))
    arms = _arms(exp)
    for kind, f, md in (('P', df, cost['md_steps']), ('Ph', half, md_half)):
        k = _pass_id(kind, a.seed)
        m = _store_run(pool, k, rd, f, dict(interfaces=intfs, lambda_model=model_path, md_steps=int(md),
                                            cycles=cost['cycles'] if kind == 'P' else cost['cycles'] // 2), system, cap)
        arms[f'{kind}_s{a.seed}'] = dict(pass_index=k, md_steps=int(md), frames=m['n_frames'], pieces=m['n_pieces'])
        common.say(f'ingest seed {a.seed} {kind}: pass {k}, {m["n_frames"]} frames, {m["n_pieces"]} pieces, '
                   f'{md} MD steps', log=_log(exp))
    _save_arms(exp, arms)


# ------------------------------------------------------------------------------------------------------------------
# launches
# ------------------------------------------------------------------------------------------------------------------
class EM:
    """The Euler-Maruyama step of the Z system (force with gravity, as the validated FFS / brute force)."""

    def __init__(self, system, rng):
        from loop.ase_calcs import load_toytis_potential
        import toy2d
        pot = load_toytis_potential(system['potential'], system['potential_kwargs'])
        self.F = toy2d.force_function(pot)
        self.c = system['dt'] / system['gamma']
        self.sig = np.sqrt(2 * system['kT'] * system['dt'] / system['gamma'])
        self.la, self.lb = system['lambda_a'], system['lambda_b']
        self.rng = rng

    def run(self, y0, x0, nmax):
        """Trajectories from (y0, x0) for up to nmax steps, each stopped at its first frame in A (y <= lambda_A) or B
        (y >= lambda_B). Returns a list of (n_i, 2) arrays (start frame included) and the steps taken."""
        n = len(y0)
        Y = np.full((nmax + 1, n), np.nan); X = np.full((nmax + 1, n), np.nan)
        Y[0], X[0] = y0, x0
        y, x = np.array(y0, float), np.array(x0, float)
        idx = np.arange(n)
        length = np.full(n, nmax + 1)
        done = (y <= self.la) | (y >= self.lb)
        length[done] = 1
        y, x, idx = y[~done], x[~done], idx[~done]
        steps = 0
        for t in range(1, nmax + 1):
            if not idx.size:
                break
            _, fy, fx = self.F(y, x)
            y = y + self.c * fy + self.sig * self.rng.standard_normal(y.size)
            x = x + self.c * fx + self.sig * self.rng.standard_normal(x.size)
            Y[t, idx], X[t, idx] = y, x
            steps += idx.size
            stop = (y <= self.la) | (y >= self.lb)
            length[idx[stop]] = t + 1
            y, x, idx = y[~stop], x[~stop], idx[~stop]
        return [np.c_[Y[:length[i], i], X[:length[i], i]] for i in range(n)], steps


def launch_points(exp_pool, model, system, n_cand, rng):
    """Candidate launch frames: physical (not mirrored) frames of the infinit pieces strictly between the states,
    `n_cand` at random, with logit q under lambda_1."""
    cvs = []
    for k in INF:
        with np.load(exp_pool.path(k)) as z:
            n = z['piece_n']; mir = np.repeat(z['piece_mirrored'], n)
            cv = z['cv'][:, 1:][~mir]
        s = cv[:, system['state_cv']]
        cvs.append(cv[(s > system['lambda_a']) & (s < system['lambda_b'])])
    cv = np.concatenate(cvs)
    cv = cv[rng.choice(len(cv), min(n_cand, len(cv)), replace=False)].astype(float)
    return cv, _lz(model.q(cv))


def make_launches(a, budget, seed, model, system, cand, zc, log):
    rng = np.random.default_rng(10_000 + seed)
    em = EM(system, rng)
    lo, hi = np.quantile(zc, 0.001), np.quantile(zc, 0.999)
    edges = np.linspace(lo, hi, a.bins + 1)
    b = np.clip(np.searchsorted(edges, zc, side='right') - 1, 0, a.bins - 1)
    members = [np.flatnonzero(b == i) for i in range(a.bins)]
    members = [m for m in members if len(m)]
    nb = len(members)

    turn = [0]

    def pick(n):                                                # round robin over the bins (the position carries over
        bi = (turn[0] + np.arange(n)) % nb                      # between batches), a random frame in each
        turn[0] += n
        sel = np.array([members[i][rng.integers(len(members[i]))] for i in bi])
        rng.shuffle(sel)
        return sel

    trajs, starts, kinds, steps = [], [], [], 0
    budget_abs = a.abs_frac * budget
    while steps < budget_abs:                                   # runs to absorption (cap lmax)
        sel = pick(a.abs_batch)
        tr, st = em.run(cand[sel, 0], cand[sel, 1], a.lmax)
        trajs += tr; starts += list(sel); kinds += ['abs'] * len(tr); steps += st
    n_abs, steps_abs = len(trajs), steps
    while steps < budget:                                       # short trajectories (stopped in A or B)
        n = int(min(a.short_batch, max(1, np.ceil((budget - steps) / a.ls))))
        sel = pick(n)
        tr, st = em.run(cand[sel, 0], cand[sel, 1], a.ls)
        trajs += tr; starts += list(sel); kinds += ['short'] * len(tr); steps += st
    ends = np.array([0 if t[-1, 0] <= system['lambda_a'] else 1 if t[-1, 0] >= system['lambda_b'] else -1 for t in trajs])
    kinds = np.array(kinds)
    common.say(f'launches (rng {seed}): budget {budget}, {steps} MD steps; {n_abs} to absorption ({steps_abs} steps, '
               f'{np.mean(ends[kinds == "abs"] == 1) if n_abs else 0:.2f} reach B), {len(trajs) - n_abs} short of {a.ls} '
               f'steps ({np.mean(ends[kinds == "short"] == 1):.3f} reach B, {np.mean(ends[kinds == "short"] == 0):.3f} A); '
               f'{nb} bins of logit q_1 [{lo:.1f}, {hi:.1f}]', log=log)
    return trajs, np.array(starts), kinds, ends, b, steps


def _launch_pieces(trajs, bins_of, model, system):
    """Pool layout of launched trajectories: one piece and one 'path' per trajectory, weight 1, not mirrored. Rows
    are numbered from ceil(n / 4) so that the pool's burn-in (0.2 of the rows) drops none of them and the MC blocks
    cover the launches evenly (the order is already random)."""
    n = len(trajs)
    lens = np.array([len(t) for t in trajs], int)
    xy = np.concatenate(trajs)
    lam = np.asarray(model(xy), float)
    r0 = int(np.ceil(n / 4))
    pieces = dict(cv=np.c_[lam, xy], t=np.concatenate([np.arange(m) for m in lens]).astype(np.int32),
                  piece=np.repeat(np.arange(n, dtype=np.int32), lens), piece_n=lens, piece_w=np.ones(n),
                  piece_pnr=np.arange(n, dtype=np.int64), piece_ens=np.asarray(bins_of, np.int32),
                  piece_row=(r0 + np.arange(n)).astype(np.int32), piece_mirrored=np.zeros(n, bool),
                  piece_core=np.zeros((n, 2), np.int32), piece_tainted=np.zeros(n, bool), weighting='none',
                  reversed_pieces=False)
    pieces['q_exact'] = ase_systems.exact_committor(system)(xy).astype(np.float32)
    raw = dict(ops=[np.c_[np.asarray(model(t), float), t] for t in trajs], pnr=list(range(n)), rows=list(r0 + np.arange(n)),
               W=np.ones((n, 1)), ptype=['LAU'] * n, gen_ens=list(bins_of), tainted=[False] * n, move=['launch'] * n)
    return pieces, raw, r0 + n


def launch(a):
    exp = _exp(a.exp)
    cfg, model_path, intfs = _src()
    system = cfg['system_spec']
    model = LambdaModel(model_path)
    pool = pool_mod.Pool(exp)
    arms = _arms(exp)
    P, Ph = arms[f'P_s{a.seed}'], arms[f'Ph_s{a.seed}']
    rng = np.random.default_rng(20_000 + a.seed)
    cand, zc = launch_points(pool, model, system, a.candidates, rng)
    for kind, budget, sub in (('L', P['md_steps'], 0), ('Lh', P['md_steps'] - Ph['md_steps'], 1)):
        t0 = time.time()
        trajs, starts, kinds, ends, b, steps = make_launches(a, budget, 2 * a.seed + sub, model, system, cand, zc, _log(exp))
        pieces, raw, n_rows = _launch_pieces(trajs, b[starts], model, system)
        k = _pass_id(kind, a.seed)
        m = pool.store(k, pieces, raw, dict(kind='launch', mode='launch', lambda_a=system['lambda_a'],
                                            lambda_b=system['lambda_b'], cv_names=system['cv_names'],
                                            state_cv=system['state_cv'], n_rows=n_rows, interfaces=intfs,
                                            lambda_model=model_path, md_steps=int(steps), workdir=exp,
                                            launch=dict(ls=a.ls, abs_frac=a.abs_frac, lmax=a.lmax, bins=a.bins)))
        arms[f'{kind}_s{a.seed}'] = dict(pass_index=k, md_steps=int(steps), frames=m['n_frames'], pieces=m['n_pieces'],
                                         n_abs=int((kinds == 'abs').sum()), abs_reach_b=int(((kinds == 'abs') & (ends == 1)).sum()),
                                         short_reach_b=int(((kinds == 'short') & (ends == 1)).sum()),
                                         seconds=round(time.time() - t0, 1))
        _save_arms(exp, arms)


# ------------------------------------------------------------------------------------------------------------------
# views, fits and scores
# ------------------------------------------------------------------------------------------------------------------
_FRAME = ('cv', 'lam', 't', 'piece', 'path', 'w', 'incl', 'ens', 'pass_of_frame', 'src_idx', 'q_np', 'q_exact', 'A',
          'B', 'inside', 'valid', 'q', 'z', 'frame_split')
_PIECE = ('piece_pass', 'piece_path', 'piece_split', 'piece_mirrored', 'piece_row', 'piece_ens', 'piece_w', 'piece_block')


def merge(views):
    """One View from several (each already normalised to mean weight 1 per frame): frame and piece arrays
    concatenated, piece indices offset."""
    out = copy.copy(views[0])
    off, pieces = 0, []
    for v in views:
        pieces.append(v.piece + off)
        off += len(v.piece_pass)
    for f in _FRAME:
        if all(hasattr(v, f) for v in views):
            setattr(out, f, np.concatenate([getattr(v, f) for v in views]))
    out.piece = np.concatenate(pieces).astype(np.int32)
    for f in _PIECE:
        setattr(out, f, np.concatenate([getattr(v, f) for v in views]))
    out.interfaces_of_pass = {}
    for v in views:
        out.interfaces_of_pass.update(getattr(v, 'interfaces_of_pass', {}))
    out.n = len(out.cv)
    out.path_id = np.unique(out.path)
    out.has_exact = all(getattr(v, 'has_exact', False) for v in views)
    return out


def select(pool, passes, burn_in, cap, max_frames, seed):
    v = pool.select(passes=passes, lineage='exclude', burn_in=burn_in, cap=cap, block=0.1, val_blocks=(1,),
                    test_blocks=(3,), need_labels=False, seed=seed, max_frames=max_frames)
    v.w = (v.w / v.w.mean()).astype(np.float32)
    return v


def arm_view(a, pool, arm, mode, cfg):
    """The data of one fit: arm in {base, P, L, M}, mode in {inf (with the infinit subsample), alone}."""
    cap, bi = cfg['pool'].get('cap_paths_per_ensemble'), float(cfg['pool'].get('burn_in', 0.2))
    parts = []
    if mode == 'inf' or arm == 'base':
        parts.append(select(pool, INF, bi, cap, a.inf_frames, seed=a.seed))       # the same subsample for every arm
    if arm == 'P':
        parts.append(select(pool, [_pass_id('P', a.seed)], bi, cap, None, a.seed))
    elif arm == 'L':
        parts.append(select(pool, [_pass_id('L', a.seed)], bi, None, None, a.seed))
    elif arm == 'M':
        parts.append(select(pool, [_pass_id('Ph', a.seed)], bi, cap, None, a.seed))
        parts.append(select(pool, [_pass_id('Lh', a.seed)], bi, None, None, a.seed))
    return parts[0] if len(parts) == 1 else merge(parts)


def _band_table(zq, zx, w, qx):
    out = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        s = (qx >= lo) & (qx < hi)
        if s.sum() < 20:
            out.append(dict(band=[lo, hi], n=int(s.sum()))); continue
        d = zq[s] - zx[s]
        out.append(dict(band=[lo, hi], n=int(s.sum()), w=float(w[s].sum() / w.sum()), bias=float(np.average(d, weights=w[s])),
                        rms=float(np.sqrt(np.average(d * d, weights=w[s]))), bias_u=float(d.mean()), rms_u=float(np.sqrt((d * d).mean()))))
    return out


def _eval_set(exp):
    f = os.path.join(exp, 'eval_set.npz')
    if not os.path.isfile(f):
        src = pool_mod.Pool(_src()[0]['loop_dir'])
        v = src.select(passes=EVAL, lineage='include', burn_in=0.0, cap=None, need_labels=False, seed=0, max_frames=2_000_000)
        ok = v.inside
        np.savez(f, cv=v.cv[ok], q_exact=v.q_exact[ok], w=v.w[ok])
    z = np.load(f)
    return z['cv'], z['q_exact'].astype(float), z['w'].astype(float)


def fit(a):
    from loop import teacher, student, diagnostics
    exp = _exp(a.exp)
    _, model_path, intfs = _src()
    fcfg = controller.load_loop_config(FIT_CFG)
    pool = pool_mod.Pool(exp)
    tag = f'{a.arm}_{a.mode}_s{a.seed}'
    out_dir = os.path.join(exp, 'fits'); os.makedirs(out_dir, exist_ok=True)
    log = _log(exp)
    t0 = time.time()
    view = arm_view(a, pool, a.arm, a.mode, fcfg)
    m1 = LambdaModel(model_path)
    seed_q = np.full(view.n, np.nan)
    seed_q[view.inside] = m1.q(view.cv[view.inside])
    _, arr = teacher.front(view, [view.lambda_b], n_front=20)
    tc = dict(fcfg['teacher'], max_frames=view.n + 1)           # no further thinning: the arm data are used whole
    if a.iters:
        tc['max_iter'] = a.iters
    common.say(f'fit {tag}: {view.n} frames ({view.inside.sum()} inside), {len(view.path_id)} paths/trajectories, '
               f'{arr[0]} reach B; teacher {tc["max_iter"]} iterations', log=log)
    res = teacher.fit(view, tc, seed_q=seed_q, provisional_b=None, seed=a.seed, log=log)
    q = res['q']
    view.q_np = q.astype(np.float32)
    view.has_labels = True
    view.valid = view.inside & np.isfinite(view.q_np)
    view.q = np.clip(np.nan_to_num(q, nan=0.5), common.Q_CLIP, 1 - common.Q_CLIP)
    view.z = common.logit(view.q)
    ok = view.valid
    labels = _band_table(_lz(q[ok]), _lz(view.q_exact[ok]), view.w[ok].astype(float), view.q_exact[ok].astype(float))
    # student and export (loop_z_dp_u settings), no warm start: every arm from the same untrained network
    params, sinfo = student.train(view, fcfg, seed=a.seed, init_params=None, log=log, specs=list(view.cv_names))
    exp_d = student.build_export(view, params, fcfg, front_s=None, blend=bool(fcfg['student'].get('blend', True)), log=log)
    mpath = os.path.join(out_dir, f'lambda_{tag}.npz')
    np.savez(mpath, **exp_d)
    model = LambdaModel(mpath)
    cv_e, qx_e, w_e = _eval_set(exp)
    m_e = model.q(cv_e)
    lam_bands = _band_table(_lz(m_e), _lz(qx_e), w_e, qx_e)

    class _V:                                                   # the evaluation set as diagnostics.exact_metrics reads it
        pass
    ev = _V(); ev.cv, ev.q_exact, ev.w, ev.n, ev.dim, ev.has_exact = cv_e, qx_e, w_e, len(cv_e), cv_e.shape[1], True
    ev.inside = np.ones(len(cv_e), bool)
    em = diagnostics.exact_metrics(model, ev, LEVELS)
    data_bands = [dict(band=b_['band'], n=b_['n'], w=b_.get('w')) for b_ in labels]
    r = dict(tag=tag, arm=a.arm, mode=a.mode, seed=a.seed, frames=int(view.n), inside=int(view.inside.sum()),
             paths=int(len(view.path_id)), reach_b=int(arr[0]), teacher=dict(seconds=res['seconds'], iterations=res['iterations'],
             frames=res['n_frames'], final={k: v[-1] for k, v in res['history'].items() if v}),
             student=dict(selected_epoch=sinfo.get('selected_epoch'), held_frac=sinfo.get('held_frac'), sdzqs=sinfo.get('sdzqs')),
             labels=labels, lam=lam_bands, exact_metrics=em, data=data_bands, model=mpath, seconds=round(time.time() - t0, 1),
             arms=_arms(exp))
    manifest.write(os.path.join(out_dir, f'{tag}.json'), **r)
    common.say(f'fit {tag}: done in {time.time() - t0:.0f} s; lambda logit RMSE {em["logit_rmse"]:.3f}, iso spread '
               f'{em["spread_mean"]:.3f}', log=log)


# ------------------------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------------------------
def report(a):
    exp = _exp(a.exp)
    d = os.path.join(exp, 'fits')
    rs = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith('.json')] if os.path.isdir(d) else []
    if not rs:
        print('no fits yet'); return
    groups = {}
    for r in rs:
        groups.setdefault((r['arm'], r['mode']), []).append(r)
    order = [('base', 'inf'), ('P', 'inf'), ('L', 'inf'), ('M', 'inf'), ('P', 'alone'), ('L', 'alone'), ('M', 'alone')]
    bands = [f'{b[0]:.0e}-{b[1]:.2g}' for b in zip(BANDS[:-1], BANDS[1:])]

    def ms(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            return '      -      '
        return f'{np.mean(vals):6.3f}' + (f' +-{np.std(vals, ddof=1) / np.sqrt(len(vals)):.3f}' if len(vals) > 1 else '       ')

    print(f'{exp}: {len(rs)} fits; mean +- standard error over seeds\n')
    print('lambda on the evaluation set: logit RMSE (binned) | iso spread (mean over levels) | frames | reach B')
    for key in order:
        if key in groups:
            g = groups[key]
            print(f'  {key[0]:4s} {key[1]:5s} (n={len(g)})  {ms([r["exact_metrics"]["logit_rmse"] for r in g])} | '
                  f'{ms([r["exact_metrics"]["spread_mean"] for r in g])} | {int(np.mean([r["frames"] for r in g])):9d} | '
                  f'{int(np.mean([r["reach_b"] for r in g])):6d}')
    for what, title in (('lam', 'lambda on the evaluation set, logit m - logit q_exact per band of exact q'),
                        ('labels', 'teacher labels on the arm\'s own frames, logit q_teacher - logit q_exact per band')):
        for stat in ('bias', 'rms'):
            print(f'\n{title}: {stat} (weighted)')
            print(f'  {"":10s} ' + ' '.join(f'{b:>13s}' for b in bands))
            for key in order:
                if key in groups:
                    g = groups[key]
                    print(f'  {key[0]:4s} {key[1]:5s} ' + ' '.join(ms([r[what][i].get(stat) for r in g]) for i in range(len(bands))))
    print('\nwhere the data lie: share of the frames (weighted) per band of exact q')
    for key in order:
        if key in groups:
            g = groups[key]
            print(f'  {key[0]:4s} {key[1]:5s} ' + ' '.join(ms([r['data'][i].get('w') for r in g]) for i in range(len(bands))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', default='launch_test')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('setup'); s.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3]); s.add_argument('--cycles', type=int, default=3400)
    for name in ('run-p', 'ingest'):
        s = sub.add_parser(name); s.add_argument('seed', type=int)
    s = sub.add_parser('launch'); s.add_argument('seed', type=int)
    s.add_argument('--ls', type=int, default=200, help='steps of a short trajectory')
    s.add_argument('--abs-frac', type=float, default=0.25, help='share of the budget run to absorption')
    s.add_argument('--lmax', type=int, default=200_000, help='cap of a run to absorption (steps)')
    s.add_argument('--bins', type=int, default=20, help='launch strata (bins of logit q under lambda_1)')
    s.add_argument('--candidates', type=int, default=2_000_000, help='candidate launch frames drawn from the infinit paths')
    s.add_argument('--abs-batch', type=int, default=8, help='runs to absorption per batch (the budget overshoots by <= one batch)')
    s.add_argument('--short-batch', type=int, default=4000)
    s = sub.add_parser('fit'); s.add_argument('seed', type=int); s.add_argument('arm', choices=['base', 'P', 'L', 'M'])
    s.add_argument('mode', choices=['inf', 'alone'])
    s.add_argument('--iters', type=int, help='teacher iterations (default: the fit config)')
    s.add_argument('--inf-frames', type=int, default=2_500_000, help='frames of the infinit subsample (+inf fits)')
    sub.add_parser('report')
    a = ap.parse_args()
    if a.cmd == 'fit' and a.arm == 'base':
        a.mode = 'inf'
    {'setup': setup, 'run-p': run_p, 'ingest': ingest, 'launch': launch, 'fit': fit, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
