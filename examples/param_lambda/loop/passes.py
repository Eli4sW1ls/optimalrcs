"""Passes k >= 1 of the loop (controller.cmd_loop calls run_passes):

    pool -> teacher (NPNEq, warm start) -> student (M3, warm start) -> export lambda_k
    -> diagnostics of lambda_{k-1} on the newest data (prospective) and stability against lambda_{k-1}
    -> plan (natural coordinate, ratio cap, length budget) -> gate (spacing checks on the stored paths under
       lambda_k, at the saved-frame resolution) -> seed paths -> infretis run of `cycles_per_pass` cycles with
       lambda_k frozen -> ingest -> re-check every spacing test on the pass's own paths
    -> stop test; freeze.

Every step writes into <loop_dir>/pass_<k>/ (manifest.json at the end) and state.json, so an interrupted pass is
redone from its last completed step.
"""
import json
import os
import time

import numpy as np

import common
from lambda_runtime import LambdaModel

from loop import diagnostics, gpu, interfaces as itf, manifest, screen, seed as seed_mod, student, teacher
from loop.backends import infretis_backend as be


def _view(pool, cfg, need_labels=False):
    pc = cfg['pool']
    return pool.select(passes=pc.get('passes', 'all'), lineage=pc.get('lineage', 'exclude'),
                       burn_in=float(pc.get('burn_in', 0.0)), cap=pc.get('cap_paths_per_ensemble'),
                       block=float(pc.get('block', 0.1)), val_blocks=tuple(pc.get('val_blocks', [1])),
                       test_blocks=tuple(pc.get('test_blocks', [3])), need_labels=need_labels,
                       max_frames=pc.get('max_frames'))


def _model_path(cfg, k):
    d = os.path.join(cfg['loop_dir'], 'models')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'lambda_{k}.npz')


def teach(cfg, pool, k, prev_model, log, cvs=None):
    """Labels for every frame of the selection (written back to the pool). `cvs`: the CVs of the fit (default
    all; the screening of the previous pass sets them)."""
    view = _view(pool, cfg)
    tc = dict(cfg['teacher'])
    if cvs is not None:
        tc['cvs'] = list(cvs)
    seed_q = view.q_np.astype(float).copy()                     # old frames keep their q
    if prev_model is not None:
        new = ~np.isfinite(seed_q) & view.inside
        if new.any():
            seed_q[new] = prev_model.q(view.cv[new])            # new frames start from the current model
    prov = None
    n_b = int(tc.get('n_b', cfg['pass0'].get('n_b', 20)))
    if tc.get('provisional_b', True):
        _, arrivals = teacher.front(view, [view.lambda_b], n_front=n_b)
        if arrivals[0] < n_b:                                   # B not reached often enough: fit to the front
            lv = sorted(set(float(v) for ints in view.interfaces_of_pass.values() for v in ints))
            prov, _ = teacher.front(view, lv[:-1], n_front=int(tc.get('n_front', 50)))
            common.say(f'teacher: {arrivals[0]} paths reach B (< {n_b}); provisional B at s >= {prov}', log=log)
    res = teacher.fit(view, tc, seed_q=seed_q, provisional_b=prov, seed=int(cfg.get('seed', 0)) + k, log=log)
    pool.set_labels_from_view(view, res['q'])
    from loop import snapshot_labels
    snapshot_labels.save(cfg['loop_dir'], k)                    # the pool labels are overwritten next pass
    out = dict(n_frames=res['n_frames'], seconds=res['seconds'], iterations=res['iterations'], provisional_b=prov,
               cvs=res['features'],
               final={key: v[-1] for key, v in res['history'].items() if v})
    if view.has_exact:
        ok = np.isfinite(res['q']) & view.inside
        out['rms_vs_exact'] = float(np.sqrt(np.average((res['q'][ok] - view.q_exact[ok]) ** 2, weights=view.w[ok])))
    return out


def learn(cfg, pool, k, prev_params, log, specs=None, view=None, front_s=None):
    """Student and export. `front_s`: the teacher's provisional B (state CV), or None: beyond it the exported
    lambda rises linearly in the state CV to lambda_B (lambda_runtime), instead of saturating where the labels were
    fitted to q = 1 (loop_stag pass 1: lambda flat at 0.307 from x = 0 to 1.2, no frames in the upper windows)."""
    view = view if view is not None else _view(pool, cfg, need_labels=True)
    params, info = student.train(view, cfg, seed=int(cfg.get('seed', 0)) + k,
                                 init_params=prev_params if cfg['student'].get('warm_start', True) else None, log=log,
                                 specs=specs)
    exp = student.build_export(view, params, cfg, front_s=front_s, blend=bool(cfg['student'].get('blend', True)), log=log)
    path = _model_path(cfg, k)
    np.savez(path, **exp)
    np.savez(path.replace('.npz', '_score.npz'), **params)
    return path, params, info, view


def run_passes(cfg, state, pool, log=None, max_passes=None):
    sc = cfg['stop']
    run = cfg['run']
    system = cfg['system_spec']
    mode = run.get('mode', 'staple')
    k0 = len(state.data['passes']) + 1
    prev_model = prev_params = None
    if state.data['passes']:
        last = state.data['passes'][-1]
        prev_model = LambdaModel(last['model'])
        prev_params = dict(np.load(last['model'].replace('.npz', '_score.npz')))
    n_done = 0
    k = k0
    while True:
        if k - 1 >= int(sc.get('max_passes', 5)) or state.data['md_steps'] >= float(sc.get('max_md_steps', np.inf)):
            common.say(f'stop: pass or MD-step budget reached ({k - 1} passes, {state.data["md_steps"]} MD steps)', log=log)
            break
        if max_passes is not None and n_done >= max_passes:
            break
        # the teacher sees every CV unless [screen] teacher = "kept": a CV the teacher never sees cannot shape the
        # labels, so the student could not bring it back in a later screening (loop_zw, 2026-11-17)
        # `[screen] cvs`: a fixed CV subset for teacher and student (loop_zu_xy: the embedding without its w)
        cvs = cfg.get('screen', {}).get('cvs')
        if cvs is None and state.data['passes'] and cfg.get('screen', {}).get('teacher', 'all') == 'kept':
            cvs = state.data['passes'][-1].get('cvs_kept')
        rec = one_pass(cfg, pool, k, prev_model, prev_params, mode, log, cvs=cvs)
        state.data['passes'].append(rec)
        state.data['md_steps'] += rec['cost']['md_steps']
        state.save()
        prev_model, prev_params = LambdaModel(rec['model']), dict(np.load(rec['model'].replace('.npz', '_score.npz')))
        n_done += 1
        if stop_test(cfg, state, log):
            state.data['done'] = True
            state.save()
            break
        k += 1
    if state.data.get('done') or (state.data['passes'] and k - 1 >= int(sc.get('max_passes', 5))):
        freeze(cfg, state, pool, log)


def one_pass(cfg, pool, k, prev_model, prev_params, mode, log, cvs=None):
    t_pass = time.time()
    pdir = os.path.join(cfg['loop_dir'], f'pass_{k}')
    os.makedirs(pdir, exist_ok=True)
    system, run, ic = cfg['system_spec'], cfg['run'], cfg['interfaces']
    common.say(f'== pass {k}', log=log)
    # A pass interrupted during its run resumes it with the same lambda and interfaces (fit.json, written before the
    # run): GPU fits are not bit-reproducible, so a refit would place slightly different interfaces and the
    # partial run could not be continued.
    fit_file = os.path.join(pdir, 'fit.json')
    if os.path.isfile(fit_file):
        f = manifest.read(fit_file)
        model_path, tinfo, sinfo, diag, plan, placed, scr, specs, fit_seconds, gpu_wait = (
            f['model'], f['teacher'], f['student'], f['diagnostics'], f['plan'], f['placed'], f['screen'], f['specs'],
            f['fit_seconds'], f['gpu_wait_seconds'])
        model = LambdaModel(model_path)
        intfs = placed['interfaces']
        common.say(f'pass {k}: fits, lambda and interfaces from {fit_file} (resuming)', log=log)
    else:
        # 1. teacher (all CVs by default), 1b. CV screening (the student's inputs), 2. student and export: all on the GPU
        with gpu.exclusive(log=log, what=f'fits of pass {k}') as gpu_wait:
            t0 = time.time()
            tinfo = teach(cfg, pool, k, prev_model, log, cvs=cvs)
            common.say(f'teacher: {tinfo}', log=log)
            view = _view(pool, cfg, need_labels=True)
            specs = list(cvs) if cvs is not None else list(view.cv_names)
            scr = None
            if screen.enabled(cfg, view):
                # candidates: every CV of the system, so a CV dropped earlier can come back
                scr = screen.run(view, cfg, list(view.cv_names), seed=int(cfg.get('seed', 0)) + k, log=log)
                specs = scr['kept']
            model_path, params, sinfo, view = learn(cfg, pool, k, prev_params, log, specs=specs, view=view,
                                                    front_s=tinfo.get('provisional_b'))
        model = LambdaModel(model_path)
        fit_seconds = time.time() - t0
        # 3. diagnostics: prospective (old model on the newest data, before this refit used them) and stability
        newest = pool.passes()[-1]
        diag = dict(new=dict(zq=diagnostics.zq_profile(model, view, view.pass_of_frame == newest),
                             ood=diagnostics.ood_by_ensemble(model, view, view.pass_of_frame == newest)))
        levels = np.exp(np.linspace(np.log(1e-3), np.log(0.95), 10))
        if prev_model is not None:
            diag['prospective'] = dict(zq=diagnostics.zq_profile(prev_model, view, view.pass_of_frame == newest),
                                       ood=diagnostics.ood_by_ensemble(prev_model, view, view.pass_of_frame == newest))
            ref_cv, ref_w = diagnostics.reference_set(view)
            diag['stability'] = diagnostics.stability(model, prev_model, ref_cv, ref_w, levels)
        ex = diagnostics.exact_metrics(model, view, levels)
        if ex is not None:
            diag['exact'] = ex
        common.say(f'diagnostics: SD(Z_q) new {diag["new"]["zq"]["sdzqs"]:.3f}'
                   + (f', previous model on the newest data {diag["prospective"]["zq"]["sdzqs"]:.3f}' if prev_model else '')
                   + (f'; exact: spread {ex["spread_mean"]:.4f}, logit RMSE {ex["logit_rmse"]:.3f}' if ex else ''), log=log)
        # 4. interfaces: plan + gate on all stored paths under the new lambda
        plan = itf.plan_natural(model, view, ic, log=log)
        paths = itf.path_arrays(pool, model, state_cv=system['state_cv'], lineage='include')
        placed = itf.place(model, paths, plan['lam'], ic, mode, log=log)
        intfs = placed['interfaces']
        if not placed['check']['ok']:
            common.say(f'WARNING: the placed set still fails {placed["check"]["fails"]} on the stored paths', log=log)
        manifest.write(fit_file, model=model_path, teacher=tinfo,
                       student={key: v for key, v in sinfo.items() if key != 'checkpoints'}, diagnostics=diag,
                       plan=plan, placed=dict(interfaces=intfs, removed=placed['removed'],
                                              refilled=placed.get('refilled', []), check=_slim(placed['check'])),
                       screen=scr, specs=specs, fit_seconds=fit_seconds, gpu_wait_seconds=gpu_wait)
    # 5-6. seeds, the run, ingest, re-check on the pass's own paths
    cost, raw, recheck = run_and_check(cfg, pool, pdir, model, model_path, intfs, mode,
                                       int(run.get('cycles_per_pass', 1000)), int(cfg.get('seed', 0)) + k,
                                       f'pass {k}', 'pass', log)
    lengths = _path_lengths(raw, len(intfs))
    rec = dict(k=k, model=model_path, model_md5=manifest.md5(model_path), interfaces=intfs,
               interfaces_md5=manifest.md5_of(intfs), plan=plan, removed=placed['removed'], refilled=placed.get('refilled', []),
               placement_check=_slim(placed['check']), recheck=_slim(recheck), teacher=tinfo,
               student={key: v for key, v in sinfo.items() if key != 'checkpoints'}, diagnostics=diag,
               screen=scr, cvs_kept=specs,
               cost=dict(cost, fit_seconds=fit_seconds, gpu_wait_seconds=gpu_wait, pass_seconds=time.time() - t_pass),
               lengths=lengths,
               pool_pass=pool.passes()[-1], data_md5=manifest.md5(pool.path(pool.passes()[-1])), seed=int(cfg.get('seed', 0)) + k)
    manifest.write(os.path.join(pdir, 'manifest.json'), config={key: v for key, v in cfg.items() if key != 'system_spec'},
                   system=cfg['system_spec'], **rec)
    return rec


def _extend_until_sampled(cfg, run_dir, target_file, mode, log):
    """The pass_stop = "paths" rule on a finished block: resume the run by check_every cycles until every ensemble
    holds min_paths_per_ensemble accepted paths (after the burn-in, unweighted: the weights only set the
    distribution of the same data) and the run min_paths_total accepted paths (all rows). Hard stops: the total
    reaches hard_paths_per_ensemble x n_ens (a stuck ensemble does not recover with more cycles: loop_zu_xy_b_s2_x12
    pass 9, ensemble 7 had 11, 13, 15 paths at 1000, 1250, 1500 cycles), or max_cycles.
    The target is written before each extension, so an interrupted pass resumes to it."""
    run = cfg['run']
    need_total = int(run.get('min_paths_total', 0))
    need_ens = int(run.get('min_paths_per_ensemble', 0))
    hard_per_ens = run.get('hard_paths_per_ensemble')
    warn = int(run.get('warn_paths_per_ensemble', 0))
    step = int(run.get('check_every', 250))
    cap = int(run.get('max_cycles', 4000))
    burn = float(cfg['pool'].get('burn_in', 0.0))
    weighting = cfg['pool'].get('weighting') if mode == 'staple' else None
    ext_wall = 0.0
    while True:
        done = be.cost(run_dir)['cycles']
        c = be.ensemble_counts(run_dir, burn_in=burn, weighting=weighting)
        worst = int(np.argmin(c['count']))
        hard = int(hard_per_ens) * len(c['count']) if hard_per_ens else None
        ok = c['total'] >= need_total and c['count'][worst] >= need_ens
        common.say(f'pass stop: cycle {done}, {c["total"]} accepted paths (min {need_total}'
                   + (f', hard stop {hard}' if hard else '') + f'); per ensemble after the burn-in {c["count"]} '
                   f'(fewest: ensemble {worst}, {c["count"][worst]} of {need_ens})', log=log)
        hit_hard = hard is not None and c['total'] >= hard
        if ok or hit_hard or done >= cap:
            if not ok:
                common.say(f'WARNING: {"hard path stop " + str(hard) if hit_hard else "max_cycles " + str(cap)} reached '
                           f'with {c["total"]} accepted paths, ensemble {worst} at {c["count"][worst]} of {need_ens}', log=log)
            low = [e for e, v in enumerate(c['count']) if v < warn]
            if low:
                common.say(f'WARNING: fewer than {warn} accepted paths in ensembles {low} '
                           f'({[c["count"][e] for e in low]})', log=log)
            return dict(rule='paths', min_paths_total=need_total, min_paths_per_ensemble=need_ens, hard_paths=hard,
                        stopped_by='target' if ok else ('hard_paths' if hit_hard else 'max_cycles'), cycles=done, total=c['total'], kept=c['kept'],
                        count=c['count'], reached=bool(ok), worst_ensemble=worst, low_paths=low,
                        extension_wall_seconds=ext_wall)
        target = min(done + step, cap)
        manifest.write(target_file, cycles=target)
        ext_wall += float(be.resume(run_dir, target).get('wall_seconds') or 0.0)


def _slim(res):
    return {key: v for key, v in res.items() if key not in ('rows',)} | dict(rows=[{a: b for a, b in r.items()} for r in res['rows']])


def _path_lengths(raw, n_ens):
    """Median and 90th percentile of the accepted path lengths per ensemble (weighted by occupancy > 0)."""
    out = {}
    L = np.array([len(o) for o in raw['ops']])
    W = np.asarray(raw['W'])
    for e in range(n_ens):
        sel = W[:, e] > 0 if len(W) else np.zeros(0, bool)
        if sel.any():
            out[e] = dict(median=float(np.median(L[sel])), p90=float(np.percentile(L[sel], 90)), n=int(sel.sum()))
    return out


def stop_test(cfg, state, log):
    """All of: stability spread below threshold at every resolved level; prospective SD(Z_q) within tolerance of
    the refit; interfaces unchanged within tolerance and every check passing on the last pass's own paths with
    no unverified interface; OOD fraction below threshold and not rising."""
    sc = cfg['stop']
    if len(state.data['passes']) < 2:
        return False
    cur, prev = state.data['passes'][-1], state.data['passes'][-2]
    d = cur['diagnostics']
    why = []
    st = d.get('stability')
    if st:
        vals = [v for v in st['new_on_old'] + st['old_on_new'] if v is not None and np.isfinite(v)]
        if not vals or max(vals) > float(sc.get('stability', 3 * st['floor'])):
            why.append(f'stability {max(vals) if vals else np.nan:.4f}')
    pz, nz = d.get('prospective', {}).get('zq', {}).get('sdzqs'), d['new']['zq']['sdzqs']
    if pz is None or pz > (1 + float(sc.get('zq_tolerance', 0.2))) * nz:
        why.append(f'prospective SD(Z_q) {pz} vs refit {nz:.3f}')
    a, b = np.asarray(cur['interfaces']), np.asarray(prev['interfaces'])
    if len(a) != len(b) or np.max(np.abs(a - b)) > float(sc.get('interface_tol', 0.02)) * (a[-1] - a[0]):
        why.append('interfaces changed')
    if not cur['recheck']['ok'] or cur['recheck']['unverified']:
        why.append('spacing checks fail on the pass\'s own paths')
    ood, ood_prev = d['new']['ood'].get('all', 0), prev['diagnostics']['new']['ood'].get('all', 0)
    if ood > float(sc.get('ood', 0.01)) or ood > 1.5 * ood_prev + 1e-4:
        why.append(f'OOD {ood:.4f} (previous {ood_prev:.4f})')
    common.say('stop test: ' + ('all criteria met' if not why else 'continue (' + '; '.join(why) + ')'), log=log)
    return not why


def freeze(cfg, state, pool, log, name='production', start=None, start_paths=None):
    """The production configuration: the last lambda, interfaces and seed paths, as an infretis run directory that
    is not started (`run.py loop <config> --production` runs it).

    The hand-off is gated (constraint 9): the frozen interfaces must be exactly those of a run with the frozen
    lambda whose re-check on its own paths passed. If the last pass passes, its interfaces are used. Otherwise, with
    lambda frozen, up to `production.verify_rounds` short verification runs (`production.verify_cycles` cycles):
    re-place (thin) the interfaces on all stored paths, run, re-check on the run's own paths. Thinning only removes
    interfaces, so the rounds settle. If none passes, nothing is written except freeze.json.

    `name` / `start`: a second production from an earlier frozen set (`start` = its interfaces), e.g. after a change
    to placement (refill): at least one verification round is then forced (place, run, re-check), and everything is
    written under the name (<name>/, <name>_verify_<r>/, <name>_seeds/, <name>_freeze.json).

    Thinning in the verification rounds uses the paths of the run that failed (`own_paths`), not the pool: the
    failing run's statistics are what the gate will see again, and on the pool (thousands of paths from other
    interface sets) a marginal failure is diluted and the set never changes (Z: three rounds kept all 13 interfaces
    while the same ensemble failed in the runs). The first round uses the pool, or `start_paths` when given (the
    own paths of an earlier failed verification run, with `start` its interface set)."""
    last = state.data['passes'][-1]
    out = os.path.join(cfg['loop_dir'], name)
    pre = '' if name == 'production' else f'{name}_'
    if os.path.isdir(out):
        common.say(f'freeze: {out} exists, left as is', log=log)
        return out
    pc = cfg.get('production', {})
    system, ic = cfg['system_spec'], cfg['interfaces']
    run = dict(cfg['run'], **pc.get('run', {}))                              # production sampler settings
    mode = run.get('mode', 'staple')
    model = LambdaModel(last['model'])
    intfs, rc, source = last['interfaces'], last['recheck'], f'pass {last["k"]}'
    if start is not None:                                        # re-place from a given set: force a verification
        intfs, rc, source = list(start), dict(ok=False, fails=['re-placement forced']), 'start set'
    rounds = []
    own = start_paths
    for r in range(int(pc.get('verify_rounds', 3))):
        if rc['ok']:
            break
        if own is not None:
            paths = own                                          # thin on the failed run's own paths
        else:
            paths = itf.path_arrays(pool, model, state_cv=system['state_cv'], lineage='include')
        placed = itf.place(model, paths, intfs, ic, mode, log=log)
        intfs = placed['interfaces']
        vdir = os.path.join(cfg['loop_dir'], f'{pre}verify_{r + 1}')
        os.makedirs(vdir, exist_ok=True)
        cost, raw, rc = run_and_check(cfg, pool, vdir, model, last['model'], intfs, mode,
                                      int(pc.get('verify_cycles', 500)), int(cfg.get('seed', 0)) + 500 + r,
                                      f'verification {r + 1}', 'verify', log)
        state.data['md_steps'] += cost['md_steps']
        own = own_paths(raw, system['state_cv'])
        rounds.append(dict(round=r + 1, interfaces=intfs, removed=placed['removed'], refilled=placed.get('refilled', []), ok=rc['ok'], fails=rc['fails'],
                           md_steps=cost['md_steps']))
        source = f'verification {r + 1}'
        state.save()
    manifest.write(os.path.join(cfg['loop_dir'], f'{pre}freeze.json'), pass_index=last['k'], model=last['model'],
                   interfaces=intfs, ok=rc['ok'], fails=rc.get('fails', []), source=source, verification=rounds)
    if not rc['ok']:
        common.say(f'freeze REFUSED: {source} fails {rc["fails"]} on its own paths', log=log)
        return None
    seeds_dir = os.path.join(cfg['loop_dir'], f'{name}_seeds')
    dirs, tainted, report = seed_mod.build_seeds(pool, model, intfs, mode, seeds_dir)
    be.write_run(out, system, run, intfs, be.learned_order_table(system, last['model']),
                 int(pc.get('cycles', 10000)), mode, dirs, tainted_inodes=tainted,
                 shooting=run.get('shooting', 'sh'), seed=int(cfg.get('seed', 0)) + 1000, note='production')
    manifest.write(os.path.join(out, 'manifest.json'), model=last['model'], model_md5=last['model_md5'],
                   interfaces=intfs, verified_by=source, recheck=_slim(rc), verification=rounds, seeds=report,
                   passes=len(state.data['passes']), initialization_md_steps=state.data['md_steps'])
    common.say(f'freeze: production run prepared in {out} (not started); {len(intfs)} interfaces, verified by '
               f'{source} on its own paths', log=log)
    return out


def own_paths(raw, state_cv):
    """[(lam, s, w)] of a run's own accepted paths (the arrays `interfaces.check` takes)."""
    return [(o[:, 0], o[:, 1 + state_cv], float(W.max())) for o, W in zip(raw['ops'], raw['W']) if W.max() > 0]


def run_and_check(cfg, pool, pdir, model, model_path, intfs, mode, cycles, seed, label, kind, log):
    """Seeds cut from the pool, an infretis run with lambda frozen (resumed if it exists), ingested as a new pool
    pass, and every spacing check on the run's own paths (plus accepted paths with an empty shooting region).
    Returns (cost, raw paths, re-check)."""
    import shutil
    system, run, ic = cfg['system_spec'], dict(cfg['run']), cfg['interfaces']
    run_dir = os.path.join(pdir, 'run')
    # [run] pass_stop = "paths": a learning pass runs min_cycles, then blocks of check_every cycles until every
    # ensemble holds min_paths_per_ensemble accepted paths after the pool's burn-in and the run min_paths_total in
    # all; hard stops at hard_paths_per_ensemble x n_ens accepted paths and at max_cycles. An ensemble with fewer
    # than warn_paths_per_ensemble paths at the end is reported. Verification and production runs keep their
    # fixed cycle counts.
    adaptive = kind == 'pass' and run.get('pass_stop', 'cycles') == 'paths'
    target_file = os.path.join(pdir, 'run_target.json')
    if adaptive:
        cycles = int(run.get('min_cycles', 500))
    if not os.path.isfile(os.path.join(run_dir, 'restart.toml')):
        if os.path.isfile(target_file):
            os.remove(target_file)
        for d in (run_dir, os.path.join(pdir, 'seeds')):
            if os.path.isdir(d):
                shutil.rmtree(d)
        dirs, tainted, seed_report = seed_mod.build_seeds(pool, model, intfs, mode, os.path.join(pdir, 'seeds'))
        be.write_run(run_dir, system, run, intfs, be.learned_order_table(system, model_path), cycles, mode, dirs,
                     tainted_inodes=tainted, shooting=run.get('shooting', 'sh'), seed=seed,
                     note=f'seeds cut from the pool for {label}')
        manifest.write(os.path.join(pdir, 'seeds.json'), report=seed_report)
        common.say(f'run: {len(intfs)} interfaces, {cycles} cycles in {run_dir}', log=log)
        cost = be.run(run_dir)
    else:
        ran = be.read_toml(os.path.join(run_dir, 'infretis.toml'))['simulation']['interfaces']
        if len(ran) != len(intfs) or not np.allclose(ran, intfs, rtol=0, atol=1e-9):
            raise ValueError(f'{run_dir} was run with interfaces {np.round(ran, 5).tolist()}, not the requested '
                             f'{np.round(intfs, 5).tolist()}: a re-check on it would be meaningless; move the run '
                             f'directory aside (or rerun with the original set)')
        if adaptive and os.path.isfile(target_file):     # the extension the stop rule had asked for
            cycles = int(manifest.read(target_file)['cycles'])
        if be.cost(run_dir)['cycles'] < cycles:              # interrupted (stopped or suspended): continue it
            common.say(f'run: resuming {run_dir} at cycle {be.cost(run_dir)["cycles"]} of {cycles}', log=log)
            be.resume(run_dir, cycles)
        cost = be.cost(run_dir)
    if adaptive:                    # the cost of the whole run, after the extensions
        stop = _extend_until_sampled(cfg, run_dir, target_file, mode, log)
        cost = dict(be.cost(run_dir), wall_seconds=(cost.get('wall_seconds') or 0.0) + stop['extension_wall_seconds'],
                    pass_stop=stop)
    pieces = be.load_pieces(run_dir, burn_in_fraction=0.0, paths_per_ensemble=cfg['pool'].get('cap_paths_per_ensemble'),
                            weighting=cfg['pool'].get('weighting') if mode == 'staple' else None)
    raw = be.raw_paths(run_dir)
    if system.get('exact'):
        from loop import ase_systems
        pieces['q_exact'] = ase_systems.exact_committor(system)(pieces['cv'][:, 1:]).astype(np.float32)
    meta = dict(kind=kind, lambda_version=manifest.md5(model_path), lambda_model=model_path, interfaces=intfs,
                mode=mode, lambda_a=system['lambda_a'], lambda_b=system['lambda_b'], cv_names=system['cv_names'],
                state_cv=system['state_cv'], n_rows=len(raw['rows']), workdir=run_dir, md_steps=cost['md_steps'],
                wall_seconds=cost.get('wall_seconds'), cycles=cost['cycles'])
    # stored already: same directory AND the same run (cycles, path rows); a name alone is not enough, since a
    # directory can be deleted and its name reused by a later run
    stored = [(pool.meta(j).get('workdir'), pool.meta(j).get('cycles'), pool.meta(j).get('n_rows')) for j in pool.passes()]
    if (run_dir, meta['cycles'], meta['n_rows']) not in stored:
        pool.store(pool.passes()[-1] + 1, pieces, raw, meta)
    own = own_paths(raw, system['state_cv'])
    recheck = itf.check(own, intfs, ic, mode)
    recheck['empty_shooting_regions'] = itf.empty_shooting_regions(raw, intfs, mode)
    n_empty = sum(v['empty'] for v in recheck['empty_shooting_regions'].values())
    if n_empty:
        recheck['ok'] = False
        recheck['fails'].append(('accepted_paths_with_empty_shooting_region', -1, n_empty))
    common.say(f'{label} re-check on its own paths: {"OK" if recheck["ok"] else recheck["fails"]}\n'
               + itf.health_table(recheck), log=log)
    return cost, raw, recheck
