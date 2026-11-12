"""Steps 3-6 of run.py: screen, fit, evaluate, export, report (see common.py)."""
import json
import os
import time

import numpy as np

import common
import data
import evaluate as ev
import models
from lambda_runtime import LambdaModel, ScoreModel


def _dir(cfg, *parts):
    d = os.path.join(cfg['out_dir'], *parts)
    os.makedirs(d, exist_ok=True)
    return d


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [None if (isinstance(v, float) and np.isnan(v)) else v for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(_jsonable(obj), f, indent=1)


class Context:
    """Frames, the training set per feature list, the calibration frames and the selection frames."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.f = data.Frames(cfg)
        self.log = os.path.join(_dir(cfg, 'logs'), 'steps.log')
        common.say(f'== {cfg["name"]}\n{self.f.summary()}', log=self.log)
        tr = np.flatnonzero(self.f.mask('train') & self.f.valid)
        self.cal_idx = tr[np.unique(np.linspace(0, len(tr) - 1, min(len(tr), 500_000)).astype(int))]
        self.sel_mask = self.f.take_pieces('val', cfg['eval']['max_pieces_select'], seed=cfg['seed'])
        self._ts = {}

    def ts(self, specs):
        key = tuple(specs)
        if key not in self._ts:
            self._ts[key] = models.TrainSet(self.f, self.cfg, list(specs))
        return self._ts[key]

    def calibration(self, params, z=None):
        """Calibration of a score model on the training frames (path weights only)."""
        if z is None:
            z = ScoreModel(params).raw(self.f.cv[self.cal_idx])
        return ev.calibrate(z, self.f.q[self.cal_idx], self.f.w[self.cal_idx].astype(float))

    def score_sets(self, ts):
        """Standardized features of the calibration and the selection frames, for scoring on the training device."""
        return [((ts.feat(self.f.cv[i]) - ts.mu) / ts.sd).astype(np.float32)
                for i in (self.cal_idx, np.flatnonzero(self.sel_mask))]

    def select_metrics(self, params, scores=None):
        """Selection metrics on the validation pieces. `scores`: (z of the calibration frames, z of the selection
        frames) when the caller already has them (M3 checkpoints, scored on the GPU); else the numpy forward pass."""
        z_cal, z = scores if scores is not None else (None, ScoreModel(params).raw(self.f.cv[self.sel_mask]))
        cal = self.calibration(params, z_cal)
        out = ev.evaluate(self.f, self.sel_mask, z, cal, self.cfg, full=False)
        return dict(sdzqs=out['sdzqs'], spread_mean=out['spread_mean'], zq_sd=out['zq_sd'])


def _ranking(cfg):
    path = os.path.join(cfg['out_dir'], 'screen', 'importance.json')
    if not os.path.isfile(path):
        raise SystemExit(f'run `run.py screen {cfg["name"]}` first')
    with open(path) as f:
        return json.load(f)['specs']


def feature_sets(cfg, frames):
    """Feature lists of the sweep: the top-K screened features for each K, and the raw CVs."""
    ranking = _ranking(cfg)
    sets = {f'top{k}': ranking[:k] for k in cfg['features']['top_k']}
    cvs = frames.cv_names.tolist()
    if cvs not in [list(v) for v in sets.values()]:
        sets['cvs'] = cvs
    # configured sets, e.g. [features.sets.noclean] exclude = ["y", "x"]: all CVs but those
    for name, spec in cfg['features'].get('sets', {}).items():
        sets[name] = [c for c in cvs if c not in spec.get('exclude', [])]
    return sets


# ---------------------------------------------------------------------------------------------------
# M2 screening
# ---------------------------------------------------------------------------------------------------


def cmd_screen(cfg, args):
    ctx = Context(cfg)
    lib = cfg['features']['library']
    if lib == 'cvs':                       # the raw CVs (embedded systems)
        lib = ctx.f.cv_names.tolist()
    ts_tr = ctx.ts(lib)
    ts_val = models.TrainSet(ctx.f, cfg, lib, split='val', stats=(ts_tr.mu, ts_tr.sd))
    res = models.screen_gbt(ts_tr, ts_val, cfg, seed=cfg['seed'])
    out = _dir(cfg, 'screen')
    _save_json(res, os.path.join(out, 'importance.json'))
    common.say('M2 permutation importance (validation MSE increase):', log=ctx.log)
    for s, v, e in zip(res['specs'], res['importance'], res['importance_sd']):
        common.say(f'  {s:8s} {v:10.4f} +- {e:.4f}', log=ctx.log)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(lib) + 1.2))
    ax.barh(res['specs'][::-1], res['importance'][::-1], xerr=res['importance_sd'][::-1], color='0.4')
    ax.set_xlabel('permutation importance (increase of weighted MSE of logit q, validation)')
    ax.set_title(f'{cfg["name"]}: M2 gradient-boosted trees, feature screening')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'importance.png'), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------------------------------


def cmd_fit(cfg, args):
    ctx = Context(cfg)
    out = _dir(cfg, 'fit')
    runs_dir = _dir(cfg, 'fit', 'runs')
    sets = feature_sets(cfg, ctx.f)
    results_path = os.path.join(out, 'results.json')
    results = json.load(open(results_path)) if os.path.isfile(results_path) and not args.fresh else []
    done = {r['id'] for r in results}

    def record(run):
        results[:] = [r for r in results if r['id'] != run['id']] + [run]
        _save_json(results, results_path)

    if args.only in (None, 'm1'):
        c = cfg['m1']
        for set_name, specs in sets.items():
            ts = ctx.ts(specs)
            todo = [('logit', C, models.SparseLogit(C, c['max_frames'], seed=cfg['seed'])) for C in c['l1_C']]
            todo += [('gam', a, models.SplineGAM(c['gam_knots'], a, c['max_frames'])) for a in c['gam_ridge']]
            for fam, hp, m in todo:
                rid = f'm1_{fam}_{set_name}_{hp:g}'
                if rid in done:
                    continue
                t0 = time.time()
                m.fit(ts)
                secs = time.time() - t0
                sel = ctx.select_metrics(m.export())
                np.savez(os.path.join(runs_dir, rid + '.npz'), **m.export())
                extra = dict(n_nonzero=getattr(m, 'n_nonzero', None))
                record(dict(id=rid, family=f'M1-{fam}', name=m.name, features=specs, set=set_name, hp=hp,
                            train_seconds=secs, **sel, **extra))
                common.say(f'{rid:28s} {secs:6.1f} s  sdzqs {sel["sdzqs"]:8.3f}  spread {sel["spread_mean"]:.4f}',
                           log=ctx.log)

    if args.only in (None, 'm3'):
        c = cfg['m3']
        archs, betas = c['archs'], c['betas']
        set_items = list(sets.items())
        if args.quick:
            archs, betas, set_items = archs[:1], betas[:2], set_items[:1]
        elif getattr(args, 'lean', False):
            # the short sweep of [m3.lean]: by default the raw CVs (or the largest top-K set), two sizes, beta 0 / 10
            lean = c.get('lean', {})
            archs, betas = lean.get('archs', [[64, 64], [128, 128, 128]]), lean.get('betas', [0.0, 10.0])
            names = lean.get('sets', ['cvs'] if 'cvs' in sets else [set_items[-1][0]])
            set_items = [(k, v) for k, v in set_items if k in names]
        for set_name, specs in set_items:
            ts = ctx.ts(specs)
            sets_dev = ctx.score_sets(ts)
            for arch in archs:
                for beta in betas:
                    rid = f'm3_{set_name}_{"x".join(map(str, arch))}_b{beta:g}'
                    if rid in done:
                        continue
                    ckpts = []

                    def on_ckpt(epoch, params, scores):
                        sel = ctx.select_metrics(params, scores)
                        ckpts.append(dict(epoch=epoch, **sel))
                        common.say(f'   {rid} epoch {epoch:3d}: sdzqs {sel["sdzqs"]:8.3f} '
                                   f'spread {sel["spread_mean"]:.4f}', log=ctx.log)

                    m = models.MLP(arch, beta, cfg, seed=cfg['seed'])
                    m.fit(ts, on_checkpoint=on_ckpt, score_sets=sets_dev)
                    best = int(np.argmin([k['sdzqs'] for k in ckpts]))   # selected by the dynamical metric
                    m.use_checkpoint(best)
                    np.savez(os.path.join(runs_dir, rid + '.npz'), **m.export())
                    np.savez(os.path.join(runs_dir, rid + '_checkpoints.npz'),
                             **{f'e{c["epoch"]}__{k}': v for c in m.checkpoints for k, v in c['params'].items()})
                    record(dict(id=rid, family='M3', name=m.name, features=specs, set=set_name, arch=arch, beta=beta,
                                train_seconds=m.train_seconds, selected_epoch=m.epoch, checkpoints=ckpts,
                                history=m.history, sdzqs=ckpts[best]['sdzqs'],
                                spread_mean=ckpts[best]['spread_mean'], zq_sd=ckpts[best]['zq_sd']))
                    common.say(f'{rid:28s} {m.train_seconds:6.1f} s  epoch {m.epoch}  '
                               f'sdzqs {ckpts[best]["sdzqs"]:8.3f}', log=ctx.log)


FAST_TOL = 0.10          # M3-fast: the cheapest M3 run within 10% of the best validation SD(Z_q)


def _n_params(arch, n_in):
    sizes = [n_in] + list(arch) + [1]
    return sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:]))


def best_runs(cfg):
    """Per family, the run with the lowest validation selection metric (SD of the stopped Z_q); and 'M3-fast',
    the M3 run with the fewest parameters among those within FAST_TOL of the best (cost is priority 1)."""
    with open(os.path.join(cfg['out_dir'], 'fit', 'results.json')) as f:
        results = json.load(f)
    best = {}
    for r in results:
        if r['family'] not in best or r['sdzqs'] < best[r['family']]['sdzqs']:
            best[r['family']] = r
    if 'M3' in best:
        ok = [r for r in results if r['family'] == 'M3' and r['sdzqs'] <= (1 + FAST_TOL) * best['M3']['sdzqs']]
        fast = min(ok, key=lambda r: (_n_params(r['arch'], len(r['features'])), r['sdzqs']))
        if fast['id'] != best['M3']['id']:
            best['M3-fast'] = fast
    return results, best


# ---------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------


def _candidates(ctx, include_all=True):
    """(id, label, score params or ('ref', key)) of every candidate to evaluate."""
    cfg, f = ctx.cfg, ctx.f
    results, best = best_runs(cfg)
    cands = [('seed', models.Seed(f, cfg).name, models.Seed(f, cfg).export()), ('q_np', 'q_np (nonparametric)', 'q_np')]
    if f.has_exact:
        cands.append(('exact', 'exact committor', 'exact'))
    runs = results if include_all else list(best.values())
    for r in runs:
        p = dict(np.load(os.path.join(cfg['out_dir'], 'fit', 'runs', r['id'] + '.npz')))
        cands.append((r['id'], r['name'] + f' [{r["set"]}]', p))
    return cands, best


def _ref_scores(ctx, key, idx):
    q = ctx.f.q[idx] if key == 'q_np' else ctx.f.q_exact[idx].astype(float)
    return common.logit(q)


def summarize(table, cfg, has_exact):
    """Mean spreads over the same planned levels for every candidate (evaluate.spread_levels): on test/OOD the
    reference is the exact committor if there is one; on the validation paths it is q_np only (no exact
    information in anything used for selection)."""
    cids = [c for c in table if not c.startswith('_')]
    for split in ('test', 'ood', 'val'):
        if split not in table['q_np']:
            continue
        if split == 'val':
            use = np.asarray(table['q_np']['val']['spread_n']) >= 50
        else:
            use = ev.spread_levels(table['exact' if has_exact else 'q_np'][split], cfg['eval']['slab_half_logit'])
        table['_spread_levels_' + split] = ev.planned_levels(cfg)[use]
        for cid in cids:
            table[cid][split].update(ev.spread_summary(table[cid][split], use))


def cmd_summarize(cfg, args):
    path = os.path.join(cfg['out_dir'], 'eval', 'metrics.json')
    with open(path) as fh:
        table = json.load(fh)
    summarize(table, cfg, 'exact' in table)
    table['_best'] = {fam: r['id'] for fam, r in best_runs(cfg)[1].items()}
    _save_json(table, path)
    common.say(f'{cfg["name"]}: spread levels test {np.round(table["_spread_levels_test"], 4)}')


def cmd_evaluate(cfg, args):
    ctx = Context(cfg)
    f = ctx.f
    out_dir = _dir(cfg, 'eval')
    cands, best = _candidates(ctx)
    table = {}
    for split in ('test', 'ood'):
        mask = f.take_pieces(split)
        idx = np.flatnonzero(mask)
        ref = ev.reference_turns(f, mask, cfg)
        common.say(f'-- {split}: {len(idx)} frames; reference turns {ref}', log=ctx.log)
        for cid, label, p in cands:
            t0 = time.time()
            if isinstance(p, str):
                z, zcal = _ref_scores(ctx, p, idx), _ref_scores(ctx, p, ctx.cal_idx)
                cal = ev.calibrate(zcal, f.q[ctx.cal_idx], f.w[ctx.cal_idx].astype(float))
            else:
                z, cal = ScoreModel(p).raw(f.cv[idx]), ctx.calibration(p)
            res = ev.evaluate(f, mask, z, cal, cfg, ref_turns=ref)
            res.update(label=label)
            table.setdefault(cid, {})[split] = res
            common.say(f'   {cid:28s} zqs_z {res["zqs_z"]:6.2f}  sdzqs {res["sdzqs"]:8.3f}  step {res["step_ratio"]:.3f}  '
                       f'turns/q_np {res.get("turn_ratio_qnp", np.nan):.3f}  ({time.time() - t0:.0f} s)', log=ctx.log)
        table['_ref_turns_' + split] = ref
    # validation metrics of every candidate with the final definitions: for the selection-sensitivity check
    vidx = np.flatnonzero(ctx.sel_mask)
    vals = {}
    for cid, label, p in cands:
        if isinstance(p, str):
            z, zcal = _ref_scores(ctx, p, vidx), _ref_scores(ctx, p, ctx.cal_idx)
            cal = ev.calibrate(zcal, f.q[ctx.cal_idx], f.w[ctx.cal_idx].astype(float))
        else:
            z, cal = ScoreModel(p).raw(f.cv[vidx]), ctx.calibration(p)
        vals[cid] = ev.evaluate(f, ctx.sel_mask, z, cal, cfg, full=False)
    for cid in vals:
        table[cid]['val'] = vals[cid]
    summarize(table, cfg, f.has_exact)
    table['_best'] = {fam: r['id'] for fam, r in best.items()}
    _save_json(table, os.path.join(out_dir, 'metrics.json'))
    if not args.quick:
        tis_plots(ctx, cands, best)


def cmd_tis(cfg, args):
    """Only the TIS diagnostics of the selected candidates (as at the end of evaluate)."""
    ctx = Context(cfg)
    cands, best = _candidates(ctx, include_all=False)
    tis_plots(ctx, cands, best)


def tis_plots(ctx, cands, best):
    """The optimalrcs TIS diagnostics (plots_tis) and path-weighted feps of the selected candidates, per split,
    with CommittorNE(path_weights, group_traj) and A/B traps as in the notebooks; r = g(lambda)."""
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    sys.path.insert(0, os.path.join(common.REPO, 'optimalrcs-cupy'))
    import cupy as cp
    import optimalrcs as orc_cp
    cfg, f = ctx.cfg, ctx.f
    keep = {'seed', 'q_np', 'exact'} | {r['id'] for r in best.values()}
    for split in ('test', 'ood'):
        mask = f.take_pieces(split)
        idx = np.flatnonzero(mask)
        A, B = f.A[idx], f.B[idx]
        q = orc_cp.CommittorNE(boundary0=A, boundary1=B, i_traj=f.piece[idx], t_traj=f.t[idx].astype(float),
                               path_weights=f.w[idx].astype(float), group_traj=f.path[idx])
        q.set_fixed_traj_length_trap(A | B, 2 ** 16)
        for cid, label, p in cands:
            if cid not in keep:
                continue
            if isinstance(p, str):
                z, zcal = _ref_scores(ctx, p, idx), _ref_scores(ctx, p, ctx.cal_idx)
                cal = ev.calibrate(zcal, f.q[ctx.cal_idx], f.w[ctx.cal_idx].astype(float))
            else:
                z, cal = ScoreModel(p).raw(f.cv[idx]), ctx.calibration(p)
            r = ev.apply_cal(cal, z)
            r[A], r[B] = 0.0, 1.0
            q.r_traj = cp.asarray(r, dtype=cp.float64)
            d = _dir(cfg, 'eval', 'tis_plots', split)
            q.plots_tis()
            plt.gcf().suptitle(f'{cfg["name"]} {split}: {label}', y=1.02)
            plt.gcf().savefig(os.path.join(d, f'{cid}_tis.png'), dpi=110, bbox_inches='tight')
            plt.close('all')
            q.plots_feps()
            plt.gcf().suptitle(f'{cfg["name"]} {split}: {label} (path-weighted)', y=1.02)
            plt.gcf().savefig(os.path.join(d, f'{cid}_feps_weighted.png'), dpi=110, bbox_inches='tight')
            plt.close('all')
        del q
        cp.get_default_memory_pool().free_all_blocks()
    common.say('TIS plots in ' + os.path.join(cfg['out_dir'], 'eval', 'tis_plots'), log=ctx.log)


# ---------------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------------


def build_export(ctx, score_params):
    """LambdaModel parameters: the score model + its calibration + A/B handling + OOD statistics."""
    cfg, f = ctx.cfg, ctx.f
    p = dict(score_params)
    zk, qk, dk = ctx.calibration(score_params)
    tr = np.flatnonzero(f.mask('train') & f.valid)
    sm = ScoreModel(score_params)
    F = sm.features(f.cv[tr[::max(1, len(tr) // 400_000)]])
    mean = F.mean(0)
    cov = np.atleast_2d(np.cov(F, rowvar=False))
    icov = np.linalg.pinv(cov)
    d2 = np.einsum('ni,ij,nj->n', F - mean, icov, F - mean)
    cv_tr = f.cv[tr]
    p.update(cal_z=zk, cal_q=qk, cal_d=dk, lambda_a=f.lambda_a, lambda_b=f.lambda_b,
             state_cv=cfg['data']['state_cv'], ramp=cfg['export']['ramp'], scale_eps=cfg['export']['scale_eps'],
             cv_min=cv_tr.min(0), cv_max=cv_tr.max(0), maha_mean=mean, maha_icov=icov,
             maha_thr=float(np.quantile(d2, cfg['export']['ood_quantile'])))
    return p


def benchmark(model, cv, reps=20000):
    """us per frame: one-frame calls (as the sampler makes them) and one batch call."""
    rows = [np.array(c, float) for c in cv[:reps]]
    for r in rows[:200]:
        model.one(r)
    t0 = time.perf_counter()
    for r in rows:
        model.one(r)
    one = (time.perf_counter() - t0) / len(rows) * 1e6
    big = np.asarray(cv[:200_000], float)
    t0 = time.perf_counter()
    model(big)
    batch = (time.perf_counter() - t0) / len(big) * 1e6
    return dict(us_per_frame_one=one, us_per_frame_batch=batch)


def cmd_export(cfg, args):
    ctx = Context(cfg)
    f = ctx.f
    results, best = best_runs(cfg)
    out = os.path.join(common.HERE, 'exported')
    os.makedirs(out, exist_ok=True)
    seed = models.Seed(f, cfg)
    todo = [('seed', seed.export(), None)] + [(fam, dict(np.load(os.path.join(cfg['out_dir'], 'fit', 'runs',
                                                                              r['id'] + '.npz'))), r)
                                              for fam, r in best.items()]
    summary = {}
    inside = np.flatnonzero(f.valid & f.mask('test'))
    for fam, sp, r in todo:
        params = build_export(ctx, sp)
        tag = fam.lower().replace(' ', '_')
        path = os.path.join(out, f'{cfg["name"]}_{tag}.npz')
        np.savez(path, **params)
        model = LambdaModel(path)
        bench = benchmark(model, f.cv[inside])
        # transparency of the A/B handling: how much of the data lies in the ramps and how far h is from m there
        cv = f.cv[inside]
        s = cv[:, cfg['data']['state_cv']]
        m = model.q(cv)
        h = model.blend(s, m)
        in_ramp = (s - f.lambda_a < model.ramp_w) | (f.lambda_b - s < model.ramp_w)
        w = f.w[inside].astype(float)
        ramp = dict(ramp_width=model.ramp_w, weight_fraction_in_ramps=float(w[in_ramp].sum() / w.sum()),
                    max_abs_h_minus_m=float(np.max(np.abs(h - m))),
                    rms_h_minus_m_in_ramps=float(np.sqrt(np.average((h - m)[in_ramp] ** 2, weights=w[in_ramp])))
                    if in_ramp.any() else 0.0)
        summary[fam] = dict(file=os.path.relpath(path, common.REPO), run=None if r is None else r['id'],
                            n_calibration_knots=len(params['cal_z']), **bench, **ramp,
                            train_seconds=None if r is None else r['train_seconds'])
        common.say(f'{fam:10s} -> {path}: {bench["us_per_frame_one"]:.1f} us/frame (one), '
                   f'{bench["us_per_frame_batch"]:.3f} us/frame (batch); ramps hold {100 * ramp["weight_fraction_in_ramps"]:.2f}% '
                   f'of the weight, max |h - m| {ramp["max_abs_h_minus_m"]:.3g}', log=ctx.log)
    _save_json(summary, os.path.join(_dir(cfg, 'eval'), 'export.json'))


def cmd_report(systems, args):
    import report
    report.write(systems)
