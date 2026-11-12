"""Training variants: the lean M3 sweep under each setting of configs/<variants>.toml, every run evaluated on the test
paths. The main results (fit/, eval/, exported/) are not touched; output in <out_dir>/variants/<variants>/.

    python run.py variants z_br cc_br --variants loss_variants

A variants file maps a variant name to overrides of config sections, e.g.
    [noise_1e-3]
    train = {noise_q = 1e-3}
Every variant uses the same frames, splits, selection rule (validation SD(Z_q)) and checkpoints as `fit --lean`.
"""
import copy
import json
import os
import time
import tomllib

import numpy as np

import common
import evaluate as ev
import models
import steps
from lambda_runtime import ScoreModel


def _merge(cfg, over):
    out = copy.deepcopy(cfg)
    for sec, kv in over.items():
        out.setdefault(sec, {}).update(kv)
    return out


def _levels_low(ref, half):
    """Levels that the exact committor reaches (>= 50 frames) but the standard mean drops because q_np is unreliable
    there (label noise at small q, or a sparsely visited region): what a noise-aware loss is meant to improve."""
    std = ev.spread_levels(ref, half)
    return (np.asarray(ref['spread_n']) >= 50) & ~std, std


def cmd_variants(cfg, args):
    with open(os.path.join(common.HERE, 'configs', args.variants + '.toml'), 'rb') as fh:
        variants = tomllib.load(fh)
    ctx = steps.Context(cfg)
    f = ctx.f
    out = steps._dir(cfg, 'variants', args.variants)
    runs_dir = steps._dir(cfg, 'variants', args.variants, 'runs')
    path = os.path.join(out, 'results.json')
    rows = json.load(open(path)) if os.path.isfile(path) and not args.fresh else []
    done = {r['id'] for r in rows}

    lean = cfg['m3'].get('lean', {})
    archs, betas = lean.get('archs', [[64, 64], [128, 128, 128]]), lean.get('betas', [0.0, 10.0])
    sets = steps.feature_sets(cfg, f)
    set_name = lean.get('sets', ['cvs'] if 'cvs' in sets else [list(sets)[-1]])[0]
    specs = sets[set_name]

    mask = f.take_pieces('test')
    idx = np.flatnonzero(mask)
    ref_turns = ev.reference_turns(f, mask, cfg)
    half = cfg['eval']['slab_half_logit']
    refs = {}
    for key in ('q_np', 'exact') if f.has_exact else ('q_np',):
        z, zcal = steps._ref_scores(ctx, key, idx), steps._ref_scores(ctx, key, ctx.cal_idx)
        cal = ev.calibrate(zcal, f.q[ctx.cal_idx], f.w[ctx.cal_idx].astype(float))
        refs[key] = ev.evaluate(f, mask, z, cal, cfg, ref_turns=ref_turns)
    low, std = _levels_low(refs['exact' if f.has_exact else 'q_np'], half)

    def summary(res):
        s = dict(test_sdzqs=res['sdzqs'], test_zqs_z=res['zqs_z'], step_ratio=res['step_ratio'],
                 turn_ratio_exact=res.get('turn_ratio_exact'))
        for name, use in (('std', std), ('low', low)):
            for key in ('spread', 'spread_exact'):
                if key in res:
                    v = np.asarray(res[key], float)[use]
                    s[f'{key}_{name}'] = float(np.nanmean(v)) if np.isfinite(v).any() else None
        return s

    seeds = args.seeds or [cfg['seed']]
    for vname, over in variants.items():
        vcfg = _merge(cfg, over)
        ts = models.TrainSet(f, vcfg, list(specs))
        score_sets = ctx.score_sets(ts)
        for seed, arch, beta in ((s, a, b) for s in seeds for a in archs for b in betas):
                rid = f'{vname}__{set_name}_{"x".join(map(str, arch))}_b{beta:g}'
                rid += f'_s{seed}' if seed != cfg['seed'] else ''
                if rid in done:
                    continue
                ckpts = []

                def on_ckpt(epoch, params, scores):
                    ckpts.append(dict(epoch=epoch, **ctx.select_metrics(params, scores)))

                m = models.MLP(arch, beta, vcfg, seed=seed).fit(ts, on_ckpt, score_sets)
                best = int(np.argmin([c['sdzqs'] for c in ckpts]))
                m.use_checkpoint(best)
                p = m.export()
                np.savez(os.path.join(runs_dir, rid + '.npz'), **p)
                t0 = time.time()
                res = ev.evaluate(f, mask, ScoreModel(p).raw(f.cv[idx]), ctx.calibration(p), cfg, ref_turns=ref_turns)
                row = dict(id=rid, variant=vname, overrides=over, seed=seed, arch=arch, beta=beta, epoch=m.epoch,
                           train_seconds=m.train_seconds, val_sdzqs=ckpts[best]['sdzqs'],
                           val_spread=ckpts[best]['spread_mean'], checkpoints=ckpts,
                           spread_levels=res['spread'], spread_exact_levels=res.get('spread_exact'), **summary(res))
                rows[:] = [r for r in rows if r['id'] != rid] + [row]
                steps._save_json(rows, path)
                common.say(f'{rid:36s} {m.train_seconds:5.1f} s  val sdzqs {row["val_sdzqs"]:6.3f}  test spread '
                           f'(exact) std {row.get("spread_exact_std", np.nan):.4f} low '
                           f'{row.get("spread_exact_low") or np.nan:.3f}  sdzqs {row["test_sdzqs"]:.3f}  '
                           f'({time.time() - t0:.0f} s eval)', log=ctx.log)

    ref_rows = {k: summary(v) for k, v in refs.items()}
    steps._save_json(dict(levels=ev.planned_levels(cfg), std_levels=std, low_levels=low, refs=ref_rows),
                     os.path.join(out, 'levels.json'))
    write_table(cfg, rows, ref_rows, variants, ev.planned_levels(cfg), std, low, os.path.join(out, 'summary.md'))


def write_table(cfg, rows, refs, variants, levels, std, low, path):
    """Per variant: the run the selection rule picks (validation SD(Z_q)) for each seed, as mean +- SD over seeds,
    and the median over all runs (selection noise)."""
    exact = 'spread_exact_std' in next(iter(refs.values()))
    key = 'spread_exact' if exact else 'spread'
    seeds = sorted({r.get('seed', cfg['seed']) for r in rows})
    lines = [f'# {cfg["name"]}: training variants ({len(rows)} runs, seeds {seeds})', '',
             f'Test paths. Spread {"vs the exact committor" if exact else "vs q_np"}: "std" = the report\'s level set '
             f'(q = {", ".join(f"{q:.2g}" for q in levels[std])}); "unrel." = levels the exact committor reaches but '
             f'the report drops because q_np is unreliable there (too few frames or label noise; '
             f'q = {", ".join(f"{q:.2g}" for q in levels[low]) or "none"}). Selected: the lean run with the lowest '
             f'validation SD(Z_q), per seed; mean ± SD over seeds.', '',
             '| variant | overrides | selected runs | spread std | spread unrel. | test SD(Z_q) | step ratio | '
             'median over all runs: spread std / unrel. | train s (all runs) |', '|---|---|---|---|---|---|---|---|---|']
    fmt = lambda v, d=4: '–' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{d}f}'

    def ms(vals, d=4):
        v = np.array([x for x in vals if x is not None], float)
        if not len(v):
            return '–'
        return f'{v.mean():.{d}f}' + (f' ± {v.std(ddof=1):.{d}f}' if len(v) > 1 else '')

    for name in ('q_np', 'exact'):
        if name in refs:
            r = refs[name]
            lines.append(f'| {name} | | | {fmt(r.get(key + "_std"))} | {fmt(r.get(key + "_low"))} | '
                         f'{fmt(r["test_sdzqs"], 3)} | {fmt(r["step_ratio"], 3)} | | |')
    for vname, over in variants.items():
        rs = [r for r in rows if r['variant'] == vname]
        if not rs:
            continue
        sel = [min([r for r in rs if r.get('seed', cfg['seed']) == s], key=lambda r: r['val_sdzqs'])
               for s in seeds if any(r.get('seed', cfg['seed']) == s for r in rs)]
        med = lambda k: float(np.median([r[k] for r in rs if r.get(k) is not None])) if any(
            r.get(k) is not None for r in rs) else None
        ov = '; '.join(f'{s}.{k}={v}' for s, kv in over.items() for k, v in kv.items()) or '(as configured)'
        names = ', '.join(f'{"x".join(map(str, r["arch"]))} β={r["beta"]:g}' for r in sel)
        lines.append(f'| {vname} | {ov} | {names} | {ms([r.get(key + "_std") for r in sel])} | '
                     f'{ms([r.get(key + "_low") for r in sel], 3)} | {ms([r["test_sdzqs"] for r in sel], 3)} | '
                     f'{ms([r["step_ratio"] for r in sel], 3)} | {fmt(med(key + "_std"))} / {fmt(med(key + "_low"))} | '
                     f'{sum(r["train_seconds"] for r in rs):.0f} |')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    common.say('\n'.join(lines))
