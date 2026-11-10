"""Step 6: figures and the markdown report docs/param_lambda_report.md from the saved results.

Reads <out>/eval/metrics.json, <out>/eval/export.json, <out>/fit/results.json, <out>/screen/importance.json and
the exported models; writes figures to <out>/report/ and the report to docs/.
"""
import json
import os

import numpy as np

import common

DOC = os.path.join(common.REPO, 'docs', 'param_lambda_report.md')
# fixed colour per entity (validated categorical slots 1-4; references in neutral ink, dashed for exact)
STYLE = {
    'M1-logit': dict(color='#2a78d6', marker='o', ls='-', label='M1 L1-logistic'),
    'M1-gam': dict(color='#eb6834', marker='s', ls='-', label='M1 spline GAM'),
    'M3': dict(color='#1baf7a', marker='^', ls='-', label='M3 MLP'),
    'M3-fast': dict(color='#1baf7a', marker='v', ls='--', label='M3 MLP, smallest within 10% (cost)'),
    'seed': dict(color='#eda100', marker='D', ls='-', label='seed (current order parameter)'),
    'q_np': dict(color='#52514e', marker='', ls='-', label='q_np (nonparametric, not exportable)'),
    'exact': dict(color='#0b0b0b', marker='', ls='--', label='exact committor (reference)'),
}
TOL_SPREAD, TOL_ZQS = 0.10, 0.5         # "M1 within tolerance of M3" (confirmed in step 1)
FAMILIES = ['M1-logit', 'M1-gam', 'M3', 'M3-fast']


def _load(system):
    cfg = common.load_config(system)
    d = cfg['out_dir']
    with open(os.path.join(d, 'eval', 'metrics.json')) as f:
        metrics = json.load(f)
    with open(os.path.join(d, 'fit', 'results.json')) as f:
        runs = {r['id']: r for r in json.load(f)}
    exp_path = os.path.join(d, 'eval', 'export.json')
    export = json.load(open(exp_path)) if os.path.isfile(exp_path) else {}
    with open(os.path.join(d, 'screen', 'importance.json')) as f:
        screen = json.load(f)
    return cfg, metrics, runs, export, screen


def _rows(metrics):
    """(entity key, candidate id) of the rows of the main table."""
    rows = [('seed', 'seed'), ('q_np', 'q_np')]
    if 'exact' in metrics:
        rows.append(('exact', 'exact'))
    for fam in FAMILIES:
        if fam in metrics['_best']:
            rows.append((fam, metrics['_best'][fam]))
    return rows


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8c8b86',
                         'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
                         'legend.frameon': False, 'lines.linewidth': 2, 'lines.markersize': 5})
    return plt


def fig_spread(cfg, metrics, out):
    import evaluate as ev
    plt = _plt()
    levels = ev.planned_levels(cfg)
    has_exact = 'spread_exact' in metrics['seed']['test']
    nrow = 2 if has_exact else 1
    fig, axes = plt.subplots(nrow, 2, figsize=(10, 3.4 * nrow), sharex=True, squeeze=False)
    for j, split in enumerate(('test', 'ood')):
        for i, (key, title) in enumerate([('spread', 'logit q_np'), ('spread_exact', 'logit q (exact)')][:nrow]):
            ax = axes[i, j]
            for ent, cid in _rows(metrics):
                v = np.array(metrics[cid][split][key], dtype=float)
                st = STYLE[ent]
                ax.plot(levels, v, color=st['color'], marker=st['marker'], ls=st['ls'], label=st['label'])
            ax.set(xscale='log', yscale='log', title=f'{"test paths" if split == "test" else "OOD block (later cycles)"}',
                   ylabel=f'variance of {title} in the slab')
            if i == nrow - 1:
                ax.set_xlabel('planned interface: committor level $q_k$ (slab $|$logit $m$ - logit $q_k| <$ '
                              f'{cfg["eval"]["slab_half_logit"]})')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f'{cfg["name"]}: isosurface spread per planned interface (lower = interfaces closer to isocommittors)')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8)
    path = os.path.join(out, 'spread_per_interface.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_zq(cfg, metrics, out):
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for j, split in enumerate(('test', 'ood')):
        for ent, cid in _rows(metrics):
            r = metrics[cid][split]
            st = STYLE[ent]
            for i, key in enumerate(('zq_z', 'zq_sd')):
                axes[i, j].plot(r['zq_lags'], r[key], color=st['color'], marker=st['marker'], ls=st['ls'],
                                label=st['label'])
        axes[0, j].axhline(2, color='#8c8b86', lw=1, ls=':')
        axes[0, j].set(title='test paths' if split == 'test' else 'OOD block', ylabel='max |stopped $Z_q$ / s.e.| over q')
        axes[1, j].set(xscale='log', xlabel='lag (frames)', ylabel='SD over q of stopped $Z_q$')
        axes[0, j].set_xscale('log')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f'{cfg["name"]}: stopped $Z_q$ of g($\\lambda$) at several lags (path weights, groups = paths)')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8)
    path = os.path.join(out, 'zq_lags.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_map(cfg, metrics, export, out):
    """2D: the planned interfaces of each exported lambda (isolines of g(z) = q_k) over the exact isocommittors."""
    import data
    from lambda_runtime import LambdaModel
    import evaluate as ev
    f = data.Frames(cfg)
    if f.dim != 2 or not f.has_exact:
        return None
    plt = _plt()
    sub = np.flatnonzero(f.valid)[::max(1, f.n // 400_000)]
    y, x = f.cv[sub, 0], f.cv[sub, 1]
    gy = np.linspace(f.lambda_a, f.lambda_b, 161)
    gx = np.linspace(*np.quantile(x, [0.002, 0.998]), 121)
    Y, X = np.meshgrid(gy, gx, indexing='ij')
    cv = np.c_[Y.ravel(), X.ravel()]
    qx = data._exact_q(cfg, cv, cv[:, 0] <= f.lambda_a, cv[:, 0] >= f.lambda_b, f.lambda_a, f.lambda_b).reshape(Y.shape)
    H, _, _ = np.histogram2d(y, x, bins=[np.linspace(gy[0], gy[-1], 41), np.linspace(gx[0], gx[-1], 31)])
    levels = ev.planned_levels(cfg)
    ents = [e for e in ['seed'] + FAMILIES if e in export]
    fig, axes = plt.subplots(1, len(ents), figsize=(3.6 * len(ents), 4.2), sharey=True, squeeze=False)
    for ax, ent in zip(axes[0], ents):
        m = LambdaModel(os.path.join(common.REPO, export[ent]['file']))
        mq = m.q(cv).reshape(Y.shape)
        ax.pcolormesh(np.linspace(gy[0], gy[-1], 41), np.linspace(gx[0], gx[-1], 31), np.log1p(H).T, cmap='Greys',
                      alpha=0.35, shading='flat')
        ax.contour(gy, gx, qx.T, levels=levels, colors='#0b0b0b', linewidths=1.0, linestyles='--')
        ax.contour(gy, gx, mq.T, levels=levels, colors=STYLE[ent]['color'], linewidths=1.6)
        ax.set(title=STYLE[ent]['label'], xlabel=str(f.cv_names[0]))
    axes[0, 0].set_ylabel(str(f.cv_names[1]))
    fig.suptitle(f'{cfg["name"]}: planned interfaces $g(\\lambda) = q_k$ (colour) vs exact isocommittors (black dashed); '
                 'grey: frame density')
    fig.tight_layout()
    path = os.path.join(out, 'interfaces_map.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _label(ent, cid, runs):
    """Row label: the entity, and for fitted models the selected run's features and settings."""
    base = STYLE[ent]['label'].split(' (')[0]
    r = runs.get(cid)
    if r is None:
        return base
    if ent in ('M3', 'M3-fast'):
        hp = f'{"x".join(map(str, r["arch"]))}, β={r["beta"]:g}, epoch {r["selected_epoch"]}'
    else:
        hp = r['name'].split(' ', 2)[-1]
    return f'{base} [{", ".join(r["features"])}; {hp}]'


BETA_RAMP = ['#6da7ec', '#2a78d6', '#1c5cab', '#0d366b']      # one hue, light -> dark: beta is ordered


def fig_selection(cfg, metrics, runs, out):
    """M3 checkpoints of the best run's feature set: validation SD(Z_q) (the selection metric) and isosurface
    spread per epoch, with the exact committor and q_np on the same validation paths for scale."""
    import evaluate as ev
    import steps
    best = runs.get(metrics['_best'].get('M3'))
    if best is None:
        return None
    ctx = steps.Context(cfg)
    f = ctx.f
    ref = {}
    idx = np.flatnonzero(ctx.sel_mask)
    for key in ('q_np', 'exact'):
        if key == 'exact' and not f.has_exact:
            continue
        z, zc = steps._ref_scores(ctx, key, idx), steps._ref_scores(ctx, key, ctx.cal_idx)
        cal = ev.calibrate(zc, f.q[ctx.cal_idx], f.w[ctx.cal_idx].astype(float))
        ref[key] = ev.evaluate(f, ctx.sel_mask, z, cal, cfg, full=False)
    plt = _plt()
    archs = cfg['m3']['archs']
    fig, axes = plt.subplots(1, len(archs), figsize=(3.6 * len(archs), 3.4), sharey='row', squeeze=False)
    for j, arch in enumerate(archs):
        for b, beta in enumerate(cfg['m3']['betas']):
            rid = f'm3_{best["set"]}_{"x".join(map(str, arch))}_b{beta:g}'
            r = runs.get(rid)
            if r is None:
                continue
            ep = [c['epoch'] for c in r['checkpoints']]
            axes[0, j].plot(ep, [c['sdzqs'] for c in r['checkpoints']], color=BETA_RAMP[b % 4], marker='o',
                            label=f'β = {beta:g}')
            axes[0, j].plot([r['selected_epoch']], [r['sdzqs']], marker='o', ms=10, mfc='none', color='#0b0b0b')
        for name, st in (('exact', STYLE['exact']), ('q_np', STYLE['q_np'])):
            if name in ref:
                axes[0, j].axhline(ref[name]['sdzqs'], color=st['color'], ls=st['ls'], lw=1.2,
                                   label=st['label'].split(' (')[0])
        axes[0, j].set_title(f'{"x".join(map(str, arch))} [{best["set"]}]')
        axes[0, j].set_xlabel('epoch')
    axes[0, 0].set_ylabel('validation SD($Z_q$), max over lags')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{cfg["name"]}: M3 checkpoints (circled: selected by SD($Z_q$))')
    fig.tight_layout()
    path = os.path.join(out, 'm3_checkpoints.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, ref


DIFF_LIM = 0.15          # colour range of the difference maps (as staple_compare)
MAP_ROWS = ['q_np', 'seed', 'M1-logit', 'M1-gam', 'M3', 'M3-fast']


def _candidate_q(ent, f, idx, export):
    """Calibrated committor g(lambda) of an entity at the frames idx (q_np itself for 'q_np')."""
    from lambda_runtime import LambdaModel
    if ent == 'q_np':
        return f.q[idx]
    return LambdaModel(os.path.join(common.REPO, export[ent]['file'])).q(f.cv[idx])


def _binned(coords, edges, val, w, min_count=20):
    sample = np.column_stack(coords)
    n = np.histogramdd(sample, bins=edges)[0]
    sw = np.histogramdd(sample, bins=edges, weights=w)[0]
    s = np.histogramdd(sample, bins=edges, weights=w * val)[0]
    return np.where(n >= min_count, s / np.where(sw > 0, sw, 1), np.nan)


def _rms_transition(diff, ref):
    t = (ref > 0.05) & (ref < 0.95)
    return float(np.sqrt(np.nanmean(diff[t] ** 2))) if t.any() else np.nan


def fig_committor_maps(cfg, export, out):
    """2D: path-weighted mean of g(lambda) per (y, x) bin on the test paths, and its difference from the exact
    committor (RMS over the bins with 0.05 < exact < 0.95), as the staple_compare maps.
    1D: the same along the order parameter, against the exact committor."""
    import data
    f = data.Frames(cfg)
    if not f.has_exact:
        return None
    plt = _plt()
    idx = np.flatnonzero(f.take_pieces('test') & f.valid)
    w = f.w[idx].astype(float)
    ents = [e for e in MAP_ROWS if e == 'q_np' or e in export]
    qs = {e: _candidate_q(e, f, idx, export) for e in ents}
    qx = f.q_exact[idx].astype(float)
    if f.dim == 1:
        x = f.cv[idx, 0]
        e1 = np.linspace(f.lambda_a, f.lambda_b, 101)
        mid = (e1[1:] + e1[:-1]) / 2
        ref = _binned((x,), (e1,), qx, w)
        fig, (ax, axd) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax.plot(mid, ref, color=STYLE['exact']['color'], ls='--', lw=2, label=STYLE['exact']['label'], zorder=5)
        for e in ents:
            P = _binned((x,), (e1,), qs[e], w)
            st = STYLE[e]
            lab = f'{st["label"]} (RMS {_rms_transition(P - ref, ref):.4f})'
            ax.plot(mid, P, color=st['color'], ls=st['ls'], lw=1.5, label=lab)
            axd.plot(mid, P - ref, color=st['color'], ls=st['ls'], lw=1.5)
        axd.axhline(0, color='#8c8b86', lw=1)
        name = str(f.cv_names[0])
        ax.set(ylabel='q (path-weighted mean per bin)',
               title=f'{cfg["name"]}: committor along the order parameter {name} (test paths)')
        axd.set(xlabel=f'order parameter {name}', ylabel='$-$ exact')
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        path = os.path.join(out, 'committor_profiles.png')
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    y, x = f.cv[idx, 0], f.cv[idx, 1]
    e1 = np.linspace(*np.percentile(y, [0.5, 99.5]), 81)
    e2 = np.linspace(*np.percentile(x, [0.5, 99.5]), 33)
    m1, m2 = (e1[1:] + e1[:-1]) / 2, (e2[1:] + e2[:-1]) / 2
    ref = _binned((y, x), (e1, e2), qx, w)
    contours = None
    try:
        import toy2d
        d = cfg['data']
        settings = toy2d.read_full_settings(d['tis_dir'])
        pot = toy2d.potential_from_settings(settings, d.get('potential'), d.get('potential_kwargs'))
        g1, g2 = np.linspace(e1[0], e1[-1], 120), np.linspace(e2[0], e2[-1], 60)
        V = toy2d.potential_grid(pot, g1, g2) / settings['temperature']
        contours = (g1, g2, V - np.nanmin(V), np.arange(1, 16, 1.0))
    except Exception as err:                                # contours are decoration only
        common.say(f'no potential contours: {err}')

    def panel(ax, M, title, cmap, lo, hi, label):
        ax.grid(False)
        im = ax.pcolormesh(e1, e2, M.T, cmap=cmap, vmin=lo, vmax=hi)
        if contours is not None:
            g1, g2, V, lev = contours
            ax.contour(g1, g2, V.T, levels=lev, colors='0.45', linewidths=0.5)
        ax.contour(m1, m2, np.nan_to_num(ref, nan=0.5).T, levels=[0.1, 0.5, 0.9], colors='k', linewidths=0.8,
                   linestyles='--')
        ax.set(title=title, xlabel=f'order parameter {f.cv_names[0]}', ylabel=str(f.cv_names[1]),
               xlim=(e1[0], e1[-1]), ylim=(e2[0], e2[-1]))
        fig.colorbar(im, ax=ax, label=label)

    fig, axes = plt.subplots(len(ents) + 1, 2, figsize=(12, 3.6 * (len(ents) + 1)), squeeze=False)
    panel(axes[0, 0], ref, 'exact committor', 'coolwarm', 0, 1, '$q$')
    H = np.histogram2d(y, x, bins=(e1, e2), weights=w)[0]
    panel(axes[0, 1], np.where(H > 0, np.log10(np.where(H > 0, H, 1)), np.nan), 'test frames (log10 path weight)',
          'Greys', None, None, '')
    for k, e in enumerate(ents, start=1):
        M = _binned((y, x), (e1, e2), qs[e], w)
        lab = STYLE[e]['label'].split(' (')[0]
        panel(axes[k, 0], M, f'{lab}: g($\\lambda$)', 'coolwarm', 0, 1, '$q$')
        panel(axes[k, 1], M - ref, f'{lab} $-$ exact, RMS {_rms_transition(M - ref, ref):.4f}', 'PuOr',
              -DIFF_LIM, DIFF_LIM, '')
    fig.suptitle(f'{cfg["name"]}: committor maps on the test paths (dashed: exact q = 0.1, 0.5, 0.9; grey: '
                 'potential, 1 kT steps)', y=1.0)
    fig.tight_layout()
    path = os.path.join(out, 'committor_maps.png')
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return path


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '–'
    if isinstance(v, (int, np.integer)):
        return str(v)
    a = abs(v)
    if a != 0 and (a < 10 ** -nd or a >= 1e4):
        return f'{v:.2e}'
    return f'{v:.{nd}f}'


def _spread(r, key):
    v, miss = r.get(key + '_mean'), r.get(key + '_missing', 0)
    return _fmt(v) + (f' ({miss} missing)' if miss else '')


COLS = [
    ('spread (q_np)', lambda r: _spread(r, 'spread')),
    ('spread (exact)', lambda r: _spread(r, 'spread_exact')),
    ('zqs_z', lambda r: r['zqs_z']),
    ('SD(Z_q)', lambda r: r['sdzqs']),
    ('step ratio', lambda r: r['step_ratio']),
    ('Δ RMSE lag 1 / 8', lambda r: f'{_fmt(r["inc_rmse"]["1"])} / {_fmt(r["inc_rmse"]["8"])}'),
    ('sign-change ratio', lambda r: r['sign_ratio']),
    ('turns / q_np', lambda r: r.get('turn_ratio_qnp')),
    ('turns / exact', lambda r: r.get('turn_ratio_exact')),
    ('crossings / q_np', lambda r: r.get('cross_ratio_qnp')),
    ('Δr² / q_np (lag 1)', lambda r: r['dr2_ratio']['1']),
    ('Spearman', lambda r: r['spearman']),
    ('logit RMSE (q_np / exact)', lambda r: f'{_fmt(r["logit_rmse"])} / {_fmt(r.get("logit_rmse_exact"))}'),
]


def _table(metrics, rows, split, export, runs, labels=None):
    head = ['candidate'] + [c for c, _ in COLS] + ['µs/frame (one / batch)', 'training s']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for ent, cid in rows:
        r = metrics[cid][split]
        cells = [labels[ent] if labels else r['label']]
        for _, fn in COLS:
            v = fn(r)
            cells.append(v if isinstance(v, str) else _fmt(v))
        e = export.get(ent, {})
        cells.append(f'{e["us_per_frame_one"]:.1f} / {e["us_per_frame_batch"]:.2f}' if e else '–')
        run = runs.get(cid)
        cells.append(f'{run["train_seconds"]:.1f}' if run else '–')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def sensitivity_table(metrics, runs):
    """Per family: the run selected by the validation SD(Z_q) (the rule) and the one a selection by validation
    isosurface spread (q_np, no exact information) would take, with their test metrics."""
    if not any('val' in metrics[c] for c in runs if c in metrics):
        return ''
    head = ['family', 'selected by', 'run', 'val SD(Z_q)', 'val spread (q_np)', 'test spread (q_np)',
            'test spread (exact)', 'test zqs_z', 'test SD(Z_q)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for fam in FAMILIES:
        cands = [c for c, r in runs.items() if r['family'] == fam and c in metrics and 'val' in metrics[c]]
        if not cands:
            continue
        by_sd = min(cands, key=lambda c: runs[c]['sdzqs'])
        by_sp = min(cands, key=lambda c: metrics[c]['val']['spread_mean'])
        for how, c in (('SD(Z_q) (rule)', by_sd), ('spread', by_sp)):
            v, t = metrics[c]['val'], metrics[c]['test']
            lines.append('| ' + ' | '.join([fam, how, c, _fmt(runs[c]['sdzqs']), _fmt(v['spread_mean']),
                                             _fmt(t['spread_mean']), _fmt(t.get('spread_exact_mean')),
                                             _fmt(t['zqs_z']), _fmt(t['sdzqs'])]) + ' |')
    return '\n'.join(lines)


def recommendation(metrics, runs):
    """M1 (the better of the two M1 families on the validation selection metric) vs M3, on the test paths."""
    best = metrics['_best']
    m1 = [f for f in ('M1-logit', 'M1-gam') if f in best]
    if not m1 or 'M3' not in best:
        return None, 'not enough models'
    m1 = min(m1, key=lambda f: runs[best[f]]['sdzqs'])
    a, b = metrics[best[m1]]['test'], metrics[best['M3']]['test']
    ok_spread = a['spread_mean'] <= (1 + TOL_SPREAD) * b['spread_mean']
    ok_zq = a['zqs_z'] <= b['zqs_z'] + TOL_ZQS
    choice = m1 if (ok_spread and ok_zq) else 'M3'
    why = (f'{STYLE[m1]["label"]} vs M3 on the test paths: spread {a["spread_mean"]:.4g} vs {b["spread_mean"]:.4g} '
           f'({"within" if ok_spread else "outside"} +{100 * TOL_SPREAD:.0f}%), zqs_z {a["zqs_z"]:.2f} vs {b["zqs_z"]:.2f} '
           f'({"within" if ok_zq else "outside"} +{TOL_ZQS})')
    return choice, why


def write(systems):
    parts = []
    summary_rows = []
    for system in systems:
        try:
            cfg, metrics, runs, export, screen = _load(system)
        except FileNotFoundError as e:
            parts.append(f'## {system}\n\nNot run yet ({os.path.basename(str(e.filename))} missing).\n')
            continue
        out = os.path.join(cfg['out_dir'], 'report')
        os.makedirs(out, exist_ok=True)
        rel = lambda p: os.path.relpath(p, os.path.dirname(DOC))
        sel = fig_selection(cfg, metrics, runs, out)
        figs = [fig_committor_maps(cfg, export, out), fig_map(cfg, metrics, export, out),
                fig_spread(cfg, metrics, out), fig_zq(cfg, metrics, out), sel[0] if sel else None]
        rows = _rows(metrics)
        labels = {ent: _label(ent, cid, runs) for ent, cid in rows}
        choice, why = recommendation(metrics, runs)
        summary_rows.append((system, choice, why))
        s = [f'## {system}\n']
        s.append(f'Data: `{os.path.basename(cfg["data"]["tis_dir"])}`, loader `{" ".join(cfg["data"]["loader_args"])}`'
                 + (f' (tis_data {cfg["data"]["tis_data_rev"]})' if cfg['data'].get('tis_data_rev') else '') + '.\n')
        s.append('M2 screening (permutation importance of the gradient-boosted trees on validation paths): '
                 + ', '.join(f'`{sp}` {v:.3g}' for sp, v in list(zip(screen['specs'], screen['importance']))[:6]) + '.\n')
        s.append('### Test paths (held-out MC cycle blocks)\n')
        s.append(_table(metrics, rows, 'test', export, runs, labels) + '\n')
        s.append('### OOD block (the last 15% of the cycles)\n')
        s.append(_table(metrics, rows, 'ood', export, runs, labels) + '\n')
        rt = lambda d: ', '.join(f'{k.replace("_", " ")} {v:.1f}' for k, v in d.items())
        s.append('Mean spreads use the planned levels q_k = ' + '; '.join(
            f'{sp}: ' + ', '.join(f'{q:.2g}' for q in metrics['_spread_levels_' + sp]) for sp in ('test', 'ood'))
            + ' (enough frames, and q_np resolves the committor there: on the exact isocommittor slab logit q_np '
              'varies by less than 10x the slab width; per-level values in the figure).\n')
        s.append(f'Reference rates per 1000 frames (sampler turn rule, planned interfaces): test: '
                 f'{rt(metrics["_ref_turns_test"])}; OOD: {rt(metrics["_ref_turns_ood"])}.\n')
        if export:
            e = next(iter(export.values()))
            s.append(f'A/B ramps (width {e["ramp_width"]:.4g} = {cfg["export"]["ramp"]} × (λ_B − λ_A)): '
                     f'{100 * e["weight_fraction_in_ramps"]:.2f}% of the test frame weight lies in them; there h differs '
                     f'from the calibrated committor m by RMS ' + ', '.join(f'{k} {v["rms_h_minus_m_in_ramps"]:.3g}'
                                                                           for k, v in export.items()) + '.\n')
        if sel:
            s.append('Validation selection metric SD(Z_q) of the references on the same paths: '
                     + ', '.join(f'{k} {v["sdzqs"]:.3f} (spread {v["spread_mean"]:.3f})' for k, v in sel[1].items())
                     + '; fitted runs: ' + f'{min(r["sdzqs"] for r in runs.values()):.3f} … '
                     f'{max(r["sdzqs"] for r in runs.values()):.3f}.\n')
        sens = sensitivity_table(metrics, runs)
        if sens:
            s.append('Selection sensitivity: the rule selects by validation SD(Z_q); a selection by validation '
                     'isosurface spread (q_np only) would take:\n')
            s.append(sens + '\n')
        s.append(f'Recommendation: **{STYLE[choice]["label"] if choice in STYLE else choice}**. {why}.\n')
        for p in figs:
            if p:
                s.append(f'![{os.path.basename(p)}]({rel(p)})\n')
        tis_dir = os.path.join(cfg['out_dir'], 'eval', 'tis_plots')
        s.append(f'TIS diagnostics (plots_tis) and path-weighted feps of every row: [`{rel(tis_dir)}`]({rel(tis_dir)}).\n')
        s.append('<details><summary>All runs (test paths)</summary>\n')
        all_rows = [(r['family'], cid) for cid, r in runs.items() if cid in metrics]
        s.append(_table(metrics, [(cid, cid) for _, cid in all_rows], 'test', {}, runs,
                        {cid: f'{cid} (val SD(Z_q) {runs[cid]["sdzqs"]:.3g})' for _, cid in all_rows}) + '\n')
        s.append('</details>\n')
        parts.append('\n'.join(s))
    intro_path = os.path.join(common.HERE, 'report_intro.md')
    intro = open(intro_path).read() if os.path.isfile(intro_path) else '# Parametric order parameter λ(x) for StapleTIS\n'
    head = [intro, '## Recommendation rule per system\n',
            'Generated by `examples/param_lambda/run.py report`; every number comes from the configs in '
            '`examples/param_lambda/configs/` (fixed seeds).\n',
            '| system | recommended | reason |', '|---|---|---|']
    head += [f'| {s} | {STYLE[c]["label"] if c in STYLE else c} | {w} |' for s, c, w in summary_rows]
    with open(DOC, 'w') as f:
        f.write('\n'.join(head) + '\n\n' + '\n\n'.join(parts) + '\n')
    common.say('wrote ' + DOC)
