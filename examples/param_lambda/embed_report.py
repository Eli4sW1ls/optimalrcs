"""Embedding test: figures in <out>/<system>/report_embed/ and docs/param_lambda_embedding.md.

    OMP_NUM_THREADS=1 python embed_report.py [z_emb cc_emb cc_emb_np]

Needs run_embedded.sh (embed, screen, fit --lean, evaluate, export) for each system. Candidates: the best M1
(L1-logistic, GAM) over all feature sets, and per feature set ('cvs' = all 31 CVs, 'noclean' = without y and x)
the M3 run picked by the validation SD(Z_q) and M3-fast (fewest parameters within 10% of it); for comparison the
base system's M3 on (y, x). Test paths throughout.
"""
import json
import os
import sys

import numpy as np

import common
import embed
import steps
from lambda_runtime import LambdaModel, ScoreModel
from report import _binned, _plt, _rms_transition

DOC = os.path.join(common.REPO, 'docs', 'param_lambda_embedding.md')
INTRO = os.path.join(common.HERE, 'embed_intro.md')
GROUP_ORDER = ['relevant (y, x)', 'copy, noise 0.01', 'copy, noise 0.03', 'copy, noise 0.1', 'fast irrelevant', 'slow w']
# one hue per group kind (validated report slots), copies light -> dark with the noise, irrelevant in neutral grey
GROUP_COLOR = {'relevant (y, x)': '#2a78d6', 'copy, noise 0.01': '#f6b48a', 'copy, noise 0.03': '#eb6834',
               'copy, noise 0.1': '#a3401a', 'fast irrelevant': '#9a9893', 'slow w': '#eda100'}
# candidates: fixed colour per entity
CAND = {
    'exact': dict(color='#0b0b0b', ls='--', label='exact committor'),
    'base M3': dict(color='#1baf7a', ls='-', label='M3 on (y, x) only, same sweep'),
    'M3 cvs': dict(color='#2a78d6', ls='-', label='M3, all 31 CVs'),
    'M3-fast cvs': dict(color='#2a78d6', ls=':', label='M3-fast, all 31 CVs'),
    'M3 noclean': dict(color='#eb6834', ls='-', label='M3, without clean y, x'),
    'M3-fast noclean': dict(color='#eb6834', ls=':', label='M3-fast, without clean y, x'),
    'M1-logit': dict(color='#eda100', ls='-', label='M1 L1-logistic (best set)'),
    'M1-gam': dict(color='#eda100', ls='-.', label='M1 GAM (best set)'),
}
FAST_TOL = steps.FAST_TOL


def select(cfg):
    """{entity: run} of the embedded system (see the module docstring)."""
    with open(os.path.join(cfg['out_dir'], 'fit', 'results.json')) as fh:
        results = json.load(fh)
    out = {}
    for fam in ('M1-logit', 'M1-gam'):
        rs = [r for r in results if r['family'] == fam]
        if rs:
            out[fam] = min(rs, key=lambda r: r['sdzqs'])
    for s in ('cvs', 'noclean'):
        rs = [r for r in results if r['family'] == 'M3' and r['set'] == s]
        if not rs:
            continue
        best = min(rs, key=lambda r: r['sdzqs'])
        ok = [r for r in rs if r['sdzqs'] <= (1 + FAST_TOL) * best['sdzqs']]
        out[f'M3 {s}'] = best
        out[f'M3-fast {s}'] = min(ok, key=lambda r: (steps._n_params(r['arch'], len(r['features'])), r['sdzqs']))
    return out


def attribution(f, idx, params, n_pairs=50_000, seed=0):
    """Two attributions over the steps of the transition region (1e-3 < q_exact < 0.99) of the frames idx, with
    the gradient g_i = dz/dx_i from central differences at the midpoint of each lag-1 step:

    - dynamic: share of E[(g_i dx_i)^2], the per-step motion of the score (jitter);
    - static: share of E[(g_i sd_i)^2], how much z changes when x_i varies over its spread (a slow coordinate that
      is constant along a path has no dynamic share, but a static one if the model uses it as an offset).

    Also the RMS effect E[(g_i sd_i)^2]^(1/2) in logit units."""
    piece, qx = f.piece[idx], f.q_exact[idx]
    ok = (piece[1:] == piece[:-1]) & f.valid[idx][1:] & f.valid[idx][:-1] & (qx[:-1] > 1e-3) & (qx[:-1] < 0.99)
    i0 = np.flatnonzero(ok)
    i0 = np.sort(np.random.default_rng(seed).choice(i0, min(n_pairs, len(i0)), replace=False))
    x0, x1 = f.cv[idx[i0]], f.cv[idx[i0 + 1]]
    xm, dx = (x0 + x1) / 2, x1 - x0
    w = f.w[idx[i0]].astype(float)
    sd = np.sqrt(np.average((xm - np.average(xm, axis=0, weights=w)) ** 2, axis=0, weights=w))
    sm = ScoreModel(params)
    h = 1e-3 * f.cv[idx[:200_000]].std(0)
    dyn, sta = np.zeros(f.dim), np.zeros(f.dim)
    for i in range(f.dim):
        e = np.zeros(f.dim)
        e[i] = h[i]
        g = (sm.raw(xm + e) - sm.raw(xm - e)) / (2 * h[i])
        dyn[i] = np.average((g * dx[:, i]) ** 2, weights=w)
        sta[i] = np.average((g * sd[i]) ** 2, weights=w)
    return dyn / dyn.sum(), sta / sta.sum(), np.sqrt(sta)


def _metrics_row(m):
    t = m['test']
    return dict(spread=t.get('spread_exact_mean'), sdzqs=t['sdzqs'], zqs_z=t['zqs_z'], step=t['step_ratio'],
                turns=t.get('turn_ratio_exact'), rmse=t.get('rmse_q_exact_mid'))


def analyse(system):
    cfg = common.load_config(system)
    base = cfg['embed']['base']
    ctx = steps.Context(cfg)
    f = ctx.f
    with open(os.path.join(cfg['out_dir'], 'eval', 'metrics.json')) as fh:
        metrics = json.load(fh)
    with open(os.path.join(common.OUT_ROOT, base, 'eval', 'metrics.json')) as fh:
        base_metrics = json.load(fh)
    with open(os.path.join(cfg['out_dir'], 'screen', 'importance.json')) as fh:
        screen = json.load(fh)
    sel = select(cfg)
    test = np.flatnonzero(f.take_pieces('test'))
    names = f.cv_names.tolist()
    grp = embed.groups(names)
    out = dict(system=system, base=base, labels=cfg['embed'].get('labels', 'base'), names=names, groups=grp,
               screen=screen, rows={}, attribution={}, trace={})
    # the 2D comparison: the base system's runs of the same lean sweep on (y, x), the same selection rule
    base_cfg = common.load_config(base)
    lean = base_cfg['m3'].get('lean', {})
    with open(os.path.join(base_cfg['out_dir'], 'fit', 'results.json')) as fh:
        base_runs = [r for r in json.load(fh) if r['family'] == 'M3' and r['set'] in lean.get('sets', ['cvs'])
                     and r['arch'] in lean.get('archs', []) and r['beta'] in lean.get('betas', [])]
    b = min(base_runs, key=lambda r: r['sdzqs'])
    bp = dict(np.load(os.path.join(base_cfg['out_dir'], 'fit', 'runs', b['id'] + '.npz')))
    out['rows']['base M3'] = dict(_metrics_row(base_metrics[b['id']]), run=b['id'], features='y, x (2)',
                                  model='x'.join(map(str, b['arch'])) + f' β={b["beta"]:g}', train_s=b['train_seconds'])
    out['rows']['exact'] = _metrics_row(metrics['exact'])
    out['rows']['q_np'] = _metrics_row(metrics['q_np'])
    base_ctx = steps.Context(base_cfg)
    base_lm = LambdaModel(steps.build_export(base_ctx, bp))
    del base_ctx
    bench_cv = f.cv[test[:20000]]
    models = {}
    for ent, r in sel.items():
        p = dict(np.load(os.path.join(cfg['out_dir'], 'fit', 'runs', r['id'] + '.npz')))
        lm = LambdaModel(steps.build_export(ctx, p))
        models[ent] = lm
        bench = steps.benchmark(lm, bench_cv, reps=5000)
        arch = 'x'.join(map(str, r['arch'])) + f' β={r["beta"]:g}' if 'arch' in r else f'hp {r["hp"]:g}'
        out['rows'][ent] = dict(_metrics_row(metrics[r['id']]), run=r['id'], features=f'{r["set"]} ({len(r["features"])})',
                                model=arch, us_one=bench['us_per_frame_one'], train_s=r['train_seconds'],
                                n_nonzero=r.get('n_nonzero'))
        share, static, effect = attribution(f, test, p)
        by_group = lambda v: {g: float(v[[k for k, gg in enumerate(grp) if gg == g]].sum()) for g in GROUP_ORDER}
        out['attribution'][ent] = by_group(share)
        out['attribution'][ent]['top'] = [(names[k], float(share[k])) for k in np.argsort(share)[::-1][:5]]
        out.setdefault('static', {})[ent] = by_group(static)
        out.setdefault('w_effect', {})[ent] = float(effect[names.index('w')])
        common.say(f'{system} {ent:16s} {r["id"]:32s} spread {out["rows"][ent]["spread"]:.4f}  '
                   f'step {out["rows"][ent]["step"]:.3f}  us {bench["us_per_frame_one"]:.1f}  '
                   f'shares {", ".join(f"{g}: {v:.3f}" for g, v in out["attribution"][ent].items() if g != "top")}')
    # one test piece for the traces: 200-3000 frames, the widest range of logit q_exact (pieces start at the
    # interface below their ensemble, so few run from near A to B); the median of the 10 widest
    piece, qx = f.piece[test], np.clip(f.q_exact[test].astype(float), 1e-6, 1 - 1e-6)
    u, inv = np.unique(piece, return_inverse=True)
    n = np.bincount(inv)
    lo, hi = np.ones(len(u)), np.zeros(len(u))
    np.minimum.at(lo, inv, qx)
    np.maximum.at(hi, inv, qx)
    width = np.where((n > 200) & (n < 3000), common.logit(hi) - common.logit(lo), -np.inf)
    top = np.argsort(width)[::-1][:10]
    if np.isfinite(width[top[0]]):
        p_ = u[top[len(top) // 2]]
        ii = test[piece == p_]
        cv = f.cv[ii]
        out['trace'] = {'t': f.t[ii].tolist(), 'exact': f.q_exact[ii].astype(float).tolist(),
                        'A': f.A[ii].tolist(), 'B': f.B[ii].tolist()}
        out['trace']['base M3'] = base_lm.q(cv[:, :2]).tolist()
        for ent, lm in models.items():
            out['trace'][ent] = lm.q(cv).tolist()
    # projections on (y, x): binned mean of the calibrated committor, test frames
    yx = f.cv[test][:, :2]
    v = f.valid[test]
    edges = [np.linspace(*np.quantile(yx[v, k], [0.001, 0.999]), 61) for k in range(2)]
    wv = f.w[test].astype(float)[v]
    out['maps'] = {'edges': [e.tolist() for e in edges],
                   'exact': _binned((yx[v, 0], yx[v, 1]), edges, f.q_exact[test][v].astype(float), wv).tolist()}
    out['maps']['base M3'] = _binned((yx[v, 0], yx[v, 1]), edges, base_lm.q(yx[v]), wv).tolist()
    for ent in ('M3 cvs', 'M3 noclean'):
        if ent in models:
            out['maps'][ent] = _binned((yx[v, 0], yx[v, 1]), edges, models[ent].q(f.cv[test][v]), wv).tolist()
    d = os.path.join(cfg['out_dir'], 'report_embed')
    os.makedirs(d, exist_ok=True)
    steps._save_json(out, os.path.join(d, 'analysis.json'))
    return out


# ---------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------


def fig_screen(res, path):
    plt = _plt()
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 6.5), squeeze=False)
    for ax, r in zip(axes[0], res):
        sc = r['screen']
        imp = np.maximum(np.asarray(sc['importance'], float), 1e-6)
        names = sc['specs']
        g = embed.groups(names)
        yy = np.arange(len(names))[::-1]
        ax.barh(yy, imp, color=[GROUP_COLOR[x] for x in g], height=0.7)
        ax.set_yticks(yy, names, fontsize=7)
        ax.set_xscale('log')
        ax.set_xlim(1e-6, imp.max() * 3)
        ax.set_xlabel('permutation importance (validation MSE of logit q)')
        ax.set_title(f'{r["system"]}: M2 screening')
        ax.grid(axis='y', visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUP_ORDER]
    fig.legend(handles, GROUP_ORDER, loc='lower center', ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_attribution(res, path):
    plt = _plt()
    ents = [e for e in CAND if e not in ('exact', 'base M3')]
    fig, axes = plt.subplots(len(res), 2, figsize=(12, 1.0 + 0.42 * len(ents) * len(res)), squeeze=False)
    for row, r in zip(axes, res):
        es = [e for e in ents if e in r['attribution']]
        yy = np.arange(len(es))[::-1]
        for ax, key, what in ((row[0], 'attribution', 'per-step motion (jitter)'),
                              (row[1], 'static', 'dependence (static)')):
            left = np.zeros(len(es))
            for g in GROUP_ORDER:
                val = np.array([r[key][e][g] for e in es])
                ax.barh(yy, val, left=left, color=GROUP_COLOR[g], height=0.7, edgecolor='white', linewidth=1)
                left += val
            ax.set_yticks(yy, [CAND[e]['label'] for e in es] if ax is row[0] else [], fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_title(f'{r["system"]} ({"2D q_np" if r["labels"] == "base" else "NPNEq on all CVs"}): {what}',
                         fontsize=9)
            ax.grid(axis='y', visible=False)
    axes[-1, 0].set_xlabel('share of E[(∂z/∂x_i · Δx_i)²] per lag-1 step')
    axes[-1, 1].set_xlabel('share of E[(∂z/∂x_i · SD(x_i))²]')
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUP_ORDER]
    fig.legend(handles, GROUP_ORDER, loc='lower center', ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_traces(res, path):
    """Per system: the calibrated committor along one test piece (logit scale), and below it the deviation from
    the exact committor, logit lambda - logit q_exact (the jitter of the noisy inputs shows there)."""
    plt = _plt()
    rs = [r for r in res if r['trace']]
    fig, axes = plt.subplots(2 * len(rs), 1, figsize=(9, 5.2 * len(rs)), squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.3, 1] * len(rs)))
    for k, r in enumerate(rs):
        ax, ad = axes[2 * k, 0], axes[2 * k + 1, 0]
        tr = r['trace']
        t = np.arange(len(tr['exact']))
        qx = np.clip(np.asarray(tr['exact'], float), 1e-5, 1 - 1e-5)
        inside = ~np.asarray(tr['A']) & ~np.asarray(tr['B'])
        for ent in ('exact', 'base M3', 'M3 cvs', 'M3 noclean', 'M1-gam'):
            if ent not in tr:
                continue
            q = np.clip(np.asarray(tr[ent], float), 1e-5, 1 - 1e-5)
            st = dict(color=CAND[ent]['color'], ls=CAND[ent]['ls'], label=CAND[ent]['label'])
            ax.plot(t, q, lw=1.6 if ent == 'exact' else 1.1, zorder=3 if ent == 'exact' else 2, **st)
            if ent != 'exact':
                dev = np.where(inside, common.logit(q) - common.logit(qx), np.nan)
                ad.plot(t, dev, lw=1.0, **st)
        ax.set_yscale('logit')
        ax.set_ylim(1e-5, 1 - 1e-5)
        ax.set_ylabel('committor (calibrated λ)')
        ax.set_title(f'{r["system"]}: one test piece', fontsize=9)
        ad.axhline(0, color='#0b0b0b', lw=0.8, ls='--')
        ad.set_ylabel('logit λ − logit q_exact')
        ad.set_xlabel('frame along the piece')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_maps(res, path):
    plt = _plt()
    ents = ['base M3', 'M3 cvs', 'M3 noclean']
    fig, axes = plt.subplots(len(res), 1 + len(ents), figsize=(3.4 * (1 + len(ents)), 3.1 * len(res)), squeeze=False)
    for row, r in zip(axes, res):
        mp = r['maps']
        ey, ex = (np.asarray(e) for e in mp['edges'])
        ref = np.asarray(mp['exact'], float)
        im = row[0].pcolormesh(ey, ex, ref.T, cmap='Greys', vmin=0, vmax=1, shading='flat')
        row[0].set_title(f'{r["system"]}: exact committor', fontsize=9)
        fig.colorbar(im, ax=row[0], fraction=0.046)
        for ax, ent in zip(row[1:], ents):
            if ent not in mp:
                ax.axis('off')
                continue
            diff = np.asarray(mp[ent], float) - ref
            im = ax.pcolormesh(ey, ex, diff.T, cmap='PuOr', vmin=-0.15, vmax=0.15, shading='flat')
            ax.set_title(f'{CAND[ent]["label"]} − exact\nRMS (0.05<q<0.95) {_rms_transition(diff, ref):.3f}',
                         fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
        for ax in row:
            ax.set_xlabel('y')
            ax.grid(False)
        row[0].set_ylabel('x')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_summary(res, path):
    """Test spread vs the exact committor and step ratio per candidate and system (two panels, one axis each)."""
    plt = _plt()
    ents = ['exact', 'base M3', 'M3 cvs', 'M3-fast cvs', 'M3 noclean', 'M3-fast noclean', 'M1-logit', 'M1-gam']
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.6 + 0.5 * len(ents)))
    yy = np.arange(len(ents))[::-1]
    for k, r in enumerate(res):
        off = (k - (len(res) - 1) / 2) * 0.22
        mk = ['o', 's', '^'][k % 3]
        for ax, key in zip(axes, ('spread', 'step')):
            for y, ent in zip(yy, ents):
                row = r['rows'].get(ent)
                if row and row.get(key) is not None:
                    ax.plot(row[key], y + off, marker=mk, ls='none', color=CAND[ent]['color'], ms=6,
                            mfc='white' if ':' in CAND[ent]['ls'] else CAND[ent]['color'])
    axes[0].set_xscale('log')
    axes[0].set_xlabel('isosurface spread vs exact committor (test; lower is better)')
    axes[1].set_xlabel('step ratio: E[Δ logit λ²] / E[Δ logit q_np²] at lag 1 (jitter)')
    for ax in axes:
        ax.set_yticks(yy, [CAND[e]['label'] for e in ents], fontsize=8)
        ax.grid(axis='y', visible=False)
    axes[1].set_yticklabels([])
    handles = [plt.Line2D([], [], marker=['o', 's', '^'][k % 3], ls='none', color='#52514e') for k in range(len(res))]
    fig.legend(handles, [r['system'] for r in res], loc='lower center', ncol=len(res), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------------------------------


def _f(v, nd=3):
    return '–' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{nd}f}'


def table(r):
    lines = ['| candidate | features | model | spread vs exact | SD(Z_q) | zqs_z | step ratio | turns / exact | '
             'jitter share: relevant / copies / fast | w: static share, RMS effect on logit λ | µs/frame | train s |',
             '|' + '---|' * 12]
    for ent in ('exact', 'q_np', 'base M3', 'M1-logit', 'M1-gam', 'M3 cvs', 'M3-fast cvs', 'M3 noclean',
                'M3-fast noclean'):
        row = r['rows'].get(ent)
        if row is None:
            continue
        a = r['attribution'].get(ent)
        share = '–' if a is None else (f'{a["relevant (y, x)"]:.2f} / '
                                       f'{sum(a[g] for g in GROUP_ORDER if g.startswith("copy")):.2f} / '
                                       f'{a["fast irrelevant"]:.3f}')
        wcol = '–' if a is None else f'{r["static"][ent]["slow w"]:.4f}, {r["w_effect"][ent]:.3f}'
        label = CAND[ent]['label'] if ent in CAND else 'q_np (labels)'
        lines.append(f'| {label} | {row.get("features", "")} | {row.get("model", "")} | {_f(row["spread"], 4)} | '
                     f'{_f(row["sdzqs"])} | {_f(row["zqs_z"], 2)} | {_f(row["step"])} | {_f(row["turns"])} | {share} | '
                     f'{wcol} | '
                     f'{_f(row.get("us_one"), 1)} | {_f(row.get("train_s"), 0)} |')
    return '\n'.join(lines)


def write(res):
    rel = lambda p: os.path.relpath(p, os.path.dirname(DOC))
    fig_dir = os.path.join(common.OUT_ROOT, 'embedding')
    os.makedirs(fig_dir, exist_ok=True)
    figs = {}
    for name, fn in (('summary', fig_summary), ('attribution', fig_attribution), ('screening', fig_screen),
                     ('traces', fig_traces), ('maps', fig_maps)):
        figs[name] = os.path.join(fig_dir, f'{name}.png')
        fn(res, figs[name])
    lines = [open(INTRO).read().rstrip() if os.path.isfile(INTRO) else '# Embedding test', '']
    lines += ['## Results (test paths)', '',
              'Spread: mean isosurface spread against the exact committor over the report\'s level set. '
              '"jitter share": the share of each CV group in the per-step motion of the score along the test paths '
              '(E[(∂z/∂x_i · Δx_i)²], 1e-3 < q_exact < 0.99). "w": its static share E[(∂z/∂w · SD(w))²] and the RMS '
              'change of logit λ over the spread of w (w hardly moves along a path, so it has no jitter share). '
              'µs/frame: one-frame calls, single-threaded BLAS.', '',
              f'![summary]({rel(figs["summary"])})', '']
    for r in res:
        lines += [f'### {r["system"]} (base {r["base"]}; labels: '
                  f'{"the 2D q_np of " + r["base"] if r["labels"] == "base" else "NPNEq on all 31 CVs"})', '',
                  table(r), '',
                  'Largest single-CV shares: ' + '; '.join(
                      f'{CAND[e]["label"]}: ' + ', '.join(f'{n} {v:.2f}' for n, v in a['top'][:3])
                      for e, a in r['attribution'].items()), '']
    for name in ('attribution', 'screening', 'traces', 'maps'):
        lines += [f'![{name}]({rel(figs[name])})', '']
    with open(DOC, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    common.say(f'wrote {DOC}')


def main():
    systems = sys.argv[1:] or ['z_emb', 'cc_emb', 'cc_emb_np']
    res = []
    for s in systems:
        cache = os.path.join(common.OUT_ROOT, s, 'report_embed', 'analysis.json')
        if os.environ.get('EMBED_REPORT_CACHED') and os.path.isfile(cache):
            res.append(json.load(open(cache)))
        else:
            res.append(analyse(s))
    write(res)


if __name__ == '__main__':
    main()
