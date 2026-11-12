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
    'base M3': dict(color='#1baf7a', ls='-', label='M3 on (y, x) only (2D run)'),
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
    """Share of each CV in the per-step motion of the score along the paths: c_i = dz/dx_i * dx_i (lag 1, central
    differences at the midpoint), share_i = E[c_i^2] / sum_j E[c_j^2], over steps in the transition region
    (1e-3 < q_exact < 0.99) of the frames idx."""
    piece, qx = f.piece[idx], f.q_exact[idx]
    ok = (piece[1:] == piece[:-1]) & f.valid[idx][1:] & f.valid[idx][:-1] & (qx[:-1] > 1e-3) & (qx[:-1] < 0.99)
    i0 = np.flatnonzero(ok)
    i0 = np.sort(np.random.default_rng(seed).choice(i0, min(n_pairs, len(i0)), replace=False))
    x0, x1 = f.cv[idx[i0]], f.cv[idx[i0 + 1]]
    xm, dx = (x0 + x1) / 2, x1 - x0
    w = f.w[idx[i0]].astype(float)
    sm = ScoreModel(params)
    h = 1e-3 * f.cv[idx[:200_000]].std(0)
    c2 = np.zeros(f.dim)
    for i in range(f.dim):
        e = np.zeros(f.dim)
        e[i] = h[i]
        g = (sm.raw(xm + e) - sm.raw(xm - e)) / (2 * h[i])
        c2[i] = np.average((g * dx[:, i]) ** 2, weights=w)
    return c2 / c2.sum()


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
    b = base_metrics['_best']['M3']
    out['rows']['base M3'] = dict(_metrics_row(base_metrics[b]), run=b, features='y, x')
    out['rows']['exact'] = _metrics_row(metrics['exact'])
    out['rows']['q_np'] = _metrics_row(metrics['q_np'])
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
        share = attribution(f, test, p)
        out['attribution'][ent] = {g: float(share[[k for k, gg in enumerate(grp) if gg == g]].sum())
                                   for g in GROUP_ORDER}
        out['attribution'][ent]['top'] = [(names[k], float(share[k])) for k in np.argsort(share)[::-1][:5]]
        common.say(f'{system} {ent:16s} {r["id"]:32s} spread {out["rows"][ent]["spread"]:.4f}  '
                   f'step {out["rows"][ent]["step"]:.3f}  us {bench["us_per_frame_one"]:.1f}  '
                   f'shares {", ".join(f"{g}: {v:.3f}" for g, v in out["attribution"][ent].items() if g != "top")}')
    # one reactive test piece for the traces: from q < 0.01 to q > 0.95, at most 3000 frames, mid-length
    piece, qx = f.piece[test], f.q_exact[test]
    cand = []
    for p_ in np.unique(piece):
        m_ = piece == p_
        if 200 < m_.sum() < 3000 and qx[m_].min() < 0.01 and qx[m_].max() > 0.95:
            cand.append((m_.sum(), p_))
    if cand:
        _, p_ = sorted(cand)[len(cand) // 2]
        ii = test[piece == p_]
        cv = f.cv[ii]
        out['trace'] = {'t': f.t[ii].tolist(), 'exact': f.q_exact[ii].astype(float).tolist(),
                        'A': f.A[ii].tolist(), 'B': f.B[ii].tolist()}
        base_lm = LambdaModel(os.path.join(common.HERE, 'exported', f'{base}_m3.npz'))
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
    base_lm = LambdaModel(os.path.join(common.HERE, 'exported', f'{base}_m3.npz'))
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
    fig, axes = plt.subplots(len(res), 1, figsize=(8.5, 1.0 + 0.42 * len(ents) * len(res)), squeeze=False)
    for ax, r in zip(axes[:, 0], res):
        es = [e for e in ents if e in r['attribution']]
        yy = np.arange(len(es))[::-1]
        left = np.zeros(len(es))
        for g in GROUP_ORDER:
            val = np.array([r['attribution'][e][g] for e in es])
            ax.barh(yy, val, left=left, color=GROUP_COLOR[g], height=0.7, edgecolor='white', linewidth=1)
            left += val
        ax.set_yticks(yy, [CAND[e]['label'] for e in es], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_title(f'{r["system"]} (labels: {"2D q_np" if r["labels"] == "base" else "NPNEq on all CVs"}): '
                     'share of each CV group in the per-step motion of the score', fontsize=9)
        ax.grid(axis='y', visible=False)
    axes[-1, 0].set_xlabel('share of E[(∂z/∂x_i · Δx_i)²], steps with 1e-3 < q_exact < 0.99 (test paths)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUP_ORDER]
    fig.legend(handles, GROUP_ORDER, loc='lower center', ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_traces(res, path):
    plt = _plt()
    rs = [r for r in res if r['trace']]
    fig, axes = plt.subplots(len(rs), 1, figsize=(9, 3.2 * len(rs)), squeeze=False)
    for ax, r in zip(axes[:, 0], rs):
        tr = r['trace']
        t = np.arange(len(tr['exact']))
        for ent in ('exact', 'base M3', 'M3 cvs', 'M3 noclean', 'M1-gam'):
            if ent in tr:
                q = np.clip(np.asarray(tr[ent], float), 1e-5, 1 - 1e-5)
                ax.plot(t, q, color=CAND[ent]['color'], ls=CAND[ent]['ls'], lw=1.2 if ent != 'exact' else 1.6,
                        label=CAND[ent]['label'], zorder=3 if ent == 'exact' else 2)
        ax.set_yscale('logit')
        ax.set_ylim(1e-5, 1 - 1e-5)
        ax.set_ylabel('committor (calibrated λ)')
        ax.set_title(f'{r["system"]}: one reactive test piece', fontsize=9)
    axes[-1, 0].set_xlabel('frame along the piece')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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
             'share: relevant / copies / fast / w | µs/frame | train s |', '|' + '---|' * 11]
    for ent in ('exact', 'q_np', 'base M3', 'M1-logit', 'M1-gam', 'M3 cvs', 'M3-fast cvs', 'M3 noclean',
                'M3-fast noclean'):
        row = r['rows'].get(ent)
        if row is None:
            continue
        a = r['attribution'].get(ent)
        share = '–' if a is None else (f'{a["relevant (y, x)"]:.2f} / '
                                       f'{sum(a[g] for g in GROUP_ORDER if g.startswith("copy")):.2f} / '
                                       f'{a["fast irrelevant"]:.3f} / {a["slow w"]:.3f}')
        label = CAND[ent]['label'] if ent in CAND else 'q_np (labels)'
        lines.append(f'| {label} | {row.get("features", "")} | {row.get("model", "")} | {_f(row["spread"], 4)} | '
                     f'{_f(row["sdzqs"])} | {_f(row["zqs_z"], 2)} | {_f(row["step"])} | {_f(row["turns"])} | {share} | '
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
              '"share": the share of each CV group in the per-step motion of the score along the test paths '
              '(E[(∂z/∂x_i · Δx_i)²], 1e-3 < q_exact < 0.99). µs/frame: one-frame calls, single-threaded BLAS.', '',
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
