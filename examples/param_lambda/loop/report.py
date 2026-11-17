#!/usr/bin/env python3
"""docs/op_loop_report.md from the loop directories (state.json, pass manifests) and the experiments.

    python -m loop.report [loop names ...]          (default: every loop directory with a state.json)

The findings at the top are written by hand in loop/report_intro.md; everything below them is generated.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import common  # noqa: E402

ROOT = os.path.join(common.OUT_ROOT, 'op_loop')
DOC = os.path.join(common.REPO, 'docs', 'op_loop_report.md')
INTRO = os.path.join(HERE, 'loop', 'report_intro.md')


def _f(x, nd=3):
    if x is None:
        return '–'
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return '–' if not np.isfinite(x) else f'{x:.{nd}f}'


def _g(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '–'
    return '–' if not np.isfinite(x) else f'{x:.{nd}g}'


def loop_section(name):
    d = os.path.join(ROOT, name)
    with open(os.path.join(d, 'state.json')) as f:
        st = json.load(f)
    out = [f'### {name}', '']
    p0 = st.get('pass0', {})
    out.append(f'Pass 0 (infinit, seed order parameter): reached B = {p0.get("reached_b")}; '
               f'{len(p0.get("pool_passes", []))} iterations stored. Total MD steps of the initialization: '
               f'{st.get("md_steps", 0):,}.')
    out.append('')
    rows = ['| pass | interfaces | teacher RMS vs exact | SD(Z_q) new | SD(Z_q) previous model | stability max | '
            'spread vs exact | logit RMSE vs exact | longest / shortest median length [1*]+ | re-check | MD steps | fit s | pass s |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for rec in st.get('passes', []):
        d_ = rec['diagnostics']
        stab = d_.get('stability')
        smax = max([v for v in (stab['new_on_old'] + stab['old_on_new']) if v is not None and np.isfinite(v)], default=np.nan) if stab else np.nan
        ex = d_.get('exact') or {}
        med = [v['median'] for k, v in rec.get('lengths', {}).items() if int(k) >= 2]
        ratio = f'{max(med):.0f} / {min(med):.0f}' if med else '–'
        rows.append(f'| {rec["k"]} | {len(rec["interfaces"])} | {_f(rec["teacher"].get("rms_vs_exact"))} | '
                    f'{_f(d_["new"]["zq"]["sdzqs"], 2)} | {_f(d_.get("prospective", {}).get("zq", {}).get("sdzqs"), 2)} | '
                    f'{_f(smax)} | {_f(ex.get("spread_mean"), 4)} | {_f(ex.get("logit_rmse"))} | {ratio} | '
                    f'{"OK" if rec["recheck"]["ok"] else "fails"} | {rec["cost"]["md_steps"]:,} | '
                    f'{_f(rec["cost"].get("fit_seconds"), 0)} | {_f(rec["cost"].get("pass_seconds"), 0)} |')
    out += rows + ['']
    if st.get('passes'):
        last = st['passes'][-1]
        out.append(f'Last interfaces: `{np.round(last["interfaces"], 5).tolist()}`')
        fails = last['recheck'].get('fails', [])
        out.append(f'Re-check of the last pass on its own paths: {"all checks pass" if not fails else fails}.')
        esr = last['recheck'].get('empty_shooting_regions')
        if esr:
            out.append('Accepted paths with an empty shooting region per ensemble: '
                       + ', '.join(f'{k}: {v["empty"]}/{v["n"]}' for k, v in esr.items()) + '.')
        out.append('')
    out += production_rows(d, st)
    return '\n'.join(out)


def production_rows(d, st):
    """The frozen set (freeze.json) and the production runs: learned lambda and the seed-lambda control."""
    out = []
    fz = os.path.join(d, 'freeze.json')
    if os.path.isfile(fz):
        with open(fz) as f:
            fr = json.load(f)
        out.append(f'Frozen set ({fr.get("source")}): `{np.round(fr["interfaces"], 4).tolist()}`.')
        for r in fr.get('verification', []):
            out.append(f'Verification round {r["round"]}: removed {[(round(v, 4), why) for v, why in r["removed"]] or "none"}'
                       + (f', refilled {[round(a["lam"], 4) for a in r["refilled"]]}' if r.get('refilled') else '')
                       + f'; {"passes" if r["ok"] else "fails " + str(r["fails"])}; {r["md_steps"]:,} MD steps.')
        out.append('')
    rows = []
    for sub, label in (('production', 'learned lambda (frozen)'), ('production_refill3', 'learned lambda, refilled interfaces'),
                       ('production_seed', 'seed lambda (control)')):
        p = os.path.join(d, sub, 'rates.json')
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            r = json.load(f)
        ex = (r.get('exact') or {}).get('rate')
        init = f'{st.get("md_steps", 0):,}' if sub.startswith('production') and sub != 'production_seed' else '0'
        label += ' (stopped early)' if r.get('stopped_early') else ''
        rows.append(f'| {label} | {len(r.get("interfaces", [])) or "–"} | {r["cost"]["cycles"]:,} | {r["cost"]["md_steps"]:,} | '
                    f'{init} | {_g(r["pcross_global"])} ({_f(r["rel_err_pcross"], 2)}) | '
                    f'{_g(r["rate_per_time"])} ({_f(r["rel_err_rate"], 2)}) | '
                    f'{_f(r["rate_per_time"] / ex, 2) + "x" if ex else "–"} |')
    if rows:
        out += ['| production run | interfaces | cycles | MD steps | MD steps of the initialization | P_cross (rel. error) | '
                'rate per time unit (rel. error) | rate / exact |', '|---|---|---|---|---|---|---|---|'] + rows
        ex = None
        for sub in ('production', 'production_refill3', 'production_seed'):
            p = os.path.join(d, sub, 'rates.json')
            if os.path.isfile(p):
                with open(p) as f:
                    ex = ex or json.load(f).get('exact')
        if ex:
            out.append(f'\nExact rate {_g(ex["rate"])} per time unit ({ex.get("source", "")}).')
        out.append('')
    return out


def staple_weight_section():
    p = os.path.join(ROOT, 'staple_weights', 'analysis.json')
    if not os.path.isfile(p):
        return ''
    with open(p) as f:
        res = json.load(f)
    out = ['### StapleTIS weights in infretis (notes section 15), flat 1D, exact committor', '',
           '| run | weighting | pieces | exact q: stopped Z_q max abs z at lags 1, 4, 16, 64, 256 | '
           'mean exact q at the first frames past M, per ensemble |', '|---|---|---|---|---|']
    for name, r in res.items():
        for wt in ('rejections', 'staple'):
            if wt not in r:
                continue
            x = r[wt]
            out.append(f'| {name} | {"path_f" if wt == "rejections" else "path_f / path_w"} | {x["n_pieces"]} | '
                       f'{", ".join(_f(v, 2) for v in x["zmax"])} | '
                       + ', '.join(f'{e}: {_f(v["mean_q"], 4)}' for e, v in x['level'].items()) + ' |')
    out.append('')
    return '\n'.join(out)


def write(names=None):
    if names is None:
        names = sorted(n for n in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, n, 'state.json')))
    parts = []
    if os.path.isfile(INTRO):
        with open(INTRO) as f:
            parts.append(f.read().rstrip() + '\n')
    parts.append('## Per-system tables (generated by `python -m loop.report`)\n')
    sw = staple_weight_section()
    if sw:
        parts.append(sw)
    for n in names:
        parts.append(loop_section(n))
    with open(DOC, 'w') as f:
        f.write('\n'.join(parts))
    print('wrote', DOC)


if __name__ == '__main__':
    write(sys.argv[1:] or None)
