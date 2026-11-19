#!/usr/bin/env python3
"""Result figures of the loop for the report: the frozen interfaces against the exact isocommittors, and the
production rates of the learned and the seed order parameter against the exact (or MD) rate.

    python -m loop.results          writes <OUT_ROOT>/op_loop/figures/results_interfaces.png, results_rates.png
                                    and results.json (the numbers of the rate figure)

Systems (RESULTS): the loop name, the production directory of the learned lambda (its freeze file holds the frozen
model and interfaces) and of the seed-lambda control. A system whose frozen model is not on this machine (e.g. a
loop run on the workstation) gets an empty interface panel that says so.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from loop import controller, pool as pool_mod, plots  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402

plt = plots.plt
RESULTS = [  # label, loop, learned production (freeze file prefix), seed control
    ('2D Z potential, Brownian', 'loop_z', 'production_refill3', 'production_seed'),
    ('1D well, Langevin: λ(x, v)', 'loop_well_lgv', 'production', 'production_seed'),
    ('two channels: loop (initial data cover one channel)', 'loop_stag', 'production', 'production_seed'),
    ('two channels: converged staple data (both channels)', 'loop_stag_staple', 'production', None),
]
OUT = os.path.join(os.path.dirname(controller.loop_dir({'name': 'x'})), 'figures')


def _freeze(cfg, sub):
    pre = '' if sub == 'production' else f'{sub}_'
    f = os.path.join(cfg['loop_dir'], f'{pre}freeze.json')
    if not os.path.isfile(f):
        return None
    fr = json.load(open(f))
    local = os.path.join(cfg['loop_dir'], 'models', os.path.basename(fr['model']))
    if not os.path.isfile(fr['model']) and os.path.isfile(local):     # run on another machine, models copied here
        fr['model'] = local
    return fr


def _rates(cfg, sub):
    if sub is None:
        return None
    f = os.path.join(cfg['loop_dir'], sub, 'rates.json')
    if not os.path.isfile(f):
        return None
    r = json.load(open(f))
    ex = (r.get('exact') or {}).get('rate')
    return dict(rate=r['rate_per_time'], rel_err=r['rel_err_rate'], exact=ex, ratio=r['rate_per_time'] / ex if ex else None,
                md_steps=(r.get('cost') or {}).get('md_steps'), cycles=(r.get('cost') or {}).get('cycles'))


def _seed_interfaces(cfg, sub):
    if sub is None:
        return []
    f = os.path.join(cfg['loop_dir'], sub, 'infretis.toml')
    if not os.path.isfile(f):
        return []
    import tomllib
    with open(f, 'rb') as fh:
        return list(tomllib.load(fh)['simulation']['interfaces'])


def fig_interfaces(rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.4), squeeze=False)
    for ax, row in zip(axes.flat, rows):
        cfg, fr = row['cfg'], row['freeze']
        title = row['label']
        lr = row['learned']
        if lr and lr.get('ratio'):
            title += f'\nlearned λ: rate {lr["ratio"]:.2f}× exact (rel. error {lr["rel_err"]:.2f})'
        sr = row['seed']
        if sr and sr.get('ratio'):
            title += f';  seed λ: {sr["ratio"]:.2f}× ({sr["rel_err"]:.2f})'
        ax.set_title(title, loc='left', fontsize=9)
        if fr is None or not os.path.isfile(fr['model']):
            ax.text(0.5, 0.5, 'frozen model not on this machine' if fr else 'no frozen production yet',
                    ha='center', va='center', transform=ax.transAxes, color=plots.MUTED)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        grid = plots.Grid(cfg, pool_mod.Pool(cfg['loop_dir']))
        grid.background(ax)
        m = LambdaModel(fr['model'])
        inner = fr['interfaces'][1:-1]
        ax.contour(grid.ys, grid.xs, grid.field(m(grid.cv)).T, levels=inner, colors=plots.BLUE, linewidths=1.3)
        h = m.h_of_lambda(np.array(inner))
        ax.contour(grid.ys, grid.xs, grid.field(grid.q_exact).T, levels=np.sort(np.unique(np.clip(h, 1e-6, 1 - 1e-6))),
                   colors=plots.RED, linewidths=0.9, linestyles='--')
        for v in row['seed_interfaces'][1:-1]:
            ax.axvline(v, color='#52514e', lw=0.7, ls=':')
        plots._mark_unvisited(ax, grid)
        names = cfg['system_spec']['cv_names']
        ax.set_xlabel(f'{names[0]} (state CV)')
        ax.set_ylabel(names[1])
    handles = [plt.Line2D([], [], color=plots.BLUE, lw=1.3), plt.Line2D([], [], color=plots.RED, lw=0.9, ls='--'),
               plt.Line2D([], [], color='#52514e', lw=0.7, ls=':')]
    fig.legend(handles, ['interfaces of the learned λ (frozen production set)', 'exact isocommittors at the same levels',
                         'interfaces of the seed λ (control run)'], loc='lower center', ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_rates(rows, path):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ys = np.arange(len(rows))[::-1]
    for y, row in zip(ys, rows):
        for key, col, dy, lab in (('learned', plots.BLUE, 0.12, 'learned λ'), ('seed', '#eb6834', -0.12, 'seed λ (state CV)')):
            r = row[key]
            if not r or not r.get('ratio'):
                continue
            lo, hi = r['ratio'] / (1 + r['rel_err']), r['ratio'] * (1 + r['rel_err'])
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=2)
            ax.plot(r['ratio'], y + dy, 'o', color=col, ms=7, label=lab if y == ys[0] else None)
    ax.axvline(1, color='#0b0b0b', lw=1, ls='--')
    ax.set_xscale('log')
    ax.set_yticks(ys, [r['label'] for r in rows], fontsize=8.5)
    ax.set_xlabel('production rate / exact rate (bars: × or ÷ (1 + relative error))')
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, fontsize=8, loc='lower right')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for label, name, learned, seed in RESULTS:
        cfg = controller.load_loop_config(name)
        rows.append(dict(label=label, name=name, cfg=cfg, freeze=_freeze(cfg, learned), learned=_rates(cfg, learned),
                         seed=_rates(cfg, seed), seed_interfaces=_seed_interfaces(cfg, seed)))
    print(fig_interfaces(rows, os.path.join(OUT, 'results_interfaces.png')))
    print(fig_rates(rows, os.path.join(OUT, 'results_rates.png')))
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump([{k: v for k, v in r.items() if k not in ('cfg',)} for r in rows], f, indent=1, default=str)


if __name__ == '__main__':
    main()
