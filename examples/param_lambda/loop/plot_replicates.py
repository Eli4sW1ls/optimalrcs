#!/usr/bin/env python3
"""Seed replicates of the loops side by side: production rate / exact (+-1 relative error), the MD cost
(initialization = pass 0 + passes, and production) and the logit RMSE against the exact committor per pass.

    python -m loop.plot_replicates [--pairs loop_zu_b:loop_zu_b_s2 ...] [--out DIR]

Default out: op_loop/figures/replicates.png. Loops without a finished production run are drawn without a rate.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import plots  # noqa: E402  (style)
from loop.plots import plt, BLUE, INK2, MUTED  # noqa: E402

OP_LOOP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       'figures_by_potential', '_shared', 'param_lambda', 'op_loop')
PAIRS = ['loop_zu_b:loop_zu_b_s2', 'loop_zw:loop_zw_s2', 'loop_zu_xy_b:loop_zu_xy_b_s2']
SEED_COL = ['#2a78d6', '#e8883a', '#3a9e6a', '#8a5cc8']


def _load(name):
    d = os.path.join(OP_LOOP, name)
    out = dict(name=name)
    try:
        with open(os.path.join(d, 'state.json')) as f:
            st = json.load(f)
    except FileNotFoundError:
        return None
    out['passes'] = [r['k'] for r in st['passes']]
    out['rmse'] = [(r['diagnostics'].get('exact') or {}).get('logit_rmse', np.nan) for r in st['passes']]
    out['init_md'] = st['pass0']['md_steps'] + sum(r['cost']['md_steps'] for r in st['passes'])
    rp = os.path.join(d, 'production', 'rates.json')
    if os.path.isfile(rp):
        with open(rp) as f:
            r = json.load(f)
        out.update(ratio=r['rate_per_time'] / r['exact']['rate'], rel=r['rel_err_rate'], prod_md=r['cost']['md_steps'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', nargs='+', default=PAIRS, help='seed0:seed1[:seed2...] per system')
    ap.add_argument('--out', default=os.path.join(OP_LOOP, 'figures'))
    a = ap.parse_args()
    groups = [[x for x in (_load(n) for n in p.split(':')) if x] for p in a.pairs]
    groups = [g for g in groups if g]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    nmax = max(len(g) for g in groups)
    width = 0.8 / nmax
    for i, g in enumerate(groups):
        for j, r in enumerate(g):
            x = i + (j - (len(g) - 1) / 2) * width
            col = SEED_COL[j % len(SEED_COL)]
            lab = (f'seed {j}' if j == 0 else f'replicate {j}') if i == 0 else None
            if 'ratio' in r:
                lo, hi = r['ratio'] / (1 + r['rel']), r['ratio'] * (1 + r['rel'])
                axs[0].errorbar(x, r['ratio'], yerr=[[r['ratio'] - lo], [hi - r['ratio']]], fmt='o', color=col, ms=7,
                                capsize=3, lw=1.5, label=lab)
                axs[0].annotate(f"{r['ratio']:.2f}x", (x, r['ratio']), xytext=(6, 0), textcoords='offset points',
                                fontsize=7, color=INK2, va='center')
            else:
                axs[0].text(x, 1.0, 'no production yet', fontsize=7, color=MUTED, ha='center', va='bottom', rotation=90)
            axs[1].bar(x, r['init_md'] / 1e6, width * 0.9, color=col, alpha=0.55, label=f'{lab}: initialization' if lab else None)
            if 'prod_md' in r:
                axs[1].bar(x, r['prod_md'] / 1e6, width * 0.9, bottom=r['init_md'] / 1e6, color=col,
                           label=f'{lab}: production' if lab else None)
            ls = ['-', '--', ':', '-.'][i % 4]
            axs[2].plot(r['passes'], r['rmse'], ls, marker='o', color=col, lw=1.6, ms=4,
                        label=r['name'])
    names = [g[0]['name'] for g in groups]
    for ax in axs[:2]:
        ax.set_xticks(range(len(groups)), names, fontsize=8)
        ax.grid(axis='y', color='#e6e5e1', lw=0.5)
    axs[0].axhline(1, color=INK2, lw=0.8, ls='--')
    axs[0].set(yscale='log', ylabel='production rate / exact rate', title='production rate (+-1 rel. error)')
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    axs[0].yaxis.set_major_locator(FixedLocator([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]))
    axs[0].yaxis.set_minor_locator(NullLocator())
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%g'))
    axs[0].legend(fontsize=7)
    axs[1].set(ylabel='MD steps [10^6]', title='MD cost: initialization (light) + production (dark)')
    axs[1].legend(fontsize=6)
    axs[2].set(xlabel='pass', ylabel='logit RMSE vs exact (binned)', title='committor accuracy per pass')
    axs[2].set_ylim(0, None)
    axs[2].grid(color='#e6e5e1', lw=0.5)
    axs[2].legend(fontsize=6, ncol=2)
    fig.suptitle('OP loop seed replicates (colour: seed; line style in the right panel: system)', x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, 'replicates.png')
    fig.savefig(p, dpi=140, bbox_inches='tight')
    print(p)


if __name__ == '__main__':
    main()
