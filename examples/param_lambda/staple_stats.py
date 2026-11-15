#!/usr/bin/env python3
"""Per-ensemble path statistics of toytis staple runs, side by side (works on unfinished runs).

    python staple_stats.py RUN_DIR [RUN_DIR ...] [--skip 0.2]

Per ensemble: the median (and 90th percentile) length of the accepted paths, and the acceptance rate, over the
cycles after the first `--skip` fraction. For StapleTIS the path lengths should not differ too much between
ensembles; an ensemble whose paths are far longer than the rest dominates the cost (and may get stuck).
"""
import argparse
import os
import re

import numpy as np

HEAD = re.compile(r'# Cycle: (\d+), status: (\w+).*path length: (\d+)')


def stats(run, skip):
    out = {}
    for e in sorted(d for d in os.listdir(run) if d.isdigit()):
        rows = []
        with open(os.path.join(run, e, 'order.txt')) as f:
            for line in f:
                if line.startswith('# Cycle'):
                    m = HEAD.match(line)
                    if m:
                        rows.append((int(m.group(1)), m.group(2) == 'ACC', int(m.group(3))))
        if not rows:
            continue
        c = np.array(rows)
        c = c[c[:, 0] >= skip * c[:, 0].max()]
        acc = c[(c[:, 1] == 1) & (c[:, 2] > 1)]
        out[e] = dict(n_cycles=int(c[:, 0].max()) + 1, acc=len(acc) / max(len(c), 1),
                      med=float(np.median(acc[:, 2])) if len(acc) else np.nan,
                      p90=float(np.percentile(acc[:, 2], 90)) if len(acc) else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--skip', type=float, default=0.2, help='leave out the first fraction of the cycles')
    a = ap.parse_args()
    for run in a.runs:
        s = stats(run, a.skip)
        if not s:
            print(f'{run}: no paths yet')
            continue
        n = max(v['n_cycles'] for v in s.values())
        body = {e: v for e, v in s.items() if int(e) >= 2}
        meds = [v['med'] for v in body.values() if np.isfinite(v['med'])]
        ratio = max(meds) / min(meds) if meds else np.nan
        print(f'{os.path.basename(os.path.normpath(run))}: {n} cycles; median length max/min over [1*] ... = {ratio:.1f}')
        print('   ens ' + ' '.join(f'{e:>7s}' for e in s))
        print('   med ' + ' '.join(f'{v["med"]:7.0f}' for v in s.values()))
        print('   p90 ' + ' '.join(f'{v["p90"]:7.0f}' for v in s.values()))
        print('   acc ' + ' '.join(f'{v["acc"]:7.2f}' for v in s.values()))


if __name__ == '__main__':
    main()
