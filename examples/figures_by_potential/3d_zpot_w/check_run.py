"""Sanity checks of a zpot_w staple run before any fitting: is the slow coordinate w equilibrated?

    python check_run.py /path/to/simulations/sim_istarzw_br_g10_dV3_s1 [--blocks 5] [--w0 0.1]

By the symmetry U(y, x, w) = U(y, Lx - x, -w), every ensemble should hold w > 0 in half of its frames. Per ensemble
and per block of cycles (accepted paths, counted once per accepted path; rejection weights are not applied):
the fraction of frames with w > 0, the fraction of paths that change channel (hysteresis at +-w0/2), and the
median path length. Fractions that stay far from 0.5 in the later blocks, or drift, mean the channels are not
equilibrated: use a later burn-in (split.id_from) or run longer.
"""
import argparse
import os
import re

import numpy as np

CYCLE = re.compile(r'# Cycle: (\d+), status: (\w+)')


def read_paths(order_file):
    """(cycle, array (n, 3) of y, x, w) of every accepted path with more than one frame."""
    out, cur, cyc, ok = [], [], None, False
    with open(order_file) as f:
        for line in f:
            if line.startswith('#'):
                m = CYCLE.match(line)
                if m:
                    if ok and len(cur) > 1:
                        out.append((cyc, np.array(cur)))
                    cur, cyc, ok = [], int(m.group(1)), m.group(2) == 'ACC'
                continue
            if ok:
                cur.append([float(v) for v in line.split()[1:4]])
    if ok and len(cur) > 1:
        out.append((cyc, np.array(cur)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--blocks', type=int, default=5)
    ap.add_argument('--w0', type=float, default=0.1)
    a = ap.parse_args()
    ens = sorted(d for d in os.listdir(a.run) if d.isdigit())
    print(f'{a.run}: {len(ens)} ensembles; per block of cycles: frac(w > 0) / frac(paths changing channel) / '
          f'median length')
    for e in ens:
        paths = read_paths(os.path.join(a.run, e, 'order.txt'))
        if not paths:
            print(f'  {e}: no accepted paths')
            continue
        cyc = np.array([c for c, _ in paths])
        edges = np.linspace(cyc.min(), cyc.max() + 1, a.blocks + 1)
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [p for c, p in paths if lo <= c < hi]
            if not sel:
                cells.append('      -        ')
                continue
            w = np.concatenate([p[:, 2] for p in sel])
            sw = 0
            for p in sel:
                ch = np.where(p[:, 2] > a.w0 / 2, 1, np.where(p[:, 2] < -a.w0 / 2, -1, 0))
                ch = ch[ch != 0]
                sw += bool(len(ch) and (ch != ch[0]).any())
            cells.append(f'{np.mean(w > 0):.2f}/{sw / len(sel):.2f}/{int(np.median([len(p) for p in sel])):5d}')
        print(f'  {e} ({len(paths):6d} paths)  ' + '  '.join(cells))


if __name__ == '__main__':
    main()
