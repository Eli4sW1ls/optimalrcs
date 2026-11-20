#!/usr/bin/env python3
"""Confirm the next pass of a loop that runs with [run] confirm_passes = true (it waits after every pass, and after
pass 0, until this is given):

    python -m loop.confirm <config> continue      # go on with the next pass
    python -m loop.confirm <config> stop          # end the passes here; the loop freezes as usual (gated)
    python -m loop.confirm <config> status        # which pass waits, and its figures

Look first at pass_<k>/interfaces_paths*.png (the interfaces of pass k, the previous set and a few of its paths).
--pass k confirms a given pass ahead of time (the loop then does not stop there)."""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, controller  # noqa: E402,F401

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('decision', choices=['continue', 'stop', 'status'])
    ap.add_argument('--pass', dest='k', type=int, help='the pass to confirm (default: the one that waits)')
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    waiting = sorted(glob.glob(os.path.join(cfg['loop_dir'], 'pass_*', 'WAITING')),
                     key=lambda p: int(os.path.basename(os.path.dirname(p)).split('_')[1]))
    if a.decision == 'status':
        if not waiting:
            print(f'{cfg["name"]}: no pass waits for confirmation')
        for w in waiting:
            print(f'{os.path.dirname(w)} waits; figures:')
            print(open(w).read().strip() or '  (none)')
        sys.exit(0)
    if a.k is not None:
        d = os.path.join(cfg['loop_dir'], f'pass_{a.k}')
    elif waiting:
        d = os.path.dirname(waiting[-1])
    else:
        sys.exit(f'{cfg["name"]}: no pass waits for confirmation (use --pass k to confirm one ahead of time)')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, a.decision.upper()), 'w') as f:
        f.write('confirmed by loop.confirm\n')
    print(f'{d}: {a.decision}')
