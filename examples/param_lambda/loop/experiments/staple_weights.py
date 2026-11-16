#!/usr/bin/env python3
"""The open StapleTIS weight question of docs/npneq_tis_notes.md section 15, measured on the flat 1D system with
its exact committor (infretis + ASE, overdamped, the settings of loop_flat1d).

Three StapleTIS runs from the same seed paths and interfaces (-0.1, 0, 0.1, 0.2, 0.3):
    inf      infinite swap (the permanent of W with the staple w)
    nn       nearest-neighbour staple swaps (staple_swap_frac = 0.5), no infinite swap
    sh       shooting only (no swaps): nothing uses w, so path_f is exact
For each run and weighting (path_f = 'rejections', path_f / path_w = 'staple'), on the pieces cut at M (forward and
mirrored): the stopped Z_q z-score of the exact committor (it must pass on unbiased data; compare between data
sets of the same size only), and the bias of the path-weighted mean of q_exact at the first frames past M against
the TIS crossing probability (a weight error shifts the share of LML/RMR against LMR/RML pieces).

    python -m loop.experiments.staple_weights setup  [--cycles 20000]
    python -m loop.experiments.staple_weights run inf|nn|sh       (each in its own unit; 1 core per worker)
    python -m loop.experiments.staple_weights analyze
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from loop import env, ase_systems, pool as pool_mod, seed as seed_mod  # noqa: E402
from loop.backends import infretis_backend as be  # noqa: E402
import common  # noqa: E402

OUT = os.path.join(common.OUT_ROOT, 'op_loop', 'staple_weights')
SOURCE = os.path.join(common.OUT_ROOT, 'op_loop', 'loop_flat1d_smoke')     # pool with pass-0 paths of flat1d
INTERFACES = [-0.1, 0.0, 0.1, 0.2, 0.3]
VARIANTS = {'inf': dict(staple_infinite_swap=True),
            'nn': dict(staple_infinite_swap=False, staple_swap_frac=0.5),
            'sh': dict(staple_infinite_swap=False, staple_swap_frac=0.0)}


class Identity:
    """lambda = x (the seed order parameter) for cutting seeds."""
    def __call__(self, cv):
        return np.atleast_2d(cv)[:, 0].astype(float)


def setup(cycles, seed=1):
    os.makedirs(OUT, exist_ok=True)
    system = ase_systems.get('flat1d')
    pool = pool_mod.Pool(SOURCE)
    seeds = os.path.join(OUT, 'seeds')
    if not os.path.isdir(seeds):
        dirs, tainted, report = seed_mod.build_seeds(pool, Identity(), INTERFACES, 'staple', seeds)
        with open(os.path.join(OUT, 'seeds.json'), 'w') as f:
            json.dump({str(k): v for k, v in report.items()}, f, indent=1, default=str)
    dirs = [os.path.join(seeds, str(i)) for i in range(len(INTERFACES))]
    for name, opts in VARIANTS.items():
        run = dict(workers=4, screen=500, **opts)
        wd = os.path.join(OUT, name if seed == 1 else f'{name}_s{seed}')
        if os.path.isdir(wd):
            continue
        be.write_run(wd, system, run, INTERFACES, be.seed_order_table(system), cycles, 'staple', dirs,
                     seed=seed, note=f'staple weight test, variant {name}, seed {seed}')
        print('prepared', wd)


def run(name):
    rec = be.run(os.path.join(OUT, name))
    print(name, rec)


def analyze(burn_in=0.1):
    sys.path.insert(0, os.path.join(env.EXAMPLES, 'figures_by_potential', '_shared', 'exact_committors'))
    mt = common.metrics_tis()
    system = ase_systems.get('flat1d')
    qx = ase_systems.exact_committor(system)
    res = {}
    names = sorted(d for d in os.listdir(OUT) if d.split('_s')[0] in VARIANTS and os.path.isdir(os.path.join(OUT, d)))
    for name in names:
        wd = os.path.join(OUT, name)
        if not os.path.isfile(os.path.join(wd, 'restart.toml')):
            continue
        res[name] = {'cost': be.cost(wd)}
        for weighting in ('rejections', 'staple'):
            p = be.load_pieces(wd, reversed_pieces=True, burn_in_fraction=burn_in, weighting=weighting)
            x = p['cv'][:, 1:]
            q = qx(x)
            piece = p['piece']
            w = np.repeat(p['piece_w'], p['piece_n'])
            w = w / w.mean()
            s = x[:, 0]
            A, B = s <= system['lambda_a'], s >= system['lambda_b']
            r = q.copy(); r[A], r[B] = 0, 1
            b = (A | B).astype(float)
            group = np.repeat(p['piece_pnr'], p['piece_n'])
            tau = mt.stop_index(piece, b)
            zmax, sd = [], []
            for lag in (1, 4, 16, 64, 256):
                _, z = mt.comp_zq_stopped_zscore(r, b, piece, w, lag, tau=tau, g_traj=group)
                zmax.append(float(np.max(np.abs(z[5:]))))
            # level test: weighted mean of q_exact at the first frames of the forward pieces per ensemble
            starts = np.r_[0, np.cumsum(p['piece_n'])[:-1]]
            level = {}
            for e in np.unique(p['piece_ens']):
                sel = (p['piece_ens'] == e) & ~p['piece_mirrored']
                if sel.sum() < 20:
                    continue
                st = starts[sel]
                level[int(e)] = dict(mean_q=float(np.average(q[st], weights=p['piece_w'][sel])), n=int(sel.sum()))
            res[name][weighting] = dict(zmax=zmax, n_pieces=int(len(p['piece_n'])), n_frames=int(len(s)), level=level)
            print(f'{name:3s} {weighting:10s}: {len(p["piece_n"])} pieces; exact q, stopped Z_q max|z| per lag '
                  f'{np.round(zmax, 2).tolist()}; mean q at first frames past M per ensemble '
                  + ', '.join(f'{e}: {v["mean_q"]:.4f}' for e, v in level.items()))
    with open(os.path.join(OUT, 'analysis.json'), 'w') as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['setup', 'run', 'analyze'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('--cycles', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    if a.cmd == 'setup':
        setup(a.cycles, a.seed)
    elif a.cmd == 'run':
        run(a.name)
    else:
        analyze()


if __name__ == '__main__':
    main()
