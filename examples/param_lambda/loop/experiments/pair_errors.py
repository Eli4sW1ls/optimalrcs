#!/usr/bin/env python3
"""Where does the error of P_cross(lambda_B) come from? Per interface pair of a production run (offline: the
running estimates of `loop.pcross_errors`, <run>/error_analysis/pcross_runav.txt).

    python -m loop.experiments.pair_errors [<loop>/<run> ...] [--json out.json]

Per pair (lambda_i, lambda_{i+1}): the local crossing probability p_i = P_cross(i+1) / P_cross(i) (running
estimates of the same snapshots), its relative block error (inftools' recursive blocking on the running series,
plateau over the longer half of the block lengths), the share of the summed squared relative errors, the effective
number of independent samples n_eff = (1 - p) / (p rel^2) (the error of a binomial estimate with n_eff trials), and
the model's ratio q(lambda_{i+1}) / q(lambda_i) that interface placement saw (h = model.h_of_lambda; the pairs that
touch a state have no model ratio). If the pairs were independent, sqrt(sum rel_i^2) would equal the relative error
of P_cross(lambda_B); the check column compares the two.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from loop import env  # noqa: E402,F401
from loop.plots import _read_runav  # noqa: E402

OP = os.path.join(HERE, '..', 'figures_by_potential', '_shared', 'param_lambda', 'op_loop')


def analyse(run):
    from inftools.istar.block_errors import compute_rel_errors_2d, plateau_error
    ea = os.path.join(run, 'error_analysis')
    head, P = _read_runav(os.path.join(ea, 'pcross_runav.txt'))
    P = P[:, 1:]                                             # Pcross_0+ ... Pcross_{n-1}+
    ok = np.all(P > 0, axis=1)
    P = P[np.argmax(ok):]                                    # from the first snapshot with every P_cross > 0
    T = len(P)
    sizes = np.arange(1, T // 5 + 1)
    p = P[:, 1:] / P[:, :-1]                                 # local crossing probabilities, running
    rel_p, _ = plateau_error(compute_rel_errors_2d(p, sizes), sizes)
    rel_B, _ = plateau_error(compute_rel_errors_2d(P[:, -1], sizes), sizes)
    pf = p[-1]
    n_eff = (1 - pf) / (pf * rel_p ** 2)
    from loop.backends import infretis_backend as be
    intf = [float(v) for v in be.read_toml(os.path.join(run, 'infretis.toml'))['simulation']['interfaces']]
    mpath = os.path.join(run, 'manifest.json')
    man = json.load(open(mpath)) if os.path.isfile(mpath) else {}
    ratio = [np.nan] * len(pf)
    try:
        if 'model' not in man:
            raise KeyError('seed lambda run')
        from lambda_runtime import LambdaModel
        h = LambdaModel(man['model']).h_of_lambda(np.array(intf[1:-1]))
        for i in range(1, len(intf) - 2):                    # inner pairs only
            ratio[i] = float(h[i] / h[i - 1])
    except Exception as e:                                   # seed-lambda runs have no learned model
        print(f'  (no model ratios: {e})')
    share = rel_p ** 2 / np.sum(rel_p ** 2)
    return dict(run=run, interfaces=intf, p=pf.tolist(), rel=rel_p.tolist(), share=share.tolist(),
                n_eff=n_eff.tolist(), model_ratio=ratio, rel_B=float(rel_B),
                rel_sum=float(np.sqrt(np.sum(rel_p ** 2))), n_snap=T)


def main():
    argv = sys.argv[1:]
    out_json = None
    if '--json' in argv:
        i = argv.index('--json')
        out_json, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    runs = argv or sorted(os.path.join(L, s) for L in os.listdir(OP) if os.path.isdir(os.path.join(OP, L))
                          for s in os.listdir(os.path.join(OP, L))
                          if os.path.isfile(os.path.join(OP, L, s, 'error_analysis', 'pcross_runav.txt')))
    res = []
    for r in runs:
        d = analyse(os.path.join(OP, r))
        d['name'] = r
        res.append(d)
        print(f"\n== {r}: rel. error of P_B {d['rel_B']:.2f}, sqrt(sum of pair rel^2) {d['rel_sum']:.2f}")
        print('  pair  lambda_i -> lambda_i+1     p_i     rel   share  n_eff  model q ratio')
        for i, (pp, rr, sh, ne, mr) in enumerate(zip(d['p'], d['rel'], d['share'], d['n_eff'], d['model_ratio'])):
            flag = '  <==' if sh > 0.25 else ''
            print(f"  {i:2d}-{i + 1:<2d} {d['interfaces'][i]:8.4f} -> {d['interfaces'][i + 1]:8.4f}  {pp:7.4f}  {rr:5.2f}  "
                  f"{sh:5.2f}  {ne:6.0f}  {'' if not np.isfinite(mr) else f'{mr:7.1f}'}{flag}")
    if out_json:
        json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
