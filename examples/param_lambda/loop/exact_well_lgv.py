#!/usr/bin/env python3
"""Exact committor q(x, v) of the M5 system `well_lgv`: the 1D cos well with ASE's Langevin integrator
(Vanden-Eijnden-Ciccotti; exact_tools.vec_step, equal to ase.md.langevin.Langevin to round-off, see
tests/test_loop.py::test_vec_step_matches_ase), kT 1, gamma 1, m 1, dt 0.005, A: x <= -0.35, B: x >= 0.35, checked
every step (infretis with subcycles 1 saves and tests every step).

    python -m loop.exact_well_lgv [--n-shoot 1000000] [--levels 701x501,1401x1001,2801x1001]

1. Grid solve (exact_tools.solve_underdamped_1d) at three resolutions; error estimate = change between the two finest
   grids on a sample of Boltzmann (x, v) points between A and B.
2. Independent check: GPU shooting at the 10 points of figures_by_potential/1d_cos_well/exact/well_exact.py.
3. The difference from the toytis BBK committor (q_exact_BBK.npz): how much the integrator matters at dt = 0.005.
Output: figures_by_potential/1d_cos_well/exact/q_exact_ASE.npz (+ .log). Needs CuPy (GPU); takes the loops' GPU lock.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from loop import env, gpu  # noqa: E402

sys.path.insert(0, os.path.join(env.EXAMPLES, 'figures_by_potential', '_shared', 'exact_committors'))
import exact_tools as E  # noqa: E402

OUT = os.path.join(env.EXAMPLES, 'figures_by_potential', '1d_cos_well', 'exact')
A, B, DT, GAMMA, KT = -0.35, 0.35, 0.005, 1.0, 1.0
POINTS = [(-0.349, -0.2), (-0.30, 0.0), (-0.28, 0.6), (-0.20, -0.5), (-0.20, 0.3), (-0.10, 0.8), (0.0, 0.0),
          (0.18, 0.7), (0.30, -0.3), (0.345, 0.3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-shoot', type=int, default=1_000_000)
    ap.add_argument('--levels', default='701x501,1401x1001,2801x1001',
                    help='grids nx x nv, coarse to fine (the finest is saved; 2801x1001 needs ~6 GB of GPU memory)')
    a = ap.parse_args()
    levels = [tuple(int(n) for n in lv.split('x')) for lv in a.levels.split(',')]
    log = open(os.path.join(OUT, 'q_exact_ASE.log'), 'w')

    def say(*s):
        t = ' '.join(str(x) for x in s)
        print(t, flush=True)
        log.write(t + '\n')
        log.flush()

    coef = E.vec_coefficients(DT, GAMMA, KT, 1.0)
    # test points: Boltzmann in (x, v) between A and B (rejection on exp(-V / kT)), v ~ N(0, kT / m)
    rng = np.random.default_rng(0)
    x = rng.uniform(A, B, 2_000_000)
    x = x[rng.uniform(size=x.size) < np.exp(-(E.coswell_potential(x) - E.coswell_potential(x).min()) / KT)][:300_000]
    v = rng.normal(0, np.sqrt(KT), x.size)
    near = np.minimum(x - A, B - x) < 0.02
    with gpu.exclusive(what='exact_well_lgv'):
        sols, vals = {}, {}
        for lv in levels:
            t0 = time.time()
            sols[lv], res = E.solve_underdamped_1d(E.vec_step, coef, E.coswell_force, A, B, *lv, verbose=False)
            vals[lv] = E.eval_q_underdamped(E.vec_step, coef, E.coswell_force, sols[lv], x, v)
            say(f'grid {lv[0]:5d} x {lv[1]:5d}: residual {res:.1e}, {time.time() - t0:.0f} s')
        say('changes on Boltzmann points: max over all | max > 0.02 from A/B | RMS')
        for l1, l2 in zip(levels[:-1], levels[1:]):
            d = vals[l2] - vals[l1]
            say(f'  {l1} -> {l2}: {np.abs(d).max():.1e} | {np.abs(d[~near]).max():.1e} | {np.sqrt(np.mean(d ** 2)):.1e}')
        best = sols[levels[-1]]
        qb = vals[levels[-1]]
        sym = np.abs(qb + E.eval_q_underdamped(E.vec_step, coef, E.coswell_force, best, -x, -v) - 1)
        say(f'symmetry |q(x,v) + q(-x,-v) - 1|: max {sym.max():.1e}')
        t0 = time.time()
        p, se, mean_steps = E.shoot_underdamped_1d(E.vec_step, coef, E.coswell_force, A, B, POINTS, a.n_shoot, seed=1)
        qg = E.eval_q_underdamped(E.vec_step, coef, E.coswell_force, best, *np.array(POINTS).T)
        say(f'shooting, {a.n_shoot} trajectories per point ({time.time() - t0:.0f} s, mean {mean_steps:.0f} steps):')
        for (px, pv), pi, si, gi in zip(POINTS, p, se, qg):
            say(f'  (x, v) = ({px:+.3f}, {pv:+.2f}): shooting {pi:.5f} +- {si:.5f} | grid {gi:.5f} | '
                f'diff {gi - pi:+.5f} ({(gi - pi) / si:+.1f} s.e.)')
        say(f'chi^2 = {np.sum(((qg - p) / se) ** 2):.1f} for {len(POINTS)} points')
    bbk_file = os.path.join(OUT, 'q_exact_BBK.npz')
    if os.path.isfile(bbk_file):
        z = np.load(bbk_file)
        sb = dict(xs=z['xs'], vs=z['vs'], m=z['m'], A=float(z['A']), B=float(z['B']), c_map=float(z['c_map']))
        qbbk = E.eval_q_underdamped(E.bbk_step, E.bbk_coefficients(DT, GAMMA), E.coswell_force, sb, x, v)
        d = qb - qbbk
        say(f'ASE Langevin - toytis BBK on Boltzmann points: RMS {np.sqrt(np.mean(d ** 2)):.1e}, max {np.abs(d).max():.1e}, '
            f'max > 0.02 from A/B {np.abs(d[~near]).max():.1e}')
    l1 = levels[-2]
    np.savez_compressed(os.path.join(OUT, 'q_exact_ASE.npz'), xs=best['xs'], vs=best['vs'], m=best['m'], A=A, B=B,
                        c_map=best['c_map'], dt=DT, gamma=GAMMA, kT=KT, m_previous=sols[l1]['m'],
                        xs_previous=sols[l1]['xs'], vs_previous=sols[l1]['vs'],
                        note='ASE Langevin (exact_tools.vec_step) committor, loop/exact_well_lgv.py')
    say('wrote', os.path.join(OUT, 'q_exact_ASE.npz'))


if __name__ == '__main__':
    main()
