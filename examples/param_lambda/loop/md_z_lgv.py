#!/usr/bin/env python3
"""Long MD of `z_lgv` (the Z potential with ASE's Langevin, gamma 2, kT 0.05, m 1, dt 0.002; A: y <= 0.05,
B: y >= 0.85): the reference rate of loop_z_lgv.

    python -m loop.md_z_lgv [--walkers 16384] [--steps 4000000] [--out DIR]

As loop/md_well_lgv.py, in 2D: W independent walkers (CuPy if available), started at (y, x) = (0.02, 0.15) with
Maxwell velocities, --burn-in steps before counting; the ASE step of exact_tools.vec_step_xieta applied per Cartesian
component (independent draws per component, as ase.md.langevin) with the Z force including gravity
(exact_tools.zpot_force, the force of sjoelbak_angle.RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)).
h = the state visited last; k_AB = n(A -> B) / (time with h = A). Errors from --groups groups of walkers.
Writes <out>/rate.json (default figures_by_potential/2d_zpot/langevin_md_ase/), which ase_systems.z_lgv reads as its
reference rate.

Cost: the overdamped Z (gamma 10) has k_AB = 1.13e-5 per time unit; at gamma 2 expect a few times more, i.e. about
1e-7 transitions per walker step at dt 0.002: 16384 walkers x 4M steps ~ 6.5e10 walker steps ~ several thousand
transitions. Check `n_ab` (error ~ 1 / sqrt(n_ab)) and extend with --steps if needed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from loop import env  # noqa: E402

sys.path.insert(0, os.path.join(env.EXAMPLES, 'figures_by_potential', '_shared', 'exact_committors'))
import exact_tools as E  # noqa: E402

A, B, DT, GAMMA, KT = 0.05, 0.85, 0.002, 2.0, 0.05
START = (0.02, 0.15)
DEFAULT_OUT = os.path.join(env.EXAMPLES, 'figures_by_potential', '2d_zpot', 'langevin_md_ase')


def force2(q, xp):
    """Force on the stacked positions q = (y, x) (shape (2, W))."""
    fy, fx = E.zpot_force(q[0], q[1], xp)
    return xp.stack([fy, fx])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walkers', type=int, default=16384)
    ap.add_argument('--steps', type=int, default=4_000_000, help='steps per walker after the burn-in')
    ap.add_argument('--burn-in', type=int, default=20_000)
    ap.add_argument('--groups', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out', default=DEFAULT_OUT)
    a = ap.parse_args()
    try:
        if a.cpu:
            raise ImportError
        import cupy as xp
        rs = xp.random.RandomState(a.seed)
    except ImportError:
        xp = np
        rs = np.random.RandomState(a.seed)
    os.makedirs(a.out, exist_ok=True)
    W, G = a.walkers, a.groups
    assert W % G == 0, '--walkers must be a multiple of --groups'
    coef = E.vec_coefficients(DT, GAMMA, KT, 1.0)
    q = xp.stack([xp.full(W, START[0]), xp.full(W, START[1])])
    v = rs.standard_normal((2, W)) * np.sqrt(KT)
    h = xp.zeros(W, dtype=xp.int8)                           # 0: last in A, 1: last in B
    n_ab = xp.zeros(W, dtype=xp.int64)
    n_ba = xp.zeros(W, dtype=xp.int64)
    t_a = xp.zeros(W, dtype=xp.int64)
    t0 = time.time()
    for step in range(-a.burn_in, a.steps):
        q, v = E.vec_step_xieta(q, v, rs.standard_normal((2, W)), rs.standard_normal((2, W)), force2, coef, xp)
        in_a, in_b = q[0] <= A, q[0] >= B
        if step >= 0:
            n_ab += (in_b & (h == 0))
            n_ba += (in_a & (h == 1))
        h = xp.where(in_a, 0, xp.where(in_b, 1, h)).astype(xp.int8)
        if step >= 0:
            t_a += (h == 0)
        if step > 0 and step % 500_000 == 0:
            print(f'step {step} of {a.steps}: {int(n_ab.sum())} A->B, {time.time() - t0:.0f} s', flush=True)
    to_np = (lambda z: z) if xp is np else (lambda z: z.get())
    n_ab, n_ba, t_a = to_np(n_ab), to_np(n_ba), to_np(t_a)
    t_b = a.steps - t_a
    g = lambda z: z.reshape(G, -1).sum(1)
    k_ab_g = g(n_ab) / (g(t_a) * DT)
    k_ba_g = g(n_ba) / np.maximum(g(t_b) * DT, 1e-300)
    out = dict(system='z_lgv', integrator='ase Langevin (exact_tools.vec_step, per component)', dt=DT, gamma=GAMMA, kT=KT,
               A=A, B=B, start=START, walkers=W, steps_per_walker=a.steps, burn_in=a.burn_in, seed=a.seed,
               walker_steps=W * a.steps, n_ab=int(n_ab.sum()), n_ba=int(n_ba.sum()),
               fraction_last_in_a=float(t_a.sum() / (W * a.steps)),
               k_ab=float(n_ab.sum() / (t_a.sum() * DT)), k_ab_se=float(k_ab_g.std(ddof=1) / np.sqrt(G)),
               k_ba=float(n_ba.sum() / max(t_b.sum() * DT, 1e-300)), k_ba_se=float(k_ba_g.std(ddof=1) / np.sqrt(G)),
               steps_per_ab=float(t_a.sum() / max(n_ab.sum(), 1)), seconds=time.time() - t0,
               note='k = transitions / time with h (last state visited) = A, per reduced time unit')
    with open(os.path.join(a.out, 'rate.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
