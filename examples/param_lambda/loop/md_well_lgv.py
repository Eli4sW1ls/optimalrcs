#!/usr/bin/env python3
"""Long MD of the M5 system `well_lgv` (1D cos well, ASE Langevin = exact_tools.vec_step, kT 1, gamma 1, m 1,
dt 0.005; A: x <= -0.35, B: x >= 0.35): the reference rate and, optionally, long (x, v) trajectories.

    python -m loop.md_well_lgv --walkers 4096 --steps 1000000 [--save-walkers 8] [--out DIR]

W independent walkers (CuPy if available, else numpy), each started at x = -0.4 with v ~ N(0, kT/m) and run
--burn-in steps before counting. h = the state visited last (A or B). Rate k_AB = n(A -> B) / (time with h = A),
k_BA likewise: the convention of TIS (flux out of A times the crossing probability). Errors from 64 groups of walkers.
Every frame of the first --save-walkers walkers (after the burn-in) goes to <out>/traj.npy, float32 [walker, step,
(x, v)] (a memmap: 8 walkers x 12.5M steps = 100M frames = 800 MB); <out>/rate.json holds the counts and rates.
The step is the ASE one (checked step by step in tests/test_loop.py), so the rate is that of the infretis runs.
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

A, B, DT, GAMMA, KT = -0.35, 0.35, 0.005, 1.0, 1.0
DEFAULT_OUT = os.path.join(env.EXAMPLES, 'figures_by_potential', '1d_cos_well', 'langevin_md_ase')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walkers', type=int, default=4096)
    ap.add_argument('--steps', type=int, default=1_000_000, help='steps per walker after the burn-in')
    ap.add_argument('--burn-in', type=int, default=20_000)
    ap.add_argument('--save-walkers', type=int, default=0)
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
    x = xp.full(W, -0.4)
    v = rs.standard_normal(W) * np.sqrt(KT)
    h = xp.zeros(W, dtype=xp.int8)                  # 0: last in A, 1: last in B
    n_ab = xp.zeros(W, dtype=xp.int64)
    n_ba = xp.zeros(W, dtype=xp.int64)
    t_a = xp.zeros(W, dtype=xp.int64)
    S = a.save_walkers
    traj = np.lib.format.open_memmap(os.path.join(a.out, 'traj.npy'), mode='w+', dtype=np.float32,
                                     shape=(S, a.steps, 2)) if S else None
    chunk = 10_000
    buf = xp.zeros((S, chunk, 2), dtype=xp.float32) if S else None
    t0 = time.time()
    for step in range(-a.burn_in, a.steps):
        x, v = E.vec_step_xieta(x, v, rs.standard_normal(W), rs.standard_normal(W), E.coswell_force, coef, xp)
        in_a, in_b = x <= A, x >= B
        if step >= 0:
            n_ab += (in_b & (h == 0))
            n_ba += (in_a & (h == 1))
        h = xp.where(in_a, 0, xp.where(in_b, 1, h)).astype(xp.int8)
        if step >= 0:
            t_a += (h == 0)
            if S:
                buf[:, step % chunk, 0] = x[:S]
                buf[:, step % chunk, 1] = v[:S]
                if step % chunk == chunk - 1 or step == a.steps - 1:
                    lo = step - step % chunk
                    traj[:, lo:step + 1] = (buf if xp is np else buf.get())[:, :step + 1 - lo]
        if step > 0 and step % 200_000 == 0:
            print(f'step {step} of {a.steps}: {int(n_ab.sum())} A->B, {time.time() - t0:.0f} s', flush=True)
    to_np = (lambda z: z) if xp is np else (lambda z: z.get())
    n_ab, n_ba, t_a = to_np(n_ab), to_np(n_ba), to_np(t_a)
    t_b = a.steps - t_a
    g = lambda z: z.reshape(G, -1).sum(1)
    k_ab_g = g(n_ab) / (g(t_a) * DT)
    k_ba_g = g(n_ba) / (g(t_b) * DT)
    k_ab = n_ab.sum() / (t_a.sum() * DT)
    k_ba = n_ba.sum() / (t_b.sum() * DT)
    out = dict(system='well_lgv', integrator='ase Langevin (exact_tools.vec_step)', dt=DT, gamma=GAMMA, kT=KT, A=A, B=B,
               walkers=W, steps_per_walker=a.steps, burn_in=a.burn_in, seed=a.seed, walker_steps=W * a.steps,
               n_ab=int(n_ab.sum()), n_ba=int(n_ba.sum()), fraction_last_in_a=float(t_a.sum() / (W * a.steps)),
               k_ab=float(k_ab), k_ab_se=float(k_ab_g.std(ddof=1) / np.sqrt(G)),
               k_ba=float(k_ba), k_ba_se=float(k_ba_g.std(ddof=1) / np.sqrt(G)),
               steps_per_ab=float(t_a.sum() / max(n_ab.sum(), 1)), seconds=time.time() - t0,
               note='k = transitions / time with h (last state visited) = A, per reduced time unit')
    if S:
        out['traj'] = dict(file='traj.npy', shape=[S, a.steps, 2], columns=['x', 'v'])
    with open(os.path.join(a.out, 'rate.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
