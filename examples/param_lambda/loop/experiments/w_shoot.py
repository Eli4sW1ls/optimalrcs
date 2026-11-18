#!/usr/bin/env python3
"""E7 shooting test for a slow CV w: does w change the dynamics of the other CVs near the transition region?

    python -m loop.experiments.w_shoot <loop config> [--points 400] [--shots 8] [--lags 10,50,200]

Data-only screening failed on the M3 pools (validation SD(Z_q), held-out Dirichlet energy, split-half reproduction of
the teacher's well difference; see w_split.py): the outcome information is a few dozen B arrivals clustered by lineage,
so a fitted w-dependence of the uncoupled w (loop_zu) reproduces as well as the real one (loop_zw). Here fresh,
independent short trajectories decide.

Points: (y, x) of pool frames whose teacher label lies in --q-range (the transition region the interfaces resolve).
From each point, overdamped Langevin (Euler-Maruyama with the system's gamma, kT, dt) is started with w = -w0 and with
w = +w0 (the two wells), `2 x shots` times each, for max(lags) steps. Per point and lag, d = mean displacement of the
other CVs with w = -w0 minus that with w = +w0, from two independent halves of the shots: d1, d2. The statistic
S = mean over points of d1 . d2 has expectation sum ||E d||^2 >= 0, exactly 0 when w does not enter the dynamics of the
other CVs (paired starting points: equal in law). z = S / s.e.(S) over points; keep w when z > --z-keep (default 3).
The test conditions on w by setting it (possible in a toy system; a molecular system needs a restrained start or
binning instead). Output: <loop dir>/w_shoot.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from loop import env, controller, passes  # noqa: E402,F401
from loop import pool as pool_mod  # noqa: E402
from loop.ase_calcs import ToytisPotential  # noqa: E402


def force_fn(system):
    c = dict(system['calculator'])
    c.pop('class')
    calc = ToytisPotential(**c)
    bd, dim = calc.base_dim, calc.dim

    def force(P):
        F = np.zeros_like(P)
        for i in range(len(P)):
            F[i, :bd] = calc.potential.potential_and_force((P[i, :bd], None))[1]
        if bd < dim:
            r = P[:, bd:] / calc.w0
            F[:, bd:] = -4.0 * calc.dV * (r * r - 1.0) * r / calc.w0
        return F
    return force


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--cv', default='w')
    ap.add_argument('--w0', type=float, default=0.1)
    ap.add_argument('--points', type=int, default=400)
    ap.add_argument('--shots', type=int, default=8, help='shots per half, well and point')
    ap.add_argument('--lags', default='10,50,200')
    ap.add_argument('--q-range', default='0.2,0.8')
    ap.add_argument('--z-keep', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    system = cfg['system_spec']
    lags = sorted(int(v) for v in a.lags.split(','))
    q_lo, q_hi = (float(v) for v in a.q_range.split(','))
    pool = pool_mod.Pool(cfg['loop_dir'])
    cfg['pool']['max_frames'] = 4_000_000
    v = passes._view(pool, cfg, need_labels=True)
    names = list(map(str, v.cv_names))
    wcol = names.index(a.cv)
    others = [i for i in range(len(names)) if i != wcol]
    rng = np.random.default_rng(a.seed)
    cand = np.flatnonzero(v.inside & np.isfinite(v.q_np) & (v.q_np >= q_lo) & (v.q_np <= q_hi))
    pts = v.cv[rng.choice(cand, min(a.points, len(cand)), replace=False)].copy()
    del v
    n_p, n_s = len(pts), a.shots
    force = force_fn(system)
    kT, gamma, dt = system['kT'], system['gamma'], system['dt']
    sig = np.sqrt(2 * kT * dt / gamma)
    disp = {}                                   # (well, lag) -> (n_p, 2, n_s, n_other)
    t0 = time.time()
    for well, w in (('-', -a.w0), ('+', a.w0)):
        P0 = np.repeat(pts, 2 * n_s, axis=0)
        P0[:, wcol] = w
        P = P0.copy()
        for step in range(1, lags[-1] + 1):
            P = P + dt * force(P) / gamma + sig * rng.standard_normal(P.shape)
            if step in lags:
                disp[(well, step)] = (P - P0)[:, others].reshape(n_p, 2, n_s, len(others))
    secs = time.time() - t0
    out = dict(config=a.config, cv=a.cv, w0=a.w0, points=n_p, shots_per_half=n_s, q_range=[q_lo, q_hi],
               walker_steps=int(2 * n_p * 2 * n_s * lags[-1]), seconds=secs, lags={})
    for lag in lags:
        dm, dp = disp[('-', lag)].mean(2), disp[('+', lag)].mean(2)          # (n_p, 2 halves, n_other)
        d = dm - dp
        s_i = np.sum(d[:, 0] * d[:, 1], axis=1)                              # per point d1 . d2
        S, se = float(s_i.mean()), float(s_i.std(ddof=1) / np.sqrt(n_p))
        noise = float(np.sqrt(np.mean(np.sum((d[:, 0] - d[:, 1]) ** 2, axis=1)) / 2))
        out['lags'][lag] = dict(S=S, se=se, z=S / se, rms_effect=float(np.sqrt(max(S, 0.0))),
                                rms_noise_per_point=noise, mean_d=d.mean((0, 1)).tolist(),
                                rms_disp=float(np.sqrt(np.mean(np.sum(disp[('-', lag)] ** 2, axis=-1)))))
    zmax = max(r['z'] for r in out['lags'].values())
    out['keep'] = bool(zmax > a.z_keep)
    out['z_max'] = zmax
    with open(os.path.join(cfg['loop_dir'], 'w_shoot.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(a.config, f'{n_p} points, {secs:.0f} s, keep {a.cv}: {out["keep"]} (max z {zmax:.1f})')
    for lag, r in out['lags'].items():
        print(f'  lag {lag:4d}: z {r["z"]:7.2f}  effect RMS {r["rms_effect"]:.4g}  noise per point {r["rms_noise_per_point"]:.4g}'
              f'  RMS displacement {r["rms_disp"]:.4g}  mean d {np.round(r["mean_d"], 5).tolist()}')


if __name__ == '__main__':
    main()
