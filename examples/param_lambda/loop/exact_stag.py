"""Exact committor and rate of the M3 two-channel system (ase_systems.stag_br): toy2d.exact_committor_brownian
(finite volumes for the overdamped generator, the boundary shift of a check every step, TPT rate) on the state
coordinate x in (-1.75, 1.75) and the transverse y in (-2.05, 2.05).

    python -m loop.exact_stag [--h 0.01]     -> figures_by_potential/2d_staggered_channels/exact/brownian_g1_dt0.005/q_tpt_h<h>.npz

Result (2026-11-17): k_AB = 1.1476e-4 per time unit (h 0.01; 1.1507e-4 at h 0.02), 1.74e6 MD steps per A -> B
transition; each channel carries half of the reactive flux; at x = 0 the committor is 0.997 in the upper channel
(past its barrier at x = -0.6) and 0.003 in the lower one (barrier at +0.6). About 50 s on one core at h 0.01.
"""
import argparse
import os
import time

import numpy as np

from loop import env, ase_systems


def solve(system, h=0.01):
    import toy2d
    from loop.ase_calcs import load_toytis_potential
    pot = load_toytis_potential(system['potential'], system['potential_kwargs'])
    lo, hi = system['exact']['ranges']
    return toy2d.exact_committor_brownian(pot, system['kT'], system['gamma'], system['dt'], system['lambda_a'],
                                          system['lambda_b'], h=h, y_range=tuple(lo), x_range=tuple(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=float, default=0.01)
    a = ap.parse_args()
    s = ase_systems.get('stag_br')
    t0 = time.time()
    r = solve(s, a.h)
    out = os.path.join(env.REPO, s['exact']['file'].replace('h0.01', f'h{a.h:g}'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **r)
    ys, xs, Q = r['y'], r['x'], r['q']
    i = np.searchsorted(ys, 0.0)
    bv = (r['V'] - r['V'].min()) / s['kT']
    J = np.exp(-(bv[i] + bv[i - 1]) / 2) * (Q[i] - Q[i - 1])
    print(f'h {a.h}: k_AB {r["k_ab"]:.5g} per time unit, {r["steps_per_ab"]:.4g} steps per transition, '
          f'upper-channel flux share {J[xs > 0].sum() / J.sum():.4f}, {time.time() - t0:.0f} s -> {out}')


if __name__ == '__main__':
    main()
