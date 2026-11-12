"""Exact committors of the 1D Brownian (overdamped) toytis dynamics used in the tests:

  * long MD, flat     : gamma 5, dt 2e-4, kT 1, A: x <= -0.1, B: x >= 0.3 (V = 0 between A and B)
  * long MD, two bumps: same, V = 0.5 (1 - cos(2 pi (x + 0.1) / 0.2)) on [-0.1, 0.3]
  * staple, flat      : sim_istar0br10intf, gamma 1, dt 2e-4, kT 1, A: x <= -0.1, B: x >= 0.8

Chain: x' = x + dt/(m gamma) F(x) + N(0, sigma^2), sigma^2 = 2 kT dt/(m gamma); A/B checked every step.
q(x) = P(x' >= B) + int_A^B N(x'; mean(x), sigma) q(x') dx'. q is piecewise linear on a grid over [A, B]
(nodes = one-sided limits at A and B) and the Gaussian is integrated against it exactly (error functions),
so the only approximation is the piecewise-linear q: O(h^2), checked by halving h (Richardson).
Checked independently by GPU shooting. Compared with the continuous-limit formula used before
(q of the generator, boundaries shifted by 0.5826 sigma).
"""
import os, sys, time
import numpy as np
from scipy.special import erf
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
FIGROOT = os.path.dirname(os.path.dirname(HERE))                    # examples/figures_by_potential
sys.path.insert(0, f'{FIGROOT}/_shared/exact_committors')           # exact_tools
import exact_tools as E
import cupy as cp
OUT = f'{HERE}/'; os.makedirs(OUT, exist_ok=True)
log = open(OUT + 'brownian1d_exact.log', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

SYSTEMS = {
    'longmd_flat': dict(gamma=5.0, dt=2e-4, A=-0.1, B=0.3, force=E.flat_force, V=lambda x: 0 * x),
    'longmd_two_bumps': dict(gamma=5.0, dt=2e-4, A=-0.1, B=0.3, force=E.two_bumps_force, V=E.two_bumps_potential),
    'staple_flat_br10': dict(gamma=1.0, dt=2e-4, A=-0.1, B=0.8, force=E.flat_force, V=lambda x: 0 * x),
}
KT = 1.0


def solve(sysd, n):
    A, B, dt, g = sysd['A'], sysd['B'], sysd['dt'], sysd['gamma']
    sigma = np.sqrt(2 * KT * dt / g)
    xs = np.linspace(A, B, n); h = xs[1] - xs[0]
    mean = xs + dt / g * sysd['force'](xs)
    Phi = lambda t: 0.5 * (1 + erf((t[None, :] - mean[:, None]) / (sigma * np.sqrt(2))))
    dens = lambda t: np.exp(-0.5 * ((t[None, :] - mean[:, None]) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    H = lambda t: mean[:, None] * Phi(t) - sigma ** 2 * dens(t)            # int^t y N(y) dy
    I0 = Phi(xs[1:]) - Phi(xs[:-1])                                       # (rows, cells)
    I1 = H(xs[1:]) - H(xs[:-1])
    Wm = np.zeros((n, n))
    Wm[:, :-1] += (xs[1:][None, :] * I0 - I1) / h                         # left node of each cell
    Wm[:, 1:] += (I1 - xs[:-1][None, :] * I0) / h                         # right node
    b = 1 - 0.5 * (1 + erf((B - mean) / (sigma * np.sqrt(2))))
    q = np.linalg.solve(np.eye(n) - Wm, b)
    return xs, q, sigma


def continuous_shifted(sysd, x, sigma):
    """The formula used before: generator committor with A, B shifted outwards by 0.5826 sigma."""
    s = 0.5826 * sigma
    g = np.linspace(sysd['A'] - s, sysd['B'] + s, 200001)
    w = np.exp(sysd['V'](g) / KT)
    c = np.concatenate([[0], np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(g))])
    return np.interp(x, g, c / c[-1])


def shoot(sysd, points, n_per, seed=0):
    """GPU shooting; absorbed trajectories are removed at once (so their outcome cannot be overwritten)."""
    A, B, dt, g = sysd['A'], sysd['B'], sysd['dt'], sysd['gamma']
    sigma = np.sqrt(2 * KT * dt / g)
    rs = cp.random.RandomState(seed)
    x = cp.repeat(cp.asarray(points, dtype=cp.float64), n_per)
    out = cp.full(x.size, -1.0); idx = cp.arange(x.size)
    while idx.size:
        x = x + dt / g * sysd['force'](x, cp) + sigma * rs.standard_normal(idx.size)
        a, b = x <= A, x >= B
        out[idx[a]] = 0.0; out[idx[b]] = 1.0
        keep = ~(a | b); x, idx = x[keep], idx[keep]
    p = cp.asnumpy(out).reshape(len(points), n_per).mean(1)
    return p, np.sqrt(p * (1 - p) / n_per)


results = {}
for name, sysd in SYSTEMS.items():
    say(f'===== {name}: gamma {sysd["gamma"]}, dt {sysd["dt"]}, A {sysd["A"]}, B {sysd["B"]} =====')
    sols = {}
    L = sysd['B'] - sysd['A']
    sigma = np.sqrt(2 * KT * sysd['dt'] / sysd['gamma'])
    ns = [int(round(L / (sigma / k))) + 1 for k in (10, 20, 40)]          # h = sigma/10, /20, /40
    for n in ns:
        t0 = time.time(); xs, q, sigma = solve(sysd, n); sols[n] = (xs, q)
        say(f'  n = {n} (h = sigma/{(n - 1) * sigma / L:.0f}): {time.time() - t0:.1f} s')
    xe = np.linspace(sysd['A'], sysd['B'], 4001)
    ev = {n: np.interp(xe, *sols[n]) for n in ns}
    d1, d2 = ev[ns[1]] - ev[ns[0]], ev[ns[2]] - ev[ns[1]]
    say(f'  change h/10 -> h/20: max {np.abs(d1).max():.1e}; h/20 -> h/40: max {np.abs(d2).max():.1e} '
        f'(ratio {np.abs(d1).max() / max(np.abs(d2).max(), 1e-300):.1f}; 4 = second order)')
    q_rich = ev[ns[2]] + d2 / 3                                            # Richardson, second order
    err = np.abs(d2) / 3
    say(f'  Richardson error estimate of the finest grid: max {err.max():.1e}')
    q_old = continuous_shifted(sysd, xe, sigma)
    say(f'  continuous formula with 0.5826 sigma shift - exact: max {np.abs(q_old - q_rich).max():.2e}, RMS {np.sqrt(np.mean((q_old - q_rich) ** 2)):.2e}')
    pts = list(np.round(sysd['A'] + L * np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]), 4))
    t0 = time.time(); p, se = shoot(sysd, pts, 4_000_000, seed=1)
    qg = np.interp(pts, xe, q_rich)
    say(f'  shooting, 4M trajectories per point ({time.time() - t0:.0f} s):')
    for x0, pp, ss, qq in zip(pts, p, se, qg):
        say(f'    x = {x0:+.4f}: shooting {pp:.5f} +- {ss:.5f} | exact {qq:.5f} | diff {qq - pp:+.5f} ({(qq - pp) / ss:+.1f} sigma)')
    np.savez(OUT + f'q_exact_{name}.npz', x=xe, q=q_rich, err=err, q_continuous_shift=q_old, sigma=sigma, **{k: v for k, v in sysd.items() if not callable(v)})
    results[name] = dict(xe=xe, q=q_rich, err=err, q_old=q_old, pts=pts, p=p, se=se, qg=qg, sysd=sysd)

fig, axes = plt.subplots(len(SYSTEMS), 4, figsize=(22, 4.2 * len(SYSTEMS)))
for row, (name, r) in zip(axes, results.items()):
    sysd = r['sysd']; xg = np.linspace(sysd['A'] - 0.05, sysd['B'] + 0.05, 1000)
    row[0].plot(xg, sysd['V'](xg), 'k'); row[0].set(title=f'{name}: potential = F(x)/kT (Boltzmann)', xlabel='x', ylabel='V / kT')
    for b in (sysd['A'], sysd['B']):
        row[0].axvline(b, color='C3', ls='--')
    row[1].plot(r['xe'], r['q'], 'k', label='exact (discrete chain)'); row[1].plot(r['xe'], r['q_old'], 'C1--', label='continuous + 0.5826σ shift')
    row[1].set(title='committor', xlabel='x', ylabel='q'); row[1].legend(frameon=False)
    row[2].plot(r['xe'], (r['q_old'] - r['q']) * 1e3, 'C1', label='continuous + shift − exact')
    row[2].plot(r['xe'], r['err'] * 1e3, 'k:', label='error estimate of exact (Richardson)')
    row[2].set(title='differences', xlabel='x', ylabel='×10⁻³'); row[2].legend(frameon=False)
    row[3].errorbar(r['pts'], (r['qg'] - r['p']) * 1e4, yerr=2e4 * r['se'], fmt='o', capsize=3)
    row[3].axhline(0, color='k', lw=0.8); row[3].set(title='exact − shooting (4M traj./point, ±2 s.e.)', xlabel='x', ylabel='×10⁻⁴')
fig.tight_layout(); fig.savefig(OUT + 'brownian1d_exact.png', dpi=100)
say('ALL DONE')
