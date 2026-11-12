"""A->B transition rate of the Brownian Z potential from many independent GPU walkers, as a
fit-free check of the reference (committor_reference.py: ~4.4e7 steps per A->B transition) against
the 16-seed long MD (37 A->B transitions in 1.06e9 steps, i.e. ~2.9e7 steps per transition).
Same overdamped update as the toytis ndLangevinEngine (x += dt/gamma F + N(0, sqrt(2 kT dt/gamma))),
force vectorized as in committor_reference.py and checked against the potential class.
States as in the MD analysis: A: y <= 0.1, B: y >= 0.85, checked every STRIDE = 20 steps.
    python rate_walkers.py N_WALKERS N_STEPS
"""
import sys, time, numpy as np, cupy as cp
TOY = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis'
sys.path.insert(0, TOY)
from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
pot = RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)
KT, GAMMA, DT = 0.05, 10.0, 0.002
SIGMA = np.sqrt(2 * KT * DT / GAMMA)
LA, LB, STRIDE = 0.1, 0.85, 20
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
BURN = 50_000


def force(y, x):
    """Force of the 45-degree sjoelbak (from committor_reference.py), returns (fy, fx)."""
    fx = cp.where(x < 0, -2000 * x, cp.where(x > 0.3, -2000 * (x - 0.3), 0.0))
    fy = cp.where(y < 0, -2000 * y, cp.where(y > 0.9, -2000 * (y - 0.9), 0.0))
    bb = 0.45 - (-1.0) * 0.15
    Dl = cp.abs(-1.0 * x - y + bb) / np.sqrt(2.0); sgn = cp.sign(y - (-1.0) * x - bb)
    on = Dl < 0.2 / 4
    dVdD = -2 * np.pi * 0.3 / 0.2 * cp.sin(2 * np.pi * Dl / 0.2)
    fx = fx + cp.where(on, sgn * dVdD * (-1.0 / np.sqrt(2.0)), 0.0)
    fy = fy + cp.where(on, sgn * dVdD * (-1.0 / np.sqrt(2.0)), 0.0)
    for yc in (0.025, 0.875):
        Dw = cp.abs(y - yc); onw = Dw < 0.2 / 4
        fy = fy + cp.where(onw, -0.15 * 2 * np.pi / 0.2 * cp.sin(2 * np.pi * Dw / 0.2) * cp.sign(y - yc), 0.0)
    fy = fy - 0.2
    fy = fy + cp.where((y >= 0.1) & (y <= 0.825), -0.2 / (0.825 - 0.1), 0.0)
    return fy, fx


# check the force against the class on random points inside the box
rng = np.random.default_rng(1)
for yy, xx in zip(rng.uniform(-0.02, 0.92, 400), rng.uniform(-0.02, 0.32, 400)):
    fy_c, fx_c = force(cp.asarray([yy]), cp.asarray([xx]))
    _, f_ref = pot.potential_and_force((np.array([yy, xx]), np.zeros(2)))
    assert np.allclose([float(fy_c[0]), float(fx_c[0])], f_ref, atol=1e-9), (yy, xx, f_ref)
print('force checked at 400 random points', flush=True)

cp.random.seed(0)
Y = cp.full(NW, 0.03) + 0.005 * cp.random.standard_normal(NW)
X = cp.full(NW, 0.15) + 0.05 * cp.random.standard_normal(NW)
state = cp.zeros(NW, dtype=cp.int8)            # last visited state: 0 = A, 1 = B
n_ab = n_ba = 0; t0 = time.time()
for step in range(1, BURN + NS + 1):
    fy, fx = force(Y, X)
    Y = Y + DT / GAMMA * fy + SIGMA * cp.random.standard_normal(NW)
    X = X + DT / GAMMA * fx + SIGMA * cp.random.standard_normal(NW)
    if step % STRIDE == 0:
        inA, inB = Y <= LA, Y >= LB
        ab = inB & (state == 0); ba = inA & (state == 1)
        if step > BURN:
            n_ab += int(ab.sum()); n_ba += int(ba.sum())
        state = cp.where(inB, 1, cp.where(inA, 0, state)).astype(cp.int8)
    if step % 200_000 == 0 and step > BURN:
        tot = NW * (step - BURN)
        print(f'{step - BURN} steps/walker, {tot:.2e} walker-steps: {n_ab} A->B, {n_ba} B->A -> '
              f'{tot / max(n_ab, 1):.3e} steps per A->B ({time.time() - t0:.0f} s)', flush=True)
tot = NW * NS
print(f'FINAL: {tot:.3e} walker-steps, {n_ab} A->B (+- {np.sqrt(n_ab):.0f}), {n_ba} B->A: '
      f'{tot / n_ab:.3e} steps per A->B transition (reference ~4.4e7, long MD ~2.9e7)', flush=True)
