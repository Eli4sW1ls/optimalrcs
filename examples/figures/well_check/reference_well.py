"""Brute-force committor q(x, v) for sim_istarwell2rv: CosWellWalls, underdamped Langevin (engine.py integrator)."""
import os, numpy as np, cupy as cp
OUT = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(f"{OUT}/reference.npz"):
    print("reference exists", flush=True); raise SystemExit
dt, gamma, kT, m = 0.005, 1.0, 1.0, 1.0
la, lb = -0.35, 0.35
gdt = gamma * dt; c0 = np.exp(-gdt); c1 = (1 - c0) / gdt; c2 = (1 - c1) / gdt
a1, a2 = c1 * dt, c2 * dt ** 2 / m
b1, b2 = (c1 - c2) * dt / m, c2 * dt / m
s_rr = (dt * kT / (gamma * m)) * (2. - (3. - 4. * c0 + c0 ** 2) / gdt)
s_vv = (1.0 - c0 ** 2) * kT / m
s_rv = (kT / (m * gamma)) * (1.0 - c0) ** 2
L = np.linalg.cholesky(np.array([[s_rr, s_rv], [s_rv, s_vv]]))
L00, L10, L11 = float(L[0, 0]), float(L[1, 0]), float(L[1, 1])

def force(x):
    """CosWellWalls force (default parameters) inside [-0.35, 0.35]."""
    f = cp.zeros_like(x)
    mid = (x >= -0.25) & (x <= 0.25)
    f = cp.where(mid, cp.pi * 1.2 / 0.5 * cp.sin(2 * cp.pi * (x + 0.25) / 0.5), f)
    left = (x < -0.25) & (x >= -0.35)
    f = cp.where(left, -0.15 * cp.pi / 0.2 * cp.sin(cp.pi * (x + 0.35) / 0.1), f)
    right = (x > 0.25) & (x <= 0.35)
    f = cp.where(right, -0.15 * cp.pi / 0.2 * cp.sin(cp.pi * (x - 0.35) / 0.1), f)
    return f

xs = np.linspace(la + 1e-3, lb - 1e-3, 29)
vs = np.linspace(-3.5, 3.5, 29)
n_rep = 600
X, V = np.meshgrid(xs, vs, indexing="ij")
x = cp.asarray(np.repeat(X.ravel(), n_rep)); v = cp.asarray(np.repeat(V.ravel(), n_rep))
outcome = cp.full(x.shape, -1.0)
alive = cp.arange(x.size)
for step in range(2_000_000):
    z0 = cp.random.standard_normal(alive.size); z1 = cp.random.standard_normal(alive.size)
    xa, va = x[alive], v[alive]
    fa = force(xa)
    xn = xa + a1 * va + a2 * fa + L00 * z0
    vn = c0 * va + b1 * fa + L10 * z0 + L11 * z1 + b2 * force(xn)
    x[alive], v[alive] = xn, vn
    outcome[alive[xn <= la]] = 0.0; outcome[alive[xn >= lb]] = 1.0
    if step % 100 == 0:
        alive = alive[outcome[alive] < 0]
        if alive.size == 0:
            break
q = cp.asnumpy(outcome).reshape(len(xs), len(vs), n_rep)
np.savez(f"{OUT}/reference.npz", xs=xs, vs=vs, q=q.mean(2), se=q.std(2) / np.sqrt(n_rep))
print(f"reference done after {step} steps; q(x, v=0):", np.round(q.mean(2)[::4, 14], 2), flush=True)
