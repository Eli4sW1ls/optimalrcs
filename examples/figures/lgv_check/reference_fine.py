"""Brute-force committor q(x, v) for sim_istar0lgv15intf on a grid refined near A (engine.py integrator, flat region)."""
import os, numpy as np, cupy as cp
OUT = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(f"{OUT}/reference_fine.npz"):
    print("reference exists, skipping", flush=True); raise SystemExit
dt, gamma, kT, m = 2e-4, 1.0, 1.0, 1.0
la, lb = -0.1, 1.3
gdt = gamma * dt; c0 = np.exp(-gdt); c1 = (1 - c0) / gdt
a1 = c1 * dt
s_rr = (dt * kT / (gamma * m)) * (2. - (3. - 4. * c0 + c0 ** 2) / gdt)
s_vv = (1.0 - c0 ** 2) * kT / m
s_rv = (kT / (m * gamma)) * (1.0 - c0) ** 2
L = np.linalg.cholesky(np.array([[s_rr, s_rv], [s_rv, s_vv]]))
L00, L10, L11 = float(L[0, 0]), float(L[1, 0]), float(L[1, 1])
xs = np.r_[np.linspace(la + 5e-4, 0.4, 21), np.linspace(0.45, lb - 5e-4, 18)]
vs = np.linspace(-3.5, 3.5, 29)
n_rep = 600
X, V = np.meshgrid(xs, vs, indexing="ij")
x = cp.asarray(np.repeat(X.ravel(), n_rep)); v = cp.asarray(np.repeat(V.ravel(), n_rep))
outcome = cp.full(x.shape, -1.0)
alive = cp.arange(x.size)
for step in range(1_000_000):
    z0 = cp.random.standard_normal(alive.size); z1 = cp.random.standard_normal(alive.size)
    xa, va = x[alive], v[alive]
    xa = xa + a1 * va + L00 * z0                 # force is zero between the walls
    va = c0 * va + L10 * z0 + L11 * z1
    x[alive], v[alive] = xa, va
    outcome[alive[xa <= la]] = 0.0; outcome[alive[xa >= lb]] = 1.0
    if step % 200 == 0:
        alive = alive[outcome[alive] < 0]
        if alive.size == 0:
            break
q = cp.asnumpy(outcome).reshape(len(xs), len(vs), n_rep)
assert (q >= 0).all()
np.savez(f"{OUT}/reference_fine.npz", xs=xs, vs=vs, q=q.mean(2), se=q.std(2) / np.sqrt(n_rep))
print(f"reference done after {step} steps", flush=True)
