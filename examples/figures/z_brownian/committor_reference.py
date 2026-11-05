"""Exact committor of the Brownian Z (angled sjoelbak, 45 deg, Lx = 0.3) potential.

Solves div(exp(-V_eff/kT) grad q) = 0 by finite volumes; V_eff = potential + gravity * y
(the force the toytis engine integrates). q = 0 for y <= lambda_A - s, q = 1 for y >= lambda_B + s,
with s = 0.5826 sigma the shift for absorption checked once per step (sigma = sqrt(2 kT dt / gamma)).
Also the A->B rate from transition path theory and a GPU shooting check at a few points.
Coordinates: y is the order parameter (channel axis), x the transverse coordinate.
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TOY = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis"
sys.path.insert(0, TOY)
from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
OUT = os.path.dirname(os.path.abspath(__file__))

KT, GAMMA, DT = 0.05, 10.0, 0.002
LA = float(os.environ.get('LAMBDA_A', 0.05))
LB = float(os.environ.get('LAMBDA_B', 0.85))
STRIDE = int(os.environ.get('STRIDE', 1))          # MD steps between the frames A/B are checked on
SIGMA = np.sqrt(2 * KT * DT / GAMMA)
S = 0.5826 * SIGMA * np.sqrt(STRIDE)
TAG = '' if (LA, LB, STRIDE) == (0.05, 0.85, 1) else f'_A{LA}_B{LB}_s{STRIDE}'

pot = RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)

H = float(os.environ.get('GRID_H', 0.002))
ys = np.arange(-0.04 + H / 2, 0.94, H)            # order parameter
xs = np.arange(-0.04 + H / 2, 0.34, H)            # transverse
V = np.array([[pot.effective_potential((np.array([y, x]), np.zeros(2))) for x in xs] for y in ys])  # V[iy, ix]
beta_v = V / KT
beta_v -= beta_v.min()
rho = np.exp(-beta_v)                              # unnormalized Boltzmann weight per cell

ny, nx = len(ys), len(xs)
idx = np.arange(ny * nx).reshape(ny, nx)
fixed0 = (ys <= LA - S)[:, None] & np.ones(nx, bool)
fixed1 = (ys >= LB + S)[:, None] & np.ones(nx, bool)
free = ~(fixed0 | fixed1)

rows, cols, vals = [], [], []
b = np.zeros(ny * nx)
diag = np.zeros(ny * nx)
def add_edges(i_from, i_to, w):
    """Edge weights w between cells i_from and i_to (both directions)."""
    for a_, c_ in ((i_from, i_to), (i_to, i_from)):
        rows.append(a_); cols.append(c_); vals.append(w)
        np.add.at(diag, a_, -w)
w_y = np.exp(-(beta_v[1:, :] + beta_v[:-1, :]) / 2)   # between (iy, ix) and (iy+1, ix)
w_x = np.exp(-(beta_v[:, 1:] + beta_v[:, :-1]) / 2)   # between (iy, ix) and (iy, ix+1)
add_edges(idx[:-1, :].ravel(), idx[1:, :].ravel(), w_y.ravel())
add_edges(idx[:, :-1].ravel(), idx[:, 1:].ravel(), w_x.ravel())
Lmat = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                     shape=(ny * nx, ny * nx)).tocsr() + sp.diags(diag)
qfix = np.zeros(ny * nx); qfix[fixed1.ravel()] = 1.0
F = np.flatnonzero(free.ravel()); C = np.flatnonzero(~free.ravel())
A_ff = Lmat[F][:, F]; rhs = -Lmat[F][:, C] @ qfix[C]
q = qfix.copy(); q[F] = spla.spsolve(A_ff.tocsc(), rhs)
Q = q.reshape(ny, nx)

# Transition path theory rate: k_AB = D sum rho |grad q|^2 / sum rho (1 - q) over the domain
D_coef = KT / GAMMA
grad2 = np.zeros_like(Q)
grad2[:-1, :] += w_y * (Q[1:, :] - Q[:-1, :]) ** 2 / H ** 2 / 2
grad2[1:, :] += w_y * (Q[1:, :] - Q[:-1, :]) ** 2 / H ** 2 / 2
grad2[:, :-1] += w_x * (Q[:, 1:] - Q[:, :-1]) ** 2 / H ** 2 / 2
grad2[:, 1:] += w_x * (Q[:, 1:] - Q[:, :-1]) ** 2 / H ** 2 / 2
flux = D_coef * grad2.sum()
pi_a = (rho * (1 - Q)).sum()
k_ab = flux / pi_a
steps_per_transition = 1 / (k_ab * DT)
print(f"grid {ny} x {nx}; Brownian step sigma {SIGMA:.4f}, boundary shift {S:.4f}")
print(f"TPT rate A->B: {k_ab:.3e} per time unit -> mean {steps_per_transition:.3e} steps "
      f"(dt {DT}, gamma {GAMMA}) per A->B transition in a long MD", flush=True)
np.savez(f"{OUT}/committor_reference_h{H}{TAG}.npz", y=ys, x=xs, q=Q, V_eff=V, kT=KT, gamma=GAMMA, dt=DT,
         lambda_a=LA, lambda_b=LB, stride=STRIDE, shift=S, k_ab=k_ab)

# GPU shooting check with the engine's Brownian update x += dt/gamma F + N(0, sigma)
try:
    if os.environ.get('SKIP_SHOOTING'): raise ImportError
    import cupy as cp
    pts = [(0.30, 0.15), (0.40, 0.05), (0.45, 0.15), (0.50, 0.25), (0.55, 0.10), (0.62, 0.15)]
    n_rep = 4000
    Y = cp.asarray(np.repeat([p[0] for p in pts], n_rep)); X = cp.asarray(np.repeat([p[1] for p in pts], n_rep))
    out = cp.full(Y.shape, -1.0); alive = cp.arange(Y.size)
    p = pot
    a = p.a if hasattr(p, "a") else -np.tan(np.radians(45))
    def force(y, x):
        """Vectorized force of the 45-degree sjoelbak (identical to sjoelbak.py), returns (fy, fx)."""
        fx = cp.where(x < 0, -2000 * x, cp.where(x > 0.3, -2000 * (x - 0.3), 0.0))
        fy = cp.where(y < 0, -2000 * y, cp.where(y > 0.9, -2000 * (y - 0.9), 0.0))
        bb = 0.45 - (-1.0) * 0.15                         # line y = -x + 0.6
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
    # check the vectorized force against the potential class at the start points
    for (y0, x0) in pts:
        fy_c, fx_c = force(cp.asarray([y0]), cp.asarray([x0]))
        _, f_ref = pot.potential_and_force((np.array([y0, x0]), np.zeros(2)))
        assert np.allclose([float(fy_c[0]), float(fx_c[0])], f_ref, atol=1e-9), (y0, x0, f_ref)
    for step in range(5_000_000):
        ya, xa = Y[alive], X[alive]
        fy, fx = force(ya, xa)
        ya = ya + DT / GAMMA * fy + SIGMA * cp.random.standard_normal(ya.size)
        xa = xa + DT / GAMMA * fx + SIGMA * cp.random.standard_normal(xa.size)
        Y[alive], X[alive] = ya, xa
        out[alive[ya <= LA]] = 0.0; out[alive[ya >= LB]] = 1.0
        if step % 500 == 0:
            alive = alive[out[alive] < 0]
            if alive.size == 0:
                break
    res = cp.asnumpy(out).reshape(len(pts), n_rep)
    for (y0, x0), r in zip(pts, res):
        iy, ix = np.argmin(np.abs(ys - y0)), np.argmin(np.abs(xs - x0))
        done = r >= 0
        print(f"  check at (y={y0:.2f}, x={x0:.2f}): shooting q = {r[done].mean():.3f} +- "
              f"{r[done].std() / np.sqrt(done.sum()):.3f} ({done.sum()} absorbed) | PDE q = {Q[iy, ix]:.3f}", flush=True)
except ImportError:
    print("CuPy not available, skipping the shooting check")

fig, ax = plt.subplots(figsize=(10, 4.2))
inside = (ys >= -0.01) & (ys <= 0.91)
ext = [ys[inside][0], ys[inside][-1], xs[0], xs[-1]]
im = ax.imshow(Q[inside].T, origin="lower", extent=ext, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
vv = V[inside]; lev = np.linspace(np.percentile(vv, 1), np.percentile(vv, 92), 14)
ax.contour(ys[inside], xs, vv.T, levels=lev, colors="0.3", linewidths=0.6)
ax.contour(ys[inside], xs, Q[inside].T, levels=[0.1, 0.25, 0.5, 0.75, 0.9], colors="k", linewidths=1.2, linestyles="--")
for lam in (LA, LB):
    ax.axvline(lam, color="k", lw=0.8)
ax.set(xlabel="order parameter y", ylabel="x", xlim=(-0.01, 0.91), ylim=(-0.01, 0.31),
       title="Brownian Z potential: exact committor (colour, dashed = 0.1/0.25/0.5/0.75/0.9), V_eff contours (grey)")
fig.colorbar(im, ax=ax, label="q")
fig.tight_layout(); fig.savefig(f"{OUT}/committor_reference_h{H}{TAG}.png", dpi=140)
for (y0, x0) in [(0.45, 0.15), (0.50, 0.25), (0.62, 0.15)]:
    print(f"  PDE q at (y={y0}, x={x0}): {Q[np.argmin(np.abs(ys - y0)), np.argmin(np.abs(xs - x0))]:.3f}", flush=True)
print("ALL DONE", flush=True)
