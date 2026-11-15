"""Free energy along the committor, F(q) = -kT ln p(q), for the 2D Langevin systems (cc, zpot, stag).

    python lgv2d_fq.py cc|zpot|stag

p(q) is the equilibrium density of q(y, x, v_y, v_x) over the phase points between A and B. The transition region is
rare under the Boltzmann density (stag: ~e^-10 of the population), so N_POINTS phase points are drawn from the
tempered position density exp(-LAMBDA V/kT) (cells of a fine grid, uniform within the cell; Maxwell velocities, which
are exact) and every point is weighted by exp(-(1 - LAMBDA) V/kT) to give the equilibrium p(q). The committor of each is
estimated by N_SHOTS shooting trajectories (lgv2d_tools.shoot, the run's BBK step). Bin errors from the weights:
sqrt(sum w^2)/sum w.
The estimate q-hat has the binomial standard error sqrt(q(1 - q)/N_SHOTS) (<= 0.5/sqrt(N_SHOTS)), which smears the
histogram by about that much: the F(q) bins next to 0 and 1 are the least reliable. F is in kT, shifted so that
its lowest bin is 0; the error bars are the Poisson errors 1/sqrt(counts) of the bins.
Output: <potential>/exact/langevin_g<gamma>/fq.npz and fq.png.
"""
import os
import sys
import time
import zlib

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXAMPLES = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import lgv2d_tools as L

KEY = sys.argv[1]
S = L.SYSTEMS[KEY]
CFG = {"cc": dict(N_POINTS=20000, N_SHOTS=2000, dy=0.005, dx=0.005, V_CUT=14, LAMBDA=0.35),
       "zpot": dict(N_POINTS=8000, N_SHOTS=1000, dy=0.005, dx=0.005, V_CUT=14, LAMBDA=0.35),
       "stag": dict(N_POINTS=8000, N_SHOTS=1000, dy=0.02, dx=0.02, V_CUT=16, LAMBDA=0.35)}[KEY]
if os.environ.get("LGV_SMOKE"):
    CFG.update(N_POINTS=200, N_SHOTS=100)
OUT = os.environ.get("LGV_OUT") or os.path.join(EXAMPLES, "figures_by_potential", S["folder"], "exact",
                                                f"langevin_g{S['gamma']:g}")
os.makedirs(OUT, exist_ok=True)
kT = S["kT"]

# Boltzmann position density between A and B on a fine grid
ey = np.arange(S["A"], S["B"] + 1e-9, CFG["dy"]); ex = np.arange(*S["x_range"], CFG["dx"])
cy, cx = (ey[1:] + ey[:-1]) / 2, (ex[1:] + ex[:-1]) / 2
GY, GX = np.meshgrid(cy, cx, indexing="ij")
V = np.asarray(S["potential"](GY, GX, np)) / kT
V -= V.min()
allowed = V < CFG["V_CUT"]
if KEY == "stag":                      # the forbidden rectangle (V ~ 0 inside, but unreachable behind the walls)
    p_ = L.STAG
    allowed &= ~((GY > p_["xl"] - 0.05) & (GY < p_["xr"] + 0.05) & (GX > p_["yb"] - 0.05) & (GX < p_["yt"] + 0.05))
lam = CFG["LAMBDA"]
w = np.where(allowed, np.exp(-lam * V), 0.0).ravel(); w /= w.sum()          # sampling probability of every cell
wt_cell = np.where(allowed, np.exp(-(1 - lam) * V), 0.0)                    # weight = Boltzmann / sampling probability
rng = np.random.default_rng(zlib.crc32(("fq" + KEY).encode()))
cell = rng.choice(w.size, CFG["N_POINTS"], p=w)
iy, ix = np.unravel_index(cell, GY.shape)
W = wt_cell[iy, ix]; W /= W.sum()
y0 = ey[iy] + CFG["dy"] * rng.random(cell.size); x0 = ex[ix] + CFG["dx"] * rng.random(cell.size)
sv = np.sqrt(kT / S["m"]); vy0 = sv * rng.standard_normal(cell.size); vx0 = sv * rng.standard_normal(cell.size)
print(f"{S['name']}: {CFG['N_POINTS']} phase points x {CFG['N_SHOTS']} shots; cells allowed {allowed.sum()}", flush=True)

f = os.path.join(OUT, "fq.npz")
if os.path.exists(f) and not os.environ.get("LGV_SMOKE"):
    d = np.load(f); q, se = d["q"], d["se"]; W = d["weight"]; y0 = d["y"]; print("loaded", f)
else:
    t0 = time.time()
    q, se, st = L.shoot(S, y0, x0, vy0, vx0, n_shots=CFG["N_SHOTS"], seed=5)
    np.savez(f if not os.environ.get("LGV_SMOKE") else os.path.join(OUT, "fq_smoke.npz"), y=y0, x=x0, vy=vy0, vx=vx0, q=q, se=se,
             steps=st, n_shots=CFG["N_SHOTS"], weight=W, lam=lam,
             note="phase points from exp(-lam V/kT) between A and B; weight = equilibrium weight of each point")
    print(f"shooting: {time.time() - t0:.0f} s; max s.e. {se.max():.4f}", flush=True)

nb = 50
edges = np.linspace(0, 1, nb + 1); mid = (edges[1:] + edges[:-1]) / 2
cnt = np.histogram(q, edges)[0]
mass = np.histogram(q, edges, weights=W)[0]; mass2 = np.histogram(q, edges, weights=W ** 2)[0]
ok = mass > 0
Fq = np.full(nb, np.nan); Fq[ok] = -np.log(mass[ok] * nb)
Fe = np.full(nb, np.nan); Fe[ok] = np.sqrt(mass2[ok]) / mass[ok]
Fq -= np.nanmin(Fq)
# F(y) of the same ensemble, for comparison
cy_ = np.histogram(y0, np.linspace(S["A"], S["B"], 41), weights=W)[0]; my = (np.linspace(S["A"], S["B"], 41)[1:] + np.linspace(S["A"], S["B"], 41)[:-1]) / 2
Fy = np.full(cy_.size, np.nan); Fy[cy_ > 0] = -np.log(cy_[cy_ > 0]); Fy -= np.nanmin(Fy)
barrier = np.nanmax(Fq[(mid > 0.1) & (mid < 0.9)]) if np.isfinite(Fq[(mid > 0.1) & (mid < 0.9)]).any() else np.nan
print(f"F(q): max over 0.1 < q < 0.9 is {barrier:.2f} kT; fraction of phase points with 0.1 < q < 0.9: "
      f"{np.sum(W[(q > 0.1) & (q < 0.9)]):.2e} (equilibrium weight; {np.sum((q > 0.1) & (q < 0.9))} sampled points)", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].errorbar(mid, Fq, yerr=Fe, fmt="o-", ms=3, capsize=2, color="k")
axes[0].set(xlabel="q", ylabel="F(q) / kT", title=f"{S['name']}, Langevin γ = {S['gamma']:g}: free energy along the committor "
            f"({q.size} phase points, {int(CFG['N_SHOTS'])} shots each)", ylim=(-0.3, None))
axes[0].title.set_fontsize(9)
axes[1].plot(my, Fy, "C0-o", ms=3, label="F(y) of the same points")
axes[1].set(xlabel="order parameter y", ylabel="F / kT", title="for comparison: free energy along the order parameter")
axes[1].axhline(barrier, color="k", ls=":", lw=0.8, label=f"F(q) max over 0.1 < q < 0.9 = {barrier:.1f} kT")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fq.png" if not os.environ.get("LGV_SMOKE") else "fq_smoke.png"), dpi=110); plt.close(fig)
print("ALL DONE")
