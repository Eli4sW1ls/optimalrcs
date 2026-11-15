"""Refine F(q) of the Z potential (Langevin): add N_EXTRA phase points in a band around the ridge, where the first batch of
lgv2d_fq.py has few points (q = 0.1-0.6 is a thin region).

Batch 2 is drawn from exp(-LAMBDA V/kT) restricted to |x + y - 0.6| < BAND (cells of the same fine grid, Maxwell
velocities). The two batches form a mixture proposal m = (N1 p1 + N2 p2) / (N1 + N2); every point of both gets the weight
exp(-V/kT) / m(cell), which keeps the equilibrium estimate unbiased. Rewrites fq.npz (both batches, new weights);
then run `python lgv2d_fq.py zpot` to redraw fq.png from it.
"""
import os, sys, time, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
EXAMPLES = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import lgv2d_tools as L

S = L.SYSTEMS["zpot"]; kT = S["kT"]
N_EXTRA, N_SHOTS, BAND, LAM, DY = 10000, 1000, 0.12, 0.35, 0.005
OUT = os.path.join(EXAMPLES, "figures_by_potential", S["folder"], "exact", "langevin_g2")
f = os.path.join(OUT, "fq.npz")
d = dict(np.load(f))
if "n_batches" in d and int(d["n_batches"]) > 1:
    sys.exit("already refined")
N1 = d["q"].size

ey = np.arange(S["A"], S["B"] + 1e-9, DY); ex = np.arange(*S["x_range"], DY)
cy, cx = (ey[1:] + ey[:-1]) / 2, (ex[1:] + ex[:-1]) / 2
GY, GX = np.meshgrid(cy, cx, indexing="ij")
V = np.asarray(S["potential"](GY, GX, np)) / kT; V -= V.min()
allowed = V < 14
band = allowed & (np.abs(GX + GY - 0.6) < BAND)
p1 = np.where(allowed, np.exp(-LAM * V), 0.0); p1 /= p1.sum()
p2 = np.where(band, np.exp(-LAM * V), 0.0); p2 /= p2.sum()
rng = np.random.default_rng(zlib.crc32(b"fq-zpot-refine"))
cell = rng.choice(p2.size, N_EXTRA, p=p2.ravel())
iy, ix = np.unravel_index(cell, GY.shape)
y2 = ey[iy] + DY * rng.random(N_EXTRA); x2 = ex[ix] + DY * rng.random(N_EXTRA)
sv = np.sqrt(kT / S["m"]); vy2 = sv * rng.standard_normal(N_EXTRA); vx2 = sv * rng.standard_normal(N_EXTRA)
print(f"batch 2: {N_EXTRA} points in the ridge band ({band.sum()} cells), {N_SHOTS} shots", flush=True)
t0 = time.time()
q2, se2, st2 = L.shoot(S, y2, x2, vy2, vx2, n_shots=N_SHOTS, seed=77)
print(f"shooting: {time.time() - t0:.0f} s", flush=True)

y = np.r_[d["y"], y2]; x = np.r_[d["x"], x2]
jy = np.clip(((y - S["A"]) / DY).astype(int), 0, GY.shape[0] - 1); jx = np.clip(((x - ex[0]) / DY).astype(int), 0, GY.shape[1] - 1)
mix = (N1 * p1[jy, jx] + N_EXTRA * p2[jy, jx]) / (N1 + N_EXTRA)
W = np.exp(-V[jy, jx]) / mix; W /= W.sum()
out = dict(y=y, x=x, vy=np.r_[d["vy"], vy2], vx=np.r_[d["vx"], vx2], q=np.r_[d["q"], q2], se=np.r_[d["se"], se2],
           steps=np.r_[d["steps"], st2], n_shots=N_SHOTS, weight=W, lam=LAM, n_batches=2,
           note="batch 1: exp(-lam V/kT) between A and B; batch 2: the same in the band |x + y - 0.6| < 0.12; "
                "weight = exp(-V/kT) / mixture density, normalized")
np.savez(f, **out)
inb = (np.abs(x + y - 0.6) < BAND)
print(f"points with 0.1 < q < 0.6: {np.sum((out['q'] > 0.1) & (out['q'] < 0.6))} (batch 1: {np.sum((d['q'] > 0.1) & (d['q'] < 0.6))}); "
      f"effective sample size {1 / np.sum(W ** 2):.0f} of {W.size}", flush=True)
