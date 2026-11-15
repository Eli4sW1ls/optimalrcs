"""Refine F(q) of the staggered channels (Langevin): add N_EXTRA phase points at the two barrier tops, where the first batch
of lgv2d_fq.py has almost none (the committor is a step there, bimodal 0/1 elsewhere).

Batch 2: uniform over the cells of the same fine grid in the bands op in [-0.9, -0.3] (upper channel, transverse > 0) and
op in [0.3, 0.9] (lower channel, transverse < 0), Maxwell velocities. Mixture proposal m = (N1 p1 + N2 p2) / (N1 + N2),
weights exp(-V/kT) / m(cell) (unbiased); rewrites fq.npz, then run `python lgv2d_fq.py stag` to redraw fq.png.
"""
import os, sys, time, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
EXAMPLES = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import lgv2d_tools as L

S = L.SYSTEMS["stag"]; kT = S["kT"]
N_EXTRA, N_SHOTS, LAM, DY, V_CUT = 10000, 1000, 0.35, 0.02, 16
OUT = os.path.join(EXAMPLES, "figures_by_potential", S["folder"], "exact", "langevin_g0.1")
f = os.path.join(OUT, "fq.npz")
d = dict(np.load(f))
if "n_batches" in d and int(d["n_batches"]) > 1:
    sys.exit("already refined")
N1 = d["q"].size

ey = np.arange(S["A"], S["B"] + 1e-9, DY); ex = np.arange(*S["x_range"], DY)
cy, cx = (ey[1:] + ey[:-1]) / 2, (ex[1:] + ex[:-1]) / 2
GY, GX = np.meshgrid(cy, cx, indexing="ij")           # GY = order parameter, GX = transverse
V = np.asarray(S["potential"](GY, GX, np)) / kT; V -= V.min()
P = L.STAG
allowed = (V < V_CUT) & ~((GY > P["xl"] - 0.05) & (GY < P["xr"] + 0.05) & (GX > P["yb"] - 0.05) & (GX < P["yt"] + 0.05))
band = allowed & (((GY > -0.9) & (GY < -0.3) & (GX > 0)) | ((GY > 0.3) & (GY < 0.9) & (GX < 0)))
p1 = np.where(allowed, np.exp(-LAM * V), 0.0); p1 /= p1.sum()
p2 = np.where(band, 1.0, 0.0); p2 /= p2.sum()
rng = np.random.default_rng(zlib.crc32(b"fq-stag-refine"))
cell = rng.choice(p2.size, N_EXTRA, p=p2.ravel())
iy, ix = np.unravel_index(cell, GY.shape)
y2 = ey[iy] + DY * rng.random(N_EXTRA); x2 = ex[ix] + DY * rng.random(N_EXTRA)
sv = np.sqrt(kT / S["m"]); vy2 = sv * rng.standard_normal(N_EXTRA); vx2 = sv * rng.standard_normal(N_EXTRA)
print(f"batch 2: {N_EXTRA} points in the barrier-top bands ({band.sum()} cells), {N_SHOTS} shots", flush=True)
t0 = time.time()
q2, se2, st2 = L.shoot(S, y2, x2, vy2, vx2, n_shots=N_SHOTS, seed=78)
print(f"shooting: {time.time() - t0:.0f} s", flush=True)

y = np.r_[d["y"], y2]; x = np.r_[d["x"], x2]
jy = np.clip(((y - S["A"]) / DY).astype(int), 0, GY.shape[0] - 1); jx = np.clip(((x - ex[0]) / DY).astype(int), 0, GY.shape[1] - 1)
mix = (N1 * p1[jy, jx] + N_EXTRA * p2[jy, jx]) / (N1 + N_EXTRA)
W = np.exp(-V[jy, jx]) / mix; W /= W.sum()
out = dict(y=y, x=x, vy=np.r_[d["vy"], vy2], vx=np.r_[d["vx"], vx2], q=np.r_[d["q"], q2], se=np.r_[d["se"], se2],
           steps=np.r_[d["steps"], st2], n_shots=N_SHOTS, weight=W, lam=LAM, n_batches=2,
           note="batch 1: exp(-lam V/kT) between A and B; batch 2: uniform in the barrier-top bands; "
                "weight = exp(-V/kT) / mixture density, normalized")
np.savez(f, **out)
print(f"points with 0.1 < q < 0.9: {np.sum((out['q'] > 0.1) & (out['q'] < 0.9))} (batch 1: {np.sum((d['q'] > 0.1) & (d['q'] < 0.9))}); "
      f"effective sample size {1 / np.sum(W ** 2):.0f} of {W.size}", flush=True)
