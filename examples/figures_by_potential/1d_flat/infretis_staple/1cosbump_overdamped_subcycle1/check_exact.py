"""infretis_staple_1d.ipynb on 1cosbump_overdamped_subcycle1: the reference was wrong, and the two weightings are the same.

1. The run is a plain RETIS run (mode retis, infinite swap, shooting 'sh' only): path_w is 1 for every occupied
   (path, ensemble) pair, so 'staple' (path_f / path_w) and 'rejections' (path_f) are the same numbers.
2. The notebook compared the fit with a LINEAR exact committor, which holds for a flat potential. This run has a cosine
   bump (deltaf = 1 kT) between -0.1 and 0.1, so the committor is steepest at the top of the bump.
Here: the exact committor of the discrete chain the ASE engine runs (x' = x + mu F(x) + sigma xi, mu = dt/(m gamma),
sigma^2 = 2 kT mu; A: x <= -0.1, B: x >= 0.1 checked every step), checked by GPU shooting, and the notebook's fit
repeated and compared with it.

    python check_exact.py          (needs the GPU; ~6 min)
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp
from ase import units

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REPO = os.path.dirname(EXAMPLES)
sys.path.insert(0, f"{REPO}/optimalrcs-cupy/")
sys.path.insert(0, EXAMPLES)
sys.path.insert(0, os.path.join(EXAMPLES, "figures_by_potential", "_shared", "exact_committors"))
import optimalrcs
import tis_data
import seed_runs
import exact_tools as E

plt.show = lambda *a, **k: None
TIS_DIR = os.path.join(EXAMPLES, "data", "1cosbump_overdamped_subcycle1")
H5 = os.path.join(TIS_DIR, "paths.h5")
OUT = HERE
log = open(os.path.join(OUT, "check_exact.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


# ---- the chain (infretis.toml: timestep 0.00020361 fs-units, friction 0.49113 /fs-units, T = 11604.518 K, mass 1)
DT = 0.00020361011421549486 * units.fs
GAMMA = 0.4911347375119244 / units.fs
KT = units.kB * 11604.518121550082
MASS = 1.0
A, B = -0.1, 0.1
BL, BR, DF = -0.1, 0.1, 1.0                       # bump: bleft, bright, deltaf (kT)
L = BR - BL
SIGMA = np.sqrt(2 * KT * DT / (MASS * GAMMA)); MU = DT / (MASS * GAMMA)
say(f"dt {DT:.3e}, gamma {GAMMA:.4f}, kT {KT:.6f}: mu {MU:.3e}, sigma {SIGMA:.6f}; bump {DF} kT on [{BL}, {BR}]")


def force(x, xp=np):
    inb = (x >= BL) & (x <= BR)
    return xp.where(inb, -DF * np.pi / L * xp.sin(2 * np.pi * (x - BL) / L), 0.0)


def potential(x):
    return np.where((x >= BL) & (x <= BR), DF / 2 * (1 - np.cos(2 * np.pi * (x - BL) / L)), 0.0)


# ---- 1. exact committor of the chain, h = sigma/10, /20, /40 (Richardson)
ns = [int(round((B - A) / (SIGMA / k))) + 1 for k in (10, 20, 40)]
sols = {}
for n in ns:
    t0 = time.time(); sols[n] = E.solve_brownian_1d(force, DT, GAMMA, KT, A, B, n, m=MASS)
    say(f"  n = {n}: {time.time() - t0:.1f} s")
xe = np.linspace(A, B, 4001)
ev = {n: np.interp(xe, *sols[n]) for n in ns}
d1, d2 = ev[ns[1]] - ev[ns[0]], ev[ns[2]] - ev[ns[1]]
q_exact = ev[ns[2]] + d2 / 3
say(f"exact committor: change h/10 -> h/20 max {np.abs(d1).max():.1e}, h/20 -> h/40 max {np.abs(d2).max():.1e}; "
    f"Richardson error estimate {np.abs(d2).max() / 3:.1e}")
# the references the notebook / the continuous theory would give
s_shift = 0.5826 * SIGMA
lin = np.clip((xe - A + s_shift) / (B - A + 2 * s_shift), 0, 1)
g = np.linspace(A - s_shift, B + s_shift, 200001); w_ = np.exp(potential(g) / KT)
cum = np.r_[0, np.cumsum((w_[1:] + w_[:-1]) / 2 * np.diff(g))]
cont = np.interp(xe, g, cum / cum[-1])
say(f"max |linear - exact| = {np.abs(lin - q_exact).max():.4f}  (RMS {np.sqrt(np.mean((lin - q_exact) ** 2)):.4f}); "
    f"max |continuous+shift - exact| = {np.abs(cont - q_exact).max():.4f} (RMS {np.sqrt(np.mean((cont - q_exact) ** 2)):.4f})")

# ---- 2. shooting check of the exact committor
pts = np.array([-0.09, -0.05, -0.02, 0.0, 0.02, 0.05, 0.09])
t0 = time.time()
rs = cp.random.RandomState(3)
n_shots = 2_000_000
x = cp.repeat(cp.asarray(pts), n_shots); idx = cp.arange(x.size); out = cp.full(x.size, -1.0)
while idx.size:
    x = x + MU * force(x, cp) + SIGMA * rs.standard_normal(idx.size)
    a, b = x <= A, x >= B
    out[idx[a]] = 0.0; out[idx[b]] = 1.0
    k = ~(a | b); x, idx = x[k], idx[k]
p = cp.asnumpy(out).reshape(len(pts), n_shots).mean(1); se = np.sqrt(p * (1 - p) / n_shots)
qg = np.interp(pts, xe, q_exact)
say(f"shooting, {n_shots} trajectories per point ({time.time() - t0:.0f} s):")
for x0, pp, ss, qq in zip(pts, p, se, qg):
    say(f"  x = {x0:+.3f}: shooting {pp:.5f} +- {ss:.5f} | exact {qq:.5f} | diff {qq - pp:+.5f} ({(qq - pp) / ss:+.1f} s.e.)")
say(f"  chi^2 = {np.sum(((qg - p) / se) ** 2):.1f} for {len(pts)} points")

# ---- 3. the notebook's fit again, compared with the right reference
res = {}
for weighting in ("staple", "rejections"):
    parser = argparse.ArgumentParser(); tis_data.add_data_arguments(parser)
    a = parser.parse_args([TIS_DIR, "--burn-in-fraction", "0.4", "--thin", "1", "--weights", weighting, "--h5", H5,
                           "--lambda-a", str(A), "--lambda-b", str(B), "--paths-per-ensemble", "5000"])
    data = tis_data.prepare_data(a)
    res[weighting] = data
d_s, d_r = res["staple"], res["rejections"]
say(f"staple vs rejections: path_weights identical {np.array_equal(d_s.path_weights, d_r.path_weights)}, "
    f"frames identical {np.array_equal(d_s.lam, d_r.lam)}")
data = d_s
fr = ~(data.boundary0 | data.boundary1)
qe_frames = np.interp(data.lam, xe, q_exact); lin_frames = np.interp(data.lam, xe, lin)

FIT_KW = dict(ny=6, print_step=500, lazy_basis=True, save_min_metric="max_z_zq_stopped",
              metrics_print=("iter", "delta_r2", "max_z_zq_stopped", "max_sd_zq_stopped", "delta_x", "time_elapsed"))
X_gpu = cp.asarray(data.X)
t0 = time.time()
results, q = seed_runs.run_seeds(
    optimalrcs,
    lambda: (lambda qq: (qq.set_fixed_traj_length_trap(data.boundary0 | data.boundary1, 2 ** 16), qq)[1])(
        optimalrcs.CommittorNE(boundary0=data.boundary0, boundary1=data.boundary1, i_traj=data.i_traj,
                               t_traj=data.t_traj, path_weights=data.path_weights, group_traj=data.group_traj)),
    comp_y=lambda: X_gpu[:, np.random.randint(X_gpu.shape[1])], fit_kw=FIT_KW, seeds=[0], max_iter=2000, chunk=1000,
    out_dir=os.path.join(OUT, "refit"), envelope=0.01, gamma=0.1, warmup_iter=1000, warmup_envelope=1,
    warmup_gamma=0.1, plots=("metrics",), plot_options=dict(metrics=["delta_r2", "max_z_zq_stopped", "delta_x"]))
r = results[0]["r"]
say(f"fit: {time.time() - t0:.0f} s; final zqs_z {float(q.metrics_history['max_z_zq_stopped'][-1]):.3f}")
np.savez(os.path.join(OUT, "refit", "committor.npz"), r=r, lam=data.lam, X_all=data.X_all, i_traj=data.i_traj,
         t_traj=data.t_traj, group_traj=data.group_traj, path_weights=data.path_weights,
         boundary0=data.boundary0, boundary1=data.boundary1, q_exact=qe_frames)

t = fr & (qe_frames > 0.05) & (qe_frames < 0.95)
W = data.path_weights


def rms(a, b):
    return np.sqrt(np.sum(W[t] * (a[t] - b[t]) ** 2) / np.sum(W[t]))


say(f"path-weighted RMS over the frames with 0.05 < q_exact < 0.95: fit - exact {rms(r, qe_frames):.4f}; "
    f"fit - linear (the notebook's reference) {rms(r, lin_frames):.4f}; linear - exact {rms(lin_frames, qe_frames):.4f}")
nb = 80
edges = np.linspace(A, B, nb + 1); mid = (edges[1:] + edges[:-1]) / 2
bi = np.clip(np.digitize(data.lam, edges) - 1, 0, nb - 1)
num = np.bincount(bi[fr], weights=(W * r)[fr], minlength=nb); den = np.bincount(bi[fr], weights=W[fr], minlength=nb)
P = num / np.maximum(den, 1e-300)
qb = np.interp(mid, xe, q_exact); lb = np.interp(mid, xe, lin)
tb = (qb > 0.05) & (qb < 0.95)
say(f"per bin along lambda: RMS fit - exact {np.sqrt(np.mean((P - qb)[tb] ** 2)):.4f}, max |fit - exact| "
    f"{np.abs((P - qb)[tb]).max():.4f}; RMS fit - linear {np.sqrt(np.mean((P - lb)[tb] ** 2)):.4f}")

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(mid, P, label="fit (staple = rejections weights)")
axes[0].plot(xe, q_exact, "k--", label="exact committor of the chain (with the bump)")
axes[0].plot(xe, lin, "C3:", label="linear reference (flat potential), used in the notebook")
axes[0].errorbar(pts, p, yerr=2 * se, fmt="ks", ms=4, capsize=3, label="shooting, ±2 s.e.")
axes[0].set(ylabel="q"); axes[0].legend(frameon=False); axes[0].grid(alpha=0.3)
axes[1].plot(mid, P - qb, label="fit − exact")
axes[1].plot(mid, P - lb, "C3:", label="fit − linear (the notebook's plot)")
axes[1].axhline(0, color="k", lw=0.8); axes[1].set(xlabel="order parameter λ", ylabel="difference"); axes[1].legend(frameon=False)
axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "check_exact.png"), dpi=110)
np.savez(os.path.join(OUT, "q_exact_1cosbump.npz"), x=xe, q=q_exact, err=np.abs(d2) / 3, q_linear=lin, q_continuous_shift=cont,
         sigma=SIGMA, mu=MU, kT=KT, deltaf=DF)
say("ALL DONE")
