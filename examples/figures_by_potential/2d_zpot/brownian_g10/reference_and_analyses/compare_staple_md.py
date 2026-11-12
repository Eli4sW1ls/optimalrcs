"""Committor of the Brownian Z potential on the potential: staple NPNEq (forward pieces only, and
forward + mirrored pieces) vs long-MD NPq vs exact.

- staple, forward only: figures/istarz_br_g10/committor.npz (3000 paths per ensemble, rejection weights)
- staple, forward + mirrored: figures/istarz_br_g10_reversed/committor.npz (--reversed, 1500 paths per
  ensemble); q from r_continued.npy if the longer run has written it, else the 10000-iteration fit
- MD: npq_md/npq_md.npz (16 seeds, A: y <= 0.1, B: y >= 0.85, one row per 20 steps)
- exact: committor_reference.py, solved for each data set's own A, B and saving stride
Maps show the mean q per (y, x) bin with at least MIN_N frames.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(HERE)
MIN_N = 20
ref_st = np.load(f"{HERE}/committor_reference_h0.001.npz")                     # A 0.05, B 0.85, stride 1
ref_md = np.load(f"{HERE}/committor_reference_h0.001_A0.1_B0.85_s20.npz")      # A 0.1, B 0.85, stride 20

def interp(ref, y, x):
    f = RegularGridInterpolator((ref["y"], ref["x"]), ref["q"], bounds_error=False, fill_value=None)
    return np.clip(f(np.c_[y, x]), 0, 1)

ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
cy, cx = (ey[1:] + ey[:-1]) / 2, (ex[1:] + ex[:-1]) / 2
def bin_mean(y, x, v, m):
    s = np.histogram2d(y[m], x[m], bins=(ey, ex), weights=v[m])[0]
    n = np.histogram2d(y[m], x[m], bins=(ey, ex))[0]
    return np.where(n >= MIN_N, s / np.maximum(n, 1), np.nan)

PTS = [(0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]
SHOOT = [0.639, 0.664, 0.698, 0.747]
fits = {}
def add_fit(key, label, y, x, q, free, ref):
    e = interp(ref, y, x)
    pts = [q[free & ((y - a) ** 2 + (x - b) ** 2 < 0.01 ** 2)].mean() for a, b in PTS]
    plateau = free & (e > 0.55) & (e < 0.85)
    k = np.clip(np.digitize(e[free], np.linspace(0, 1, 51)) - 1, 0, 49)
    n = np.bincount(k, minlength=50)
    rows = np.where(n >= MIN_N, np.bincount(k, weights=q[free], minlength=50) / np.maximum(n, 1), np.nan)
    fits[key] = dict(label=label, Q=bin_mean(y, x, q, free), E=bin_mean(y, x, e, free), pts=pts,
                     plateau=np.mean(q[plateau] - e[plateau]), rows=rows)

K = 4      # every K-th frame of the staple fits: enough for per-bin means, and keeps memory low
def staple(path, r_override=None):
    d = np.load(path)
    X = d["X_all"][::K].astype(np.float32); b = (d["boundary0"] | d["boundary1"])[::K]
    r = (np.load(r_override, mmap_mode="r")[::K] if r_override else d["r"][::K]).astype(np.float32)
    d.close()
    return X[:, 0], X[:, 1], r, ~b
add_fit("fwd", "staple, forward pieces only", *staple(f"{FIG}/istarz_br_g10/committor.npz"), ref_st)
r_cont = f"{FIG}/istarz_br_g10_reversed/r_continued.npy"
have_cont = os.path.exists(r_cont)
rev_label = "staple, forward + mirrored" + (" (continued)" if have_cont else " (10000 it.)")
add_fit("rev", rev_label, *staple(f"{FIG}/istarz_br_g10_reversed/committor.npz", r_cont if have_cont else None), ref_st)
md = np.load(f"{HERE}/npq_md/npq_md.npz")
add_fit("md", "long MD NPq (16 seeds)", md["y"], md["x"], md["q_npq"],
        (md["y"] > float(md["lambda_a"])) & (md["y"] < float(md["lambda_b"])), ref_md)

def rms(f):
    D = f["Q"] - f["E"]; m = np.isfinite(D) & (f["E"] > 0.05) & (f["E"] < 0.95)
    return np.sqrt(np.mean(D[m] ** 2))
print("fit                                   RMS per bin  plateau (fit - exact)  q at the shooting points "
      + " ".join(f"{v:.3f}" for v in SHOOT))
for f in fits.values():
    print(f"{f['label']:38s} {rms(f):.4f}       {f['plateau']:+.4f}                " + " ".join(f"{v:.3f}" for v in f["pts"]))

gy, gx = np.meshgrid(cy, cx, indexing="ij")
E_grid = interp(ref_st, gy.ravel(), gx.ravel()).reshape(gy.shape)
V = ref_st["V_eff"]; vy, vx = ref_st["y"], ref_st["x"]
inside = (vy >= -0.01) & (vy <= 0.91)
lev = np.linspace(np.percentile(V[inside], 1), np.percentile(V[inside], 92), 14)
ext = [ey[0], ey[-1], ex[0], ex[-1]]
F, R, M_ = fits["fwd"], fits["rev"], fits["md"]
panels = [
    (F["Q"], F["label"], "coolwarm", 1), (R["Q"], R["label"], "coolwarm", 1),
    (M_["Q"], M_["label"], "coolwarm", 1), (E_grid, "exact committor (A: y ≤ 0.05)", "coolwarm", 1),
    (F["Q"] - F["E"], f"forward only − exact (RMS {rms(F):.3f})", "PuOr", 0.15),
    (R["Q"] - R["E"], f"forward + mirrored − exact (RMS {rms(R):.3f})", "PuOr", 0.15),
    (M_["Q"] - M_["E"], f"MD NPq − exact (RMS {rms(M_):.3f})", "PuOr", 0.15),
    (R["Q"] - F["Q"], "forward + mirrored − forward only", "PuOr", 0.15),
]
fig, axes = plt.subplots(2, 4, figsize=(26, 8))
for ax, (M, title, cmap, vmax) in zip(axes.ravel(), panels):
    vmin = 0 if vmax == 1 else -vmax
    im = ax.imshow(M.T, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.contour(vy[inside], vx, V[inside].T, levels=lev, colors="0.4", linewidths=0.5)
    if vmax == 1:
        ax.contour(cy, cx, np.nan_to_num(M, nan=0.5).T, levels=[0.1, 0.5, 0.9], colors="k", linewidths=1, linestyles="--")
    ax.set(title=title, xlabel="order parameter y", ylabel="x", xlim=(0.04, 0.86), ylim=(-0.01, 0.31))
    fig.colorbar(im, ax=ax)
fig.suptitle("Brownian Z potential (γ = 10, kT = 0.05): committor on the potential, mean per bin "
             f"(≥ {MIN_N} frames; each fit compared with the exact q for its own A, B and saving stride)")
fig.tight_layout(); fig.savefig(f"{HERE}/compare_staple_md_map.png", dpi=110)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
mid = (np.linspace(0, 1, 51)[1:] + np.linspace(0, 1, 51)[:-1]) / 2
for f in fits.values():
    a1.plot(mid, f["rows"], "o-", ms=3, label=f["label"])
a1.plot([0, 1], [0, 1], "k--", lw=1, label="exact")
a1.set(xlabel="exact q", ylabel="fitted q (mean per bin of exact q)", title="per-frame agreement"); a1.legend(frameon=False); a1.grid(alpha=0.3)
xs = np.arange(len(PTS))
a2.errorbar(xs, SHOOT, yerr=[0.014, 0.014, 0.013, 0.013], fmt="ks", capsize=4, label="engine shooting (1200 per point)")
for k, f in enumerate(fits.values()):
    a2.plot(xs + 0.1 * (k + 1), f["pts"], "o", label=f["label"])
a2.set(xticks=xs, xticklabels=[f"({a:.2f}, {b:.2f})" for a, b in PTS], xlabel="point (y, x) on the plateau", ylabel="q",
       title="plateau points"); a2.legend(frameon=False, fontsize=9); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{HERE}/compare_staple_md_rows.png", dpi=110)
print("saved compare_staple_md_map.png, compare_staple_md_rows.png")
