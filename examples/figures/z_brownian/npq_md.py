"""NPq (equilibrium committor, CuPy optimalrcs) on the 16-seed Brownian long MD of the Z potential.

Data: toytis md_run.py output md_zbr_g10_s*/md_path.txt (columns step y x vy vx, one row per 20 steps).
A: y <= LAMBDA_A, B: y >= LAMBDA_B, checked on saved rows. Only the rows outside A/B plus the A/B row
before and after each excursion are kept (steps inside a state carry no information for NPq); every
excursion is its own trajectory segment. CVs: the two positions (y, x).

Fits: all seeds together, then the two halves (seeds 1-8, 9-16) separately as an error estimate.
Compared with the exact committor of committor_reference.py, solved for the same A, B and stride.
Usage: python npq_md.py [n_iter]
"""
import os, sys, glob, time, importlib.util, subprocess
import numpy as np, pandas as pd, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

REPO = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs"
TOY = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis"
sys.path.insert(0, os.path.join(REPO, "optimalrcs-cupy"))
spec = importlib.util.spec_from_file_location("orc", os.path.join(REPO, "optimalrcs-cupy", "optimalrcs.py"))
orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npq_md")
os.makedirs(OUT, exist_ok=True)

FILES = sorted(glob.glob(f"{TOY}/simulations/md_zbr_g10_s*/md_path.txt"),
               key=lambda p: int(p.split("_s")[-1].split("/")[0]))
LAMBDA_A, LAMBDA_B = 0.1, 0.85     # A at 0.1 instead of 0.05: q < 0.003 up to y = 0.3, 3x fewer rows
STRIDE = 20                        # md_run --stride: MD steps per saved row
BURN_IN = 5_000                    # saved rows dropped per seed (start point relaxation)
N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
CHUNK = 5_000_000


def load(path):
    """Kept rows (y, x) and their row indices of one seed, plus a 2D histogram of all rows."""
    ys, xs, idx_all, hist = [], [], [], np.zeros((92, 32))
    pending = None
    start = 0
    for df in pd.read_csv(path, sep=r"\s+", comment="#", header=None, usecols=[1, 2],
                          chunksize=CHUNK, dtype=np.float64):
        rows = df.to_numpy(); idx = np.arange(start, start + len(rows)); start += len(rows)
        sel = idx >= BURN_IN; rows, idx = rows[sel], idx[sel]
        if len(idx) == 0:
            continue
        hist += np.histogram2d(rows[:, 0], rows[:, 1], bins=(92, 32), range=((-0.01, 0.91), (-0.01, 0.31)))[0]
        out = (rows[:, 0] > LAMBDA_A) & (rows[:, 0] < LAMBDA_B)
        left_of_first = False
        if pending is not None:
            rows, idx = np.vstack([pending[0], rows]), np.r_[pending[1], idx]
            left_of_first, out = pending[3], np.r_[pending[2], out]
        keep = out | np.r_[left_of_first, out[:-1]] | np.r_[out[1:], False]
        ys.append(rows[:-1][keep[:-1]]); idx_all.append(idx[:-1][keep[:-1]])
        pending = (rows[-1:], idx[-1:], out[-1:], bool(out[-2]) if len(out) > 1 else left_of_first)
    if pending is not None and (pending[2][0] or pending[3]):
        ys.append(pending[0]); idx_all.append(pending[1])
    return np.concatenate(ys), np.concatenate(idx_all), hist


cache = f"{OUT}/kept_rows_A{LAMBDA_A}_B{LAMBDA_B}.npz"
if os.path.exists(cache):
    d = np.load(cache); Z, row, seed, hist2d = d["Z"], d["row"], d["seed"], d["hist2d"]
    print(f"loaded {cache}", flush=True)
else:
    Z, row, seed, hist2d = [], [], [], 0
    for k, path in enumerate(FILES):
        t0 = time.time()
        z, r, h = load(path)
        Z.append(z); row.append(r); seed.append(np.full(len(r), k + 1)); hist2d = hist2d + h
        print(f"seed {k + 1}: {int(h.sum())} rows after burn-in, kept {len(r)} ({time.time() - t0:.0f} s)", flush=True)
    Z, row, seed = np.concatenate(Z), np.concatenate(row), np.concatenate(seed)
    np.savez(cache, Z=Z, row=row, seed=seed, hist2d=hist2d)

y, x = Z[:, 0], Z[:, 1]
in_a, in_b = y <= LAMBDA_A, y >= LAMBDA_B
new_seg = np.r_[True, (seed[1:] != seed[:-1]) | (np.diff(row) != 1)]
i_traj = np.cumsum(new_seg)
# A->B and B->A transitions: consecutive A/B visits within a seed that differ
ab = np.flatnonzero(in_a | in_b)
switch = (seed[ab][1:] == seed[ab][:-1]) & (in_b[ab][1:] != in_b[ab][:-1])
print(f"kept rows {len(y)}, segments {i_traj[-1]}, rows in A {in_a.sum()}, in B {in_b.sum()}, "
      f"A<->B transitions {switch.sum()}, total rows after burn-in {int(hist2d.sum())}", flush=True)

# exact committor for the same A, B and absorption checked on saved rows
ref_file = f"{os.path.dirname(OUT)}/committor_reference_h0.001_A{LAMBDA_A}_B{LAMBDA_B}_s{STRIDE}.npz"
if not os.path.exists(ref_file):
    print("solving the reference committor ...", flush=True)
    env = dict(os.environ, GRID_H="0.001", LAMBDA_A=str(LAMBDA_A), LAMBDA_B=str(LAMBDA_B),
               STRIDE=str(STRIDE), SKIP_SHOOTING="1")
    subprocess.run([sys.executable, f"{os.path.dirname(OUT)}/committor_reference.py"], env=env, check=True)
ref = np.load(ref_file)
q_ref_fun = RegularGridInterpolator((ref["y"], ref["x"]), ref["q"], bounds_error=False, fill_value=None)
q_ref = np.clip(q_ref_fun(Z), 0, 1)


def fit(mask, label, seed_rng):
    """NPq on the rows in mask; returns q per row (nan elsewhere) and the metrics history."""
    Y = cp.asarray(Z[mask])
    q = orc.Committor(boundary0=in_a[mask], boundary1=in_b[mask], i_traj=i_traj[mask], t_traj=row[mask].astype(float))
    q.set_boundary_traps()   # excursions start/end in A or B: Z_q metrics and plots as for the continuous runs
    np.random.seed(seed_rng); cp.random.seed(seed_rng)
    print(f"--- fit {label}: {mask.sum()} rows, {N_ITER} iterations ---", flush=True)
    t0 = time.time()
    q.fit_transform(lambda: Y[:, np.random.randint(2)], ny=6, max_iter=N_ITER, print_step=250,
                    metrics_print=("iter", "delta_r2", "max_sd_zq", "auc", "delta_x"), save_min_delta_zq=False)
    r = np.full(len(y), np.nan); r[mask] = cp.asnumpy(q.r_traj)
    print(f"fit {label} done in {time.time() - t0:.0f} s", flush=True)
    if label == "all":
        for name, call in (("feps", q.plots_feps), ("obs_pred", q.plots_obs_pred),
                           ("obs_pred_log", lambda: q.plots_obs_pred(log_scale=True))):
            call(); plt.gcf().savefig(f"{OUT}/{name}.png", dpi=120); plt.close("all")
    hist = {k: np.asarray(v) for k, v in q.metrics_history.items()}
    del q, Y; cp.get_default_memory_pool().free_all_blocks()
    return r, hist


r_all, h_all = fit(np.ones(len(y), bool), "all", 0)
r_h1, _ = fit(seed <= 8, "seeds 1-8", 1)
r_h2, _ = fit(seed > 8, "seeds 9-16", 2)
np.savez(f"{OUT}/npq_md.npz", y=y, x=x, seed=seed, row=row, i_traj=i_traj, q_npq=r_all, q_half1=r_h1, q_half2=r_h2,
         q_ref=q_ref, lambda_a=LAMBDA_A, lambda_b=LAMBDA_B, stride=STRIDE, hist2d=hist2d,
         **{f"metric_{k}": v for k, v in h_all.items()})

# --- comparison with the exact committor ---------------------------------------------------------
free = ~(in_a | in_b)
err = r_all[free] - q_ref[free]
trans = free & (q_ref > 0.05) & (q_ref < 0.95)
print(f"RMS(NPq - exact) over all free rows {np.sqrt(np.mean(err ** 2)):.4f}, "
      f"over rows with 0.05 < q_exact < 0.95 {np.sqrt(np.mean((r_all[trans] - q_ref[trans]) ** 2)):.4f} "
      f"({trans.sum()} rows)", flush=True)

# binned maps (mean q per bin) of the fit and the reference
ey, ex = np.linspace(LAMBDA_A, LAMBDA_B, 76), np.linspace(-0.01, 0.31, 33)
def bin_mean(v, m):
    s = np.histogram2d(y[m], x[m], bins=(ey, ex), weights=v[m])[0]
    n = np.histogram2d(y[m], x[m], bins=(ey, ex))[0]
    return np.where(n >= 20, s / np.maximum(n, 1), np.nan), n
Q_npq, n_bin = bin_mean(r_all, free)
Q_ref_b, _ = bin_mean(q_ref, free)
Q_h1, _ = bin_mean(r_h1, free & (seed <= 8)); Q_h2, _ = bin_mean(r_h2, free & (seed > 8))
tb = (Q_ref_b > 0.05) & (Q_ref_b < 0.95)
print(f"per bin, 0.05 < q_exact < 0.95: RMS(NPq - exact) {np.sqrt(np.nanmean((Q_npq - Q_ref_b)[tb] ** 2)):.4f}, "
      f"RMS(half1 - half2) / 2 {np.sqrt(np.nanmean((Q_h1 - Q_h2)[tb] ** 2)) / 2:.4f} (statistical error of the full fit)",
      flush=True)
mask_ref = (ref["y"] >= -0.01) & (ref["y"] <= 0.91)
V = ref["V_eff"][mask_ref]; lev = np.linspace(np.percentile(V, 1), np.percentile(V, 92), 14)

fig, axes = plt.subplots(2, 2, figsize=(15, 7.5))
ext = [ey[0], ey[-1], ex[0], ex[-1]]
panels = [(Q_npq, "NPq (all 16 seeds)", "coolwarm", (0, 1)),
          (Q_ref_b, "exact committor (same A, B, stride 20)", "coolwarm", (0, 1)),
          (Q_npq - Q_ref_b, "NPq − exact", "PuOr", (-0.15, 0.15)),
          (Q_h1 - Q_h2, "NPq seeds 1-8 − NPq seeds 9-16", "PuOr", (-0.15, 0.15))]
for ax, (M, title, cmap, (vmin, vmax)) in zip(axes.ravel(), panels):
    im = ax.imshow(M.T, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.contour(ref["y"][mask_ref], ref["x"], V.T, levels=lev, colors="0.4", linewidths=0.5)
    if vmin == 0:
        ax.contour((ey[1:] + ey[:-1]) / 2, (ex[1:] + ex[:-1]) / 2, np.nan_to_num(M, nan=0.5).T,
                   levels=[0.1, 0.5, 0.9], colors="k", linewidths=1, linestyles="--")
    ax.set(title=title, xlabel="order parameter y", ylabel="x", xlim=(-0.01, 0.91), ylim=(-0.01, 0.31))
    for lam in (LAMBDA_A, LAMBDA_B):
        ax.axvline(lam, color="k", lw=0.8)
    fig.colorbar(im, ax=ax)
fig.suptitle(f"Brownian Z long MD, NPq on (y, x): {len(y)} kept rows, {switch.sum()} A↔B transitions, "
             f"{N_ITER} iterations (bins with ≥ 20 rows)")
fig.tight_layout(); fig.savefig(f"{OUT}/npq_vs_exact_map.png", dpi=130)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
bins = np.linspace(0, 1, 51); mid = (bins[1:] + bins[:-1]) / 2
k = np.digitize(q_ref[free], bins) - 1; k = np.clip(k, 0, 49)
for v, lab in ((r_all[free], "all seeds"),):
    m = np.bincount(k, weights=v, minlength=50) / np.maximum(np.bincount(k, minlength=50), 1)
    axes[0].plot(mid, m, "o-", ms=3, label=f"NPq {lab} (mean per bin of exact q)")
axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set(xlabel="exact q", ylabel="NPq", title="per-row agreement")
axes[0].legend(frameon=False)
it = h_all.get("iter", np.arange(len(h_all["delta_x"])))
axes[1].semilogy(it, h_all["delta_x"], label="|dx|")
axes[1].semilogy(it, h_all["max_sd_zq"], label="max sd Zq")
axes[1].set(xlabel="iteration", title="convergence (all seeds)"); axes[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/npq_vs_exact_rows.png", dpi=130)
print(f"figures and data in {OUT}\nALL DONE", flush=True)
