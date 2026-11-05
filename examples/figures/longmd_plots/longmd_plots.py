"""Standard optimalrcs validation plots (metrics, FEPs + Z_q, observed vs predicted) for the long
overdamped MD runs, flat and two bumps.

One 5M-frame block per run (block 0, seed 0, same settings as ../longmd_npq/npq_longmd.py), fitted
with the full metric set so the metric curves can be shown. Also q(lambda) against the exact committor.
"""
import os, sys, importlib.util
import numpy as np, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs"
sys.path.insert(0, os.path.join(REPO, "optimalrcs-cupy"))
spec = importlib.util.spec_from_file_location("orc", os.path.join(REPO, "optimalrcs-cupy", "optimalrcs.py"))
orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc)
OUT = os.path.dirname(os.path.abspath(__file__))
SRC = "/mnt/tw06_biommeda_pastime1/11.2024_StapleTIS_Elias/simulations/long_MD"
LA, LB, DT, GAMMA, KT = -0.1, 0.3, 2e-4, 5.0, 1.0
BLOCK, N_ITER, PRINT_STEP = 5_000_000, 1500, 50
GRID = np.linspace(LA, LB, 101)
MID = (GRID[1:] + GRID[:-1]) / 2
METRICS = ('iter', 'cross_entropy', 'mse', 'max_sd_zq', 'max_grad_zq', 'delta_r2', 'auc', 'delta_x', 'time_elapsed')

def v_two_bumps(x):
    v = np.zeros_like(x)
    m = (x >= -0.1) & (x <= 0.3)
    v[m] = 0.5 * (1 - np.cos(2 * np.pi * (x[m] + 0.1) / 0.2))
    return v

def exact_q(x, potential):
    s = 0.5826 * np.sqrt(2 * KT / GAMMA * DT)
    g = np.linspace(LA - s, LB + s, 20001)
    w = np.exp(potential(g) / KT)
    c = np.concatenate([[0], np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(g))])
    return np.interp(x, g, c / c[-1])

def save(path, title):
    fig = plt.gcf(); fig.suptitle(title, y=1.03)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)

plt.show = lambda: None                                  # plots_* methods end with plt.show()
RUNS = {"flat": ("longmd_flatbr50M", lambda x: np.zeros_like(x)),
        "two_bumps": ("longmd_2b_br", v_two_bumps)}
for name, (folder, potential) in RUNS.items():
    x = np.load(f"{SRC}/{folder}/md_path.npy", mmap_mode="r")[:BLOCK, 0].astype(np.float64)
    q = orc.Committor(boundary0=x <= LA, boundary1=x >= LB)
    y = cp.asarray(x); np.random.seed(0); cp.random.seed(0)
    print(f"=== {name}", flush=True)
    q.fit_transform(lambda: y, ny=6, max_iter=N_ITER, print_step=PRINT_STEP, metrics_print=METRICS,
                    save_min_delta_zq=False)
    title = f"long MD, {name.replace('_', ' ')} (block 0, 5M frames)"

    # metrics vs iteration: blue dotted = all iterations (left axis), red = second half (right axis)
    q.plots_metrics(['cross_entropy', 'max_sd_zq', 'delta_r2']); save(f"{OUT}/{name}_metrics.png", title)
    q.plots_metrics(['mse', 'auc', 'delta_x']); save(f"{OUT}/{name}_metrics2.png", title)
    # F(q), F(natural q), Z_q profiles at lags 1..2^15
    q.plots_feps(delta_t_sim=DT); save(f"{OUT}/{name}_feps.png", title)
    # observed vs predicted committor, its deviation, ROC
    q.plots_obs_pred(); save(f"{OUT}/{name}_obs_pred.png", title)

    # q(lambda) against the exact committor
    r = cp.asnumpy(q.r_traj)
    idx = np.digitize(x, GRID) - 1; inside = (idx >= 0) & (idx < len(MID))
    qb = np.bincount(idx[inside], weights=r[inside], minlength=len(MID)) / np.maximum(
        np.bincount(idx[inside], minlength=len(MID)), 1)
    ex = exact_q(MID, potential)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(MID, qb, color="#2a78d6", lw=2, label="NPq, mean q per λ bin")
    a1.plot(MID, ex, "--", color="#0b0b0b", lw=1.3, label="exact (overdamped)")
    a1.set(xlabel="order parameter λ", ylabel="q", title=title); a1.legend(frameon=False); a1.grid(alpha=0.3)
    a2.plot(MID, qb - ex, color="#2a78d6", lw=1.5); a2.axhline(0, color="#0b0b0b", lw=1, ls="--")
    a2.set(xlabel="order parameter λ", ylabel="NPq − exact",
           title=f"RMS {np.sqrt(np.mean((qb - ex) ** 2)):.4f}"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}_q_vs_lambda.png", dpi=130); plt.close(fig)

    np.savez(f"{OUT}/{name}_metrics_history.npz", **{k: np.asarray([float(v) for v in vals])
                                                     for k, vals in q.metrics_history.items()})
    del q, y; cp.get_default_memory_pool().free_all_blocks()
print("ALL DONE", flush=True)
