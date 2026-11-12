"""NPq (equilibrium committor, CuPy optimalrcs) on the long overdamped MD runs, flat and two bumps.

Each 50M-frame trajectory is split into ten 5M-frame blocks, fitted separately; the spread
between blocks is the statistical error. A: lambda <= -0.1, B: lambda >= 0.3 (the iSTAR runs'
first and last interfaces). The exact committor of the overdamped dynamics is saved alongside.
"""
import os, sys, io, contextlib, importlib.util, time
import numpy as np, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs"
sys.path.insert(0, os.path.join(REPO, "optimalrcs-cupy"))
spec = importlib.util.spec_from_file_location("orc", os.path.join(REPO, "optimalrcs-cupy", "optimalrcs.py"))
orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc)
OUT = os.path.dirname(os.path.abspath(__file__))
SRC = "/mnt/tw06_biommeda_pastime1/11.2024_StapleTIS_Elias/simulations/long_MD"
LA, LB, DT, GAMMA, KT = -0.1, 0.3, 2e-4, 5.0, 1.0
BLOCK, N_ITER = 5_000_000, 1500
GRID = np.linspace(LA, LB, 101)                          # bin edges for q(lambda)
MID = (GRID[1:] + GRID[:-1]) / 2

def v_two_bumps(x):
    v = np.zeros_like(x)
    m = (x >= -0.1) & (x <= 0.3)
    v[m] = 0.5 * (1 - np.cos(2 * np.pi * (x[m] + 0.1) / 0.2))
    return v

def exact_q(x, potential):
    """Overdamped committor: int_{A-s}^{x} e^{V/kT} / int_{A-s}^{B+s} e^{V/kT}, s = 0.5826 sigma
    (absorption checked only at saved frames; V = 0 at both boundaries)."""
    s = 0.5826 * np.sqrt(2 * KT / GAMMA * DT)
    g = np.linspace(LA - s, LB + s, 20001)
    w = np.exp(potential(g) / KT)
    c = np.concatenate([[0], np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(g))])
    return np.interp(x, g, c / c[-1])

RUNS = {"flat": ("longmd_flatbr50M", lambda x: np.zeros_like(x)),
        "two_bumps": ("longmd_2b_br", v_two_bumps)}
results = {}
for name, (folder, potential) in RUNS.items():
    x_all = np.load(f"{SRC}/{folder}/md_path.npy")[:, 0].astype(np.float64)
    # sanity check of the force: mean step / (dt/gamma) per lambda bin vs -dV/dx
    dx = np.diff(x_all[:10_000_000]); xm = x_all[:10_000_000 - 1]
    for x0 in (-0.05, 0.05, 0.15, 0.25):
        sel = np.abs(xm - x0) < 0.01
        f_data = dx[sel].mean() / (DT / GAMMA)
        f_pot = -(potential(np.array([x0 + 1e-4])) - potential(np.array([x0 - 1e-4])))[0] / 2e-4
        print(f"{name}: force at {x0:+.2f}: data {f_data:+6.2f}, potential {f_pot:+6.2f}", flush=True)
    q_blocks = []
    for k in range(len(x_all) // BLOCK):
        x = x_all[k * BLOCK:(k + 1) * BLOCK]
        q = orc.Committor(boundary0=x <= LA, boundary1=x >= LB)
        y = cp.asarray(x); np.random.seed(k); cp.random.seed(k); t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            q.fit_transform(lambda: y, ny=6, max_iter=N_ITER, print_step=500,
                            metrics_print=('iter', 'delta_r2', 'max_sd_zq', 'delta_x'), save_min_delta_zq=False)
        r = cp.asnumpy(q.r_traj); h = q.metrics_history
        idx = np.digitize(x, GRID) - 1; inside = (idx >= 0) & (idx < len(MID))
        qb = np.bincount(idx[inside], weights=r[inside], minlength=len(MID)) / np.maximum(
            np.bincount(idx[inside], minlength=len(MID)), 1)
        q_blocks.append(qb)
        rms = np.sqrt(np.mean((qb - exact_q(MID, potential)) ** 2))
        print(f"{name} block {k}: {time.time() - t0:4.0f} s | |dx| {h['delta_x'][-1]:.2e} | sdzq {h['max_sd_zq'][-1]:.2f} "
              f"| RMS vs exact {rms:.4f}", flush=True)
        del q, y; cp.get_default_memory_pool().free_all_blocks()
    q_blocks = np.array(q_blocks)
    results[name] = q_blocks
    np.savez(f"{OUT}/npq_{name}.npz", lambda_edges=GRID, lambda_mid=MID, q_blocks=q_blocks,
             q_mean=q_blocks.mean(0), q_se=q_blocks.std(0, ddof=1) / np.sqrt(len(q_blocks)),
             q_exact=exact_q(MID, potential), lambda_a=LA, lambda_b=LB)
    np.savetxt(f"{OUT}/npq_{name}.csv", np.c_[MID, q_blocks.mean(0), q_blocks.std(0, ddof=1) / np.sqrt(len(q_blocks)),
                                               exact_q(MID, potential)],
               header="lambda q_npq_mean q_npq_stderr q_exact", fmt="%.6f")
    del x_all

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
for col, name in enumerate(RUNS):
    qb = results[name]; ex = exact_q(MID, RUNS[name][1])
    ax = axes[0, col]
    for row in qb:
        ax.plot(MID, row, color="#2a78d6", lw=0.6, alpha=0.4)
    ax.plot(MID, qb.mean(0), color="#2a78d6", lw=2, label="NPq, mean of 10 blocks of 5M frames")
    ax.plot(MID, ex, "--", color="#0b0b0b", lw=1.3, label="exact (overdamped)")
    ax.set(title=f"long MD, {name.replace('_', ' ')}", ylabel="q")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
    ax = axes[1, col]
    se = qb.std(0, ddof=1) / np.sqrt(len(qb))
    ax.fill_between(MID, qb.mean(0) - ex - 2 * se, qb.mean(0) - ex + 2 * se, color="#2a78d6", alpha=0.25, lw=0)
    ax.plot(MID, qb.mean(0) - ex, color="#2a78d6", lw=1.5)
    ax.axhline(0, color="#0b0b0b", lw=1, ls="--")
    ax.set(xlabel="order parameter λ", ylabel="NPq − exact (±2 s.e.)"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/npq_longmd.png", dpi=130)
print("ALL DONE", flush=True)
