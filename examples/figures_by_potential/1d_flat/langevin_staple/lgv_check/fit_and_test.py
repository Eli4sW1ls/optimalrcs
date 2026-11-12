"""Fit (--fd-velocity, gamma 0.1, 100 paths/ensemble), then: does the true committor pass the stopped Z_q,
and how far is the fit from it, overall and near A. Each stage saves its result and is skipped on rerun."""
import os, sys, io, contextlib, argparse, importlib.util, time
import numpy as np, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
EX = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples"
sys.path.insert(0, EX); sys.path.insert(0, os.path.join(EX, "..", "optimalrcs-cupy"))
spec = importlib.util.spec_from_file_location("orc", os.path.join(EX, "..", "optimalrcs-cupy", "optimalrcs.py"))
orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc)
import tis_data, metrics_tis as M
OUT = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
args = p.parse_args(["/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istar0lgv15intf",
                     "--paths-per-ensemble", "100", "--seed", "0", "--fd-velocity"])
with contextlib.redirect_stdout(io.StringIO()):
    d = tis_data.prepare_data(args)
x, v = d.X[:, 0], d.X[:, 1] / 2e-4
b = d.boundary0 | d.boundary1
ref = np.load(f"{OUT}/reference_fine.npz")
interp = RegularGridInterpolator((ref["xs"], ref["vs"]), ref["q"], bounds_error=False, fill_value=None)
q_ref = np.clip(interp(np.c_[np.clip(x, ref["xs"][0], ref["xs"][-1]), np.clip(v, ref["vs"][0], ref["vs"][-1])]), 0, 1)
q_ref[d.boundary0] = 0; q_ref[d.boundary1] = 1
print(f"{len(x)} frames; |v| > 3.5 in {np.mean(np.abs(v) > 3.5) * 100:.2f}% (clipped to the grid)", flush=True)

def stopped_test(r, label):
    tau = M.stop_index(d.i_traj, b)
    worst = (0, None, None)
    for dt in M.ldt0:
        lx, z = M.comp_zq_stopped_zscore(cp.asarray(r), b, d.i_traj, d.path_weights, dt, tau=tau, g_traj=d.group_traj)
        z, lx = cp.asnumpy(z), cp.asnumpy(lx)
        k = np.argmax(np.abs(z[5:]))
        if abs(z[5:][k]) > worst[0]:
            worst = (abs(z[5:][k]), dt, lx[5:][k])
    print(f"{label:28s}: stopped Z_q max|z| {worst[0]:5.2f} (lag {worst[1]}, q = {worst[2]:.2f})", flush=True)

stopped_test(q_ref, "reference committor")

fit_file = f"{OUT}/fit_g0.1_p100.npz"
if os.path.exists(fit_file):
    r = np.load(fit_file)["r"]
else:
    q = orc.CommittorNE(d.boundary0, d.boundary1, i_traj=d.i_traj, t_traj=d.t_traj,
                        path_weights=d.path_weights, group_traj=d.group_traj)
    q.set_fixed_traj_length_trap(b, 2 ** 16)
    X = cp.asarray(d.X); np.random.seed(0); cp.random.seed(0)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        q.fit_transform(lambda: X[:, np.random.randint(2)], gamma=0.1, ny=6, max_iter=4000, print_step=500,
                        metrics_print=('iter', 'delta_x'), save_min_delta_zq=False)
    r = cp.asnumpy(q.r_traj)
    np.savez(fit_file, r=r)
    print(f"fit done in {time.time() - t0:.0f} s, |dx| {q.metrics_history['delta_x'][-1]:.4f}", flush=True)
    del q, X; cp.get_default_memory_pool().free_all_blocks()
stopped_test(r, "fitted q")

inside = ~b & (np.abs(v) < 3.5)
err = r - q_ref
for name, sel in (("all", inside), ("x < 0.05", inside & (x < 0.05)), ("0.05 < x < 0.3", inside & (x > 0.05) & (x < 0.3)),
                  ("x < 0.3, |v| < 0.5", inside & (x < 0.3) & (np.abs(v) < 0.5)), ("x > 0.3", inside & (x > 0.3))):
    print(f"  fit - reference, {name:20s}: RMS {np.sqrt(np.mean(err[sel]**2)):.3f}, bias {err[sel].mean():+.3f}, "
          f"{sel.sum():8d} frames", flush=True)

fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
for ax, v0 in zip(axes, (-1.0, 0.0, 0.5, 1.0)):
    j = np.argmin(np.abs(ref["vs"] - v0))
    ax.errorbar(ref["xs"], ref["q"][:, j], yerr=ref["se"][:, j], fmt="k.-", ms=3, lw=1, label="reference")
    sel = inside & (np.abs(v - v0) < 0.125)
    edges = np.r_[ref["xs"][0], (ref["xs"][1:] + ref["xs"][:-1]) / 2, ref["xs"][-1]]
    k = np.digitize(x[sel], edges) - 1
    fb = [r[sel][k == i].mean() if (k == i).sum() > 30 else np.nan for i in range(len(ref["xs"]))]
    ax.plot(ref["xs"], fb, "-", color="#d6604d", lw=2, label="fit")
    ax.set(title=f"v = {v0:+.1f}", xlabel="x", xlim=(-0.1, 1.3)); ax.grid(alpha=0.3)
axes[0].set_ylabel("q"); axes[0].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fit_vs_reference_slices.png", dpi=110)
print("ALL DONE", flush=True)
