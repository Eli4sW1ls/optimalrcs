"""Reference for base_documented_1d_langevin.ipynb: same CommittorNE fit, but y drawn between
lambda and the true velocity v, no history. Separates "history cannot represent v" from setup problems."""
import sys, numpy as np, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/optimalrcs-cupy/")
import optimalrcs, plots
plt.show = lambda: None
SRC = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/md_istarwell2_100M"
data = np.load(f"{SRC}/md_path.npy", mmap_mode="r")[:5_000_000]
lam, v = np.array(data[:, 0]), np.array(data[:, 1])
ys = [cp.asarray(lam), cp.asarray(v)]
np.random.seed(0); cp.random.seed(0)
q = optimalrcs.CommittorNE(boundary0=lam <= -0.35, boundary1=lam >= 0.35,
                           i_traj=np.ones_like(lam), t_traj=np.arange(len(lam)))
q.fit_transform(lambda: ys[np.random.randint(2)], ny=6, max_iter=1500, print_step=500, save_min_delta_zq=False)
q.plots_feps(delta_t_sim=0.005); plt.gcf().savefig("ref_feps.png", dpi=110, bbox_inches="tight"); plt.close("all")
q.plots_obs_pred(); plt.gcf().savefig("ref_obs_pred.png", dpi=110, bbox_inches="tight"); plt.close("all")
r = cp.asnumpy(q.r_traj); el = np.linspace(-0.35, 0.35, 71); ev = np.linspace(-3, 3, 61)
n = np.histogram2d(lam, v, [el, ev])[0]; s = np.histogram2d(lam, v, [el, ev], weights=r)[0]
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(el, ev, np.where(n > 20, s / np.maximum(n, 1), np.nan).T, cmap="coolwarm", vmin=0, vmax=1)
fig.colorbar(im, ax=ax, label="$q$"); ax.set(xlabel="$\\lambda$", ylabel="$v$", title="reference: y = λ or true v")
fig.savefig("ref_q_lambda_v.png", dpi=110, bbox_inches="tight")
print("REF DONE", flush=True)
