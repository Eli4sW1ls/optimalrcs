"""History 0,1 fit as in base_documented_1d_langevin.ipynb, but with the original notebook's
gamma = 0.2 and envelope_scale = 0.001. Reports q against the observed committor per lambda bin."""
import sys, numpy as np, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/optimalrcs-cupy/")
import optimalrcs
plt.show = lambda: None
optimalrcs.envelope_scale = 0.001
a = np.load("/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/md_istarwell2_100M/md_path.npy", mmap_mode="r")[:5_000_000]
lam, v = np.array(a[:, 0]), np.array(a[:, 1])
s = np.where(lam <= -0.35, 0, np.where(lam >= 0.35, 1, -1)); idx = np.nonzero(s >= 0)[0]
hitB = s[idx[np.searchsorted(idx, np.arange(len(lam))).clip(max=len(idx) - 1)]] == 1
y = cp.asarray(lam); np.random.seed(0); cp.random.seed(0)
q = optimalrcs.CommittorNE(boundary0=lam <= -0.35, boundary1=lam >= 0.35, i_traj=np.ones_like(lam), t_traj=np.arange(len(lam)))
for it in range(3):
    q.fit_transform(lambda: y, history_delta_t=[0, 1], history_type=['y(t-d),y(t)'], gamma=0.2, ny=6,
                    max_iter=500, print_step=500, save_min_delta_zq=False)
    r = cp.asnumpy(q.r_traj)
    out = []
    for x0 in (-0.3, -0.2, 0.0, 0.2, 0.3):
        m = (s < 0) & (np.abs(lam - x0) < 0.01)
        out.append(f"{x0:+.1f}: q {r[m].mean():.3f} obs {hitB[m].mean():.3f}")
    print(f"after {500 * (it + 1)}: " + " | ".join(out), flush=True)
q.plots_obs_pred(); plt.gcf().savefig("gamma02_obs_pred.png", dpi=110, bbox_inches="tight"); plt.close("all")
el = np.linspace(-0.35, 0.35, 71); ev = np.linspace(-3, 3, 61)
n = np.histogram2d(lam, v, [el, ev])[0]; sm = np.histogram2d(lam, v, [el, ev], weights=r)[0]
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(el, ev, np.where(n > 20, sm / np.maximum(n, 1), np.nan).T, cmap="coolwarm", vmin=0, vmax=1)
fig.colorbar(im, ax=ax, label="$q$"); ax.set(xlabel="$\\lambda$", ylabel="$v$", title="history 0,1, gamma = 0.2")
fig.savefig("gamma02_q_lambda_v.png", dpi=110, bbox_inches="tight")
print("GAMMA DONE", flush=True)
