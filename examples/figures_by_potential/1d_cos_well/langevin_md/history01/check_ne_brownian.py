"""Check: CommittorNE (y = lambda, no history) on the Brownian two-bumps long MD, block 0, against the
exact committor. Tells whether the Langevin failure comes from CommittorNE on a long equilibrium trajectory."""
import sys, numpy as np, cupy as cp
sys.path.insert(0, "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/optimalrcs-cupy/")
import optimalrcs
x = np.load("/mnt/tw06_biommeda_pastime1/11.2024_StapleTIS_Elias/simulations/long_MD/longmd_2b_br/md_path.npy",
            mmap_mode="r")[:5_000_000, 0].astype(np.float64)
ex = np.loadtxt("../longmd_npq/npq_two_bumps.csv")           # lambda q_mean q_se q_exact
y = cp.asarray(x); np.random.seed(0); cp.random.seed(0)
q = optimalrcs.CommittorNE(boundary0=x <= -0.1, boundary1=x >= 0.3, i_traj=np.ones_like(x), t_traj=np.arange(len(x)))
for it in range(3):                                           # 3 x 500 iterations, report q(lambda) error each time
    q.fit_transform(lambda: y, ny=6, max_iter=500, print_step=500, save_min_delta_zq=False)
    edges = np.linspace(-0.1, 0.3, 101); idx = np.digitize(x, edges) - 1; ins = (idx >= 0) & (idx < 100)
    r = cp.asnumpy(q.r_traj)
    qb = np.bincount(idx[ins], weights=r[ins], minlength=100) / np.bincount(idx[ins], minlength=100)
    print(f"after {500 * (it + 1)} iterations: RMS vs exact {np.sqrt(np.mean((qb - ex[:, 3]) ** 2)):.4f}, "
          f"q at lambda = 0, 0.1, 0.2: {qb[25]:.3f} {qb[50]:.3f} {qb[75]:.3f} (exact {ex[25, 3]:.3f} {ex[50, 3]:.3f} {ex[75, 3]:.3f})", flush=True)
print("CHECK DONE", flush=True)
