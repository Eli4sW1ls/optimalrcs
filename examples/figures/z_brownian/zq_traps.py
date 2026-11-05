"""Z_q profiles of the 16-seed Brownian Z NPq fit, without and with A/B treated as traps.

The data are excursions between A and B (npq_md.py), so every segment starts and ends in a boundary.
Without traps comp_zq drops the boundary terms of frames within dt of a segment edge; with
set_boundary_traps the profiles equal those of the continuous trajectories. For the exact committor
Z_q(q, dt) should equal N_AB, the number of A->B transitions, at every q and dt.
"""
import os, sys, importlib.util
import numpy as np, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs"
sys.path.insert(0, os.path.join(REPO, "optimalrcs-cupy"))
spec = importlib.util.spec_from_file_location("orc", os.path.join(REPO, "optimalrcs-cupy", "optimalrcs.py"))
orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc)
import cut_profiles
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "npq_md")

d = np.load(f"{OUT}/npq_md.npz")
y, i_traj, seed, row = d["y"], d["i_traj"], d["seed"], d["row"]
LA, LB = float(d["lambda_a"]), float(d["lambda_b"])
in_a, in_b = y <= LA, y >= LB
ab = np.flatnonzero(in_a | in_b)
same = seed[ab][1:] == seed[ab][:-1]
n_ab = int((same & in_a[ab][:-1] & in_b[ab][1:]).sum())
n_ba = int((same & in_b[ab][:-1] & in_a[ab][1:]).sum())
LDT = [2 ** k for k in range(16)]
colors = plt.cm.viridis(np.linspace(0, 0.95, len(LDT)))


def profiles(r, traps):
    q = orc.Committor(boundary0=in_a, boundary1=in_b, i_traj=i_traj, t_traj=row.astype(float))
    q.r_traj = cp.asarray(np.where(in_a, 0.0, np.where(in_b, 1.0, r)))
    if traps:
        q.set_boundary_traps()
    out = []
    for dt in LDT:
        e, z = cut_profiles.comp_zq(q.r_traj, q.b_traj, q.i_traj, q.future_boundary, q.past_boundary, dt=dt)
        out.append((cp.asnumpy(e)[:-2], cp.asnumpy(z)[:-1]))
    return out


panels = [(d["q_npq"], False, "NPq, as plotted before (no traps)"),
          (d["q_npq"], True, "NPq, A/B as traps"),
          (d["q_ref"], True, "exact committor, A/B as traps")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (r, traps, title) in zip(axes, panels):
    for (e, z), dt, c in zip(profiles(r, traps), LDT, colors):
        ax.plot(e, z, color=c, lw=1.2, label=f"Δt = {dt}")
    ax.axhline(n_ab, color="k", ls="--", lw=1.2, label=f"N_AB = {n_ab}")
    ax.set(title=title, xlabel="q", ylim=(0, 50))
    ax.grid(alpha=0.3)
axes[0].set_ylabel("$Z_q$")
axes[2].legend(fontsize=8, ncol=2, frameon=False, loc="lower center")
fig.suptitle(f"Brownian Z long MD, 16 seeds: {len(y)} rows in {i_traj[-1]} excursions, "
             f"{n_ab} A→B and {n_ba} B→A transitions (Δt in saved rows, 1 row = 20 steps)")
fig.tight_layout(); fig.savefig(f"{OUT}/zq_traps.png", dpi=130)
print(f"saved {OUT}/zq_traps.png")
