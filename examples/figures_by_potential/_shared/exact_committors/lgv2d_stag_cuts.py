"""Staggered channels (Langevin, script defaults): the committor along the channel centres through the barrier tops,
on a fine line (the 0.1 grid of lgv2d_exact.py does not resolve the step there).

Upper channel (transverse 1.3): order parameter -1.0 ... 0.0 (barrier top at -0.6); lower channel (transverse -1.3):
0.0 ... 1.0 (barrier top at +0.6). Velocity-averaged (Maxwell) and at v_op = -s, 0, +s (s = sqrt(kT/m)), v_tr = 0;
N shots per point. Output: cuts.npz, cuts.png in 2d_staggered_channels/exact/langevin_g0.1/.
"""
import os
import sys
import time
import zlib

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lgv2d_tools as L

S = L.SYSTEMS["stag"]
N = int(os.environ.get("CUT_SHOTS", 10000))
OUT = os.path.join(HERE, "..", "..", "2d_staggered_channels", "exact", "langevin_g0.1")
sv = np.sqrt(S["kT"] / S["m"])
lines = {"upper": (np.linspace(-1.0, 0.0, 101), 1.3), "lower": (np.linspace(0.0, 1.0, 101), -1.3)}
res = {}
f = os.path.join(OUT, "cuts.npz")
if os.path.exists(f):
    res = {k: v for k, v in np.load(f).items()}
for ch, (op, tr) in lines.items():
    for lab, v in (("qbar", None), ("minus", -sv), ("zero", 0.0), ("plus", sv)):
        key = f"{ch}_{lab}"
        if key in res:
            continue
        t0 = time.time()
        vv = (None, None) if v is None else (np.full(op.size, v), np.zeros(op.size))
        q, se, st = L.shoot(S, op, np.full(op.size, tr), *vv, n_shots=N, seed=zlib.crc32(key.encode()))
        res[key], res[key + "_se"] = q, se
        res[f"{ch}_op"] = op
        np.savez(f, **res)
        print(f"{key}: {time.time() - t0:.0f} s, s.e. max {se.max():.4f}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, (ch, (op, tr)) in zip(axes, lines.items()):
    V = S["potential"](op, np.full(op.size, tr), np) / S["kT"]
    for lab, col, name in (("qbar", "k", "Maxwell average"), ("minus", "C0", "v_op = -√(kT/m)"),
                           ("zero", "C2", "v = 0"), ("plus", "C3", "v_op = +√(kT/m)")):
        ax.errorbar(op, res[f"{ch}_{lab}"], yerr=2 * res[f"{ch}_{lab}_se"], color=col, lw=1, label=name)
    ax2 = ax.twinx(); ax2.plot(op, V - V.min(), color="0.6", ls=":"); ax2.set_ylabel("V / kT along the cut", color="0.5")
    ax.set(xlabel="order parameter (toytis x)", ylabel="q", ylim=(-0.02, 1.02),
           title=f"{ch} channel, transverse {tr:+g}: q through the barrier top ({N} shots, ±2 s.e.)")
    ax.legend(frameon=False, fontsize=8, loc="center left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cuts.png"), dpi=110); plt.close(fig)
print("ALL DONE")
