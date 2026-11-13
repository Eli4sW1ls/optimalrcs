"""Reference committors of the 2D Langevin runs by GPU shooting (see lgv2d_tools.py).

    python lgv2d_exact.py cc|zpot|stag

For the system, in figures_by_potential/<potential>/exact/langevin_g<gamma>/:
  * qbar(y, x): the velocity-averaged committor (Maxwell velocities, one per trajectory) on a position grid, for
    the points with V - V_min < V_CUT kT (where frames can be); N_QBAR shots per point;
  * q(y, x | v): the committor at fixed velocities v_y = -s, 0, +s (s = sqrt(kT/m), along the order parameter),
    v_x = 0; N_V shots per point (not for the Z potential, whose run has no velocities);
  * cc only: the committor of FRAMES random frames of test_cc_lgv (their recorded y, x, v_y, v_x), N_FRAME shots;
  * landscape.png, maps.png (committors, isolines, standard errors), frames.png (cc), and the values in npz files.
Every map is saved as soon as it is done, and reused on a rerun.
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
EXAMPLES = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, EXAMPLES)
import lgv2d_tools as L

KEY = sys.argv[1]
S = L.SYSTEMS[KEY]
SETTINGS = {
    "cc": dict(ny=36, nx=35, V_CUT=10, N_QBAR=20000, N_V=10000, FRAMES=300, N_FRAME=20000),
    "zpot": dict(ny=33, nx=13, V_CUT=10, N_QBAR=10000, N_V=0, FRAMES=0, N_FRAME=0),
    "stag": dict(ny=33, nx=39, V_CUT=13, N_QBAR=10000, N_V=10000, FRAMES=0, N_FRAME=0),
}[KEY]
if os.environ.get("LGV_SMOKE"):          # quick test: few shots, few frames, output elsewhere
    SETTINGS.update(N_QBAR=200, N_V=200 if SETTINGS["N_V"] else 0, FRAMES=min(SETTINGS["FRAMES"], 20), N_FRAME=200,
                    ny=8, nx=7)
OUT = os.environ.get("LGV_OUT") or os.path.join(EXAMPLES, "figures_by_potential", S["folder"], "exact",
                                                f"langevin_g{S['gamma']:g}")
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "lgv2d_exact.log"), "a")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


kT = S["kT"]; sv = np.sqrt(kT / S["m"])
say(f"===== {S['name']}: kT {kT}, gamma {S['gamma']}, dt {S['dt']}, A: y <= {S['A']}, B: y >= {S['B']} "
    f"({S['runs']}); {time.ctime()} =====")

# position grid (y = order parameter between A and B, x transverse), points where frames can be
gy = np.linspace(*S["y_range"], SETTINGS["ny"] + 2)[1:-1]
gx = np.linspace(*S["x_range"], SETTINGS["nx"])
GY, GX = np.meshgrid(gy, gx, indexing="ij")
V = np.asarray(S["potential"](GY, GX, np)) / kT
V -= V.min()
use = V < SETTINGS["V_CUT"]
if KEY == "stag":       # inside the forbidden rectangle V is ~0 again (Gaussian walls), but it cannot be reached
    p_ = L.STAG
    use &= ~((GY > p_["xl"] - 0.05) & (GY < p_["xr"] + 0.05) & (GX > p_["yb"] - 0.05) & (GX < p_["yt"] + 0.05))
say(f"grid {len(gy)} x {len(gx)}, {use.sum()} points with V - V_min < {SETTINGS['V_CUT']} kT")


def run_map(name, vy, vx, n):
    f = os.path.join(OUT, f"{name}.npz")
    if os.path.exists(f):
        d = np.load(f); say(f"  {name}: loaded {f}"); return d["q"], d["se"], d["steps"]
    t0 = time.time()
    yy, xx = GY[use], GX[use]
    vv = (None, None) if vy is None else (np.full(yy.size, vy), np.full(yy.size, vx))
    q, se, st = L.shoot(S, yy, xx, *vv, n_shots=n, seed=zlib.crc32(name.encode()))
    Q, SE, ST = (np.full(GY.shape, np.nan) for _ in range(3))
    Q[use], SE[use], ST[use] = q, se, st
    np.savez(f, y=gy, x=gx, q=Q, se=SE, steps=ST, n_shots=n, vy=np.nan if vy is None else vy,
             vx=np.nan if vx is None else vx, **{k: v for k, v in S.items() if isinstance(v, (int, float))})
    say(f"  {name}: {n} shots per point, {time.time() - t0:.0f} s; s.e. max {np.nanmax(SE):.4f}, "
        f"mean steps to absorption {np.nanmean(ST):.0f}")
    return Q, SE, ST


maps = {"qbar": run_map("qbar_maxwell", None, None, SETTINGS["N_QBAR"])}
if SETTINGS["N_V"]:
    for lab, v in (("minus", -sv), ("zero", 0.0), ("plus", sv)):
        maps[f"v_{lab}"] = run_map(f"q_vy_{lab}", v, 0.0, SETTINGS["N_V"])

# per-frame references (cc: test_cc_lgv frames with their recorded velocities)
if SETTINGS["FRAMES"]:
    f = os.path.join(OUT, "frames_test_cc_lgv.npz")
    if os.path.exists(f):
        fr = dict(np.load(f)); say(f"  frames: loaded {f}")
    else:
        import argparse
        import tis_data
        p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
        a = p.parse_args([f"{L.TOYTIS}/simulations/test_cc_lgv", "--paths-per-ensemble", "300"])
        data = tis_data.prepare_data(a)
        free = np.flatnonzero(~(data.boundary0 | data.boundary1))
        pick = np.sort(np.random.default_rng(3).choice(free, SETTINGS["FRAMES"], replace=False))
        y, x, vy, vx = (data.X_all[pick, k] for k in range(4))
        t0 = time.time()
        q, se, st = L.shoot(S, y, x, vy, vx, n_shots=SETTINGS["N_FRAME"], seed=11)
        fr = dict(y=y, x=x, vy=vy, vx=vx, q=q, se=se, steps=st, frame_index=pick, n_shots=SETTINGS["N_FRAME"])
        np.savez(f, **fr, note="frames of test_cc_lgv as loaded by tis_data.prepare_data(--paths-per-ensemble 300)")
        say(f"  frames: {len(q)} frames of test_cc_lgv x {SETTINGS['N_FRAME']} shots, {time.time() - t0:.0f} s; "
            f"s.e. max {se.max():.4f}")
    # how much the velocity matters: frame committor vs the velocity-averaged one at its position
    from scipy.interpolate import RegularGridInterpolator
    Qb = maps["qbar"][0]
    ib = RegularGridInterpolator((gy, gx), np.nan_to_num(Qb, nan=0.5), bounds_error=False, fill_value=None)
    qb_at = np.clip(ib(np.c_[fr["y"], fr["x"]]), 0, 1)
    say(f"  frames: RMS(q(frame) - qbar(position)) = {np.sqrt(np.mean((fr['q'] - qb_at) ** 2)):.3f} "
        f"(how much the velocity changes the committor)")

# ------------------------------------------------------------------ figures
fy = np.linspace(min(S["y_range"][0], S["A"]) - 0.05 * np.ptp(S["y_range"]), S["B"] + 0.05 * np.ptp(S["y_range"]), 300)
fx = np.linspace(*S["x_range"], 200)
FY, FX = np.meshgrid(fy, fx, indexing="ij")
VF = np.asarray(S["potential"](FY, FX, np)) / kT if KEY != "zpot" else \
    np.asarray(S["potential"](FY[::3, ::3], FX[::3, ::3], np)) / kT
if KEY == "zpot":
    fy, fx = fy[::3], fx[::3]
VF -= VF.min()
lev = np.arange(0, 12.5, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
im = axes[0].pcolormesh(fy, fx, np.minimum(VF, 12).T, cmap="viridis", shading="auto")
fig.colorbar(im, ax=axes[0], label="V / kT (clipped at 12)")
for b in (S["A"], S["B"]):
    axes[0].axvline(b, color="w", ls="--", lw=0.8)
axes[0].set(xlabel="order parameter y", ylabel="x", title=f"{S['name']}: V/kT; dashed: A, B")
Fprof = -np.log(np.exp(-VF).sum(1))
axes[1].plot(fy, Fprof - Fprof.min(), "k"); axes[1].set(xlabel="y", ylabel="F(y) / kT", ylim=(0, 12),
                                                     title="equilibrium free energy along y")
for b in (S["A"], S["B"]):
    axes[1].axvline(b, color="C3", ls="--", lw=0.8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "landscape.png"), dpi=110); plt.close(fig)

names = list(maps)
fig, axes = plt.subplots(2, len(names), figsize=(6.5 * len(names), 9), squeeze=False)
titles = {"qbar": "velocity-averaged q̄(y, x) (Maxwell v)", "v_minus": "q(y, x | v_y = -√(kT/m), v_x = 0)",
          "v_zero": "q(y, x | v = 0)", "v_plus": "q(y, x | v_y = +√(kT/m), v_x = 0)"}
for k, nm in enumerate(names):
    Q, SE, _ = maps[nm]
    im = axes[0, k].pcolormesh(gy, gx, Q.T, cmap="coolwarm", vmin=0, vmax=1, shading="nearest")
    fig.colorbar(im, ax=axes[0, k], label="q")
    axes[0, k].contour(fy, fx, VF.T, levels=lev, colors="0.5", linewidths=0.4)
    if np.isfinite(Q).sum() > 10:
        axes[0, k].contour(gy, gx, np.nan_to_num(Q, nan=np.nan).T, levels=[0.1, 0.5, 0.9], colors="k",
                           linewidths=1, linestyles="--")
    axes[0, k].set(title=titles[nm], xlabel="y (order parameter)", ylabel="x")
    im = axes[1, k].pcolormesh(gy, gx, SE.T, cmap="magma", shading="nearest")
    fig.colorbar(im, ax=axes[1, k], label="standard error")
    axes[1, k].set(title=f"standard error (max {np.nanmax(SE):.4f})", xlabel="y", ylabel="x")
fig.suptitle(f"{S['name']}, Langevin γ = {S['gamma']:g}, kT = {kT}, dt = {S['dt']}: committor by shooting")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "maps.png"), dpi=100); plt.close(fig)

if SETTINGS["FRAMES"]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    sc = axes[0].scatter(fr["y"], fr["q"], c=fr["vy"] / sv, cmap="coolwarm", vmin=-2, vmax=2, s=12)
    fig.colorbar(sc, ax=axes[0], label="v_y / √(kT/m)")
    axes[0].set(xlabel="y", ylabel="q (frame)", title=f"{len(fr['q'])} frames of test_cc_lgv, {fr['n_shots']} shots each")
    axes[1].scatter(qb_at, fr["q"], c=fr["vy"] / sv, cmap="coolwarm", vmin=-2, vmax=2, s=12)
    axes[1].plot([0, 1], [0, 1], "k:")
    axes[1].set(xlabel="q̄ at the frame's position (velocity-averaged)", ylabel="q of the frame (its own velocity)",
                title="how much the velocity matters")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "frames.png"), dpi=110); plt.close(fig)
say("ALL DONE")
