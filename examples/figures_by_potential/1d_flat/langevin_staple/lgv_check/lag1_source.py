"""Where does the true committor's lag-1 inconsistency near A come from? Split by ensemble and by segment length."""
import os, sys, io, contextlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
EX = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples"
sys.path.insert(0, EX); sys.path.insert(0, os.path.join(EX, ".."))
import tis_data
from optimalrcs import metrics_tis as M
OUT = os.path.dirname(os.path.abspath(__file__))
SIM = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istar0lgv15intf"
with contextlib.redirect_stdout(io.StringIO()):
    intf = tis_data.read_interfaces(SIM)
la, lb = intf[0], intf[-1]
lam, vv, seg, ens, w, grp = [], [], [], [], [], []
n_seg = 0
for k in range(1, 15):
    with contextlib.redirect_stdout(io.StringIO()):
        c, o, t, wt, pth = tis_data.load_tis_data(SIM, intf, ensemble_glob=f"{k:03d}", paths_per_ensemble=100,
                                                 seed=k, fd_velocity=True)
    # load_tis_data indexes folders by their own number, so a single folder keeps its M
    for piece, wp, g in zip(c, wt, pth):
        lam.append(piece[:, 0]); vv.append(piece[:, -1] / 2e-4)
        seg.append(np.full(len(piece), n_seg)); ens.append(np.full(len(piece), k))
        w.append(np.full(len(piece), float(wp))); grp.append(np.full(len(piece), 10 ** 6 * k + g)); n_seg += 1
lam, vv, seg, ens, w, grp = map(np.concatenate, (lam, vv, seg, ens, w, grp))
w /= w.mean()
b = ((lam <= la) | (lam >= lb)).astype(float)
ref = np.load(f"{OUT}/reference_fine.npz")
interp = RegularGridInterpolator((ref["xs"], ref["vs"]), ref["q"], bounds_error=False, fill_value=None)
q = np.clip(interp(np.c_[np.clip(lam, ref["xs"][0], ref["xs"][-1]), np.clip(vv, ref["vs"][0], ref["vs"][-1])]), 0, 1)
q[lam <= la] = 0; q[lam >= lb] = 1
seg_len = np.bincount(seg)[seg]
print(f"{len(lam)} frames, {n_seg} segments; segments of <= 3 frames: {np.mean(np.bincount(seg) <= 3) * 100:.1f}%", flush=True)

def zmax(sel, dt):
    _, z = M.comp_zq_stopped_zscore(q[sel], b[sel], seg[sel], w[sel], dt, g_traj=grp[sel])
    k = np.argmax(np.abs(z[5:])); return f"{abs(z[5:][k]):4.1f} (q {(k + 5.5) / 200:.2f})"

for name, sel in (("all", np.ones_like(b, bool)), ("without 001 [0+-]", ens != 1), ("only 001 [0+-]", ens == 1),
                  ("only 002 [1*]", ens == 2), ("segments > 20 frames", seg_len > 20), ("segments <= 20 frames", seg_len <= 20)):
    print(f"{name:24s}: {sel.sum():8d} frames | max|z| lag 1 {zmax(sel, 1)} | lag 16 {zmax(sel, 16)}", flush=True)
print("ALL DONE", flush=True)
