"""True committor's stopped Z_q at lag 1 and 16 with the velocity estimated over d frames."""
import os, sys, io, contextlib, argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator
EX = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples"
sys.path.insert(0, EX); sys.path.insert(0, os.path.join(EX, ".."))
import tis_data
from optimalrcs import metrics_tis as M
OUT = os.path.dirname(os.path.abspath(__file__))
ref = np.load(f"{OUT}/reference_fine.npz")
interp = RegularGridInterpolator((ref["xs"], ref["vs"]), ref["q"], bounds_error=False, fill_value=None)
for d in (1, 4, 8, 16):
    p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
    args = p.parse_args(["/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istar0lgv15intf",
                         "--paths-per-ensemble", "100", "--seed", "0", "--fd-velocity", str(d)])
    with contextlib.redirect_stdout(io.StringIO()):
        dd = tis_data.prepare_data(args)
    x, v = dd.X[:, 0], dd.X[:, 1] / 2e-4
    b = dd.boundary0 | dd.boundary1
    q = np.clip(interp(np.c_[np.clip(x, ref["xs"][0], ref["xs"][-1]), np.clip(v, ref["vs"][0], ref["vs"][-1])]), 0, 1)
    q[dd.boundary0] = 0; q[dd.boundary1] = 1
    tau = M.stop_index(dd.i_traj, b)
    out = []
    for dt in (1, 2, 4, 16):
        lx, z = M.comp_zq_stopped_zscore(q, b, dd.i_traj, dd.path_weights, dt, tau=tau, g_traj=dd.group_traj)
        k = np.argmax(np.abs(z[5:])); out.append(f"{abs(z[5:][k]):4.1f} (q {lx[5:][k]:.2f})")
    print(f"d = {d:2d}: true q max|z| at lags 1, 2, 4, 16: " + " | ".join(out), flush=True)
print("ALL DONE", flush=True)
