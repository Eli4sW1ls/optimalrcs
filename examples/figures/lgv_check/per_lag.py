"""Worst |z| of the stopped Z_q per lag, for the reference committor and the fit (100 paths/ensemble)."""
import os, sys, io, contextlib, argparse
import numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
EX = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples"
sys.path.insert(0, EX); sys.path.insert(0, os.path.join(EX, "..", "optimalrcs-cupy"))
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
r_fit = np.load(f"{OUT}/fit_g0.1_p100.npz")["r"]
tau = M.stop_index(d.i_traj, b)
print("lag   | reference: max|z| (q)   | fit: max|z| (q)", flush=True)
for dt in M.ldt0:
    row = []
    for r in (q_ref, r_fit):
        lx, z = M.comp_zq_stopped_zscore(cp.asarray(r), b, d.i_traj, d.path_weights, dt, tau=tau, g_traj=d.group_traj)
        z, lx = cp.asnumpy(z)[5:], cp.asnumpy(lx)[5:]
        k = np.argmax(np.abs(z)); row.append(f"{abs(z[k]):5.2f} ({lx[k]:.2f})")
    print(f"{dt:5d} | {row[0]:22s} | {row[1]}", flush=True)
