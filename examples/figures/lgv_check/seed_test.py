"""Start NPNEq from the true committor: does it stay there or drift to the near-A bias? 30 paths/ensemble."""
import os, sys, io, contextlib, argparse, importlib.util
import numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
EX = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples"
sys.path.insert(0, EX); sys.path.insert(0, os.path.join(EX, "..", "optimalrcs-cupy"))
spec = importlib.util.spec_from_file_location("orc", os.path.join(EX, "..", "optimalrcs-cupy", "optimalrcs.py"))
orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc)
import tis_data
OUT = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
args = p.parse_args(["/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istar0lgv15intf",
                     "--paths-per-ensemble", "30", "--seed", "0", "--fd-velocity"])
with contextlib.redirect_stdout(io.StringIO()):
    d = tis_data.prepare_data(args)
x, v = d.X[:, 0], d.X[:, 1] / 2e-4
b = d.boundary0 | d.boundary1
ref = np.load(f"{OUT}/reference_fine.npz")
interp = RegularGridInterpolator((ref["xs"], ref["vs"]), ref["q"], bounds_error=False, fill_value=None)
q_ref = np.clip(interp(np.c_[np.clip(x, ref["xs"][0], ref["xs"][-1]), np.clip(v, ref["vs"][0], ref["vs"][-1])]), 0, 1)
q_ref[d.boundary0] = 0; q_ref[d.boundary1] = 1
inside = ~b & (np.abs(v) < 3.5)
near = inside & (x < 0.3) & (np.abs(v) < 0.5)

def report(r, label):
    err = cp.asnumpy(r) - q_ref
    print(f"{label:16s}: RMS all {np.sqrt(np.mean(err[inside]**2)):.3f} | near A slow: bias {err[near].mean():+.3f}", flush=True)

q = orc.CommittorNE(d.boundary0, d.boundary1, i_traj=d.i_traj, t_traj=d.t_traj, seed_r=q_ref,
                    path_weights=d.path_weights, group_traj=d.group_traj)
Y = cp.asarray(d.X); np.random.seed(0); cp.random.seed(0)
report(q.r_traj, "start (true q)")
done = 0
for chunk in (100, 400, 500, 1000, 2000):
    with contextlib.redirect_stdout(io.StringIO()):
        q.fit_transform(lambda: Y[:, np.random.randint(2)], gamma=0.1, ny=6, max_iter=chunk, print_step=10 ** 6,
                        metrics_print=('iter',), save_min_delta_zq=False)
    done += chunk
    report(q.r_traj, f"after {done}")
print("ALL DONE", flush=True)
