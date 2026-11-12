"""NPNEq at lag Delta with stopped windows: sum_t f_k(t) [r(min(t+Delta, tau_t)) - r(t)] = 0.
Swapped in for npneq inside the CuPy fit_transform; 30 paths/ensemble, gamma 0.1, ny 6, 4000 iterations."""
import os, sys, io, contextlib, argparse, importlib.util, time
import numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
EX = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples"
sys.path.insert(0, EX); sys.path.insert(0, os.path.join(EX, "..", "optimalrcs-cupy"))
spec = importlib.util.spec_from_file_location("orc", os.path.join(EX, "..", "optimalrcs-cupy", "optimalrcs.py"))
orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc)
import tis_data, metrics_tis as M
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
inside = ~b & (np.abs(v) < 3.5)
near = inside & (x < 0.3) & (np.abs(v) < 0.5)
tau_np = cp.asnumpy(M.stop_index(d.i_traj, b))
W = cp.asarray(d.path_weights)

for lag in (1, 4, 16):
    u = cp.asarray(np.minimum(np.arange(len(x)) + lag, tau_np))
    starts = cp.asarray(np.flatnonzero(~b & (np.minimum(np.arange(len(x)) + lag, tau_np) > np.arange(len(x)))))

    def npneq_lag(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk=None, w_traj=None):
        fa, fb, wt = fk[:, starts], fk[:, u[starts]], W[starts]
        akj = fa @ (fb * wt - fa * (wt + gamma)).T
        bb = fa @ (-(r_traj[u[starts]] - r_traj[starts]) * wt)
        al = cp.linalg.lstsq(akj, bb, rcond=None)[0]
        return cp.clip(r_traj + al @ fk, 0, 1)

    orc.nonparametrics.npneq = npneq_lag
    Y = cp.asarray(d.X)
    q = orc.CommittorNE(d.boundary0, d.boundary1, i_traj=d.i_traj, t_traj=d.t_traj,
                        path_weights=d.path_weights, group_traj=d.group_traj)
    np.random.seed(0); cp.random.seed(0); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        q.fit_transform(lambda: Y[:, np.random.randint(2)], basis_functions=lambda r_, y_, n_, f_: orc.basis_poly_ry(r_, y_, n_, f_), gamma=0.1, ny=6,
                        max_iter=4000, print_step=500, metrics_print=('iter', 'delta_x'), save_min_delta_zq=False)
    r = cp.asnumpy(q.r_traj); dx = q.metrics_history['delta_x'][-1]
    np.savez(f"{OUT}/fit_lag{lag}.npz", r=r)
    del q, Y; cp.get_default_memory_pool().free_all_blocks()
    err = r - q_ref
    zs = []
    for dt in (1, 4, 16, 64, 256, 1024):
        _, z = M.comp_zq_stopped_zscore(cp.asarray(r), b, d.i_traj, d.path_weights, dt, tau=cp.asarray(tau_np),
                                        g_traj=d.group_traj)
        zs.append(float(cp.max(cp.abs(z[5:]))))
    print(f"fit lag {lag:2d}: {time.time() - t0:5.0f} s, |dx| {dx:.4f} | RMS all {np.sqrt(np.mean(err[inside]**2)):.3f} "
          f"| near A slow: RMS {np.sqrt(np.mean(err[near]**2)):.3f} bias {err[near].mean():+.3f} "
          f"| max|z| lags 1,4,16,64,256,1024: " + " ".join(f"{z:.1f}" for z in zs), flush=True)
print("ALL DONE", flush=True)
