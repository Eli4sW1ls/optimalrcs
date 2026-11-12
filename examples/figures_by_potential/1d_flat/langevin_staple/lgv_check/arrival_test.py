"""Near-A error of the fit: baseline vs a joint (r, x, v) basis vs an extra CV x + v/gamma.
30 paths/ensemble, gamma 0.1, ny 6, 4000 iterations, compared with the fine brute-force reference."""
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
x, v = d.X[:, 0], d.X[:, 1] / 2e-4          # position, velocity (gamma = m = 1, dt = 2e-4)
b = d.boundary0 | d.boundary1
ref = np.load(f"{OUT}/reference_fine.npz")
interp = RegularGridInterpolator((ref["xs"], ref["vs"]), ref["q"], bounds_error=False, fill_value=None)
q_ref = np.clip(interp(np.c_[np.clip(x, ref["xs"][0], ref["xs"][-1]), np.clip(v, ref["vs"][0], ref["vs"][-1])]), 0, 1)
inside = ~b & (np.abs(v) < 3.5)
near = inside & (x < 0.3) & (np.abs(v) < 0.5)
tau = M.stop_index(d.i_traj, b)

def scaled(a):
    a = cp.asarray(a); lo, hi = float(a.min()), float(a.max())
    return (2 * a - (lo + hi)) / (hi - lo)       # to [-1, 1]

xg, vg = scaled(x), scaled(v)

# x^j v^k never change, so they are computed once (for degree 6: 28 rows).
XV = {(j, k): xg ** j * vg ** k for j in range(7) for k in range(7 - j)}

def joint_basis(r, y, n, fenv):
    """Monomials r^i x^j v^k (i + j + k <= n) times the envelope; the drawn CV y is not used."""
    rs = r / cp.max(cp.abs(r))
    out = cp.empty(((n + 1) * (n + 2) * (n + 3) // 6, r.shape[0]), dtype=r.dtype)
    row, ri = 0, fenv.astype(r.dtype)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                cp.multiply(ri, XV[(j, k)], out=out[row]); row += 1
        ri = ri * rs
    return out

# Frames until an incoming particle reaches A at its current velocity (per-frame displacement v_fd).
v_frame = d.X[:, 1]
steps_to_a = np.where(v_frame < 0, (x - args.lambda_a) / np.maximum(-v_frame, 1e-12), np.inf)
arrival = np.exp(-steps_to_a / 2)
print(f"frames with an incoming particle within 5 frames of A: {np.mean(steps_to_a < 5) * 100:.2f}%", flush=True)
variants = {
    "arrival CV exp(-s/2)": dict(cvs=[x, v, arrival], basis=orc.basis_poly_ry),
}
for name, cfg in variants.items():
    Y = cp.asarray(np.stack(cfg["cvs"], 1))
    q = orc.CommittorNE(d.boundary0, d.boundary1, i_traj=d.i_traj, t_traj=d.t_traj,
                        path_weights=d.path_weights, group_traj=d.group_traj)
    np.random.seed(0); cp.random.seed(0); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        q.fit_transform(lambda: Y[:, np.random.randint(Y.shape[1])], basis_functions=cfg["basis"], gamma=0.1, ny=6,
                        max_iter=4000, print_step=500, metrics_print=('iter', 'delta_x'), save_min_delta_zq=False)
    r = cp.asnumpy(q.r_traj); dx = q.metrics_history['delta_x'][-1]
    np.savez(f"{OUT}/fit_basis_{name.split()[0]}.npz", r=r)
    del q, Y; cp.get_default_memory_pool().free_all_blocks()
    err = r - q_ref
    zs = []
    for dt in (1, 4, 16, 64, 256, 1024):
        _, z = M.comp_zq_stopped_zscore(cp.asarray(r), b, d.i_traj, d.path_weights, dt, tau=tau, g_traj=d.group_traj)
        zs.append(float(cp.max(cp.abs(z[5:]))))
    print(f"{name:22s}: {time.time() - t0:5.0f} s, |dx| {dx:.4f} | RMS all {np.sqrt(np.mean(err[inside]**2)):.3f} "
          f"| near A slow: RMS {np.sqrt(np.mean(err[near]**2)):.3f} bias {err[near].mean():+.3f} "
          f"| max|z| lags 1,4,16,64,256,1024: " + " ".join(f"{z:.1f}" for z in zs), flush=True)
print("ALL DONE", flush=True)
