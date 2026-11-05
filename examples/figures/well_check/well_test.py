"""sim_istarwell2rv: recorded velocity vs estimated (fd) velocity, NPNEq at lag 1 and 4, against the reference.
REVERSED=1 adds the mirrored pieces (--reversed); outputs then get the prefix rev_."""
import os, sys, io, contextlib, argparse, importlib.util, time
import numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
EX = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples"
sys.path.insert(0, EX); sys.path.insert(0, os.path.join(EX, "..", "optimalrcs-cupy"))
spec = importlib.util.spec_from_file_location("orc", os.path.join(EX, "..", "optimalrcs-cupy", "optimalrcs.py"))
orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc)
import tis_data, metrics_tis as M
OUT = os.path.dirname(os.path.abspath(__file__))
DT = 0.005
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
args = p.parse_args(["/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarwell2rv",
                     "--paths-per-ensemble", "2000", "--seed", "0", "--fd-velocity"]
                    + (["--reversed"] if os.environ.get("REVERSED") == "1" else []))
REV = "rev_" if os.environ.get("REVERSED") == "1" else ""   # REVERSED=1: add the mirrored pieces
with contextlib.redirect_stdout(io.StringIO()):
    d = tis_data.prepare_data(args)
x, v_rec, v_fd = d.X[:, 0], d.X[:, 1], d.X[:, 2] / DT          # columns: x, recorded v, fd estimate
b = d.boundary0 | d.boundary1
print(f"{len(x)} frames; corr(v_rec, v_fd) {np.corrcoef(v_rec, v_fd)[0, 1]:.4f}, "
      f"std(v_fd - v_rec) {np.std(v_fd - v_rec):.3f}", flush=True)
ref = np.load(f"{OUT}/reference.npz")
interp = RegularGridInterpolator((ref["xs"], ref["vs"]), ref["q"], bounds_error=False, fill_value=None)
def q_of(vel):
    q = np.clip(interp(np.c_[np.clip(x, ref["xs"][0], ref["xs"][-1]), np.clip(vel, ref["vs"][0], ref["vs"][-1])]), 0, 1)
    q[d.boundary0] = 0; q[d.boundary1] = 1
    return q
q_ref = q_of(v_rec)                                            # truth uses the true current velocity
tau_np = cp.asnumpy(M.stop_index(d.i_traj, b))
W = cp.asarray(d.path_weights)
inside = ~b & (np.abs(v_rec) < 3.5)
near_a = inside & (x < -0.25) & (np.abs(v_rec) < 0.5)
near_b = inside & (x > 0.25) & (np.abs(v_rec) < 0.5)
LAGS = (1, 2, 4, 16, 64, 256)

def zrow(r):
    out = []
    for dt in LAGS:
        _, z = M.comp_zq_stopped_zscore(cp.asarray(r), b, d.i_traj, d.path_weights, dt, tau=cp.asarray(tau_np),
                                        g_traj=d.group_traj)
        out.append(float(cp.max(cp.abs(z[5:]))))
    return " ".join(f"{z:4.1f}" for z in out)

print(f"true q with recorded v      : max|z| lags {LAGS}: {zrow(q_ref)}", flush=True)
print(f"true q with fd-estimated v  : max|z| lags {LAGS}: {zrow(q_of(v_fd))}", flush=True)

for vname, vel in (("recorded v", v_rec), ("fd v", v_fd)):
    if os.environ.get("ONLY") and not vname.startswith(os.environ["ONLY"]):   # ONLY=fd or ONLY=recorded
        continue
    for lag in (1, 4):
        n = len(x)
        u = cp.asarray(np.minimum(np.arange(n) + lag, tau_np))
        starts = cp.asarray(np.flatnonzero(~b & (np.minimum(np.arange(n) + lag, tau_np) > np.arange(n))))

        def npneq_lag(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk=None, w_traj=None, **kw):
            akj, bb = 0, 0
            for s0 in range(0, starts.size, 500_000):
                st = starts[s0:s0 + 500_000]
                fa, fb, wt = fk[:, st], fk[:, u[st]], W[st]
                akj = akj + fa @ (fb * wt - fa * (wt + gamma)).T
                bb = bb + fa @ (-(r_traj[u[st]] - r_traj[st]) * wt)
            al = cp.linalg.lstsq(akj, bb, rcond=None)[0]
            return cp.clip(r_traj + al @ fk, 0, 1)

        orc.nonparametrics.npneq = npneq_lag
        Y = cp.asarray(np.c_[x, vel])
        q = orc.CommittorNE(d.boundary0, d.boundary1, i_traj=d.i_traj, t_traj=d.t_traj,
                            path_weights=d.path_weights, group_traj=d.group_traj)
        np.random.seed(0); cp.random.seed(0); t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            q.fit_transform(lambda: Y[:, np.random.randint(2)],
                            basis_functions=lambda r_, y_, n_, f_: orc.basis_poly_ry(r_, y_, n_, f_),
                            gamma=0.1, ny=6, max_iter=4000, print_step=500, metrics_print=('iter', 'delta_x'),
                            save_min_delta_zq=False)
        r = cp.asnumpy(q.r_traj); dx = q.metrics_history['delta_x'][-1]
        np.savez(f"{OUT}/fit_{REV}{vname.split()[0]}_lag{lag}.npz", r=r)
        del q, Y; cp.get_default_memory_pool().free_all_blocks()
        err = r - q_ref
        print(f"{vname:10s} lag {lag}: {time.time() - t0:5.0f} s, |dx| {dx:.4f} | RMS all {np.sqrt(np.mean(err[inside]**2)):.3f} "
              f"| near A slow bias {err[near_a].mean():+.3f} (RMS {np.sqrt(np.mean(err[near_a]**2)):.3f}) "
              f"| near B slow bias {err[near_b].mean():+.3f} | max|z| lags {LAGS}: {zrow(r)}", flush=True)
print("ALL DONE", flush=True)
