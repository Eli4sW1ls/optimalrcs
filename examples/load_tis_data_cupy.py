"""Fit a non-equilibrium committor (CommittorNE) to (RE)TIS, REPPTIS or staple path data,
on the GPU with CuPy.

CuPy counterpart of load_tis_data.py (same arguments, same figures), using the
optimalrcs-cupy implementation instead of the TensorFlow one.

Example:
    python load_tis_data_cupy.py /path/to/sim --max-iter 20000 --history 0 1 2 4 8

Each ensemble folder `<tis_dir>/0NN/order.txt` holds the time in column 0,
the order parameter in column 1 and optional extra CVs after that. The order
parameter itself is always used as CV 0, so 1D simulations (time + order
only) work too. If `<tis_dir>/engine.py` exists, the committor is also
plotted on the potential that engine was run with.

Only accepted paths are used, each with its Monte Carlo weight, and each path
is cut at its first crossing of the ensemble's interface M (see tis_data.py).
Convergence is judged by the stopped Z_q of metrics_tis, which is valid for
such segments: `zqs_z` (the worst |Z_q / standard error| over q and lag times)
should end up at about 2 or less.
"""
import argparse
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display on the remote host, so render straight to files
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

# optimalrcs-cupy is a folder of flat modules (`import metrics`, ...), not a
# package, and its optimalrcs.py shares its name with the TensorFlow package,
# so it is loaded from its file under a name of its own.
_CUPY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "optimalrcs-cupy")
sys.path.insert(0, os.path.abspath(_CUPY_DIR))
_spec = importlib.util.spec_from_file_location("optimalrcs_cupy", os.path.join(_CUPY_DIR, "optimalrcs.py"))
optimalrcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(optimalrcs)
import nonparametrics  # noqa: E402  (the optimalrcs-cupy module)

import tis_data  # noqa: E402

# plots_feps()/plots_tis() call plt.show() themselves and never return
# their figures; with show() a no-op the figures stay open to be saved.
plt.show = lambda *args, **kwargs: None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a non-equilibrium committor to (RE)TIS, REPPTIS or staple path data (CuPy).")
    tis_data.add_data_arguments(parser)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--history", type=int, nargs="+", metavar="DT",
                        help="history delays in frames, e.g. --history 0 1 2 4 8 "
                             "(default: no history)")
    parser.add_argument("--history-shift-type", default="r(t0)",
                        choices=["r(t0)", "r(t)", "r(t-d)", "zero", "none"],
                        help="value used for y(t-d) when t-d falls before the start of the path: "
                             "r(t0) = the path's first frame, as in load_tis_data.py (default); "
                             "none = the older CuPy behaviour (0, faster)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="regularization strength")
    parser.add_argument("--ny", type=int, default=6,
                        help="maximum polynomial degree of the basis")
    parser.add_argument("--print-step", type=int, default=500)
    parser.add_argument("--min-delta-x", type=float, default=1e-6,
                        help="stop when the RC changes less than this between print steps")
    parser.add_argument("--chunk", type=int, metavar="FRAMES",
                        help="frames per block when building the basis "
                             f"(default: {nonparametrics.npneq_chunk}); lower it to save GPU memory")
    parser.add_argument("--gpu-mem-limit", type=float, metavar="GIB",
                        help="cap CuPy's GPU memory pool at this many GiB (default: no cap)")
    tis_data.add_plot_arguments(
        parser, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_cupy"))
    return parser.parse_args()


def main():
    args = parse_args()

    # CuPy allocates GPU memory on demand (nothing is reserved up front), but its
    # pool keeps freed blocks for reuse; on a shared GPU the pool can be capped.
    if args.gpu_mem_limit is not None:
        cp.get_default_memory_pool().set_limit(size=int(args.gpu_mem_limit * 2**30))
    if args.chunk is not None:
        nonparametrics.npneq_chunk = args.chunk

    os.makedirs(args.figure_dir, exist_ok=True)
    data = tis_data.prepare_data(args)

    q = optimalrcs.CommittorNE(
        boundary0=data.boundary0,
        boundary1=data.boundary1,
        i_traj=data.i_traj,
        t_traj=data.t_traj,
        path_weights=data.path_weights,
        group_traj=data.group_traj,
    )

    # Paths stop at their first A/B frame. Treat those ends as traps the path stays
    # in (longer than any lag), so the standard Z_q and delta_r2 keep the windows
    # that run into A or B instead of dropping them.
    q.set_fixed_traj_length_trap(data.boundary0 | data.boundary1, 2 ** 16)

    # The CVs are copied to the GPU once, instead of every iteration.
    X_gpu = cp.asarray(data.X, dtype=q.prec)

    def comp_y():
        return X_gpu[:, np.random.randint(X_gpu.shape[1])]

    shift_type = {"none": None, "zero": "0"}.get(args.history_shift_type, args.history_shift_type)
    print(f"Starting CommittorNE training for {args.max_iter} iterations...")
    q.fit_transform(
        comp_y,
        history_delta_t=args.history,
        history_shift_type=shift_type,
        gamma=args.gamma,
        ny=args.ny,
        max_iter=args.max_iter,
        print_step=args.print_step,
        min_delta_x=args.min_delta_x,
        # The standard Z_q, cross-entropy, mse and AUC are biased for path segments.
        metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped',
                       'delta_x', 'time_elapsed'),
        save_min_metric='max_z_zq_stopped',
    )
    print("CommittorNE training complete.")
    del X_gpu

    tis_data.save_fit_figures(q, args.figure_dir, "iter")
    tis_data.plot_committor(data, cp.asnumpy(q.r_traj), args)
    print(f"Figures written to {args.figure_dir}")


if __name__ == "__main__":
    main()
