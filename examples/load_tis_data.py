"""Fit a non-equilibrium committor (CommittorNE) to (RE)TIS, REPPTIS or staple path data.

Example:
    python load_tis_data.py /path/to/sim --max-iter 20000 --history 0 1 2 4 8

Each ensemble folder `<tis_dir>/0NN/order.txt` holds the time in column 0,
the order parameter in column 1 and optional extra CVs after that. The order
parameter itself is always used as CV 0, so 1D simulations (time + order
only) work too. If `<tis_dir>/engine.py` exists, the committor is also
plotted on the potential that engine was run with.

Only accepted paths are used, each with its Monte Carlo weight, and each path
is cut at its first crossing of the ensemble's interface M (see tis_data.py).
Convergence is judged by the stopped Z_q of optimalrcs.metrics_tis, which is
valid for such segments: `zqs_z` (the worst |Z_q / standard error| over q and
lag times) should end up at about 2 or less.
"""
import argparse
import contextlib
import os
import sys

# Has to be set before TensorFlow is imported. With --hybrid the per-frame arrays
# live in host memory, where oneDNN's CPU memory pools would keep ~90 bytes per
# frame on top of them, for only ~10% more speed.
if "--hybrid" in sys.argv[1:]:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")  # no display on the remote host, so render straight to files
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import optimalrcs
from optimalrcs import nonparametrics

import tis_data

# plots_feps()/plots_tis() call plt.show() themselves and never return
# their figures; with show() a no-op the figures stay open to be saved.
plt.show = lambda *args, **kwargs: None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a non-equilibrium committor to (RE)TIS, REPPTIS or staple path data.")
    tis_data.add_data_arguments(parser)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--history", type=int, nargs="+", metavar="DT",
                        help="history delays in frames, e.g. --history 0 1 2 4 8 "
                             "(default: no history)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="regularization strength")
    parser.add_argument("--ny", type=int, default=6,
                        help="maximum polynomial degree of the basis")
    parser.add_argument("--print-step", type=int, default=500)
    parser.add_argument("--min-delta-x", type=float, default=1e-6,
                        help="stop when the RC changes less than this between print steps")
    parser.add_argument("--hybrid", action="store_true",
                        help="keep the per-frame arrays in host memory and send only one block of "
                             "frames at a time to the GPU, so the GPU memory used does not grow "
                             "with the data (block size: --chunk)")
    parser.add_argument("--cpu", action="store_true",
                        help="run on the CPU only (much slower training; the metrics take as long)")
    parser.add_argument("--chunk", type=int, metavar="FRAMES",
                        help="frames per block (default: "
                             f"{nonparametrics.npneq_chunk}); lower it to save GPU memory")
    tis_data.add_plot_arguments(
        parser, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    return parser.parse_args()


def main():
    args = parse_args()

    if args.cpu and args.hybrid:
        raise SystemExit("--cpu and --hybrid exclude each other")
    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
    # The GPU is shared. Without this TensorFlow reserves the entire card up front,
    # which would starve anyone else already running on it.
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.chunk is not None:
        nonparametrics.npneq_chunk = args.chunk

    os.makedirs(args.figure_dir, exist_ok=True)
    data = tis_data.prepare_data(args)
    X = data.X

    # --hybrid: everything that is created in this block, including what the figures
    # compute, stays in host memory; only the fit's blocks of frames go to the GPU.
    with tf.device("/CPU:0") if args.hybrid else contextlib.nullcontext():
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

        def comp_y():
            return X[:, np.random.randint(X.shape[1])]

        print(f"Starting CommittorNE training for {args.max_iter} iterations...")
        q.fit_transform(
            comp_y,
            history_delta_t=args.history,
            gamma=args.gamma,
            ny=args.ny,
            max_iter=args.max_iter,
            print_step=args.print_step,
            min_delta_x=args.min_delta_x,
            # The standard Z_q, cross-entropy, mse and AUC are biased for path segments.
            metrics_print=('iteration', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped',
                           'delta_x', 'time_elapsed'),
            save_min_metric='max_z_zq_stopped',
            chunk_device="/GPU:0" if args.hybrid else None,
        )
        print("CommittorNE training complete.")

        tis_data.save_fit_figures(q, args.figure_dir, "iteration")
        tis_data.plot_committor(data, np.asarray(q.r_traj), args)
    print(f"Figures written to {args.figure_dir}")


if __name__ == "__main__":
    main()
