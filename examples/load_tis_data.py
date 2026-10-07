"""Fit a non-equilibrium committor (CommittorNE) to TIS path data.

Example:
    python load_tis_data.py /path/to/sim -a 0.05 -b 0.85 --max-iter 20000 \
        --history 0 1 2 4 8 --potential potentials/sjoelbak.py

Each ensemble folder `<tis_dir>/0NN/order.txt` holds the time in column 0,
the order parameter in column 1 and optional extra CVs after that. The order
parameter itself is always used as CV 0, so 1D simulations (time + order
only) work too.
"""
import argparse
import glob
import importlib.util
import os
import re

import matplotlib
matplotlib.use("Agg")  # no display on the remote host, so render straight to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import optimalrcs

# plots_feps()/plots_obs_pred() call plt.show() themselves and never return
# their figures; with show() a no-op the figures stay open to be saved.
plt.show = lambda *args, **kwargs: None

_CYCLE_RE = re.compile(r"# Cycle:\s*\d+,\s*status:\s*(\S+?),")


def load_order_file(order_file, acc_only=True):
    """Load per-path time/order/CV arrays from a single TIS `order.txt` file.

    An `order.txt` file holds one or more paths, each introduced by a
    `# Cycle: <n>, status: <FLAG>, ...` header line. Within a path, every
    frame line holds the time in column 0, the order parameter in column 1,
    and any additional collective variables (CVs) in the remaining columns.
    The returned CVs include the order parameter as their first column.
    """
    # First pass: locate path boundaries and acceptance flags by scanning
    # lines as plain text (cheap - no per-line list/float allocation).
    starts, accepted_flags = [], []
    n_data_lines = 0
    with open(order_file) as f:
        for line in f:
            match = _CYCLE_RE.match(line)
            if match:
                starts.append(n_data_lines)
                accepted_flags.append(match.group(1) == "ACC")
                continue
            if line.startswith("#"):
                continue
            n_data_lines += 1
    starts.append(n_data_lines)

    # Second pass: parse all numeric data in one bulk call with pandas' C
    # parser. This avoids building millions of Python list/string objects
    # (as a line-by-line float() loop would), which is what previously blew
    # up memory and runtime on multi-million-line order.txt files.
    data = pd.read_csv(
        order_file, comment="#", header=None, sep=r"\s+",
    ).to_numpy(dtype=float)

    cvs, order, time = [], [], []
    for i in range(len(starts) - 1):
        lo, hi = starts[i], starts[i + 1]
        if hi <= lo or not (accepted_flags[i] or not acc_only):
            continue
        block = data[lo:hi]
        time.append(block[:, 0])
        order.append(block[:, 1])
        cvs.append(block[:, 1:])
    return cvs, order, time


def load_tis_data(tis_dir, ensemble_glob="0[0-9][0-9]", acc_only=True,
                  include_zero_minus=False):
    """Load per-path time/order/CV arrays from all TIS ensemble folders in `tis_dir`.

    Ensemble folders are expected at `tis_dir/<ensemble_glob>/order.txt`,
    following the standard (RE)PPTIS output layout. The first folder is the
    [0-] ensemble and is skipped unless `include_zero_minus` is set.
    """
    cvs, order, time = [], [], []
    folders = sorted(glob.glob(os.path.join(tis_dir, ensemble_glob)))
    if not include_zero_minus:
        folders = folders[1:]
    print(f"Found {len(folders)} ensemble folders in {tis_dir}")
    for i, folder in enumerate(folders):
        order_file = os.path.join(folder, "order.txt")
        if not os.path.isfile(order_file):
            print(f"  [{i + 1}/{len(folders)}] {folder}: no order.txt, skipping")
            continue
        c, o, t = load_order_file(order_file, acc_only=acc_only)
        cvs += c
        order += o
        time += t
        n_frames = sum(len(p) for p in o)
        print(f"  [{i + 1}/{len(folders)}] {folder}: "
              f"loaded {len(c)} paths, {n_frames} frames")
    print(f"Loaded {len(cvs)} paths, {sum(len(p) for p in order)} frames total")
    return cvs, order, time


def save_open_figures(figure_dir, label):
    """Save every open figure as `<figure_dir>/<label>[_n].png`, then close them."""
    numbers = plt.get_fignums()
    for i, number in enumerate(numbers):
        suffix = "" if len(numbers) == 1 else f"_{i + 1}"
        path = os.path.join(figure_dir, f"{label}{suffix}.png")
        plt.figure(number).savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close("all")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a non-equilibrium committor to TIS path data.")
    parser.add_argument("tis_dir", help="simulation directory with the 0NN ensemble folders")
    parser.add_argument("-a", "--lambda-a", type=float, required=True,
                        help="state A: order parameter <= lambda_A")
    parser.add_argument("-b", "--lambda-b", type=float, required=True,
                        help="state B: order parameter >= lambda_B")
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
    parser.add_argument("--include-zero-minus", action="store_true",
                        help="also use the [0-] ensemble (folder 000)")
    parser.add_argument("--potential",
                        help="2D potential module (absolute, or relative to tis_dir) "
                             "to plot the committor on")
    parser.add_argument("--potential-class", default="RectangularGridWithBarrierPotential")
    parser.add_argument("--figure-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    return parser.parse_args()


def main():
    args = parse_args()

    # The GPU is shared. Without this TensorFlow reserves the entire card up front,
    # which would starve anyone else already running on it.
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    os.makedirs(args.figure_dir, exist_ok=True)

    cvs, order, time = load_tis_data(args.tis_dir, include_zero_minus=args.include_zero_minus)

    X = np.concatenate(cvs, axis=0)
    lam = np.concatenate(order, axis=0)
    t_traj = np.concatenate(time, axis=0)
    i_traj = np.concatenate([
        np.full(len(path), path_id, dtype=int)
        for path_id, path in enumerate(cvs)
    ])
    del cvs, order, time
    print(f"Using {X.shape[1]} CV(s)")

    # Define basin membership from the same operational-state definitions used in TIS.
    boundary0 = lam <= args.lambda_a  # state A: q = 0
    boundary1 = lam >= args.lambda_b  # state B: q = 1
    print(f"Frames in A: {boundary0.sum()}, in B: {boundary1.sum()}")

    # Validate the prepared data before fitting.
    same_path = i_traj[1:] == i_traj[:-1]
    assert X.shape[0] == lam.size == t_traj.size == i_traj.size
    assert not np.any(boundary0 & boundary1)
    assert np.all(np.diff(i_traj) >= 0)
    assert np.all(np.diff(t_traj)[same_path] > 0)

    q = optimalrcs.CommittorNE(
        boundary0=boundary0,
        boundary1=boundary1,
        i_traj=i_traj,
        t_traj=t_traj,
    )

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
    )
    print("CommittorNE training complete.")

    q.plots_feps()
    save_open_figures(args.figure_dir, "feps")

    q.plots_obs_pred()
    save_open_figures(args.figure_dir, "obs_pred")

    r_traj = np.asarray(q.r_traj)
    fig, ax = plt.subplots()
    ax.scatter(lam, r_traj, s=1, alpha=0.2, edgecolors="none")
    for lam_boundary in (args.lambda_a, args.lambda_b):
        ax.axvline(lam_boundary, color="k", linestyle="--", lw=0.5)
    ax.set_xlabel("order parameter")
    ax.set_ylabel("committor")
    save_open_figures(args.figure_dir, "committor_vs_order")

    if args.potential:
        if X.shape[1] < 2:
            print("Skipping the potential plot: it needs a second CV (2D simulation).")
        else:
            # Assumes the order parameter (x-axis) and CV 1 (y-axis) are the
            # coordinates the 2D potential is defined on.
            potential_file = os.path.join(args.tis_dir, args.potential)
            spec = importlib.util.spec_from_file_location("potential_module", potential_file)
            potential_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(potential_module)
            potential = getattr(potential_module, args.potential_class)()

            fig, ax = plt.subplots()
            potential.plot_potential(ax)
            sc = ax.scatter(lam, X[:, 1], c=r_traj, cmap="coolwarm", s=2, edgecolors="none")
            fig.colorbar(sc, ax=ax, label="committor")
            ax.set_xlabel("order parameter")
            ax.set_ylabel("CV[1]")
            save_open_figures(args.figure_dir, "committor_on_potential")

    print(f"Figures written to {args.figure_dir}")


if __name__ == "__main__":
    main()
