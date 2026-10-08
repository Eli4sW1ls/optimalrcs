"""Fit a non-equilibrium committor (CommittorNE) to TIS path data, on the GPU with CuPy.

CuPy counterpart of load_tis_data.py (same arguments, same figures), using the
optimalrcs-cupy implementation instead of the TensorFlow one.

Example:
    python load_tis_data_cupy.py /path/to/sim -a 0.05 -b 0.85 --max-iter 20000 \
        --history 0 1 2 4 8

Each ensemble folder `<tis_dir>/0NN/order.txt` holds the time in column 0,
the order parameter in column 1 and optional extra CVs after that. The order
parameter itself is always used as CV 0, so 1D simulations (time + order
only) work too. If `<tis_dir>/engine.py` exists, the committor is also
plotted on the potential that engine was run with.
"""
import argparse
import ast
import glob
import importlib
import importlib.util
import os
import re
import sys
import types

import matplotlib
matplotlib.use("Agg")  # no display on the remote host, so render straight to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# plots_feps()/plots_obs_pred() call plt.show() themselves and never return
# their figures; with show() a no-op the figures stay open to be saved.
plt.show = lambda *args, **kwargs: None

_CYCLE_RE = re.compile(r"# Cycle:\s*\d+,\s*status:\s*(\S+?),")


def load_order_file(order_file, acc_only=True, max_paths=None, rng=None):
    """Load per-path time/order/CV arrays from a single TIS `order.txt` file.

    An `order.txt` file holds one or more paths, each introduced by a
    `# Cycle: <n>, status: <FLAG>, ...` header line. Within a path, every
    frame line holds the time in column 0, the order parameter in column 1,
    and any additional collective variables (CVs) in the remaining columns.
    The returned CVs include the order parameter as their first column.

    If `max_paths` is given, at most that many paths are kept, drawn at random
    (in their original order) from the eligible ones.
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
    # parser, instead of building millions of Python objects line by line.
    data = pd.read_csv(
        order_file, comment="#", header=None, sep=r"\s+",
    ).to_numpy(dtype=float)

    keep = [i for i in range(len(starts) - 1)
            if starts[i + 1] > starts[i] and (accepted_flags[i] or not acc_only)]
    subsample = max_paths is not None and len(keep) > max_paths
    if subsample:
        rng = rng if rng is not None else np.random.default_rng()
        keep = np.sort(rng.choice(keep, max_paths, replace=False))

    cvs, order, time = [], [], []
    for i in keep:
        block = data[starts[i]:starts[i + 1]]
        if subsample:
            block = block.copy()  # views would keep the whole file's array alive
        time.append(block[:, 0])
        order.append(block[:, 1])
        cvs.append(block[:, 1:])
    return cvs, order, time


def load_tis_data(tis_dir, ensemble_glob="0[0-9][0-9]", acc_only=True,
                  include_zero_minus=False, paths_per_ensemble=None, seed=None):
    """Load per-path time/order/CV arrays from all TIS ensemble folders in `tis_dir`.

    Ensemble folders are expected at `tis_dir/<ensemble_glob>/order.txt`,
    following the standard (RE)PPTIS output layout. The first folder is the
    [0-] ensemble and is skipped unless `include_zero_minus` is set. With
    `paths_per_ensemble`, a random subset of that many paths is taken from
    each ensemble.
    """
    rng = np.random.default_rng(seed)
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
        c, o, t = load_order_file(order_file, acc_only=acc_only,
                                  max_paths=paths_per_ensemble, rng=rng)
        cvs += c
        order += o
        time += t
        n_frames = sum(len(p) for p in o)
        print(f"  [{i + 1}/{len(folders)}] {folder}: "
              f"loaded {len(c)} paths, {n_frames} frames")
    print(f"Loaded {len(cvs)} paths, {sum(len(p) for p in order)} frames total")
    return cvs, order, time


def load_engine_potential(sim_dir, engine_class):
    """Build the potential that `engine_class` in `<sim_dir>/engine.py` runs with.

    engine.py imports every available potential module, some of which need
    packages that may not be installed, so it is not imported. Instead its
    source is parsed for the active (last uncommented) `self.potential = ...`
    in `engine_class`, and only that potential's module is loaded.
    """
    with open(os.path.join(sim_dir, "engine.py")) as f:
        tree = ast.parse(f.read())
    modules = {alias.asname or alias.name: node.module
               for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               for alias in node.names}
    engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == engine_class)
    calls = [n.value for n in ast.walk(engine)
             if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
             and any(isinstance(t, ast.Attribute) and t.attr == "potential" for t in n.targets)]
    call = max(calls, key=lambda c: c.lineno)
    name = call.func.id

    try:
        importlib.import_module("pyretis.forcefield.potential")
    except ImportError:
        # The toy potentials only import PyRETIS for the PotentialFunction base
        # class and never call into it, so an empty stand-in is enough.
        stub = types.ModuleType("pyretis.forcefield.potential")
        stub.PotentialFunction = type("PotentialFunction", (), {})
        sys.modules.update({"pyretis": types.ModuleType("pyretis"),
                            "pyretis.forcefield": types.ModuleType("pyretis.forcefield"),
                            "pyretis.forcefield.potential": stub})

    sys.path.insert(0, sim_dir)
    try:
        module = importlib.import_module(modules[name])
    finally:
        sys.path.remove(sim_dir)
    print(f"Potential from {engine_class} in engine.py: {ast.unparse(call)}")
    return eval(compile(ast.Expression(call), "engine.py", "eval"),
                {"np": np, name: getattr(module, name)})


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
        description="Fit a non-equilibrium committor to TIS path data (CuPy).")
    parser.add_argument("tis_dir", help="simulation directory with the 0NN ensemble folders")
    parser.add_argument("-a", "--lambda-a", type=float, required=True,
                        help="state A: order parameter <= lambda_A")
    parser.add_argument("-b", "--lambda-b", type=float, required=True,
                        help="state B: order parameter >= lambda_B")
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
    parser.add_argument("--include-zero-minus", action="store_true",
                        help="also use the [0-] ensemble (folder 000)")
    parser.add_argument("--paths-per-ensemble", type=int, metavar="N",
                        help="randomly take at most N accepted paths from each ensemble "
                             "(default: all)")
    parser.add_argument("--seed", type=int,
                        help="random seed for --paths-per-ensemble")
    parser.add_argument("--n-cvs", type=int, metavar="N",
                        help="train on only the first N CVs; CV 0 is the order parameter "
                             "(default: all CVs in order.txt)")
    parser.add_argument("--chunk", type=int, metavar="FRAMES",
                        help="frames per block when building the basis "
                             f"(default: {nonparametrics.npneq_chunk}); lower it to save GPU memory")
    parser.add_argument("--gpu-mem-limit", type=float, metavar="GIB",
                        help="cap CuPy's GPU memory pool at this many GiB (default: no cap)")
    parser.add_argument("--engine-class",
                        help="engine in <tis_dir>/engine.py whose potential to plot "
                             "(default: LangevinEngine for 1 CV, ndLangevinEngine otherwise)")
    parser.add_argument("--no-potential", action="store_true",
                        help="skip the committor-on-potential plot")
    parser.add_argument("--figure-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_cupy"))
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

    cvs, order, time = load_tis_data(args.tis_dir, include_zero_minus=args.include_zero_minus,
                                     paths_per_ensemble=args.paths_per_ensemble,
                                     seed=args.seed)

    X = np.concatenate(cvs, axis=0)
    lam = np.concatenate(order, axis=0)
    t_traj = np.concatenate(time, axis=0)
    i_traj = np.concatenate([
        np.full(len(path), path_id, dtype=int)
        for path_id, path in enumerate(cvs)
    ])
    del cvs, order, time

    # Training may use fewer CVs, but plots always see the data's full dimension.
    X_all = X
    if args.n_cvs is not None:
        if not 1 <= args.n_cvs <= X_all.shape[1]:
            raise SystemExit(f"--n-cvs must be between 1 and {X_all.shape[1]} "
                             f"(the number of CVs in order.txt)")
        X = X_all[:, :args.n_cvs]
    print(f"Using {X.shape[1]} of {X_all.shape[1]} CV(s)")

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

    # The CVs are copied to the GPU once, instead of every iteration.
    X_gpu = cp.asarray(X, dtype=q.prec)

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
    )
    print("CommittorNE training complete.")
    del X_gpu

    q.plots_feps()
    save_open_figures(args.figure_dir, "feps")

    q.plots_obs_pred()
    save_open_figures(args.figure_dir, "obs_pred")

    r_traj = cp.asnumpy(q.r_traj)
    fig, ax = plt.subplots()
    ax.scatter(lam, r_traj, s=1, alpha=0.2, edgecolors="none")
    for lam_boundary in (args.lambda_a, args.lambda_b):
        ax.axvline(lam_boundary, color="k", linestyle="--", lw=0.5)
    ax.set_xlabel("order parameter")
    ax.set_ylabel("committor")
    save_open_figures(args.figure_dir, "committor_vs_order")

    if args.no_potential:
        pass
    elif not os.path.isfile(os.path.join(args.tis_dir, "engine.py")):
        print("No engine.py in tis_dir, skipping the potential plot.")
    else:
        engine_class = args.engine_class or ("LangevinEngine" if X_all.shape[1] == 1
                                             else "ndLangevinEngine")
        potential = load_engine_potential(args.tis_dir, engine_class)

        fig, ax = plt.subplots()
        if X_all.shape[1] == 1:
            # 1D: the order parameter is the position the potential is defined on.
            grid = np.linspace(lam.min(), lam.max(), 500)
            ax.plot(grid, [potential.potential_and_force((x, 0.))[0] for x in grid],
                    color="gray")
            ax.set_ylabel("potential")
            ax_q = ax.twinx()
            ax_q.scatter(lam, r_traj, c=r_traj, cmap="coolwarm", s=1, edgecolors="none")
            ax_q.set_ylabel("committor")
        else:
            # Assumes the order parameter (x-axis) and CV 1 (y-axis) are the
            # coordinates the 2D potential is defined on.
            potential.plot_potential(ax)
            sc = ax.scatter(lam, X_all[:, 1], c=r_traj, cmap="coolwarm", s=2, edgecolors="none")
            fig.colorbar(sc, ax=ax, label="committor")
            ax.set_ylabel("CV[1]")
        ax.set_xlabel("order parameter")
        save_open_figures(args.figure_dir, "committor_on_potential")

    print(f"Figures written to {args.figure_dir}")


if __name__ == "__main__":
    main()
