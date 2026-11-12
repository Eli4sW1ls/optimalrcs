"""Loading and plotting helpers shared by load_tis_data.py (TensorFlow) and load_tis_data_cupy.py.

Reads (RE)TIS, REPPTIS and staple output in the PyRETIS layout: ensemble folders
`<tis_dir>/0NN/` with `order.txt` (time in column 0, order parameter in column 1,
optional extra CVs after that) and `pathensemble.txt`.

Only accepted paths are used, each with its Monte Carlo weight from
`pathensemble.txt` (computed with tistools). Every ensemble requires its paths
to cross an interface M, which conditions the future of the frames before that
crossing; those frames are cut, so each path is used from its first frame past
M onward (see `first_frame_past_m`). What remains are segments that end at a
stopping time, which `metrics_tis` validates the committor on.
"""
import ast
import contextlib
import glob
import importlib
import io
import os
import re
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_CYCLE_RE = re.compile(r"# Cycle:\s*(\d+),\s*status:\s*(\S+?),")


def read_interfaces(tis_dir, interfaces_file=None):
    """Interface positions, read with tistools from `interfaces_file` or else
    the first of repptis.rst, retis.rst or logging.log found in `tis_dir`."""
    from tistools.reading import read_inputfile
    candidates = [interfaces_file] if interfaces_file else [
        os.path.join(tis_dir, name) for name in ("repptis.rst", "retis.rst", "logging.log")]
    for path in candidates:
        if os.path.isfile(path):
            interfaces, _, _ = read_inputfile(path)
            print(f"Interfaces from {path}: {interfaces}")
            return interfaces
    raise SystemExit(f"No interface file found (tried {', '.join(candidates)})")


def read_run_settings(tis_dir):
    """Dimension of the simulated system and whether `order.txt` also holds velocities.

    Read from the settings the toy TIS code writes to `logging.log` ('dim' and
    'v_ord'), else from `dimensions = N` in a PyRETIS repptis.rst/retis.rst.
    Older logs do not record 'v_ord'; then it is inferred from the first frame
    of an `order.txt`: with velocities a row holds 2 * dim values after the time.
    Returns (dim or None if not recorded, v_ord).
    """
    dim, v_ord = None, None
    log = os.path.join(tis_dir, "logging.log")
    if os.path.isfile(log):
        with open(log) as f:
            text = f.read()
        match = re.search(r"'dim':\s*(\d+)", text)
        dim = int(match.group(1)) if match else None
        match = re.search(r"'v_ord':\s*(True|False)", text)
        v_ord = None if match is None else match.group(1) == "True"
    if dim is None:
        for name in ("repptis.rst", "retis.rst"):
            path = os.path.join(tis_dir, name)
            if os.path.isfile(path):
                with open(path) as f:
                    match = re.search(r"^\s*dimensions\s*=\s*(\d+)", f.read(), re.MULTILINE)
                if match:
                    dim = int(match.group(1))
                    break
    if v_ord is None:
        v_ord = False
        orders = sorted(glob.glob(os.path.join(tis_dir, "0[0-9][0-9]", "order.txt")))
        if dim is not None and orders:
            with open(orders[0]) as f:
                n_values = next(len(line.split()) - 1 for line in f if not line.startswith("#"))
            v_ord = n_values == 2 * dim
    return dim, v_ord


def read_path_info(folder, ensemble_index, n_ensembles, weighting):
    """Per-cycle Monte Carlo weight and path metadata from `<folder>/pathensemble.txt`.

    Weights follow tistools. "staple" is `get_weights_staple`: each accepted
    path counts once for every cycle it stays the current path (rejected moves
    re-count it), times the high-acceptance factor for shot LML/RMR paths.
    "rejections" is `get_weights`, the re-counting alone. "none" gives every
    accepted path weight 1. Rejected cycles get weight 0.

    Returns {cycle: (weight, length, generation, core_start, core_end)}, with the
    core as recorded in `staridx` ((0, 0) for paths that are all core).
    """
    from tistools.reading import get_weights, read_pathensemble, set_flags_ACC_REJ
    pe = read_pathensemble(os.path.join(folder, "pathensemble.txt"))
    acc_flags, rej_flags = set_flags_ACC_REJ()
    with contextlib.redirect_stdout(io.StringIO()):  # the tistools helpers print regardless of verbose
        if weighting == "staple":
            from tistools.istar_analysis import get_weights_staple
            weights = get_weights_staple(ensemble_index, pe.flags, pe.generation, pe.lmrs,
                                         n_ensembles, acc_flags, rej_flags, verbose=False)
        elif weighting == "rejections":
            weights, _ = get_weights(pe.flags, acc_flags, rej_flags, verbose=False)
        else:
            weights = (pe.flags == "ACC").astype(int)
    return {cycle: (weight, length, generation, core_start, core_end)
            for cycle, weight, length, generation, (core_start, core_end)
            in zip(pe.cyclenumbers, weights, pe.lengths, pe.generation, pe.istar_idx)}


def first_frame_past_m(order, core_start, lambda_m):
    """Index of the first core frame on the far side of M, or None if there is none.

    The core is the part of the path inside the ensemble's [L, R] window; the
    frame before it is the L or R crossing it starts from. Staple paths record
    the core start in `staridx`, which is 0 for paths that are all core
    (REPPTIS, RETIS and [0-] paths), whose frame 0 is then the starting crossing.
    Backward staple extensions can cross M too, so the search starts at the core.

    The ensemble only holds paths whose core crosses M, so before that crossing
    each step is conditioned on a future crossing; from this frame on the
    dynamics is unconditioned up to the path's end (a stopping rule).
    """
    s = max(core_start, 1)
    start_side = order[s - 1] > lambda_m
    past = np.flatnonzero((order[s:] > lambda_m) != start_side)
    return s + past[0] if past.size else None


def last_frame_before_m(order, core_start, core_end, lambda_m):
    """Index of the last core frame on the far side of M from where the core ends, or None.

    The time-reversed counterpart of `first_frame_past_m`. The core's last frame
    is followed by the L or R crossing (or A/B frame) it ends at; staple paths
    record it in `staridx`, paths that are all core end at their last frame. Read
    backwards, the path satisfies the ensemble's M condition from this frame on,
    and its start is where the reversed path stops (a window exit, a completed
    turn of the backward extension, or A/B). For time-reversible dynamics at
    equilibrium the frames up to here, read backwards, are therefore a valid
    sample of the dynamics too.
    """
    e = len(order) - 2 if core_start == 0 else min(core_end, len(order) - 2)
    end_side = order[e + 1] > lambda_m
    before = np.flatnonzero((order[:e + 1] > lambda_m) != end_side)
    return before[-1] if before.size else None


def limit_piece_length(piece, max_length, long_pieces="truncate"):
    """Apply the piece-length limit: a piece longer than `max_length` frames is cut
    to its first `max_length` frames ('truncate') or dropped ('discard', returns None).

    Truncating is a stopping rule (stop at the earlier of the path's own end and a
    fixed time after the piece's start), so the truncated pieces are still valid
    stopped segments for the committor. Discarding selects pieces on their future
    (how long they take to end) and biases the data; it is for comparison only.
    """
    if max_length is None or len(piece) <= max_length:
        return piece
    if long_pieces == "discard":
        return None
    if long_pieces != "truncate":
        raise ValueError(f"long_pieces must be 'truncate' or 'discard', got {long_pieces!r}")
    return piece[:max_length]


def load_order_file(order_file, path_info, lambda_m=None, burn_in=0, max_paths=None, rng=None,
                    reversed_pieces=False, odd_columns=(), fd_velocity=False, fd_columns=(1,),
                    max_length=None, long_pieces="truncate"):
    """Load accepted paths and their Monte Carlo weights from one ensemble's `order.txt`.

    An `order.txt` file holds one or more paths, each introduced by a
    `# Cycle: <n>, status: <FLAG>, ...` header line. Within a path, every
    frame line holds the time in column 0, the order parameter in column 1,
    and any additional collective variables (CVs) in the remaining columns.
    The returned CVs include the order parameter as their first column.

    Paths are kept if accepted, not the initial load ('ld') path, from cycle
    `burn_in` on, and with a nonzero weight in `path_info` (see `read_path_info`).
    With `lambda_m`, each path is cut to start at `first_frame_past_m`; with
    `reversed_pieces` as well, the frames from its start up to
    `last_frame_before_m` are added read backwards (time t_end - t), which needs
    time-reversible dynamics at equilibrium; the columns in `odd_columns`
    (velocities) change sign there. If `max_paths` is given, at most that
    many paths are kept, drawn at random (in their original order); they keep
    their weights, so the subset stays unbiased. With `max_length`, every piece
    (after the cut at M) longer than that many frames is truncated or discarded,
    as `long_pieces` says (see `limit_piece_length`).

    Returns per-piece cvs, order, time, weight and path-number lists (both pieces
    of a path share its number), the number of paths dropped because their
    core never crosses M, and the number of pieces truncated or discarded by
    `max_length`.
    """
    # First pass: locate path boundaries, cycle numbers and acceptance flags by
    # scanning lines as plain text (cheap - no per-line list/float allocation).
    starts, cycles, accepted_flags = [], [], []
    n_data_lines = 0
    with open(order_file) as f:
        for line in f:
            match = _CYCLE_RE.match(line)
            if match:
                starts.append(n_data_lines)
                cycles.append(int(match.group(1)))
                accepted_flags.append(match.group(2) == "ACC")
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

    keep = []
    for i, (cycle, accepted) in enumerate(zip(cycles, accepted_flags)):
        if not accepted or cycle < burn_in:
            continue
        if cycle not in path_info:
            raise ValueError(f"{order_file}: cycle {cycle} is missing from pathensemble.txt")
        weight, length, generation, _, _ = path_info[cycle]
        if length != starts[i + 1] - starts[i]:
            raise ValueError(f"{order_file}: cycle {cycle} has {starts[i + 1] - starts[i]} "
                             f"frames, pathensemble.txt says {length}")
        if weight > 0 and generation != "ld":
            keep.append(i)
    subsample = max_paths is not None and len(keep) > max_paths
    if subsample:
        rng = rng if rng is not None else np.random.default_rng()
        keep = np.sort(rng.choice(keep, max_paths, replace=False))

    cvs, order, time, weights, paths = [], [], [], [], []
    n_no_crossing = n_long = 0
    for n_path, i in enumerate(keep):
        weight, _, _, core_start, core_end = path_info[cycles[i]]
        block = data[starts[i]:starts[i + 1]]
        if fd_velocity:
            # Change of the order parameter over the last d = fd_velocity frames, per frame,
            # on the whole path so that frames after the cut keep their true previous frames.
            # A longer d averages out the random position kick of single steps.
            # One velocity per position column in `fd_columns` (the order parameter, and the
            # other positions of a multi-dimensional run), appended in that order.
            idx = np.arange(len(block))
            prev = np.maximum(idx - int(fd_velocity), 0)
            vs = (block[:, fd_columns] - block[prev][:, fd_columns]) / np.maximum(idx - prev, 1)[:, None]
            if len(block) > 1:
                vs[0] = vs[1]
            block = np.column_stack([block, vs])
        pieces = [block]
        if lambda_m is not None:
            t_star = first_frame_past_m(block[:, 1], core_start, lambda_m)
            if t_star is None:
                n_no_crossing += 1
                continue
            pieces = [block[t_star:]]
            if reversed_pieces:
                t_last = last_frame_before_m(block[:, 1], core_start, core_end, lambda_m)
                if t_last is not None:
                    backwards = block[t_last::-1].copy()
                    backwards[:, 0] = block[t_last, 0] - backwards[:, 0]
                    backwards[:, list(odd_columns)] *= -1
                    if fd_velocity:
                        # Read backwards, the frames before frame i are i + 1, i + 2, ..., so
                        # the change over d frames is x[i] - x[i + d] (fewer at the path end).
                        idx = np.arange(t_last + 1)
                        nxt = np.minimum(idx + int(fd_velocity), len(block) - 1)
                        vrev = ((block[idx][:, fd_columns] - block[nxt][:, fd_columns])
                                / np.maximum(nxt - idx, 1)[:, None])
                        backwards[:, -len(fd_columns):] = vrev[::-1]
                    pieces.append(backwards)
        for piece in pieces:
            if max_length is not None and len(piece) > max_length:
                n_long += 1
                piece = limit_piece_length(piece, max_length, long_pieces)
                if piece is None:
                    continue
            if len(piece) < 2:
                continue  # no transition left to learn from
            if subsample:
                piece = piece.copy()  # views would keep the whole file's array alive
            time.append(piece[:, 0])
            order.append(piece[:, 1])
            cvs.append(piece[:, 1:])
            weights.append(weight)
            paths.append(n_path)
    return cvs, order, time, weights, paths, n_no_crossing, n_long


def load_tis_data(tis_dir, interfaces, ensemble_glob="0[0-9][0-9]", include_zero_minus=False,
                  weighting="rejections", cut_at_m=True, burn_in=0, paths_per_ensemble=None,
                  seed=None, reversed_pieces=False, odd_columns=(), fd_velocity=False, fd_columns=(1,),
                  max_length=None, long_pieces="truncate"):
    """Load per-path time/order/CV arrays and MC weights from all ensemble folders in `tis_dir`.

    Ensemble folders are expected at `tis_dir/<ensemble_glob>/`, each with an
    `order.txt` and `pathensemble.txt`, following the standard PyRETIS (RE)TIS,
    REPPTIS and staple output layout. Folder 0NN must cross interface
    max(NN - 1, 0): l_0 for [0-] and [0+-] (folders 000, 001), l_1 for 002, and
    so on. The [0-] folder is skipped unless `include_zero_minus` is set. With
    `paths_per_ensemble`, a random subset of that many paths is taken from
    each ensemble. With `reversed_pieces` (needs `cut_at_m`), every path also
    gives its piece before the last M crossing, read backwards, with the
    `order.txt` columns in `odd_columns` (velocities) negated. With `max_length`,
    pieces longer than that many frames are truncated or discarded (`long_pieces`).

    Returns per-piece cvs, order, time, weight and path-number lists; the path
    numbers are unique over all ensembles and shared by the two pieces of a path.
    """
    if reversed_pieces and not cut_at_m:
        raise ValueError("reversed pieces need the cut at M")
    rng = np.random.default_rng(seed)
    cvs, order, time, weights, paths = [], [], [], [], []
    folders = sorted(glob.glob(os.path.join(tis_dir, ensemble_glob)))
    n_ensembles = len(folders)
    print(f"Found {n_ensembles} ensemble folders in {tis_dir}")
    for folder in folders:
        k = int(os.path.basename(folder))
        if k == 0 and not include_zero_minus:
            continue
        order_file = os.path.join(folder, "order.txt")
        if not os.path.isfile(order_file):
            print(f"  {folder}: no order.txt, skipping")
            continue
        lambda_m = interfaces[max(k - 1, 0)] if cut_at_m else None
        path_info = read_path_info(folder, k, n_ensembles, weighting)
        c, o, t, w, pth, n_no_crossing, n_long = load_order_file(
            order_file, path_info, lambda_m=lambda_m, burn_in=burn_in,
            max_paths=paths_per_ensemble, rng=rng, reversed_pieces=reversed_pieces,
            odd_columns=odd_columns, fd_velocity=fd_velocity, fd_columns=fd_columns,
            max_length=max_length, long_pieces=long_pieces)
        offset = paths[-1] + 1 if paths else 0
        cvs += c
        order += o
        time += t
        weights += w
        paths += [offset + n for n in pth]
        n_frames = sum(len(p) for p in o)
        msg = (f"  {folder}: loaded {len(c)} {'pieces' if reversed_pieces else 'paths'}, "
               f"{n_frames} frames, total weight {sum(w)}")
        if lambda_m is not None:
            msg += f", cut at M = {lambda_m}"
        if n_no_crossing:
            msg += f" ({n_no_crossing} paths dropped: core never crosses M)"
        if n_long:
            msg += (f" ({n_long} pieces longer than {max_length} frames "
                    f"{'discarded' if long_pieces == 'discard' else 'truncated'})")
        print(msg)
    print(f"Loaded {len(cvs)} {'pieces' if reversed_pieces else 'paths'}, "
          f"{sum(len(p) for p in order)} frames total")
    return cvs, order, time, weights, paths


def _import_sim_module(sim_dir, module_name):
    """Import `module_name` (e.g. potentials.sjoelbak_angle) from the run directory."""
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
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(sim_dir)


def load_named_potential(sim_dir, spec, kwargs):
    """Build the potential `module:Class` from `<sim_dir>/potentials` with constructor `kwargs`."""
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"--potential must be module:Class, got {spec!r}")
    module = _import_sim_module(sim_dir, "potentials." + module_name)
    print(f"Potential from --potential: {spec}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})")
    return getattr(module, class_name)(**kwargs)


def load_engine_potential(sim_dir, engine_class):
    """Build the potential that `engine_class` in `<sim_dir>/engine.py` runs with.

    engine.py imports every available potential module, some of which need
    packages that may not be installed, so it is not imported. Instead its
    source is parsed for the active (last uncommented) `self.potential = ...`
    in `engine_class`, and only that potential's module is loaded. Returns None
    if the engine does not set its potential itself (it then comes from the
    run's settings).
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
    if not calls:
        return None
    call = max(calls, key=lambda c: c.lineno)
    name = call.func.id

    module = _import_sim_module(sim_dir, modules[name])
    print(f"Potential from {engine_class} in engine.py: {ast.unparse(call)}")
    return eval(compile(ast.Expression(call), "engine.py", "eval"),
                {"np": np, name: getattr(module, name)})


def read_order_columns(sim_dir, dim):
    """Which position each `order.txt` column is, for a `dim`-dimensional toy TIS run.

    The run's order.py computes the order columns from the phase point; for
    dim > 1 its `calculate` returns `[ph[0][i], ph[0][j], ...]`. This returns
    [i, j, ...] from the active (uncommented) return, or None if order.py has no
    such return.
    """
    path = os.path.join(sim_dir, "order.py")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        tree = ast.parse(f.read())

    def position_index(node):
        # ph[0][k] -> k
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name) and isinstance(node.slice, ast.Constant)
                and isinstance(node.value.slice, ast.Constant) and node.value.slice.value == 0):
            return node.slice.value
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.List) and len(node.value.elts) == dim:
            columns = [position_index(e) for e in node.value.elts]
            if all(c is not None for c in columns):
                return columns
    return None


def save_open_figures(figure_dir, label):
    """Save every open figure as `<figure_dir>/<label>[_n].png`, then close them."""
    numbers = plt.get_fignums()
    for i, number in enumerate(numbers):
        suffix = "" if len(numbers) == 1 else f"_{i + 1}"
        path = os.path.join(figure_dir, f"{label}{suffix}.png")
        plt.figure(number).savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close("all")


def add_data_arguments(parser):
    """Command-line options for reading and preparing the path data."""
    parser.add_argument("tis_dir", help="simulation directory with the 0NN ensemble folders")
    parser.add_argument("-a", "--lambda-a", type=float,
                        help="state A: order parameter <= lambda_A (default: first interface)")
    parser.add_argument("-b", "--lambda-b", type=float,
                        help="state B: order parameter >= lambda_B (default: last interface)")
    parser.add_argument("--interfaces-file",
                        help="file with the `interfaces = [...]` line (default: the first of "
                             "repptis.rst, retis.rst, logging.log in tis_dir)")
    parser.add_argument("--weights", choices=["rejections", "staple", "none"], default="rejections",
                        help="Monte Carlo path weights, from tistools: 'rejections' = each accepted "
                             "path counted once per cycle it stays the current path (default); "
                             "'staple' = that times the high-acceptance factor 2 for shot LML/RMR "
                             "paths; 'none' = every accepted path once. On a flat 1D staple test "
                             "only 'rejections' passed the stopped Z_q test for the exact committor.")
    parser.add_argument("--no-cut", action="store_true",
                        help="keep whole paths instead of cutting them at their first frame "
                             "past M (for comparison only: the cut frames are biased)")
    parser.add_argument("--reversed", action="store_true",
                        help="also use every path read backwards from its last M crossing to its "
                             "start, which recovers the frames the cut drops; needs time-reversible "
                             "dynamics at equilibrium. Velocity columns written with v_ord are "
                             "negated in the reversed pieces; other CVs must be even under time "
                             "reversal. Recommended for REPPTIS (halved the committor error on a "
                             "flat 1D test) and for staple paths with rejection weights (on the 2D "
                             "Brownian Z run the forward pieces alone are biased; see "
                             "docs/npneq_tis_notes.md section 9.1)")
    parser.add_argument("--fd-velocity", type=int, nargs="?", const=1, default=0, metavar="D",
                        help="add the change of each position (the order parameter, and for a "
                             "multi-dimensional run every position column) over the last D frames "
                             "(per frame; D = 1 if not given) as extra (last) CVs: a velocity estimate "
                             "from positions only, for underdamped dynamics when order.txt has no "
                             "velocities. Computed before the cut at M. A larger D averages out the "
                             "random position kick of single steps, which otherwise looks like extra "
                             "friction at lag 1.")
    parser.add_argument("--max-length", type=int, metavar="N",
                        help="limit every piece (after the cut at M) to N frames; longer pieces "
                             "are truncated or discarded, see --long-pieces (default: no limit)")
    parser.add_argument("--long-pieces", choices=["truncate", "discard"], default="truncate",
                        help="what --max-length does with longer pieces: 'truncate' keeps their "
                             "first N frames (a stopping rule, unbiased; default), 'discard' drops "
                             "them (selects on the future, biased; for comparison only)")
    parser.add_argument("--burn-in", type=int, default=0, metavar="N",
                        help="skip the first N cycles of every ensemble")
    parser.add_argument("--include-zero-minus", action="store_true",
                        help="also use the [0-] ensemble (folder 000)")
    parser.add_argument("--paths-per-ensemble", type=int, metavar="N",
                        help="randomly take at most N accepted paths from each ensemble, "
                             "keeping their weights (default: all)")
    parser.add_argument("--seed", type=int,
                        help="random seed for --paths-per-ensemble")
    parser.add_argument("--n-cvs", type=int, metavar="N",
                        help="train on only the first N CVs; CV 0 is the order parameter "
                             "(default: all CVs in order.txt)")


def add_plot_arguments(parser, default_figure_dir):
    """Command-line options for the committor-on-potential plot and the figure folder."""
    parser.add_argument("--engine-class",
                        help="engine in <tis_dir>/engine.py whose potential to plot "
                             "(default: LangevinEngine for 1 CV, ndLangevinEngine otherwise)")
    parser.add_argument("--no-potential", action="store_true",
                        help="skip the committor-on-potential plot")
    parser.add_argument("--potential", metavar="MODULE:CLASS",
                        help="potential to plot instead of the one in <tis_dir>/engine.py, as "
                             "module:Class from <tis_dir>/potentials, e.g. "
                             "sjoelbak_angle:RectangularGridWithAngledBarrierPotential (for runs that "
                             "pass the potential in their settings, which logging.log does not record "
                             "with its arguments)")
    parser.add_argument("--pot-kwargs", nargs="+", default=[], metavar="KEY=VALUE",
                        help="constructor arguments of --potential (Python literals), e.g. Lx=0.3 angle=45")
    parser.add_argument("--figure-dir", default=default_figure_dir)


def prepare_data(args):
    """Load the paths and build the per-frame arrays the fit needs.

    Fills in `args.lambda_a`/`args.lambda_b` from the interfaces when not given.
    Also returns `dim`, the simulated system's dimension if recorded (else None).
    Returns a namespace with X (the CVs to train on), X_all (all CVs, for plots),
    lam (order parameter), t_traj, i_traj (segment index), group_traj (path number:
    the forward and reversed piece of a path are not independent, and the standard
    errors of metrics_tis treat a path as one unit), path_weights (MC weight of
    each frame's path, averaging 1 so that --gamma keeps its meaning relative to
    the unweighted transition count), boundary0 (A) and boundary1 (B).
    """
    interfaces = read_interfaces(args.tis_dir, args.interfaces_file)
    if args.lambda_a is None:
        args.lambda_a = interfaces[0]
    if args.lambda_b is None:
        args.lambda_b = interfaces[-1]
    print(f"State A: order <= {args.lambda_a}, state B: order >= {args.lambda_b}")

    if args.reversed and args.no_cut:
        raise SystemExit("--reversed needs the cut at M (drop --no-cut)")
    dim, v_ord = read_run_settings(args.tis_dir)
    # order.txt columns: time, `dim` positions (order parameter first), then with
    # v_ord `dim` velocities, which change sign under time reversal.
    odd_columns = range(1 + dim, 1 + 2 * dim) if (v_ord and dim) else ()
    if args.reversed and v_ord and not dim:
        raise SystemExit("--reversed: order.txt holds velocities but the dimension is not recorded")
    cvs, order, time, weights, paths = load_tis_data(
        args.tis_dir, interfaces, include_zero_minus=args.include_zero_minus,
        weighting=args.weights, cut_at_m=not args.no_cut, burn_in=args.burn_in,
        paths_per_ensemble=args.paths_per_ensemble, seed=args.seed,
        reversed_pieces=args.reversed, odd_columns=odd_columns, fd_velocity=args.fd_velocity,
        fd_columns=list(range(1, 1 + dim)) if dim else [1],
        max_length=getattr(args, "max_length", None), long_pieces=getattr(args, "long_pieces", "truncate"))

    path_weights = np.concatenate([np.full(len(path), w, dtype=float)
                                   for path, w in zip(order, weights)])
    path_weights /= path_weights.mean()
    X_all = np.concatenate(cvs, axis=0)
    lam = np.concatenate(order, axis=0)
    t_traj = np.concatenate(time, axis=0)
    i_traj = np.concatenate([np.full(len(path), path_id, dtype=int)
                             for path_id, path in enumerate(cvs)])
    group_traj = np.concatenate([np.full(len(piece), n, dtype=int) for piece, n in zip(order, paths)])
    del cvs, order, time, weights, paths

    # Training may use fewer CVs, but plots always see the data's full dimension.
    X = X_all
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
    assert X.shape[0] == lam.size == t_traj.size == i_traj.size == path_weights.size == group_traj.size
    assert not np.any(boundary0 & boundary1)
    assert np.all(np.diff(i_traj) >= 0)
    assert np.all(np.diff(t_traj)[same_path] > 0)
    return types.SimpleNamespace(X=X, X_all=X_all, lam=lam, t_traj=t_traj, i_traj=i_traj,
                                 group_traj=group_traj, path_weights=path_weights,
                                 boundary0=boundary0, boundary1=boundary1, dim=dim)


def save_fit_figures(q, figure_dir, iteration_key):
    """Save the free-energy profiles, the stopped-segment validation plots and the
    convergence history of a fitted CommittorNE.

    The standard observed-vs-predicted plot is not saved: for path segments it
    only counts frames whose segment reaches A or B, which biases it. The Z_q
    panel of `feps` has the same problem; the stopped Z_q of `tis_diagnostics`
    replaces it. `iteration_key` is the name of the iteration counter in
    `q.metrics_history` ('iteration' for TensorFlow, 'iter' for CuPy).
    """
    q.plots_feps()
    save_open_figures(figure_dir, "feps")

    q.plots_tis()
    save_open_figures(figure_dir, "tis_diagnostics")

    # Convergence history: dotted blue is the whole run, solid red (right axis)
    # zooms in on the second half to show whether each metric has plateaued.
    if len(q.metrics_history.get(iteration_key, [])) > 1:
        q.plots_metrics(["delta_r2", "max_z_zq_stopped", "delta_x"])
        save_open_figures(figure_dir, "metrics")


def plot_committor(data, r_traj, args):
    """Save the committor against the order parameter and, if `<tis_dir>/engine.py`
    exists, on the potential that engine was run with."""
    lam, X_all = data.lam, data.X_all
    os.makedirs(args.figure_dir, exist_ok=True)
    np.savez(os.path.join(args.figure_dir, "committor.npz"), r=r_traj, lam=lam, X_all=X_all,
             i_traj=data.i_traj, t_traj=data.t_traj, path_weights=data.path_weights,
             boundary0=data.boundary0, boundary1=data.boundary1)
    print(f"  saved {os.path.join(args.figure_dir, 'committor.npz')}")
    fig, ax = plt.subplots()
    ax.scatter(lam, r_traj, s=1, alpha=0.2, edgecolors="none")
    for lam_boundary in (args.lambda_a, args.lambda_b):
        ax.axvline(lam_boundary, color="k", linestyle="--", lw=0.5)
    ax.set_xlabel("order parameter")
    ax.set_ylabel("committor")
    save_open_figures(args.figure_dir, "committor_vs_order")

    if args.no_potential:
        return
    # The number of CV columns is not the dimension: order.txt can also hold velocities.
    dim = data.dim if data.dim is not None else (1 if X_all.shape[1] == 1 else 2)
    if getattr(args, "potential", None):
        kwargs = {}
        for pair in args.pot_kwargs:
            key, _, value = pair.partition("=")
            try:
                kwargs[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                kwargs[key] = value
        potential = load_named_potential(args.tis_dir, args.potential, kwargs)
    else:
        if not os.path.isfile(os.path.join(args.tis_dir, "engine.py")):
            print("No engine.py in tis_dir, skipping the potential plot.")
            return
        engine_class = args.engine_class or ("LangevinEngine" if dim == 1 else "ndLangevinEngine")
        potential = load_engine_potential(args.tis_dir, engine_class)
        if potential is None:
            print(f"{engine_class} in engine.py does not set its potential (it comes from the run "
                  "settings); give it with --potential module:Class --pot-kwargs ... to plot it. "
                  "Skipping the potential plot.")
            return

    fig, ax = plt.subplots()
    if dim == 1:
        # 1D: the order parameter is the position the potential is defined on.
        grid = np.linspace(lam.min(), lam.max(), 500)
        ax.plot(grid, [potential.potential_and_force((x, 0.))[0] for x in grid], color="gray")
        ax.set_ylabel("potential")
        ax_q = ax.twinx()
        ax_q.scatter(lam, r_traj, c=r_traj, cmap="coolwarm", s=1, edgecolors="none")
        ax_q.set_ylabel("committor")
    else:
        # The order parameter (x-axis) and CV 1 (y-axis) are two of the positions;
        # order.py says which, so the potential can be evaluated on a grid of them.
        columns = read_order_columns(args.tis_dir, dim)
        if columns is None:
            columns = list(range(dim))
            print(f"order.py does not say which positions the order columns are; assuming {columns}")
        gx = np.linspace(lam.min(), lam.max(), 120)
        gy = np.linspace(X_all[:, 1].min(), X_all[:, 1].max(), 120)
        pot = np.empty((len(gy), len(gx)))
        pos = np.zeros(dim)
        for iy, y in enumerate(gy):
            for ix, x in enumerate(gx):
                pos[columns[0]], pos[columns[1]] = x, y
                pot[iy, ix] = potential.potential_and_force((pos.copy(), np.zeros(dim)))[0]
        # Contours over the sampled energy range, so steep walls do not take all levels.
        levels = np.linspace(*np.percentile(pot, [1, 95]), 15)
        ax.contour(gx, gy, pot, levels=levels, colors="0.55", linewidths=0.6)
        sc = ax.scatter(lam, X_all[:, 1], c=r_traj, cmap="coolwarm", s=2, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="committor")
        ax.set_ylabel("CV[1]")
    ax.set_xlabel("order parameter")
    save_open_figures(args.figure_dir, "committor_on_potential")
