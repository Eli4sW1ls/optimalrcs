"""Reading infretis runs (TIS/RETIS, REPPTIS and StapleTIS) for the committor fits.

The counterpart of the PyRETIS/toytis reader in `tis_data.py`: it returns the same per-piece lists
(cvs, order, time, weight, path number), so `tis_data.prepare_data` and everything after it are shared.

infretis layout (simulation directory):
  infretis.toml         interfaces, lambda_minus_one, mode ("staple", "pp" or plain RETIS) and the swap settings
  infretis_data.txt     one tab-separated row per finished path: number, length, max OP, [min OP, ptype,]
                        then path_f and path_w for every ensemble ([0-], [0+], [1+], ...), and for staple runs
                        the generating move and ensemble. Paths still alive at the end are not in it.
  load/<pnr>/order.txt  time and order parameter (plus any extra CVs the order parameter returns) per frame,
                        or an `inft store_txt -order` h5 archive with the same arrays as <pnr>/order.txt.
The order parameters are read with inftools (`inftools.istar.reading.OrderSource`), the staple cores with
infretis' own `StaplePath.get_sh_region`, so the definitions are those of the sampler.

Weights. path_f[p, e] is the occupancy of path p in ensemble e summed over the cycles it was alive (the
infinite-swap fraction, or 1 per cycle without infinite swap): the infretis counterpart of the repeat count
('rejections' in tis_data). path_w[p, e] is the weight the sampler put into the W matrix: 1 for valid
paths, the high-acceptance weight for wire fencing, and for StapleTIS 0.5 for LML/RMR paths in the inner
ensembles. infretis/inftools analyse with path_f / path_w ('staple' in tis_data: the toytis staple weights
are the same factor 2). Which one is right depends on how the sampler used w; see `default_weighting` and
docs/npneq_tis_notes.md (section 15).

As in tis_data, every (path, ensemble) pair with a nonzero weight gives one piece: the path cut at its
first frame past the ensemble's middle interface M (searched from the start of the ensemble's core), and
with `reversed_pieces` also the part before its last M crossing, read backwards.
"""
import os
import sys

import numpy as np

# infretis / inftools sources: the installed packages may be older copies without StapleTIS, so the
# development checkouts are put first on the path (override with $INFRETIS_SRC / $INFTOOLS_SRC).
_APPTIS = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/APPTIS"
INFRETIS_SRC = os.environ.get("INFRETIS_SRC", os.path.join(_APPTIS, "infretis"))
INFTOOLS_SRC = os.environ.get("INFTOOLS_SRC", os.path.join(_APPTIS, "inftools"))
for _src in (INFTOOLS_SRC, INFRETIS_SRC):
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


def is_infretis_run(sim_dir):
    """True if `sim_dir` is an infretis run (has infretis.toml)."""
    return os.path.isfile(os.path.join(sim_dir, "infretis.toml"))


def read_settings(sim_dir, toml_file=None):
    """Interfaces and sampling settings from infretis.toml (restart.toml is not needed).

    Returns a dict: interfaces, lambda_minus_one (None if off), mode ('staple', 'pp' or 'retis'),
    infinite_swap (staple: `staple_infinite_swap`, default False; the other modes always use it),
    staple_swap_frac (staple without infinite swap: fraction of nearest-neighbour swaps, default zeroswap),
    shooting_moves, timestep, subcycles."""
    toml_file = toml_file or os.path.join(sim_dir, "infretis.toml")
    with open(toml_file, "rb") as f:
        config = tomllib.load(f)
    sim = config["simulation"]
    lm1 = sim.get("tis_set", {}).get("lambda_minus_one", None)
    mode = sim.get("mode", "retis") or "retis"
    if mode not in ("staple", "pp"):
        mode = "retis"
    engine = config.get("engine", {})
    return dict(
        interfaces=[float(x) for x in sim["interfaces"]],
        lambda_minus_one=None if isinstance(lm1, bool) else lm1,
        mode=mode,
        infinite_swap=bool(sim.get("staple_infinite_swap", False)) if mode == "staple" else mode == "retis",
        staple_swap_frac=float(sim.get("staple_swap_frac", sim.get("zeroswap", 0.5))),
        shooting_moves=list(sim.get("shooting_moves", [])),
        timestep=float(engine.get("timestep", 1.0)),
        subcycles=int(engine.get("subcycles", 1)),
    )


def _is_number(s):
    if s == "----":
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_infretis_data(data_file, n_ens):
    """Parse infretis_data.txt (or .txt.gz) for a run with `n_ens` ensemble columns ([0-], [0+], [1+], ...;
    one per interface).

    Rows are tab separated: the staple move column holds a tuple with spaces, so the line is split on tabs,
    not on whitespace (`inftools.misc.data_helper.data_reader` and `inftools.istar.reading` each read one
    of the two layouts only). '----' is read as 0. Returns a dict of arrays in file order (the order in
    which the paths ended): pnr, length, maxop, minop (NaN for plain RETIS), ptype ('' if absent),
    path_f and path_w (n_rows, n_ens), move ('' if absent) and gen_ens (the generating ensemble number of
    staple runs, -1 = [0-], 0 = [0+]; -2 if absent)."""
    import gzip
    opener = gzip.open if str(data_file).endswith(".gz") else open
    rows = {k: [] for k in ("pnr", "length", "maxop", "minop", "ptype", "path_f", "path_w", "move", "gen_ens")}
    with opener(data_file, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = [s.strip() for s in line.rstrip("\n").split("\t")]
            fields = [s for s in fields if s != ""]
            # head: pnr, len, maxop [, minop, ptype]
            head = 5 if len(fields) >= 5 + 2 * n_ens and not _is_number(fields[4]) else 3
            if len(fields) < head + 2 * n_ens:
                raise ValueError(f"{data_file}: row with {len(fields)} fields, expected at least "
                                 f"{head + 2 * n_ens} for {n_ens} ensembles:\n{line}")
            num = lambda s: 0.0 if s == "----" else float(s)
            rows["pnr"].append(int(float(fields[0])))
            rows["length"].append(int(float(fields[1])))
            rows["maxop"].append(float(fields[2]))
            rows["minop"].append(float(fields[3]) if head == 5 else np.nan)
            rows["ptype"].append(fields[4] if head == 5 else "")
            rows["path_f"].append([num(s) for s in fields[head:head + n_ens]])
            rows["path_w"].append([num(s) for s in fields[head + n_ens:head + 2 * n_ens]])
            extra = fields[head + 2 * n_ens:]
            rows["move"].append(extra[0] if extra else "")
            try:
                rows["gen_ens"].append(int(extra[1]) if len(extra) > 1 else -2)
            except ValueError:
                rows["gen_ens"].append(-2)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["path_f"] = out["path_f"].astype(float).reshape(-1, n_ens)
    out["path_w"] = out["path_w"].astype(float).reshape(-1, n_ens)
    return out


def default_weighting(settings):
    """The weighting used when --weights is not given, and why.

    'staple' = path_f / path_w (the infretis/inftools analysis), 'rejections' = path_f. They differ only where
    path_w is not 1: wire-fencing high-acceptance weights (RETIS) and the StapleTIS 0.5 for LML/RMR paths.
    The right choice is the one that undoes how the sampler used w:
      * RETIS/REPPTIS: every move samples with w (wire fencing is a high-acceptance move, the permanent uses
        W), so path_f / path_w.
      * StapleTIS with swaps (infinite swap, or nearest-neighbour staple swaps): both split a path's
        occupancy over its ensembles in proportion to w, so path_f / path_w. On the two KCl staple runs
        (`weight_consistency`) path_f / path_w passes the cross-ensemble test (0.99-1.00 +- 0.01) and path_f
        fails it by exactly the factor 2.
      * StapleTIS without any swap (no infinite swap, staple_swap_frac 0): only shooting, which accepts
        every valid path whatever its w (as the toytis sampler, where the plain repeat counts gave the exact
        committor and the factor 2 did not): path_f.
    Staple shooting itself does not use w (no min(1, w_new / w_old) in its acceptance), so with swaps the
    run samples neither P nor w P exactly; no weighting of the output undoes that (notes section 15).
    """
    if settings["mode"] != "staple":
        return "staple", "RETIS/REPPTIS: every move samples with w"
    if settings["infinite_swap"]:
        return "staple", "StapleTIS with infinite swap: the permanent of W splits the occupancies by w"
    if settings["staple_swap_frac"] > 0:
        return "staple", "StapleTIS with nearest-neighbour swaps: their acceptance splits the occupancies by w"
    return "rejections", "StapleTIS with shooting only: no move uses w"


def weight_consistency(data, burn_in_fraction=0.2, n_blocks=20, min_paths=30):
    """Cross-ensemble test of the weightings (needs no committor).

    Under correct weights, a path valid in two ensembles e < e' has weight_e / weight_e' = C_e / C_e' (the
    ratio of the two ensembles' normalizations), the same for every such path, whatever its type. So the
    ratio sum_p weight_e / sum_p weight_e' over the paths of one class (w_e, w_e') must not depend on the
    class. Returned per ensemble pair: R(class) / R(1, 1) for the classes (0.5, 1) and (1, 0.5), with block
    standard errors, for path_f ('rejections'; 1 if the occupancies follow P, 0.5 and 2 if they follow w P)
    and path_f / path_w ('staple'). It cannot see a bias common to all ensembles a path is in (which paths
    the shooting move generates), only how each path's occupancy is split over its ensembles."""
    keep = np.arange(len(data["pnr"])) >= burn_in_fraction * len(data["pnr"])
    f, w = data["path_f"][keep], data["path_w"][keep]
    blocks = np.array_split(np.arange(len(f)), n_blocks)
    n_ens = f.shape[1]
    out = []

    def ratio(mask, e1, e2, rows):
        a, b = f[rows][mask[rows], e1].sum(), f[rows][mask[rows], e2].sum()
        return a / b if b > 0 else np.nan

    for e1 in range(2, n_ens):
        for e2 in range(e1 + 1, n_ens):
            both = (f[:, e1] > 0) & (f[:, e2] > 0)
            cls = {c: both & np.isclose(w[:, e1], c[0]) & np.isclose(w[:, e2], c[1])
                   for c in ((1, 1), (0.5, 1), (1, 0.5))}
            if cls[(1, 1)].sum() < min_paths:
                continue
            row = dict(pair=(e1, e2), n={c: int(m.sum()) for c, m in cls.items()})
            for c in ((0.5, 1), (1, 0.5)):
                if cls[c].sum() < min_paths:
                    continue
                every = np.arange(len(f))
                val = ratio(cls[c], e1, e2, every) / ratio(cls[(1, 1)], e1, e2, every)
                bs = np.array([ratio(cls[c], e1, e2, b) / ratio(cls[(1, 1)], e1, e2, b) for b in blocks])
                bs = bs[np.isfinite(bs)]
                se = bs.std(ddof=1) / np.sqrt(len(bs)) if len(bs) > 2 else np.nan
                corr = c[1] / c[0]                                    # path_f / path_w rescales the ratio by w_e'/w_e
                row[c] = dict(rejections=(val, se), staple=(val * corr, se * corr))
            out.append(row)
    return out


def _pair_weights(data, weighting, n_ens):
    """Weight of every (row, ensemble) pair, normalized per ensemble to the total occupancy (as inftools)."""
    f, w = data["path_f"], data["path_w"]
    if weighting == "staple":
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.where((f > 0) & (w > 0), f / np.where(w > 0, w, 1.0), 0.0)
    elif weighting == "rejections":
        W = np.where(f > 0, f, 0.0)
    elif weighting == "none":
        W = (f > 0).astype(float)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    tot_f, tot_W = f.sum(0), W.sum(0)
    scale = np.where(tot_W > 0, tot_f / np.where(tot_W > 0, tot_W, 1.0), 0.0)
    return W * scale


def find_bb_weight_bug(data, settings, rows, get_order):
    """Rows of the infretis_data.txt pairs hit by the last-ensemble B-to-B staple weight bug.

    In the last ensemble (L, M, R) = (lambda_{n-3}, lambda_{n-2}, lambda_B) an RMR path starts and ends in B: its
    core runs from B below M and back, with nothing to extend at either end, so the extension's coin flip gives the
    same path either way and its W-matrix weight must be 1 (infretis commit 1ed2805, 2026-11-16). Earlier infretis
    versions gave it the 0.5 of a genuine LML/RMR staple. A pair is affected when its path_w in the last ensemble
    is 0.5 and its path starts and ends at or beyond lambda_B. `rows` are candidate rows, `get_order(r)` returns the
    path's order parameters (column 0 = lambda) or None. Returns the sorted affected rows (empty for a fixed run)."""
    if settings["mode"] != "staple":
        return []
    e_last = len(settings["interfaces"]) - 1
    lam_b = settings["interfaces"][-1]
    hit = []
    for r in rows:
        if not np.isclose(data["path_w"][r, e_last], 0.5) or data["path_f"][r, e_last] <= 0:
            continue
        op = get_order(r)
        if op is not None and op[0, 0] >= lam_b and op[-1, 0] >= lam_b:
            hit.append(int(r))
    return sorted(hit)


def middle_interface(settings, e):
    """M of ensemble column e: [0-] and [0+] cross lambda_0, [i+] (column i + 1) crosses lambda_i."""
    return settings["interfaces"][max(e - 1, 0)]


def ensemble_window(settings, e):
    """(L, M, R) interfaces of ensemble column e, as infretis sets them up (`initiate_ensembles`)."""
    intf = settings["interfaces"]
    if e == 0:
        lm1 = settings["lambda_minus_one"]
        return ([lm1, (lm1 + intf[0]) / 2, intf[0]] if lm1 is not None else [float("-inf"), intf[0], intf[0]])
    if e == 1:
        return [intf[0], intf[0], intf[1]]
    return [intf[e - 2], intf[e - 1], intf[e]]


def _staple_path(op):
    """An infretis StaplePath holding only the order parameters (enough for its turn and core logic)."""
    from infretis.classes.staple_path import StaplePath
    from infretis.classes.system import System
    path = StaplePath()
    for row in op:
        frame = System()
        frame.order = row
        path.phasepoints.append(frame)
    return path


def staple_core(op, settings, e, gen_pptype=None):
    """(core_start, core_end) of a staple path in ensemble column e: infretis' shooting region
    (`StaplePath.get_sh_region`), i.e. the first and last frame of the path's segment inside the ensemble's
    window that crosses M. The degenerate [0+] window needs the path's stored pptype; `gen_pptype` =
    (generating ensemble number, 'LMR'/...) from infretis_data.txt. Returns None if infretis finds no core."""
    path = _staple_path(op)
    if gen_pptype is not None:
        path.pptype = gen_pptype
    try:
        start, end = path.get_sh_region(settings["interfaces"], ensemble_window(settings, e))
    except (ValueError, AssertionError, StopIteration, IndexError):
        return None
    if start == 0 and end == 0:
        return None
    return int(start), int(end)


def _ptype_middle(ptype):
    """'0LMR9' -> 'LMR', 'RMR' -> 'RMR'."""
    for key in ("LMR", "LML", "RML", "RMR"):
        if key in ptype:
            return key
    return None


def select_rows(data, W, ens_list, burn_in=0, burn_in_fraction=0.4, thin=1, paths_per_ensemble=None):
    """Rows of infretis_data.txt used for each ensemble column: after the burn-in (the first burn_in_fraction of the
    rows, and burn_in rows if more), not an initial load path, with a nonzero weight W[row, e]; every thin-th of
    those, at most paths_per_ensemble of them evenly spaced. Returns {e: row indices}. Used by load_infretis_data and
    by make_compact_archive, so the archive holds exactly the paths the loader reads."""
    n_rows = len(data["pnr"])
    is_load = np.array([m.startswith("('ld'") for m in data["move"]]) if data["move"][0] else data["pnr"] < len(W[0])
    first_row = max(int(burn_in), int(np.ceil(burn_in_fraction * n_rows)))
    eligible_rows = (np.arange(n_rows) >= first_row) & ~is_load
    selected = {}
    for e in ens_list:
        rows = np.flatnonzero(eligible_rows & (W[:, e] > 0))[::max(1, int(thin))]
        if paths_per_ensemble is not None and len(rows) > paths_per_ensemble:
            rows = rows[np.unique(np.round(np.linspace(0, len(rows) - 1, paths_per_ensemble)).astype(int))]
        selected[e] = rows
    return selected


def load_infretis_data(sim_dir, include_zero_minus=False, weighting=None, cut_at_m=True, burn_in=0,
                       burn_in_fraction=0.4, thin=1, paths_per_ensemble=None, reversed_pieces=False,
                       odd_columns=(), fd_velocity=False, fd_columns=(1,), max_length=None,
                       long_pieces="truncate", h5_file=None, data_file=None, load_dir=None, toml_file=None,
                       with_info=False, bb_weight_fix="auto"):
    """Per-piece cvs, order, time, weight and path-number lists from an infretis run (see the module doc).

    The options are those of `tis_data.load_tis_data`. Burn-in and thinning act per ensemble on the rows of
    infretis_data.txt in file order (the order in which the paths ended): the first `burn_in_fraction` of
    the rows (and the first `burn_in`, if more) are skipped, as are the initial load paths; then every
    `thin`-th path with a nonzero weight in the ensemble is kept, at most `paths_per_ensemble`, evenly
    spaced. Paths missing from the load folder or archive (removed with delete_old_all) are skipped and
    counted. The path number of a piece is the infretis path number, shared by the pieces it gives in
    different ensembles, which overlap in frames.

    Returns (cvs, order, time, weights, paths, settings), with `settings` from `read_settings` plus the
    weighting used; with `with_info` also a list with one dict per piece (ensemble column `ens`, the row `row`
    of infretis_data.txt, `pnr`, `mirrored`, the core `(core_start, core_end)` and the path length).

    Mirrored pieces are refused for RETIS runs: a RETIS path always starts in A, so its mirrored piece always
    ends in A, which selects on the piece's future (notes section 9).

    `bb_weight_fix` ('auto', 'on', 'off'): StapleTIS runs made with infretis before commit 1ed2805 gave the B-to-B
    RMR paths of the last ensemble w = 0.5 instead of 1 (`find_bb_weight_bug`). Their path_f is then twice what it
    should be, which pulls the committor toward B (flat 1D staple run, notes section 15.1: +0.022 -> +0.003). With
    'auto' (default) and 'on' the 'rejections' weight of these pairs is halved when such pairs are found; 'auto'
    and 'on' differ only in that 'on' also reports when none are found. 'off' keeps the stored weights. The
    'staple' weighting is not corrected (it is biased for committor pieces anyway)."""
    from tis_data import first_frame_past_m, last_frame_before_m, limit_piece_length
    from inftools.istar.reading import OrderSource

    if reversed_pieces and not cut_at_m:
        raise ValueError("reversed pieces need the cut at M")
    settings = read_settings(sim_dir, toml_file)
    if reversed_pieces and settings["mode"] == "retis":
        raise ValueError("mirrored pieces (--reversed) are not valid for RETIS ensembles: every [i+] path starts "
                         "in A, so the mirrored piece always ends in A (a condition on its future)")
    n_ens = len(settings["interfaces"])
    data_file = data_file or os.path.join(sim_dir, "infretis_data.txt")
    data = read_infretis_data(data_file, n_ens)
    n_rows = len(data["pnr"])
    if weighting is None:
        weighting, why = default_weighting(settings)
        print(f"Weights: {weighting} ({'path_f / path_w' if weighting == 'staple' else 'path_f'}), "
              f"the default for this run: {why}")
    settings["weighting"] = weighting
    print(f"infretis run {sim_dir}: mode {settings['mode']}, infinite swap {settings['infinite_swap']}, "
          f"{n_ens} ensembles, {n_rows} finished paths in {data_file}")
    print(f"Interfaces: {settings['interfaces']}")

    W = _pair_weights(data, weighting, n_ens)
    w_not1 = (data["path_f"] > 0) & (data["path_w"] > 0) & ~np.isclose(data["path_w"], 1.0)
    if w_not1.any():
        share = (data["path_f"] * w_not1).sum() / data["path_f"].sum()
        print(f"  path_w != 1 for {w_not1.sum()} (path, ensemble) pairs, {share:.1%} of the occupancy; "
              f"path_f and path_f / path_w differ there by a factor {np.unique(np.round(1 / data['path_w'][w_not1], 3))}")
    if settings["mode"] == "staple":
        print("  Note (StapleTIS): path_f / path_w is the natural measure of the stored staple paths (crossing "
              "probabilities, rates). For committor pieces use path_f (--weights rejections): an LML/RMR core is "
              "stored twice, once with each turn (the extension's coin flip, w = 1/2 each), and path_f counts the "
              "core once; see docs/op_loop_report.md.")

    ens_list = [e for e in range(n_ens) if e > 0 or include_zero_minus]
    selected = select_rows(data, W, ens_list, burn_in=burn_in, burn_in_fraction=burn_in_fraction, thin=thin,
                           paths_per_ensemble=paths_per_ensemble)

    cvs, order, time, weights, paths, info = [], [], [], [], [], []
    cache = {}
    needed = sorted(set(np.concatenate([r for r in selected.values()]).tolist())) if selected else []
    n_missing = 0
    with OrderSource(load_dir=load_dir or os.path.join(sim_dir, "load"), h5_file=h5_file) as source:
        for r in needed:
            op = source.get(int(data["pnr"][r]))
            if op is None:
                n_missing += 1
            cache[r] = op
    print(f"Order parameters from {h5_file or (load_dir or os.path.join(sim_dir, 'load'))}: "
          f"{len(needed) - n_missing} paths read" + (f", {n_missing} missing (skipped)" if n_missing else ""))

    if bb_weight_fix not in ("auto", "on", "off"):
        raise ValueError(f"bb_weight_fix must be 'auto', 'on' or 'off', not {bb_weight_fix!r}")
    e_last = n_ens - 1
    bb_rows = find_bb_weight_bug(data, settings, selected.get(e_last, []), cache.get)
    bb_fixed = set()
    if bb_rows:
        msg = (f"  B-to-B weight bug (infretis before 1ed2805): {len(bb_rows)} of the {len(selected[e_last])} paths of "
               f"the last ensemble start and end in B with path_w = 0.5 (should be 1)")
        if bb_weight_fix == "off":
            print(msg + "; NOT corrected (bb_weight_fix='off'): the committor is pulled toward B")
        elif weighting == "rejections":
            W = W.copy()
            W[bb_rows, e_last] *= 0.5
            bb_fixed = set(bb_rows)
            print(msg + "; their path_f is halved (notes section 15.1)")
        else:
            print(msg + f"; not corrected for the '{weighting}' weighting (use 'rejections' for committor pieces)")
    elif bb_weight_fix == "on" and settings["mode"] == "staple":
        print("  B-to-B weight bug: no B-to-B path of the last ensemble has path_w = 0.5, nothing changed")

    for e in ens_list:
        lambda_m = middle_interface(settings, e) if cut_at_m else None
        n_pieces = n_frames = n_no_crossing = n_no_core = n_long = n_badlen = 0
        tot_w = 0.0
        for r in selected[e]:
            op = cache.get(r)
            if op is None:
                continue
            if len(op) != data["length"][r]:
                n_badlen += 1
            block = np.column_stack([np.arange(len(op), dtype=float), op])   # time = frame index
            if fd_velocity:
                idx = np.arange(len(block))
                prev = np.maximum(idx - int(fd_velocity), 0)
                vs = (block[:, fd_columns] - block[prev][:, fd_columns]) / np.maximum(idx - prev, 1)[:, None]
                if len(block) > 1:
                    vs[0] = vs[1]
                block = np.column_stack([block, vs])
            if settings["mode"] == "staple" and e >= 1:
                mid = _ptype_middle(str(data["ptype"][r]))
                gen = (int(data["gen_ens"][r]), mid) if (mid and data["gen_ens"][r] > -2) else None
                core = staple_core(op, settings, e, gen)
                if core is None:
                    n_no_core += 1
                    continue
                core_start, core_end = core
            else:
                core_start, core_end = 0, len(block) - 1               # TIS / PPTIS: the whole path
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
                            idx = np.arange(t_last + 1)
                            nxt = np.minimum(idx + int(fd_velocity), len(block) - 1)
                            vrev = ((block[idx][:, fd_columns] - block[nxt][:, fd_columns])
                                    / np.maximum(nxt - idx, 1)[:, None])
                            backwards[:, -len(fd_columns):] = vrev[::-1]
                        pieces.append(backwards)
            for j, piece in enumerate(pieces):
                if max_length is not None and len(piece) > max_length:
                    n_long += 1
                    piece = limit_piece_length(piece, max_length, long_pieces)
                    if piece is None:
                        continue
                if len(piece) < 2:
                    continue
                piece = piece.copy()
                time.append(piece[:, 0])
                order.append(piece[:, 1])
                cvs.append(piece[:, 1:])
                weights.append(float(W[r, e]))
                paths.append(int(data["pnr"][r]))
                info.append(dict(ens=int(e), row=int(r), pnr=int(data["pnr"][r]), mirrored=j == 1,
                                 core=(int(core_start), int(core_end)), length=len(block),
                                 bb_fixed=(e == e_last and int(r) in bb_fixed)))
                n_pieces += 1
                n_frames += len(piece)
                tot_w += float(W[r, e])
        msg = (f"  ensemble {e:03d}: {len(selected[e])} paths, {n_pieces} pieces, {n_frames} frames, "
               f"total weight {tot_w:.1f}")
        if lambda_m is not None:
            msg += f", cut at M = {lambda_m}"
        if n_no_crossing:
            msg += f" ({n_no_crossing} paths dropped: core never crosses M)"
        if n_no_core:
            msg += f" ({n_no_core} paths without a core in this ensemble, dropped)"
        if n_long:
            msg += f" ({n_long} pieces longer than {max_length} frames {'discarded' if long_pieces == 'discard' else 'truncated'})"
        if n_badlen:
            msg += f" ({n_badlen} paths whose order.txt length differs from infretis_data.txt)"
        print(msg)
    print(f"Loaded {len(cvs)} pieces, {sum(len(p) for p in order)} frames total")
    if with_info:
        return cvs, order, time, weights, paths, settings, info
    return cvs, order, time, weights, paths, settings


def make_compact_archive(sim_dir, out_dir, paths_per_ensemble=5000, burn_in_fraction=0.4, thin=1):
    """Write a compressed copy of an infretis run with the order parameters of the paths the loader reads.

    out_dir gets: infretis.toml (copied), infretis_data.txt.gz (the whole path table, gzipped, so every occupancy
    and every weight is kept and the per-ensemble normalization of `_pair_weights` is the run's own), paths.h5
    (gzipped, one group per path number with dataset 'order.txt' = [time, order parameter, CVs ...], the layout of
    `inft store_txt -order`), and README.md. The h5 holds, for every ensemble, the rows that `select_rows` picks
    (paths_per_ensemble per ensemble after the burn-in) under both weightings ('staple' and 'rejections'), so the
    loader gets the same paths whichever weighting it is given. Use with
        python tis_data.py <out_dir> --data-file <out_dir>/infretis_data.txt.gz --h5 <out_dir>/paths.h5 --include-zero-minus
    (the default burn-in of the loader is the same fraction, so no extra option is needed).
    Returns a dict with the counts per ensemble."""
    import gzip
    import shutil
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    settings = read_settings(sim_dir)
    n_ens = len(settings["interfaces"])
    data = read_infretis_data(os.path.join(sim_dir, "infretis_data.txt"), n_ens)
    ens_list = list(range(n_ens))
    rows = set()
    counts = {}
    for weighting in ("staple", "rejections"):
        W = _pair_weights(data, weighting, n_ens)
        sel = select_rows(data, W, ens_list, burn_in_fraction=burn_in_fraction, thin=thin,
                          paths_per_ensemble=paths_per_ensemble)
        for e, r in sel.items():
            rows.update(int(i) for i in r)
            counts[(weighting, e)] = len(r)
    shutil.copy(os.path.join(sim_dir, "infretis.toml"), os.path.join(out_dir, "infretis.toml"))
    with open(os.path.join(sim_dir, "infretis_data.txt"), "rb") as src, \
            gzip.open(os.path.join(out_dir, "infretis_data.txt.gz"), "wb") as dst:
        shutil.copyfileobj(src, dst)

    n_frames = 0
    with h5py.File(os.path.join(out_dir, "paths.h5"), "w") as h5:
        for r in sorted(rows):
            pnr = int(data["pnr"][r])
            order = np.loadtxt(os.path.join(sim_dir, "load", str(pnr), "order.txt"), comments="#", ndmin=2)
            if len(order) != data["length"][r]:
                raise ValueError(f"path {pnr}: order.txt has {len(order)} frames, the table says {data['length'][r]}")
            h5.create_dataset(f"{pnr}/order.txt", data=order, compression="gzip", chunks=True)
            n_frames += len(order)
    print(f"{out_dir}: {len(rows)} paths ({n_frames} frames) in paths.h5; selected per ensemble: "
          + ", ".join(f"{k[0]}/{k[1]}: {v}" for k, v in sorted(counts.items())))
    return dict(paths=len(rows), frames=n_frames, counts=counts)
