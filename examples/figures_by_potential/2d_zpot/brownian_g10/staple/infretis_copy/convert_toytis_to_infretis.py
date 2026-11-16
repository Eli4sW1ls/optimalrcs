"""Write part of the toytis staple run sim_istarz_br_g10 (2D Brownian Z potential) in the infretis layout.

There is no infretis StapleTIS run of a 2D toy system yet. This copy lets the infretis reader
(`examples/infretis_data.py`) and the infretis staple notebooks run on 2D staple paths with a known exact
committor. It is a format copy, not an infretis simulation:

  infretis.toml         interfaces of the toytis run, mode "staple", no swaps (staple_infinite_swap = false,
                        staple_swap_frac = 0): the toytis sampler only shoots, so path_f is the right weight and
                        the loader picks it by default (infretis_data.default_weighting)
  infretis_data.txt     one row per (ensemble, accepted toytis path): its own path number, length, max/min order
                        parameter, path type, path_f = the toytis repeat count in that ensemble's column, path_w =
                        0.5 for LML/RMR in the inner ensembles (as infretis StapleTIS writes it), else 1; move
                        ('sh', k) and generating ensemble k - 1. Rows are sorted by MC cycle, so a burn-in on the
                        rows removes the early cycles of every ensemble.
  load/<pnr>/order.txt  time, y (order parameter), x

The cores are not copied: the reader finds them with infretis' StaplePath.get_sh_region. `--check` compares
every piece the infretis reader makes with the piece the toytis reader (tis_data.load_order_file) makes from
the same path, which also compares the two core definitions.

Usage: python convert_toytis_to_infretis.py [--paths-per-ensemble N] [--check]
"""
import argparse
import contextlib
import glob
import io
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))
sys.path.insert(0, EXAMPLES)
import tis_data  # noqa: E402

SIMS = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations"
SRC = f"{SIMS}/sim_istarz_br_g10"
DST = f"{SIMS}/sim_istarz_br_g10_infretis"
HEADER = re.compile(r"# Cycle:\s*(\d+),\s*status:\s*(\S+?),.*path type:\s*(\S+),")


def read_paths(order_file):
    """{cycle: (path type, (n, 3) array time, y, x)} of the accepted paths in a toytis order.txt."""
    out, cur, rows = {}, None, []
    with open(order_file) as f:
        for line in f:
            if line.startswith("# Cycle"):
                if cur is not None and rows:
                    out[cur[0]] = (cur[1], np.array(rows, dtype=float))
                m = HEADER.match(line)
                cur = (int(m.group(1)), m.group(3)) if (m and m.group(2) == "ACC") else None
                rows = []
            elif line.startswith("#") or cur is None:
                continue
            else:
                rows.append([float(v) for v in line.split()[:3]])
    if cur is not None and rows:
        out[cur[0]] = (cur[1], np.array(rows, dtype=float))
    return out


def convert(paths_per_ensemble):
    with contextlib.redirect_stdout(io.StringIO()):
        interfaces = tis_data.read_interfaces(SRC, None)
    n_ens = len(interfaces)
    folders = sorted(glob.glob(os.path.join(SRC, "0[0-9][0-9]")))
    entries = []                                          # (cycle, k, ptype, frames, weight)
    for folder in folders:
        k = int(os.path.basename(folder))
        if k == 0:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            info = tis_data.read_path_info(folder, k, len(folders), "rejections")
        paths = read_paths(os.path.join(folder, "order.txt"))
        cycles = [c for c in sorted(paths) if info[c][0] > 0 and info[c][2] != "ld"]
        pick = np.unique(np.round(np.linspace(0, len(cycles) - 1, min(paths_per_ensemble, len(cycles)))).astype(int))
        for c in (cycles[i] for i in pick):
            ptype, frames = paths[c]
            entries.append((c, k, ptype, frames, float(info[c][0])))
        print(f"  ensemble {k:03d}: {len(pick)} of {len(cycles)} accepted paths")
    entries.sort(key=lambda e: (e[0], e[1]))                # the order in which paths ended: MC cycle

    os.makedirs(os.path.join(DST, "load"), exist_ok=True)
    with open(os.path.join(DST, "infretis.toml"), "w") as f:
        f.write("# Format copy of the toytis staple run sim_istarz_br_g10 (see convert_toytis_to_infretis.py)\n")
        f.write("[simulation]\nmode = \"staple\"\n")
        f.write(f"interfaces = [{', '.join(repr(float(x)) for x in interfaces)}]\n")
        f.write("steps = 100000\nseed = 0\nload_dir = \"load\"\n")
        f.write("staple_infinite_swap = false\nstaple_swap_frac = 0.0\n")
        f.write("\n[simulation.tis_set]\nmaxlength = 1000000\nlambda_minus_one = false\n")
        f.write("\n[engine]\nclass = \"toytis\"\ntimestep = 0.002\nsubcycles = 1\n")
    fmt = lambda v: "----" if v == 0 else f"{v:.6f}"
    mapping = []
    with open(os.path.join(DST, "infretis_data.txt"), "w") as fh:
        fh.write("# " + "=" * 60 + "\n# \txxx\tlen\tmax OP\tmin OP\tptype\t"
                 + "\t".join(f"{e:03d}" for e in range(n_ens)) + "\n# " + "=" * 60 + "\n")
        for pnr, (c, k, ptype, frames, weight) in enumerate(entries, start=n_ens):
            f_row, w_row = [0.0] * n_ens, [0.0] * n_ens
            f_row[k] = weight
            w_row[k] = 0.5 if (k >= 2 and ptype in ("LML", "RMR")) else 1.0
            fh.write("\t".join([str(pnr), str(len(frames)), f"{frames[:, 1].max():.6f}", f"{frames[:, 1].min():.6f}",
                                ptype] + [fmt(v) for v in f_row] + [fmt(v) for v in w_row]
                               + [f"('sh', {k})", str(k - 1)]) + "\n")
            d = os.path.join(DST, "load", str(pnr))
            os.makedirs(d, exist_ok=True)
            np.savetxt(os.path.join(d, "order.txt"), frames, header="time orderp orderp1",
                       fmt=["%d", "%.6f", "%.6f"])
            mapping.append((pnr, k, c))
    np.savetxt(os.path.join(DST, "toytis_source.txt"), np.array(mapping, dtype=int), fmt="%d",
               header="pnr toytis_ensemble toytis_cycle")
    print(f"wrote {len(entries)} paths to {DST}")


def check():
    """Every infretis piece against the toytis piece of the same path (cut at M from the toytis staridx core)."""
    import infretis_data
    mapping = {int(p): (int(k), int(c)) for p, k, c in np.loadtxt(os.path.join(DST, "toytis_source.txt"), dtype=int)}
    with contextlib.redirect_stdout(io.StringIO()):
        cvs, order, time, weights, paths, settings = infretis_data.load_infretis_data(
            DST, burn_in_fraction=0.0, reversed_pieces=False)
        interfaces = settings["interfaces"]
    staridx = {}
    for k in sorted({k for k, _ in mapping.values()}):
        with contextlib.redirect_stdout(io.StringIO()):
            info = tis_data.read_path_info(os.path.join(SRC, f"{k:03d}"), k, 11, "rejections")
        staridx[k] = info
    n_same = n_diff = 0
    for o, t, p in zip(order, time, paths):
        k, c = mapping[p]
        frames = np.loadtxt(os.path.join(DST, "load", str(p), "order.txt"))
        core_start = staridx[k][c][3]
        t_star = tis_data.first_frame_past_m(frames[:, 1], core_start, interfaces[max(k - 1, 0)])
        if t_star is not None and int(t[0]) == t_star and np.allclose(o, frames[t_star:, 1]):
            n_same += 1
        else:
            n_diff += 1
    n_expected = len(mapping)
    print(f"infretis pieces {len(order)} of {n_expected} paths; identical to the toytis piece: {n_same}, "
          f"different: {n_diff}; paths without an infretis core or M crossing: {n_expected - len(order)}")
    return n_diff


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paths-per-ensemble", type=int, default=500)
    ap.add_argument("--check", action="store_true", help="only compare the existing copy with the toytis pieces")
    a = ap.parse_args()
    if not a.check:
        convert(a.paths_per_ensemble)
    sys.exit(1 if check() else 0)
