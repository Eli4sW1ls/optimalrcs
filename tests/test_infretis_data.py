"""Tests for the infretis reader in examples/infretis_data.py and its use through tis_data.prepare_data.

Most tests build a small synthetic infretis run in a temporary folder (infretis.toml, infretis_data.txt,
load/<pnr>/order.txt) and compare what the loader returns with values computed here independently: the cut at
the first frame past M, the mirrored pieces, the path selection, the weights. The staple tests use infretis'
own StaplePath for the cores; one test reads a real RETIS run if it is on disk. Skipped when infretis or
inftools cannot be imported.
"""
import argparse
import contextlib
import gzip
import io
import os
import shutil
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

EXAMPLES = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "examples"))
sys.path.insert(0, EXAMPLES)
import infretis_data  # noqa: E402  (puts the infretis/inftools checkouts on the path)
import tis_data  # noqa: E402

try:
    from inftools.istar.reading import OrderSource  # noqa: F401
    from infretis.classes.staple_path import StaplePath  # noqa: F401
    HAVE_INFRETIS = True
except ImportError:
    HAVE_INFRETIS = False

REAL_RETIS_RUN = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/F_cond_analysis/1D_ASE_infretis/flat_lgv/gamma5"


def quiet(f, *args, **kwargs):
    """Call f with its printing suppressed (the loaders report every ensemble)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


# ---------------------------------------------------------------------------------------------------
# synthetic runs
# ---------------------------------------------------------------------------------------------------

def fmt(v):
    return "----" if v == 0 else f"{v:.6f}"


def write_toml(sim_dir, interfaces, extra_sim=""):
    with open(os.path.join(sim_dir, "infretis.toml"), "w") as f:
        f.write("[simulation]\n")
        f.write(f"interfaces = [{', '.join(repr(float(x)) for x in interfaces)}]\n")
        f.write("steps = 100\nseed = 0\nload_dir = \"load\"\n")
        f.write(extra_sim)
        f.write("\n[simulation.tis_set]\nmaxlength = 100000\nlambda_minus_one = false\n")
        f.write("\n[engine]\nclass = \"turtlemd\"\ntimestep = 0.01\nsubcycles = 2\n")


def write_order(sim_dir, pnr, x, extra_cv=True):
    """load/<pnr>/order.txt: time, order parameter and (optionally) an extra CV, x**2."""
    os.makedirs(os.path.join(sim_dir, "load", str(pnr)), exist_ok=True)
    cols = [np.arange(len(x)), x] + ([x ** 2] if extra_cv else [])
    np.savetxt(os.path.join(sim_dir, "load", str(pnr), "order.txt"), np.column_stack(cols),
               header="time orderp cv", fmt=["%d"] + ["%.10f"] * (len(cols) - 1))


def retis_walk(rng, n_max=400):
    """A [0+]-like path: starts just inside A (< 0), crosses 0 at frame 1, a biased random walk until it is
    back in A or reaches B (>= 1)."""
    x = [-0.03, 0.02]
    while len(x) < n_max and 0 < x[-1] < 1:
        x.append(x[-1] + 0.012 + 0.08 * rng.standard_normal())
    return np.array(x)


def make_retis_run(sim_dir, n_paths=40, seed=0, w2_every=7):
    """Synthetic RETIS run, interfaces 0, 0.5, 1 (ensemble columns [0-], [0+], [1+]).

    Rows 0-2 are the initial load paths (pnr < n_ens, skipped by the loader). Every path has path_f in [0+];
    paths that reach 0.5 also in [1+]. path_w = 1, except every `w2_every`-th path in [1+] (w = 2, like a
    high-acceptance weight), so path_f and path_f / path_w differ. Returns the rows as dicts."""
    rng = np.random.default_rng(seed)
    interfaces = [0.0, 0.5, 1.0]
    write_toml(sim_dir, interfaces)
    rows = []
    for pnr in range(n_paths):
        x = retis_walk(rng)
        f = [0.0, float(rng.choice([0.5, 1.0, 2.0])), 0.0]
        w = [0.0, 1.0, 0.0]
        if x.max() >= 0.5:
            f[2] = float(rng.choice([0.5, 1.0, 1.5]))
            w[2] = 2.0 if pnr % w2_every == 0 else 1.0
        rows.append(dict(pnr=pnr, x=x, f=f, w=w))
        write_order(sim_dir, pnr, x)
    with open(os.path.join(sim_dir, "infretis_data.txt"), "w") as fh:
        fh.write("# " + "=" * 40 + "\n# \txxx\tlen\tmax OP\t\t000\t001\t002\n# " + "=" * 40 + "\n")
        for r in rows:
            fh.write("\t".join([f"{r['pnr']:5d}", f"{len(r['x']):5d}", f"{r['x'].max():.5f}"]
                               + [fmt(v) for v in r["f"]] + [fmt(v) for v in r["w"]]) + "\t\n")
    return rows, interfaces


def expected_weights(rows, weighting):
    """Weight per (row, ensemble column): path_f, path_f / path_w or 1, rescaled per ensemble to the total
    path_f (the normalization of the loader and inftools)."""
    f = np.array([r["f"] for r in rows])
    w = np.array([r["w"] for r in rows])
    raw = {"rejections": f, "staple": np.where(f > 0, f / np.where(w > 0, w, 1), 0), "none": (f > 0) * 1.0}[weighting]
    tot = raw.sum(0)
    return raw * np.where(tot > 0, f.sum(0) / np.where(tot > 0, tot, 1), 0)


def first_past(x, start, m):
    """First frame >= max(start, 1) on the other side of m than the frame before the core."""
    s = max(start, 1)
    side = x[s - 1] > m
    idx = [i for i in range(s, len(x)) if (x[i] > m) != side]
    return idx[0] if idx else None


def last_before(x, m):
    """Last frame <= len - 2 on the other side of m than the path's last frame (a path that is all core)."""
    side = x[-1] > m
    idx = [i for i in range(len(x) - 1) if (x[i] > m) != side]
    return idx[-1] if idx else None


def expected_retis_pieces(rows, interfaces, weighting="staple", first_row=0, reversed_pieces=False):
    """(ensemble, pnr, frames, times, weight, mirrored) of every piece, in the loader's order."""
    W = expected_weights(rows, weighting)
    out = []
    for e in (1, 2):
        m = interfaces[max(e - 1, 0)]
        for k, r in enumerate(rows):
            if k < first_row or r["pnr"] < 3 or W[k, e] == 0:
                continue
            x = r["x"]
            t = first_past(x, 0, m)
            if t is None:
                continue
            out.append((e, r["pnr"], x[t:], np.arange(t, len(x)), W[k, e], False))
            if reversed_pieces:
                tl = last_before(x, m)
                if tl is not None and tl + 1 >= 2:
                    out.append((e, r["pnr"], x[tl::-1], np.arange(tl + 1), W[k, e], True))
    return out


def staple_base_path(rng=None, noise=0.0):
    """A staple path for interfaces 0..5: starts across interface 2 going down, turns below 1, climbs past 4,
    turns and ends across interface 3. Valid (with a core) in ensemble columns 2-5. Optional noise on the
    inner frames keeps every interface crossing."""
    def seg(a, b, n):
        return list(np.linspace(a, b, n, endpoint=False))
    x = np.array([2.05] + seg(1.95, 0.55, 8) + seg(0.55, 4.55, 25) + seg(4.55, 3.05, 8) + [3.05, 2.95])
    if noise and rng is not None:
        x[1:-2] += noise * rng.uniform(-1, 1, len(x) - 3)
    return x


def make_staple_run(sim_dir, n_paths=12, seed=0, infinite_swap=True, swap_frac=0.5):
    """Synthetic StapleTIS run, interfaces 0..5 (columns [0-], [0+], [1+], ..., [4+]); every path is the base
    staple path with noise, valid in columns 2-5, w = 0.5 where it is RMR (columns 2 and 5)."""
    rng = np.random.default_rng(seed)
    interfaces = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    write_toml(sim_dir, interfaces, "mode = \"staple\"\n"
               f"staple_infinite_swap = {'true' if infinite_swap else 'false'}\nstaple_swap_frac = {swap_frac}\n")
    rows = []
    for pnr in range(n_paths):
        x = staple_base_path(rng, noise=0.04)
        f = [0.0, 0.0] + [float(v) for v in rng.choice([0.5, 1.0, 2.0], 4)]
        w = [0.0, 0.0, 0.5, 1.0, 1.0, 0.5]
        move = "('ld',)" if pnr < 2 else f"('sh', {pnr % 4 + 2})"
        rows.append(dict(pnr=pnr, x=x, f=f, w=w, move=move, ptype="RMR"))
        write_order(sim_dir, pnr, x, extra_cv=False)
    with open(os.path.join(sim_dir, "infretis_data.txt"), "w") as fh:
        fh.write("# header\n")
        for r in rows:
            fh.write("\t".join([f"{r['pnr']}", f"{len(r['x'])}", f"{r['x'].max():.5f}", f"{r['x'].min():.5f}",
                                f"0{r['ptype']}5"] + [fmt(v) for v in r["f"]] + [fmt(v) for v in r["w"]]
                               + [r["move"], "3"]) + "\n")
    return rows, interfaces


def prepare(sim_dir, *argv):
    parser = argparse.ArgumentParser()
    tis_data.add_data_arguments(parser)
    args = parser.parse_args([sim_dir, *argv])
    return quiet(tis_data.prepare_data, args), args


# ---------------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------------

class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="infretis_test_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestSettings(TempDirTest):

    def test_retis_settings_and_default_weighting(self):
        write_toml(self.dir, [0.0, 0.5, 1.0])
        self.assertTrue(infretis_data.is_infretis_run(self.dir))
        s = infretis_data.read_settings(self.dir)
        self.assertEqual(s["interfaces"], [0.0, 0.5, 1.0])
        self.assertEqual(s["mode"], "retis")
        self.assertTrue(s["infinite_swap"])
        self.assertIsNone(s["lambda_minus_one"])               # `false` in the toml: no [0-] interface
        self.assertEqual((s["timestep"], s["subcycles"]), (0.01, 2))
        self.assertEqual(infretis_data.default_weighting(s)[0], "staple")

    def test_staple_default_weighting(self):
        cases = [(True, 0.5, "staple"), (False, 0.5, "staple"), (False, 0.0, "rejections")]
        for infinite_swap, swap_frac, expected in cases:
            write_toml(self.dir, [0.0, 1.0, 2.0], "mode = \"staple\"\n"
                       f"staple_infinite_swap = {'true' if infinite_swap else 'false'}\n"
                       f"staple_swap_frac = {swap_frac}\n")
            s = infretis_data.read_settings(self.dir)
            self.assertEqual(s["mode"], "staple")
            self.assertEqual(infretis_data.default_weighting(s)[0], expected, (infinite_swap, swap_frac))

    def test_not_an_infretis_run(self):
        self.assertFalse(infretis_data.is_infretis_run(self.dir))

    def test_middle_interface_and_window(self):
        s = dict(interfaces=[0.0, 1.0, 2.0, 3.0], lambda_minus_one=-1.0)
        self.assertEqual([infretis_data.middle_interface(s, e) for e in range(4)], [0.0, 0.0, 1.0, 2.0])
        self.assertEqual(infretis_data.ensemble_window(s, 0), [-1.0, -0.5, 0.0])
        self.assertEqual(infretis_data.ensemble_window(s, 1), [0.0, 0.0, 1.0])
        self.assertEqual(infretis_data.ensemble_window(s, 3), [1.0, 2.0, 3.0])
        s["lambda_minus_one"] = None
        self.assertEqual(infretis_data.ensemble_window(s, 0)[1:], [0.0, 0.0])


class TestReadInfretisData(TempDirTest):

    def test_retis_layout(self):
        rows, _ = make_retis_run(self.dir, n_paths=15)
        d = infretis_data.read_infretis_data(os.path.join(self.dir, "infretis_data.txt"), 3)
        npt.assert_array_equal(d["pnr"], [r["pnr"] for r in rows])                  # file order
        npt.assert_array_equal(d["length"], [len(r["x"]) for r in rows])
        npt.assert_allclose(d["maxop"], [r["x"].max() for r in rows], atol=1e-5)
        npt.assert_allclose(d["path_f"], [r["f"] for r in rows], atol=1e-6)         # '----' -> 0
        npt.assert_allclose(d["path_w"], [r["w"] for r in rows], atol=1e-6)
        self.assertTrue(np.all(np.isnan(d["minop"])))
        self.assertTrue(np.all(d["ptype"] == "") and np.all(d["move"] == ""))
        self.assertTrue(np.all(d["gen_ens"] == -2))

    def test_gzip_equals_plain(self):
        make_retis_run(self.dir, n_paths=10)
        src = os.path.join(self.dir, "infretis_data.txt")
        with open(src, "rb") as fi, gzip.open(src + ".gz", "wb") as fo:
            fo.write(fi.read())
        a = infretis_data.read_infretis_data(src, 3)
        b = infretis_data.read_infretis_data(src + ".gz", 3)
        for key in a:
            npt.assert_array_equal(a[key], b[key], err_msg=key)

    def test_staple_layout_with_tuple_move(self):
        rows, _ = make_staple_run(self.dir, n_paths=6)
        d = infretis_data.read_infretis_data(os.path.join(self.dir, "infretis_data.txt"), 6)
        npt.assert_allclose(d["minop"], [r["x"].min() for r in rows], atol=1e-5)
        self.assertEqual(list(d["ptype"]), ["0RMR5"] * 6)
        self.assertEqual(list(d["move"]), [r["move"] for r in rows])                  # spaces inside the tuple
        npt.assert_array_equal(d["gen_ens"], 3)
        npt.assert_allclose(d["path_f"], [r["f"] for r in rows])
        npt.assert_allclose(d["path_w"], [r["w"] for r in rows])

    def test_short_row_raises(self):
        path = os.path.join(self.dir, "bad.txt")
        with open(path, "w") as f:
            f.write("1\t10\t0.5\t1.0\t----\n")                                         # 2 of 6 weight columns
        with self.assertRaises(ValueError):
            infretis_data.read_infretis_data(path, 3)


class TestWeights(TempDirTest):

    def test_pair_weights(self):
        rows, _ = make_retis_run(self.dir, n_paths=30)
        d = infretis_data.read_infretis_data(os.path.join(self.dir, "infretis_data.txt"), 3)
        for weighting in ("rejections", "staple", "none"):
            W = infretis_data._pair_weights(d, weighting, 3)
            npt.assert_allclose(W, expected_weights(rows, weighting), atol=1e-6, err_msg=weighting)
            npt.assert_allclose(W.sum(0), d["path_f"].sum(0), atol=1e-6)            # per-ensemble total kept
            npt.assert_array_equal(W > 0, d["path_f"] > 0)
        f, w = d["path_f"][:, 2], d["path_w"][:, 2]
        st = infretis_data._pair_weights(d, "staple", 3)[:, 2]
        rj = infretis_data._pair_weights(d, "rejections", 3)[:, 2]
        on = f > 0
        ratio = st[on] / rj[on]
        npt.assert_allclose(ratio * w[on], ratio[0] * w[on][0])                      # staple / rejections ∝ 1 / w
        with self.assertRaises(ValueError):
            infretis_data._pair_weights(d, "bogus", 3)

    def test_weight_consistency(self):
        """Occupancies split over two ensembles in proportion to w (as the infinite swap does): path_f / path_w
        gives class ratios 1, path_f gives 0.5 and 2."""
        rng = np.random.default_rng(1)
        n = 4000
        w = np.ones((n, 4))
        cls = rng.integers(0, 3, n)                                  # (1, 1), (0.5, 1), (1, 0.5)
        w[cls == 1, 2], w[cls == 2, 3] = 0.5, 0.5
        occ = rng.gamma(2.0, 1.0, n)
        f = np.zeros((n, 4))
        f[:, 2] = occ * w[:, 2] * 1.0                               # C_2 = 1
        f[:, 3] = occ * w[:, 3] * 0.6                               # C_3 = 0.6
        data = dict(pnr=np.arange(n), path_f=f, path_w=w)
        res = infretis_data.weight_consistency(data, burn_in_fraction=0.0)
        self.assertEqual(len(res), 1)
        row = res[0]
        self.assertEqual(row["pair"], (2, 3))
        for c, rej in (((0.5, 1), 0.5), ((1, 0.5), 2.0)):
            npt.assert_allclose(row[c]["rejections"][0], rej, rtol=1e-10)
            npt.assert_allclose(row[c]["staple"][0], 1.0, rtol=1e-10)


@unittest.skipUnless(HAVE_INFRETIS, "infretis / inftools not importable")
class TestLoadRetis(TempDirTest):

    def setUp(self):
        super().setUp()
        self.rows, self.interfaces = make_retis_run(self.dir, n_paths=40)

    def load(self, **kw):
        kw.setdefault("burn_in_fraction", 0.0)
        return quiet(infretis_data.load_infretis_data, self.dir, **kw)

    def check(self, got, expected):
        cvs, order, time, weights, paths, _ = got
        self.assertEqual(len(order), len(expected))
        for k, (e, pnr, x, t, w, mirrored) in enumerate(expected):
            npt.assert_allclose(order[k], x, atol=1e-9, err_msg=f"piece {k} (ensemble {e}, path {pnr})")
            npt.assert_array_equal(time[k], t)
            npt.assert_allclose(cvs[k][:, 0], order[k])                       # CV 0 is the order parameter
            npt.assert_allclose(cvs[k][:, 1], x ** 2, atol=1e-8)              # the extra CV column is kept
            npt.assert_allclose(weights[k], w, atol=1e-6)
            self.assertEqual(paths[k], pnr)                                   # infretis path number

    def test_pieces_cut_at_m(self):
        got = self.load()
        self.check(got, expected_retis_pieces(self.rows, self.interfaces))
        self.assertEqual(got[5]["weighting"], "staple")                      # the RETIS default
        # pieces of one path in [0+] and [1+] share its path number
        paths = got[4]
        self.assertTrue(any(paths.count(p) == 2 for p in set(paths)))

    def test_other_weightings(self):
        for weighting in ("rejections", "none"):
            self.check(self.load(weighting=weighting),
                       expected_retis_pieces(self.rows, self.interfaces, weighting=weighting))

    def test_reversed_pieces(self):
        got = self.load(reversed_pieces=True)
        exp = expected_retis_pieces(self.rows, self.interfaces, reversed_pieces=True)
        self.check(got, exp)
        self.assertTrue(any(m for *_, m in exp))
        with self.assertRaises(ValueError):
            self.load(reversed_pieces=True, cut_at_m=False)

    def test_no_cut(self):
        _, order, time, _, _, _ = self.load(cut_at_m=False)
        W = expected_weights(self.rows, "staple")
        n_exp = sum(1 for e in (1, 2) for k, r in enumerate(self.rows) if r["pnr"] >= 3 and W[k, e] > 0)
        self.assertEqual(len(order), n_exp)
        self.assertTrue(all(t[0] == 0 for t in time))                         # whole paths

    def test_selection(self):
        n = len(self.rows)
        # burn-in: the first half of the rows (file order) is skipped
        self.check(self.load(burn_in_fraction=0.5),
                   expected_retis_pieces(self.rows, self.interfaces, first_row=int(np.ceil(0.5 * n))))
        # burn_in (rows) wins if larger than the fraction
        self.check(self.load(burn_in=25, burn_in_fraction=0.1),
                   expected_retis_pieces(self.rows, self.interfaces, first_row=25))
        # paths_per_ensemble: evenly spaced over the eligible paths of each ensemble, first and last included
        W = expected_weights(self.rows, "staple")
        got = self.load(paths_per_ensemble=5)
        expected = []
        for e in (1, 2):
            eligible = [k for k, r in enumerate(self.rows) if r["pnr"] >= 3 and W[k, e] > 0]
            pick = [eligible[i] for i in np.unique(np.round(np.linspace(0, len(eligible) - 1, 5)).astype(int))]
            m = self.interfaces[max(e - 1, 0)]
            expected += [self.rows[k]["pnr"] for k in pick if first_past(self.rows[k]["x"], 0, m) is not None]
        self.assertEqual(got[4], expected)

    def test_thin_per_ensemble(self):
        got = self.load(thin=3)
        W = expected_weights(self.rows, "staple")
        expected = []
        for e in (1, 2):
            eligible = [k for k, r in enumerate(self.rows) if r["pnr"] >= 3 and W[k, e] > 0][::3]
            m = self.interfaces[max(e - 1, 0)]
            expected += [self.rows[k]["pnr"] for k in eligible if first_past(self.rows[k]["x"], 0, m) is not None]
        self.assertEqual(got[4], expected)

    def test_initial_load_paths_skipped(self):
        paths = self.load()[4]
        self.assertFalse(set(paths) & {0, 1, 2})

    def test_missing_order_file_skipped(self):
        victim = 10
        shutil.rmtree(os.path.join(self.dir, "load", str(victim)))
        got = self.load()
        self.assertNotIn(victim, got[4])
        exp = [p for p in expected_retis_pieces(self.rows, self.interfaces) if p[1] != victim]
        self.check(got, exp)

    def test_h5_archive_equals_load_folder(self):
        import h5py
        h5 = os.path.join(self.dir, "archive.h5")
        with h5py.File(h5, "w") as f:
            for r in self.rows:
                f[f"{r['pnr']}/order.txt"] = np.loadtxt(os.path.join(self.dir, "load", str(r["pnr"]), "order.txt"))
        a = self.load()
        shutil.rmtree(os.path.join(self.dir, "load"))
        b = self.load(h5_file=h5)
        for x, y in zip(a[:5], b[:5]):
            self.assertEqual(len(x), len(y))
            for u, v in zip(x, y):
                npt.assert_array_equal(u, v)

    def test_max_length(self):
        got = self.load(max_length=20)
        self.assertTrue(all(len(o) <= 20 for o in got[1]))
        full = self.load()
        for o_cut, o_full in zip(got[1], full[1]):
            npt.assert_array_equal(o_cut, o_full[:20])                       # truncated, not dropped


@unittest.skipUnless(HAVE_INFRETIS, "infretis / inftools not importable")
class TestPrepareData(TempDirTest):

    def test_prepare_data_switches_to_infretis(self):
        rows, interfaces = make_retis_run(self.dir, n_paths=60)
        data, args = prepare(self.dir, "--burn-in-fraction", "0", "--reversed")
        self.assertEqual(args.weights, "staple")                              # filled in from the run
        self.assertEqual((args.lambda_a, args.lambda_b), (0.0, 1.0))           # first and last interface
        self.assertIsNone(data.dim)
        self.assertEqual(data.X.shape, (len(data.lam), 2))                     # order parameter + extra CV
        npt.assert_allclose(data.path_weights.mean(), 1.0)
        npt.assert_array_equal(data.boundary0, data.lam <= 0.0)
        npt.assert_array_equal(data.boundary1, data.lam >= 1.0)
        # pieces: one index per piece; path numbers are the infretis numbers
        exp = expected_retis_pieces(rows, interfaces, reversed_pieces=True)
        self.assertEqual(data.i_traj.max() + 1, len(exp))
        starts = np.r_[0, np.flatnonzero(np.diff(data.i_traj)) + 1]
        npt.assert_array_equal(data.group_traj[starts], [p[1] for p in exp])
        npt.assert_allclose(data.lam[starts], [p[2][0] for p in exp])

    def test_explicit_weights_and_states(self):
        make_retis_run(self.dir, n_paths=30)
        data, args = prepare(self.dir, "--burn-in-fraction", "0", "--weights", "rejections", "-a", "0.1", "-b", "0.9")
        self.assertEqual(args.weights, "rejections")
        npt.assert_array_equal(data.boundary0, data.lam <= 0.1)
        npt.assert_array_equal(data.boundary1, data.lam >= 0.9)

    def test_fit_transform_tensorflow(self):
        """The prepared infretis data go straight into CommittorNE.fit_transform (TensorFlow backend)."""
        import optimalrcs.optimalrcs as optimalrcs
        make_retis_run(self.dir, n_paths=80)
        data, _ = prepare(self.dir, "--burn-in-fraction", "0")
        q = optimalrcs.CommittorNE(boundary0=data.boundary0, boundary1=data.boundary1, i_traj=data.i_traj,
                                   t_traj=data.t_traj, path_weights=data.path_weights, group_traj=data.group_traj)
        q.set_fixed_traj_length_trap(data.boundary0 | data.boundary1, 2 ** 16)
        y = data.X[:, 0].copy()
        q.fit_transform(lambda: y, max_iter=20, print_step=10, ny=3,
                        metrics_print=('iteration', 'delta_r2', 'max_z_zq_stopped', 'delta_x'),
                        save_min_metric='max_z_zq_stopped')
        r = np.asarray(q.r_traj)
        self.assertTrue(np.all(np.isfinite(r)) and r.min() >= 0 and r.max() <= 1)
        npt.assert_array_equal(r[data.boundary0], 0)
        npt.assert_array_equal(r[data.boundary1], 1)
        free = ~(data.boundary0 | data.boundary1)
        self.assertGreater(np.corrcoef(r[free], data.lam[free])[0, 1], 0.8)   # q rises along lambda
        plt.close("all")

def load_cupy():
    """The CuPy optimalrcs module (flat imports from ../optimalrcs-cupy; optimalrcs.py under an alias, since the
    TF package has the same name), or None without CuPy or a GPU."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return None
    import importlib.util
    cupy_dir = os.path.abspath(os.path.join(EXAMPLES, os.pardir, "optimalrcs-cupy"))
    sys.path.insert(0, cupy_dir)
    try:
        spec = importlib.util.spec_from_file_location("optimalrcs_cupy_infretis_test",
                                                      os.path.join(cupy_dir, "optimalrcs.py"))
        orc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orc)
    finally:
        sys.path.remove(cupy_dir)
    return orc


ORC_CUPY = load_cupy()


@unittest.skipUnless(HAVE_INFRETIS and ORC_CUPY is not None, "infretis / inftools or CuPy / GPU not available")
class TestCupyFit(TempDirTest):
    """The prepared infretis data in the CuPy CommittorNE, as load_tis_data_cupy.py and the notebooks use it."""

    def fit(self, data, lazy_basis, n_iter=30, seed=0):
        import cupy as cp
        q = ORC_CUPY.CommittorNE(boundary0=data.boundary0, boundary1=data.boundary1, i_traj=data.i_traj,
                                 t_traj=data.t_traj, path_weights=data.path_weights, group_traj=data.group_traj)
        q.set_fixed_traj_length_trap(data.boundary0 | data.boundary1, 2 ** 16)
        X = cp.asarray(data.X)
        np.random.seed(seed); cp.random.seed(seed)
        quiet(q.fit_transform, lambda: X[:, np.random.randint(X.shape[1])], gamma=0.1, max_iter=n_iter,
              print_step=10, ny=4, lazy_basis=lazy_basis, save_min_metric='max_z_zq_stopped',
              metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped', 'delta_x'))
        return q

    def check_rc(self, q, data):
        import cupy as cp
        r = cp.asnumpy(q.r_traj)
        self.assertEqual(r.shape, data.lam.shape)
        self.assertTrue(np.all(np.isfinite(r)) and r.min() >= 0 and r.max() <= 1)
        npt.assert_array_equal(r[data.boundary0], 0)
        npt.assert_array_equal(r[data.boundary1], 1)
        free = ~(data.boundary0 | data.boundary1)
        self.assertGreater(np.corrcoef(r[free], data.lam[free])[0, 1], 0.8)   # q rises along lambda
        self.assertTrue(np.all(np.isfinite(q.metrics_history['max_z_zq_stopped'])))
        return r

    def test_fit_retis(self):
        make_retis_run(self.dir, n_paths=120)
        data, _ = prepare(self.dir, "--burn-in-fraction", "0")
        q = self.fit(data, lazy_basis=True)
        self.check_rc(q, data)
        quiet(q.plots_tis, ldt=[1, 4, 16])                                    # stopped Z_q, obs-pred
        plt.close("all")

    def test_fit_retis_reversed_two_cvs(self):
        """Mirrored pieces (group_traj ties them to their path) and both CVs of order.txt."""
        make_retis_run(self.dir, n_paths=120)
        data, _ = prepare(self.dir, "--burn-in-fraction", "0", "--reversed")
        self.assertEqual(data.X.shape[1], 2)
        self.assertLess(len(np.unique(data.group_traj)), data.i_traj.max() + 1)   # pieces share paths
        self.check_rc(self.fit(data, lazy_basis=True), data)

    def test_lazy_basis_equals_full_basis(self):
        make_retis_run(self.dir, n_paths=80)
        data, _ = prepare(self.dir, "--burn-in-fraction", "0")
        # one update: the same up to rounding; over iterations the different summation order adds up a little
        import cupy as cp
        r1 = [cp.asnumpy(self.fit(data, lazy_basis=lazy, n_iter=0).r_traj) for lazy in (True, False)]
        npt.assert_allclose(r1[0], r1[1], atol=1e-9)
        r_lazy = self.check_rc(self.fit(data, lazy_basis=True, n_iter=10), data)
        r_full = self.check_rc(self.fit(data, lazy_basis=False, n_iter=10), data)
        npt.assert_allclose(r_lazy, r_full, atol=1e-4)

    def test_fit_staple(self):
        make_staple_run(self.dir, n_paths=40)
        data, _ = prepare(self.dir, "--burn-in-fraction", "0", "-a", "0.0", "-b", "5.0")
        q = self.fit(data, lazy_basis=True, n_iter=10)
        import cupy as cp
        r = cp.asnumpy(q.r_traj)
        self.assertTrue(np.all(np.isfinite(r)) and r.min() >= 0 and r.max() <= 1)

    def test_load_tis_data_cupy_script(self):
        """The CuPy fit script end to end on an infretis run: it reads the run, fits and saves committor.npz with
        the frames prepare_data makes."""
        import subprocess
        make_retis_run(self.dir, n_paths=80)
        out = os.path.join(self.dir, "figs")
        res = subprocess.run([sys.executable, os.path.join(EXAMPLES, "load_tis_data_cupy.py"), self.dir,
                              "--burn-in-fraction", "0", "--max-iter", "20", "--print-step", "10", "--lazy-basis",
                              "--no-potential", "--figure-dir", out],
                             cwd=EXAMPLES, capture_output=True, text=True, timeout=900)
        self.assertEqual(res.returncode, 0, res.stdout[-3000:] + res.stderr[-3000:])
        self.assertIn("infretis run", res.stdout)
        saved = np.load(os.path.join(out, "committor.npz"))
        data, _ = prepare(self.dir, "--burn-in-fraction", "0")
        npt.assert_array_equal(saved["lam"], data.lam)
        npt.assert_array_equal(saved["i_traj"], data.i_traj)
        r = saved["r"]
        self.assertTrue(np.all(np.isfinite(r)) and r.min() >= 0 and r.max() <= 1)
        for name in ("tis_diagnostics.png", "metrics.png", "committor_vs_order.png"):
            self.assertTrue(os.path.isfile(os.path.join(out, name)), name)


@unittest.skipUnless(HAVE_INFRETIS, "infretis / inftools not importable")
class TestStaple(TempDirTest):

    def test_core_convention(self):
        """infretis' core in every ensemble lies inside (L, R), is entered and left across L or R, crosses M."""
        settings = dict(interfaces=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], lambda_minus_one=None, mode="staple")
        rng = np.random.default_rng(3)
        for trial in range(5):
            x = staple_base_path(rng, noise=0.04 * (trial > 0))
            found = 0
            for e in range(2, 6):
                L, M, R = infretis_data.ensemble_window(settings, e)
                core = infretis_data.staple_core(x[:, None], settings, e)
                if core is None:
                    continue
                found += 1
                s, t = core
                inside = x[s:t + 1]
                self.assertTrue(np.all((inside > L) & (inside < R)), (trial, e, core))
                self.assertTrue(x[s - 1] <= L or x[s - 1] >= R, (trial, e))
                self.assertTrue(x[t + 1] <= L or x[t + 1] >= R, (trial, e))
                self.assertTrue(inside.min() < M < inside.max(), (trial, e))
            self.assertEqual(found, 4, trial)                               # valid in columns 2-5

    def test_no_core_without_turns(self):
        settings = dict(interfaces=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], lambda_minus_one=None, mode="staple")
        x = np.linspace(0.5, 4.5, 30)[:, None]                              # monotonic: no turn at either end
        self.assertTrue(all(infretis_data.staple_core(x, settings, e) is None for e in range(2, 6)))

    def test_load_staple_run(self):
        rows, interfaces = make_staple_run(self.dir, n_paths=12)
        cvs, order, time, weights, paths, settings = quiet(
            infretis_data.load_infretis_data, self.dir, burn_in_fraction=0.0, reversed_pieces=True)
        self.assertEqual(settings["weighting"], "staple")                    # infinite swap
        self.assertFalse(set(paths) & {0, 1})                                # ('ld',) rows skipped
        W = expected_weights(rows, "staple")
        k = 0
        for e in range(2, 6):
            m = interfaces[e - 1]
            for j, r in enumerate(rows):
                if j < 2:
                    continue
                x = r["x"]
                s, t_end = infretis_data.staple_core(x[:, None], settings, e)
                t = first_past(x, s, m)
                npt.assert_allclose(order[k], x[t:], atol=1e-9)              # cut from the core start, to the end
                npt.assert_array_equal(time[k], np.arange(t, len(x)))
                npt.assert_allclose(weights[k], W[j, e], atol=1e-6)
                self.assertEqual(paths[k], r["pnr"])
                k += 1
                # mirrored piece: frames up to the last M crossing of the core, read backwards
                side = x[t_end + 1] > m
                tl = [i for i in range(t_end + 1) if (x[i] > m) != side][-1]
                npt.assert_allclose(order[k], x[tl::-1], atol=1e-9)
                npt.assert_array_equal(time[k], np.arange(tl + 1))
                k += 1
        self.assertEqual(k, len(order))

    def test_staple_weights_differ_by_w(self):
        rows, _ = make_staple_run(self.dir, n_paths=10)
        st = quiet(infretis_data.load_infretis_data, self.dir, burn_in_fraction=0.0, weighting="staple")
        rj = quiet(infretis_data.load_infretis_data, self.dir, burn_in_fraction=0.0, weighting="rejections")
        self.assertEqual(len(st[3]), len(rj[3]))
        npt.assert_allclose(st[3], expected_weights(rows, "staple")[2:, 2:].T.ravel(), atol=1e-6)
        npt.assert_allclose(rj[3], expected_weights(rows, "rejections")[2:, 2:].T.ravel(), atol=1e-6)

    def test_shooting_only_defaults_to_path_f(self):
        make_staple_run(self.dir, n_paths=6, infinite_swap=False, swap_frac=0.0)
        settings = quiet(infretis_data.load_infretis_data, self.dir, burn_in_fraction=0.0)[5]
        self.assertEqual(settings["weighting"], "rejections")


@unittest.skipUnless(HAVE_INFRETIS and os.path.isfile(os.path.join(REAL_RETIS_RUN, "infretis.toml")),
                     "real infretis RETIS run not available")
class TestRealRetisRun(TempDirTest):

    def test_real_run_load_folder_vs_h5(self):
        """The RETIS example: pieces from load/ equal those from an h5 archive of the same order files, and
        every piece starts on the far side of its ensemble's M."""
        import h5py
        got = quiet(infretis_data.load_infretis_data, REAL_RETIS_RUN, burn_in_fraction=0.0)
        cvs, order, time, weights, paths, settings = got
        self.assertGreater(len(order), 10)
        pnrs = sorted(set(paths))
        h5 = os.path.join(self.dir, "archive.h5")
        with h5py.File(h5, "w") as f:
            for p in pnrs:
                f[f"{p}/order.txt"] = np.loadtxt(os.path.join(REAL_RETIS_RUN, "load", str(p), "order.txt"))
        got_h5 = quiet(infretis_data.load_infretis_data, REAL_RETIS_RUN, burn_in_fraction=0.0, h5_file=h5)
        for a, b in zip(got[:5], got_h5[:5]):
            self.assertEqual(len(a), len(b))
            for u, v in zip(a, b):
                npt.assert_array_equal(u, v)
        intf = settings["interfaces"]
        for o, t in zip(order, time):
            self.assertGreater(t[0], 0)                                       # cut at t* >= 1
            m_candidates = [intf[0], intf[1]]
            self.assertTrue(any((o[0] > m) for m in m_candidates))


if __name__ == "__main__":
    unittest.main()
