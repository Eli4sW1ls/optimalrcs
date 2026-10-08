"""Tests for the CuPy implementation in ../optimalrcs-cupy.

Mirrors the value tests and the fits of the TensorFlow suite, checks the fused
polynomial-basis kernels against the materialized basis, and compares one
iteration of the solvers with the TensorFlow implementation. Skipped when
CuPy or a GPU is not available.
"""
import importlib.util
import os
import sys
import types
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

CUPY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "optimalrcs-cupy"))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_cupy():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return None
    # The modules import each other by flat names (`import metrics`), and
    # optimalrcs.py shares its name with the TF package, so it gets an alias.
    sys.path.insert(0, CUPY_DIR)
    try:
        spec = importlib.util.spec_from_file_location("optimalrcs_cupy", os.path.join(CUPY_DIR, "optimalrcs.py"))
        orc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orc)
        import boundaries, cut_profiles, metrics, metrics_tis, nonparametrics, plots, plots_tis, polybasis
    finally:
        sys.path.remove(CUPY_DIR)
    return types.SimpleNamespace(cp=cp, orc=orc, bd=boundaries, cut_profiles=cut_profiles, metrics=metrics,
                                 metrics_tis=metrics_tis, nonparametrics=nonparametrics, plots=plots,
                                 plots_tis=plots_tis, polybasis=polybasis)


C = _load_cupy()
requires_cupy = unittest.skipIf(C is None, "CuPy with a GPU is not available")


def _load_tf():
    try:
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        import optimalrcs.boundaries as bd
        import optimalrcs.metrics_tis as metrics_tis
        import optimalrcs.nonparametrics as nonparametrics
        import optimalrcs.optimalrcs as orc
        return types.SimpleNamespace(tf=tf, bd=bd, metrics_tis=metrics_tis, nonparametrics=nonparametrics, orc=orc)
    except Exception:
        return None


def np_(a):
    return C.cp.asnumpy(a) if isinstance(a, C.cp.ndarray) else np.asarray(a)


def random_walks(mtraj, msteps):
    """Same walks as tests/test_cut_profiles.py."""
    r_traj, b_traj, i_traj = [], [], []
    for i in range(mtraj):
        x0 = np.random.random()
        for _ in range(msteps):
            x0 = min(max(x0 + 0.1 * (np.random.random() - 0.5), 0), 1)
            r_traj.append(x0)
            i_traj.append(i)
            b_traj.append(1 if x0 in (0, 1) else 0)
    return np.asarray(r_traj), np.asarray(b_traj), np.asarray(i_traj)


def lattice_segments(n_segments, n_sites=20, max_half_width=5, seed=0):
    """Same segments as tests/test_metrics_tis.py: a +-1 walk on 0..n_sites (A = 0, B = n_sites),
    each segment stopping in A/B or at its first exit from x0 +- k. The committor is x / n_sites."""
    rng = np.random.default_rng(seed)
    x_traj, i_traj = [], []
    for i in range(n_segments):
        x0 = x = int(rng.integers(1, n_sites))
        k = int(rng.integers(1, max_half_width + 1))
        path = [x]
        while 0 < x < n_sites and abs(x - x0) < k:
            x += 1 if rng.random() < 0.5 else -1
            path.append(x)
        x_traj += path
        i_traj += [i] * len(path)
    x_traj = np.asarray(x_traj, dtype=float)
    b_traj = ((x_traj == 0) | (x_traj == n_sites)).astype(float)
    return x_traj / n_sites, b_traj, np.asarray(i_traj)


def read_2f4k(max_frames=None):
    r_traj = []
    with open(os.path.join(DATA_DIR, "2f4k.CArmsd")) as f:
        for line in f:
            r_traj.append(float(line.split()[-1]))
            if max_frames is not None and len(r_traj) > max_frames:
                break
    return np.asarray(r_traj)


@requires_cupy
class TestBoundaries(unittest.TestCase):
    """The expected values of tests/test_boundaries.py."""

    def setUp(self):
        cp = C.cp
        self.r = cp.asarray([0, 0.4, 1, 0.6, 0])
        self.b = cp.asarray([1, 0, 1, 0, 1])
        self.t = cp.asarray([1, 3, 4, 7, 10])

    def check(self, obj, **expected):
        for name, values in expected.items():
            npt.assert_array_equal(np_(getattr(obj, name)), np.asarray(values), err_msg=name)

    def test_future_single_trajectory(self):
        for i_traj in (C.cp.asarray([1, 1, 1, 1, 1]), None):
            fb = C.bd.FutureBoundary(self.r, self.b, self.t, i_traj)
            self.check(fb, index=[0, 2, 2, 4, 4], r=[0, 1, 1, 0, 0], delta_i=[0, 1, 0, 1, 0],
                       delta_t=[0, 1, 0, 3, 0], index2=[2, 2, 4, 4, -1], r2=[1, 1, 0, 0, 0],
                       delta_i2=[2, 1, 2, 1, 0], delta_t2=[3, 1, 6, 3, 0], index3=[0, 2, 2, 4, 4])
        fb = C.bd.FutureBoundary(self.r, self.b)
        npt.assert_array_equal(np_(fb.delta_t), np_(fb.delta_i))
        npt.assert_array_equal(np_(fb.delta_t2), np_(fb.delta_i2))

    def test_future_two_trajectories(self):
        fb = C.bd.FutureBoundary(self.r, self.b, self.t, C.cp.asarray([1, 1, 1, 1, 2]))
        self.check(fb, index=[0, 2, 2, -1, 4], r=[0, 1, 1, 0, 0], delta_i=[0, 1, 0, 0, 0],
                   delta_t=[0, 1, 0, 0, 0], index2=[2, 2, -1, -1, -1], r2=[1, 1, 0, 0, 0],
                   delta_i2=[2, 1, 0, 0, 0], delta_t2=[3, 1, 0, 0, 0],
                   index3=[0, 2, 2, 3, 4], delta_i_to_end=[3, 2, 1, 0, 0])

    def test_past_single_trajectory(self):
        for i_traj in (C.cp.asarray([1, 1, 1, 1, 1]), None):
            pb = C.bd.PastBoundary(self.r, self.b, self.t, i_traj)
            self.check(pb, index=[0, 0, 2, 2, 4], r=[0, 0, 1, 1, 0], delta_i=[0, -1, 0, -1, 0],
                       delta_t=[0, -2, 0, -3, 0], index2=[-1, 0, 0, 2, 2], r2=[0, 0, 0, 1, 1],
                       delta_i2=[0, -1, -2, -1, -2], delta_t2=[0, -2, -3, -3, -6])
        pb = C.bd.PastBoundary(self.r, self.b)
        npt.assert_array_equal(np_(pb.delta_t), np_(pb.delta_i))
        npt.assert_array_equal(np_(pb.delta_t2), np_(pb.delta_i2))

    def test_past_two_trajectories(self):
        pb = C.bd.PastBoundary(self.r, self.b, self.t, C.cp.asarray([1, 1, 1, 2, 2]))
        self.check(pb, index=[0, 0, 2, -1, 4], r=[0, 0, 1, 0, 0], delta_i=[0, -1, 0, 0, 0],
                   delta_t=[0, -2, 0, 0, 0], index2=[-1, 0, 0, -1, -1], r2=[0, 0, 0, 0, 0],
                   delta_i2=[0, -1, -2, 0, 0], delta_t2=[0, -2, -3, 0, 0], delta_i_from_start=[0, 1, 2, 0, 1])


@requires_cupy
class TestDeltaR2(unittest.TestCase):
    """tests/test_metrics.py::TestDeltaR2."""

    def dr2(self, *args, **kwargs):
        return float(C.metrics._delta_r2(*[C.cp.asarray(a) for a in args], **kwargs))

    def nobd(self, r, dt):
        return float(C.metrics._delta_r2_eq_nobd(C.cp.asarray(r), dt=dt))

    def test_single_transitions(self):
        self.assertAlmostEqual(self.dr2(np.asarray([0.1, 0.2])), 0.01)

    def test_single_trajectory(self):
        for dt in range(1, 10):
            val = self.nobd([0., 0, 0.1, 0.2, 0.3], dt) + self.nobd([0.1, 0, 0, 0, 0], dt)
            self.assertAlmostEqual(self.dr2(np.asarray([0.1, 0, 0.1, 0.2, 0.3]), np.asarray([0, 1, 0, 0, 0]), dt=dt), val)
            val = self.nobd([0.5, 0.6, 0.7, 1.0, 1.0], dt) + self.nobd([1.0, 1.0, 1.0, 1.0, 0.9], dt)
            self.assertAlmostEqual(self.dr2(np.asarray([0.5, 0.6, 0.7, 1.0, 0.9]), np.asarray([0, 0, 0, 1, 0]), dt=dt), val)

    def test_two_trajectories(self):
        cp = C.cp
        for dt in range(1, 10):
            r = np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.1, 0.4])
            b = np.asarray([1, 0, 0, 0, 0, 0, 0, 0])
            i = np.asarray([1, 1, 1, 1, 1, 2, 2, 2])
            val = float(C.metrics._delta_r2_slow_exact(cp.asarray(r), cp.asarray(b), cp.asarray(i), dt=dt))
            self.assertAlmostEqual(self.dr2(r, b, i, dt=dt), val)
            val = self.nobd([0, 0.1, 0.2, 0.3, 0.4], dt) + self.nobd([0.3, 0.1, 0.4], dt)
            self.assertAlmostEqual(self.dr2(r, b, i, dt=dt), val)


@requires_cupy
class TestCutProfiles(unittest.TestCase):
    """tests/test_cut_profiles.py."""

    def zc1(self, r, b, **kwargs):
        cp = C.cp
        if "i_traj" in kwargs:
            kwargs["i_traj"] = cp.asarray(kwargs["i_traj"])
        lx, lz = C.cut_profiles.comp_zc1(cp.asarray(r, dtype=cp.float64), cp.asarray(b, dtype=cp.float64), **kwargs)
        return np_(lx), np_(lz)

    def zq(self, r, b, i=None, **kwargs):
        cp = C.cp
        return np_(C.cut_profiles.comp_zq(cp.asarray(r, dtype=cp.float64), cp.asarray(b, dtype=cp.float64),
                                          None if i is None else cp.asarray(i), **kwargs)[1])

    def test_zc1_values(self):
        lx, lz = self.zc1([0.4, 0.5, 0.8], [0, 0, 0])
        npt.assert_array_almost_equal(lx, np.linspace(0.4, 0.8, 1001, True))
        lz1 = np.ones(1000)
        lz1[:250], lz1[250:], lz1[-1] = 0.05, 0.15, 0
        npt.assert_array_almost_equal(lz, lz1)
        for dt in (1, 10):
            lx, lz = self.zc1([1., 0, 1], [1, 1, 1], dt=dt)
            npt.assert_array_almost_equal(lx, np.linspace(0, 1, 1001, True))
            lz1 = np.ones(1000)
            lz1[-1] = 0
            npt.assert_array_almost_equal(lz, lz1)
        lx, lz = self.zc1([0, 0.4, 1], [1, 0, 1], nbins=10)
        lz1 = np.ones(10)
        lz1[:4], lz1[4:], lz1[-1] = 0.2, 0.3, 0
        npt.assert_array_almost_equal(lz, lz1)

    def test_zc1_integral_is_delta_r2(self):
        for seed in range(10):
            np.random.seed(seed)
            r, b, i = random_walks(10, 3)
            for dt in [2 ** k for k in range(10)]:
                val = float(C.metrics._delta_r2(*map(C.cp.asarray, (r, b, i)), dt=dt)) / 2
                lx, lz = self.zc1(r, b, i_traj=i, dt=dt, nbins=100000)
                self.assertAlmostEqual(float(np.sum((lx[1:] - lx[:-1]) * lz)), val, 4)

    def test_zq_forward_backward_is_zc1(self):
        for seed in range(10):
            np.random.seed(seed)
            r, b, i = random_walks(10, 200)
            for dt in [2 ** k for k in range(10)]:
                val = (self.zq(r, b, i, dt=dt) + self.zq(r[::-1], b[::-1], i[::-1], dt=dt)) / 2
                npt.assert_array_almost_equal(self.zc1(r, b, i_traj=i, dt=dt)[1], val)

    @unittest.expectedFailure
    def test_zca1_is_zc1(self):
        # Holds for the TF implementation. CuPy's comp_zca bins up to r_max+0.001
        # into nbins+1 bins, so its profile is shifted and one bin longer.
        np.random.seed(0)
        r, b, i = random_walks(10, 200)
        _, lz1 = C.cut_profiles.comp_zca(C.cp.asarray(r), 1, i_traj=C.cp.asarray(i), dt=1)
        npt.assert_array_almost_equal(self.zc1(r, b, i_traj=i, dt=1)[1], np_(lz1))


@requires_cupy
class TestPolyBasis(unittest.TestCase):
    """The fused kernels against the materialized basis_poly_ry."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        r, b, i = random_walks(200, 300)
        cp = C.cp
        cls.r, cls.b, cls.i = cp.asarray(r), cp.asarray(b, dtype=cp.float64), cp.asarray(i)
        cls.y = cp.asarray(np.sin(7 * r) + 0.1 * np.random.standard_normal(len(r)))

    def test_columns(self):
        for ny in (1, 4, 6):
            fused = C.orc.lazy_basis_poly_ry(self.r, self.y, ny, 1 - self.b)
            full = C.orc.basis_poly_ry(self.r, self.y, ny, 1 - self.b)
            npt.assert_array_equal(np_(fused(17, 4321)), np_(full[:, 17:4321]))

    def test_npneq(self):
        for ny in (2, 6):
            fused = C.orc.lazy_basis_poly_ry(self.r, self.y, ny, 1 - self.b)
            full = C.orc.basis_poly_ry(self.r, self.y, ny, 1 - self.b)
            for gamma, stable in ((0.0, False), (0.2, False), (0.2, True)):
                a = np_(C.nonparametrics.npneq(self.r, fused, self.i, gamma, stable))
                c = np_(C.nonparametrics.npneq(self.r, full, self.i, gamma, stable))
                npt.assert_allclose(a, c, rtol=0, atol=1e-8, err_msg=f"ny={ny} gamma={gamma} stable={stable}")

    def test_npneq_weights(self):
        cp = C.cp
        w = cp.asarray(np.random.default_rng(3).integers(1, 4, size=int(self.i.max()) + 1))[self.i][:-1]
        fused = C.orc.lazy_basis_poly_ry(self.r, self.y, 4, 1 - self.b)
        full = C.orc.basis_poly_ry(self.r, self.y, 4, 1 - self.b)
        npt.assert_allclose(np_(C.nonparametrics.npneq(self.r, fused, self.i, 0.1, w_traj=w)),
                            np_(C.nonparametrics.npneq(self.r, full, self.i, 0.1, w_traj=w)), rtol=0, atol=1e-8)

    def test_npneq_weight_two_is_a_duplicated_segment(self):
        # Without regularization a transition of weight 2 counts as the same transition twice.
        cp = C.cp
        full = C.orc.basis_poly_ry(self.r, self.y, 4, 1 - self.b)
        seg0 = self.i == 0
        n0 = int(seg0.sum())
        w = cp.where(seg0, 2.0, 1.0)[:-1]
        r_w = C.nonparametrics.npneq(self.r, full, self.i, 0, w_traj=w)
        r_dup = C.nonparametrics.npneq(cp.concatenate([self.r[seg0], self.r]),
                                       cp.concatenate([full[:, seg0], full], axis=1),
                                       cp.concatenate([cp.full(n0, -1), self.i]), 0)
        npt.assert_allclose(np_(r_w), np_(r_dup[n0:]), rtol=0, atol=1e-8)

    def test_npnet(self):
        cp = C.cp
        t = cp.arange(len(self.r), dtype=cp.float64)
        rt = cp.where(self.b > 0, 0, 10 * self.r)
        fused = C.orc.lazy_basis_poly_ry(rt, self.y, 4, 1 - self.b)
        full = C.orc.basis_poly_ry(rt, self.y, 4, 1 - self.b)
        for subsample in (None, 4):
            npt.assert_allclose(np_(C.nonparametrics.npnet(rt, fused, t, self.i, 0.1, subsample=subsample)),
                                np_(C.nonparametrics.npnet(rt, full, t, self.i, 0.1, subsample=subsample)),
                                rtol=0, atol=1e-7)


@requires_cupy
class TestMetricsTis(unittest.TestCase):
    """The expected values and pass/fail cases of tests/test_metrics_tis.py."""

    def test_values(self):
        cp = C.cp
        r = cp.asarray([0.55, 0.05, 0.45, 0.25, 0.35, 0.45, 0.55])
        b = cp.asarray([0, 1, 0, 0, 0, 0, 0])
        i = cp.asarray([0, 0, 0, 1, 1, 1, 1])
        npt.assert_array_equal(np_(C.metrics_tis.stop_index(i, b)), [1, 1, 2, 6, 6, 6, 6])
        npt.assert_array_equal(np_(C.metrics_tis.stop_index(None, b)), [1, 1, 6, 6, 6, 6, 6])
        _, zq = C.metrics_tis.comp_zq_stopped(r, b, i, dt=2, nbins=10)
        npt.assert_allclose(np_(zq), [0, 0, 0.1, 0.2, 0.25, 0, 0, 0, 0, 0], atol=1e-12)

    def test_exact_committor_passes_and_wrong_rc_fails(self):
        cp = C.cp
        r, b, i = (cp.asarray(a) for a in lattice_segments(4000))
        ldt = [1, 2, 4, 8, 16, 32]
        self.assertLess(C.metrics_tis._comp_max_zq_stopped(r, b, i, ldt=ldt, nbins=20)[0], 4)
        self.assertGreater(C.metrics_tis._comp_max_zq_stopped(r ** 2, b, i, ldt=ldt, nbins=20)[0], 6)


T = _load_tf() if C is not None else None


@requires_cupy
@unittest.skipIf(T is None, "TensorFlow implementation not available")
class TestAgainstTF(unittest.TestCase):
    """One iteration from the same r and basis gives the same result."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(2)
        r, b, i = random_walks(200, 300)
        cls.r, cls.b, cls.i = r, b.astype(float), i
        cls.y = np.cos(5 * r) + 0.1 * np.random.standard_normal(len(r))

    def test_boundaries(self):
        cp = C.cp
        t = np.cumsum(np.random.default_rng(0).uniform(0.5, 1.5, len(self.r)))
        for cls_name in ("FutureBoundary", "PastBoundary"):
            ref = getattr(T.bd, cls_name)(self.r, self.b, t, self.i)
            new = getattr(C.bd, cls_name)(cp.asarray(self.r), cp.asarray(self.b), cp.asarray(t), cp.asarray(self.i))
            for name in ("index", "r", "delta_i", "index2", "r2", "delta_i2", "delta_t", "delta_t2"):
                npt.assert_allclose(np_(getattr(new, name)), np.asarray(getattr(ref, name)), err_msg=f"{cls_name}.{name}")

    def test_npneq(self):
        cp, tf = C.cp, T.tf
        for gamma, stable in ((0.0, False), (0.1, False), (0.1, True)):
            ref = T.nonparametrics.npneq(self.r, T.orc.lazy_basis_poly_ry(tf.constant(self.r), tf.constant(self.y), 6, 1 - self.b),
                                         self.i, gamma, stable).numpy()
            new = C.nonparametrics.npneq(cp.asarray(self.r), C.orc.lazy_basis_poly_ry(cp.asarray(self.r), cp.asarray(self.y), 6,
                                                                                      cp.asarray(1 - self.b)),
                                         cp.asarray(self.i), gamma, stable)
            npt.assert_allclose(np_(new), ref, rtol=0, atol=1e-7, err_msg=f"gamma={gamma} stable={stable}")

    def test_npneq_weights(self):
        cp, tf = C.cp, T.tf
        w = np.random.default_rng(4).integers(1, 4, size=self.i.max() + 1)[self.i][:-1].astype(float)
        ref = T.nonparametrics.npneq(self.r, T.orc.lazy_basis_poly_ry(tf.constant(self.r), tf.constant(self.y), 6, 1 - self.b),
                                     self.i, 0.1, w_traj=tf.constant(w)).numpy()
        new = C.nonparametrics.npneq(cp.asarray(self.r), C.orc.lazy_basis_poly_ry(cp.asarray(self.r), cp.asarray(self.y), 6,
                                                                                  cp.asarray(1 - self.b)),
                                     cp.asarray(self.i), 0.1, w_traj=cp.asarray(w))
        npt.assert_allclose(np_(new), ref, rtol=0, atol=1e-7)

    def test_metrics_tis(self):
        cp = C.cp
        r, b, i = lattice_segments(800, seed=6)
        w = np.random.default_rng(7).integers(1, 4, size=i.max() + 1)[i].astype(float)
        r = np.clip(r + 0.01 * np.random.default_rng(8).standard_normal(len(r)) * (1 - b), 0, 1)
        rc, bc, ic, wc = (cp.asarray(a) for a in (r, b, i, w))
        npt.assert_array_equal(np_(C.metrics_tis.stop_index(ic, bc)), T.metrics_tis.stop_index(i, b))
        for dt in (1, 3, 64):
            for fn in ("comp_zq_stopped", "comp_zq_stopped_zscore"):
                ref = getattr(T.metrics_tis, fn)(r, b, i, w, dt=dt, nbins=50)
                new = getattr(C.metrics_tis, fn)(rc, bc, ic, wc, dt=dt, nbins=50)
                for a, c in zip(new, ref):
                    npt.assert_allclose(np_(a), c, rtol=1e-9, atol=1e-9, err_msg=f"{fn} dt={dt}")
        for a, c in zip(C.metrics_tis.comp_obs_pred_stopped(rc, bc, ic, wc, nbins=20),
                        T.metrics_tis.comp_obs_pred_stopped(r, b, i, w, nbins=20)):
            npt.assert_allclose(np_(a), c, rtol=1e-9, atol=1e-12)
        npt.assert_allclose(C.metrics_tis._comp_max_zq_stopped(rc, bc, ic, wc, ldt=[1, 8], nbins=50),
                            T.metrics_tis._comp_max_zq_stopped(r, b, i, w, ldt=[1, 8], nbins=50), rtol=1e-9)
        for dt in (1, 4, 16):
            self.assertAlmostEqual(C.metrics_tis.dropped_fraction(ic, bc, dt, wc),
                                   T.metrics_tis.dropped_fraction(i, b, dt, w), places=12)


@requires_cupy
class TestFits(unittest.TestCase):
    """The fits of tests/test_committorne.py, test_committor.py and test_mfptne.py."""

    def tearDown(self):
        plt.close("all")

    def test_committorne_history(self):
        r_traj = read_2f4k(10000)
        q = C.orc.CommittorNE(boundary0=r_traj > 10.5, boundary1=r_traj < 1.0)
        float(C.metrics.low_bound_delta_r2_eq(q))
        y = C.cp.asarray(r_traj)
        q.fit_transform(comp_y=lambda: y, gamma=0.05, history_delta_t=list(range(9)), max_iter=10,
                        min_delta_x=1e-4, print_step=1)
        q.plots_feps(delta_t_sim=1)
        q.plots_obs_pred()
        q.plots_feps(r_traj=q.r_traj_min_sd_zq)
        q.plots_obs_pred(r_traj=q.r_traj_min_sd_zq)
        self.assertTrue(np.all(np.isfinite(np_(q.r_traj))))

    def test_committorne(self):
        r_traj = read_2f4k(10000)
        q = C.orc.CommittorNE(boundary0=r_traj > 10.5, boundary1=r_traj < 1.0)
        y = C.cp.asarray(r_traj)
        q.fit_transform(comp_y=lambda: y, envelope=lambda r, it, mi: np.ones_like(r), gamma=lambda it, mi: 0.5,
                        max_iter=1, min_delta_x=1e-4, print_step=1)
        q.fit_transform(comp_y=lambda: y, max_iter=10, min_delta_x=1e-4, print_step=1)
        q.plots_feps(delta_t_sim=1)
        q.plots_feps(delta_t_sim=1, reweight=True)
        q.plots_obs_pred()
        r = np_(q.r_traj)
        self.assertTrue(np.all((r >= 0) & (r <= 1)))
        self.assertTrue(np.all(r[np_(q.boundary0)] == 0) and np.all(r[np_(q.boundary1)] == 1))

    def test_committorne_full_trajectory(self):
        r_traj = read_2f4k()
        q = C.orc.CommittorNE(boundary0=r_traj > 10.5, boundary1=r_traj < 1.0)
        y = C.cp.asarray(r_traj)
        q.fit_transform(comp_y=lambda: y, envelope=lambda r, it, mi: np.ones_like(r), gamma=lambda it, mi: 0.5,
                        max_iter=2, min_delta_x=1e-4, print_step=1)
        q.fit_transform(comp_y=lambda: y, max_iter=10, min_delta_x=1e-4, print_step=1)
        q.plots_feps(delta_t_sim=1)
        q.plots_feps(delta_t_sim=1, reweight=True)
        q.plots_obs_pred()

    def test_committorne_path_weights(self):
        cp = C.cp
        r, b, i = lattice_segments(500, seed=4)
        w = np.random.default_rng(5).integers(1, 3, size=i.max() + 1)[i].astype(float)
        q = C.orc.CommittorNE(boundary0=(b > 0) & (r == 0), boundary1=(b > 0) & (r == 1), i_traj=i,
                              path_weights=w)
        y = cp.asarray(r)
        q.fit_transform(comp_y=lambda: y, max_iter=4, print_step=2, ny=3,
                        metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped', 'delta_x'),
                        save_min_metric='max_z_zq_stopped')
        self.assertEqual(len(q.metrics_history['max_z_zq_stopped']), 2)
        self.assertTrue(np.all(np.isfinite(q.metrics_history['max_z_zq_stopped'])))
        q.plots_tis(ldt=[1, 4, 16])

    def test_mfptne(self):
        r_traj = read_2f4k()
        y = C.cp.asarray(r_traj)
        for history in (None, [0] + [2 ** i for i in range(9)]):
            mfpt = C.orc.MFPTNE(boundary0=r_traj < 1.0)
            float(C.metrics.low_bound_i_mfpt_eq(mfpt))
            np.random.seed(0)
            mfpt.fit_transform(comp_y=lambda: y, gamma=0.01, history_delta_t=history, max_iter=10, min_delta_x=1,
                               print_step=1)
            mfpt.plots_metrics()
            mfpt.plots_feps()
            mfpt.plots_obs_pred()
            self.assertTrue(np.all(np_(mfpt.r_traj) >= 0))


if __name__ == "__main__":
    unittest.main()
