"""Tests for the stopped-segment metrics in optimalrcs/metrics_tis.py and path weights in npneq."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import tensorflow as tf

import optimalrcs.boundaries as boundaries
import optimalrcs.metrics_tis as metrics_tis
import optimalrcs.nonparametrics as nonparametrics
import optimalrcs.optimalrcs as optimalrcs


def lattice_segments(n_segments, n_sites=20, max_half_width=5, seed=0):
    """Segments of a +-1 random walk on 0..n_sites with A = {0} and B = {n_sites}.

    Each segment starts at a random interior site x0 and stops on its own: when it
    reaches A or B, or first reaches x0 +- k for a random k (a window exit, like a
    REPPTIS path reaching L or R). The exact committor is x / n_sites.
    """
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


def dropped_zscore(r, b, i, dt, nbins=200):
    """Z / standard error when windows running past a segment end are dropped (the standard Z_q)."""
    n = len(r)
    idx = np.arange(n)
    tau = metrics_tis.stop_index(i, b)
    seg = metrics_tis._segments(i, n)
    last = np.minimum.accumulate(np.where(np.r_[seg[1:] != seg[:-1], True], idx, n)[::-1])[::-1]
    reaches_ab = (b[tau] != 0) & (tau - idx <= dt)
    use = (b == 0) & (reaches_ab | (idx + dt <= last))
    u = np.minimum(np.minimum(idx + dt, tau), n - 1)
    contrib = (r[u] - r) / dt
    bins = np.clip((r * nbins).astype(int), 0, nbins - 1)
    zq = np.cumsum(np.bincount(bins[use], weights=contrib[use], minlength=nbins))
    sigma = metrics_tis._sigma_zq(seg[use], bins[use], contrib[use], nbins)
    return np.divide(zq, sigma, out=np.zeros(nbins), where=sigma > 0)


class TestStopIndex(unittest.TestCase):

    def test_values(self):
        i = np.asarray([0, 0, 0, 1, 1, 1, 1])
        b = np.asarray([0, 1, 0, 0, 0, 0, 0])
        npt.assert_array_equal(metrics_tis.stop_index(i, b), [1, 1, 2, 6, 6, 6, 6])
        npt.assert_array_equal(metrics_tis.stop_index(None, b), [1, 1, 6, 6, 6, 6, 6])


class TestZqStopped(unittest.TestCase):

    def setUp(self):
        # Values at bin centers (nbins=10) so that the binning is unambiguous.
        self.r = np.asarray([0.55, 0.05, 0.45, 0.25, 0.35, 0.45, 0.55])
        self.b = np.asarray([0, 1, 0, 0, 0, 0, 0])
        self.i = np.asarray([0, 0, 0, 1, 1, 1, 1])

    def test_values(self):
        # dt=2: frame 0 stops at the A frame 1, frame 2 at its segment end, frames 3-6 at frame 6.
        _, zq = metrics_tis.comp_zq_stopped(self.r, self.b, self.i, dt=2, nbins=10)
        npt.assert_allclose(zq, [0, 0, 0.1, 0.2, 0.25, -0.0, 0, 0, 0, 0], atol=1e-12)

    def test_dropped_fraction(self):
        # Frames left of their segment end: f0 2, f2 0, f3 3, f4 2, f5 1, f6 0 (f1 is in A).
        self.assertAlmostEqual(metrics_tis.dropped_fraction(self.i, self.b, 1), 2 / 6)
        self.assertAlmostEqual(metrics_tis.dropped_fraction(self.i, self.b, 3), 5 / 6)
        w = np.asarray([3, 1, 3, 1, 1, 1, 1.])
        self.assertAlmostEqual(metrics_tis.dropped_fraction(self.i, self.b, 1, w), 4 / 10)

    def test_dropped_fraction_with_traps(self):
        # Segment 0 ends in A at frame 2. Without the trap its windows that reach A within dt
        # but run past the end are dropped; with it they are kept.
        r = np.asarray([0.45, 0.25, 0.05, 0.55, 0.65, 0.75])
        b = np.asarray([0, 0, 1, 0, 0, 0.])
        i = np.asarray([0, 0, 0, 1, 1, 1])
        fb = boundaries.FutureBoundary(r, b, i_traj=i)
        self.assertAlmostEqual(metrics_tis.dropped_fraction(i, b, 2, future_boundary=fb), 3 / 5)
        fb.set_distance_to_end_fixed_traj_length_trap(i, b > 0, 10)
        self.assertAlmostEqual(metrics_tis.dropped_fraction(i, b, 2, future_boundary=fb), 2 / 5)

    def test_unit_weights(self):
        for dt in (1, 2, 4):
            _, z0 = metrics_tis.comp_zq_stopped_zscore(self.r, self.b, self.i, dt=dt, nbins=10)
            _, z1 = metrics_tis.comp_zq_stopped_zscore(self.r, self.b, self.i, np.ones(7), dt=dt, nbins=10)
            npt.assert_allclose(z0, z1)

    def test_standard_error_against_brute_force(self):
        r, b, i = lattice_segments(300, seed=1)
        w = np.random.default_rng(2).integers(1, 4, size=i.max() + 1)[i].astype(float)
        nbins, dt = 40, 3
        _, z = metrics_tis.comp_zq_stopped_zscore(r, b, i, w, dt=dt, nbins=nbins)
        tau = metrics_tis.stop_index(i, b)
        _, bins, contrib, starts = metrics_tis._stopped_contributions(r, b, w, dt, nbins, tau)
        per_seg = np.zeros((i.max() + 1, nbins))
        np.add.at(per_seg, (i[starts], bins[starts]), contrib[starts])
        cum = np.cumsum(per_seg, axis=1)
        sigma = np.sqrt((cum ** 2).sum(0))
        npt.assert_allclose(z, np.divide(cum.sum(0), sigma, out=np.zeros(nbins), where=sigma > 0),
                            atol=1e-12)


class TestGroups(unittest.TestCase):

    def test_standard_error_with_groups(self):
        # Segments 2k and 2k+1 form one group, like the forward and reversed piece of a path.
        r, b, i = lattice_segments(300, seed=9)
        g = i // 2
        w = np.random.default_rng(10).integers(1, 4, size=i.max() + 1)[i].astype(float)
        nbins, dt = 40, 3
        _, z = metrics_tis.comp_zq_stopped_zscore(r, b, i, w, dt=dt, nbins=nbins, g_traj=g)
        tau = metrics_tis.stop_index(i, b)
        _, bins, contrib, starts = metrics_tis._stopped_contributions(r, b, w, dt, nbins, tau)
        per_group = np.zeros((g.max() + 1, nbins))
        np.add.at(per_group, (g[starts], bins[starts]), contrib[starts])
        cum = np.cumsum(per_group, axis=1)
        sigma = np.sqrt((cum ** 2).sum(0))
        npt.assert_allclose(z, np.divide(cum.sum(0), sigma, out=np.zeros(nbins), where=sigma > 0),
                            atol=1e-12)
        # One group per segment is the default.
        _, z_seg = metrics_tis.comp_zq_stopped_zscore(r, b, i, w, dt=dt, nbins=nbins)
        _, z_g = metrics_tis.comp_zq_stopped_zscore(r, b, i, w, dt=dt, nbins=nbins, g_traj=i * 7)
        npt.assert_allclose(z_seg, z_g)


class TestStoppedSegments(unittest.TestCase):
    """Segments that end at window exits, with the exact committor known."""

    @classmethod
    def setUpClass(cls):
        cls.r, cls.b, cls.i = lattice_segments(4000)
        cls.ldt = [1, 2, 4, 8, 16, 32]

    def test_exact_committor_passes(self):
        max_z, _, _, _ = metrics_tis._comp_max_zq_stopped(self.r, self.b, self.i, ldt=self.ldt, nbins=20)
        self.assertLess(max_z, 4)

    def test_wrong_rc_fails(self):
        max_z, _, _, _ = metrics_tis._comp_max_zq_stopped(self.r ** 2, self.b, self.i, ldt=self.ldt, nbins=20)
        self.assertGreater(max_z, 6)

    def test_dropping_windows_biases_the_exact_committor(self):
        z = max(np.max(np.abs(dropped_zscore(self.r, self.b, self.i, dt, nbins=20))) for dt in (4, 8, 16))
        self.assertGreater(z, 6)

    def test_obs_pred(self):
        pred, obs, _ = metrics_tis.comp_obs_pred_stopped(self.r, self.b, self.i, nbins=20)
        npt.assert_allclose(obs, pred, atol=0.03)


class TestNpneqWeights(unittest.TestCase):

    def test_weight_two_is_a_duplicated_segment(self):
        # Without regularization a transition of weight 2 counts as the same transition twice.
        rng = np.random.default_rng(0)
        r, b, i = lattice_segments(3, seed=3)
        r = np.where(b > 0, r, rng.random(len(r)))
        fk = rng.random((4, len(r))) * (1 - b)
        seg0 = i == 0
        w = np.where(seg0, 2.0, 1.0)
        r_w = nonparametrics.npneq(tf.constant(r), tf.constant(fk), i, gamma=0, w_traj=tf.constant(w[:-1])).numpy()
        r_dup = np.r_[r[seg0], r]
        fk_dup = np.c_[fk[:, seg0], fk]
        i_dup = np.r_[np.full(seg0.sum(), -1), i]
        r_d = nonparametrics.npneq(tf.constant(r_dup), tf.constant(fk_dup), i_dup, gamma=0).numpy()
        npt.assert_allclose(r_w, r_d[seg0.sum():], atol=1e-10)


class TestCommittorNETis(unittest.TestCase):

    def test_fit_with_stopped_metrics(self):
        r, b, i = lattice_segments(500, seed=4)
        w = np.random.default_rng(5).integers(1, 3, size=i.max() + 1)[i].astype(float)
        q = optimalrcs.CommittorNE(boundary0=b * (r == 0) > 0, boundary1=b * (r == 1) > 0, i_traj=i,
                                   path_weights=w)
        q.fit_transform(lambda: r, max_iter=4, print_step=2, ny=3,
                        metrics_print=('iteration', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped',
                                       'delta_x'),
                        save_min_metric='max_z_zq_stopped')
        self.assertEqual(len(q.metrics_history['max_z_zq_stopped']), 3)
        self.assertTrue(np.all(np.isfinite(q.metrics_history['max_z_zq_stopped'])))
        q.plots_tis(ldt=[1, 4, 16])
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
