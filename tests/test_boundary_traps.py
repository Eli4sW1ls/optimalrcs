"""Tests for set_boundary_traps: Z_q of excursion-cut data equals Z_q of the continuous trajectory."""
import importlib.util
import os
import sys
import unittest

import numpy as np
import numpy.testing as npt

import optimalrcs.optimalrcs as optimalrcs
import optimalrcs.cut_profiles as cut_profiles

CUPY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "optimalrcs-cupy"))
LDT = (1, 2, 4, 16, 64, 256, 1024)


def random_walk(n=200_000, n_sites=20, seed=0):
    """+-1 random walk on 0..n_sites (reflecting), A = {0, 1}, B = {n_sites - 1, n_sites};
    r is the exact committor (x - 1) / (n_sites - 2), clipped to [0, 1]."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, int); x[0] = n_sites // 2
    steps = rng.choice((-1, 1), size=n)
    for t in range(1, n):
        x[t] = min(max(x[t - 1] + steps[t], 0), n_sites)
    r = np.clip((x - 1) / (n_sites - 2), 0, 1)
    return r, x <= 1, x >= n_sites - 1


def excursions(boundary0, boundary1):
    """Rows outside A and B plus the A/B row before and after each excursion, and a segment index
    that starts a new segment wherever kept rows are not consecutive."""
    out = ~(boundary0 | boundary1)
    keep = out | np.r_[False, out[:-1]] | np.r_[out[1:], False]
    rows = np.flatnonzero(keep)
    i_traj = np.cumsum(np.r_[True, np.diff(rows) != 1])
    return rows, i_traj


def zq_values(zq, model, dt):
    edges, z = zq(model, dt)
    return np.asarray(z)[[100, 300, 500, 700, 900]]


class TestBoundaryTrapsTF(unittest.TestCase):
    def test_excursions_with_traps_match_continuous(self):
        r, b0, b1 = random_walk()
        rows, i_traj = excursions(b0, b1)
        full = optimalrcs.Committor(boundary0=b0, boundary1=b1)
        full.r_traj = r
        cut = optimalrcs.Committor(boundary0=b0[rows], boundary1=b1[rows], i_traj=i_traj)
        cut.r_traj = r[rows]

        def zq(m, dt):
            return cut_profiles.comp_zq(m.r_traj, m.b_traj, m.i_traj, m.future_boundary, m.past_boundary, dt=dt)

        dropped = False
        for dt in LDT:
            if not np.allclose(zq_values(zq, full, dt), zq_values(zq, cut, dt)):
                dropped = True
        self.assertTrue(dropped, "without traps the cut data should lose boundary terms at large lags")
        # the continuous run ends inside B: trap its end too, or frames within dt of it lose terms
        full.set_boundary_traps()
        cut.set_boundary_traps()
        for dt in LDT:
            npt.assert_allclose(zq_values(zq, cut, dt), zq_values(zq, full, dt), rtol=1e-6, atol=1e-6)
        cut.set_boundary_traps()   # idempotent
        npt.assert_allclose(zq_values(zq, cut, 256), zq_values(zq, full, 256), rtol=1e-6, atol=1e-6)


class TestBoundaryTrapsCuPy(unittest.TestCase):
    def setUp(self):
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
        except Exception:
            self.skipTest("CuPy or a GPU is not available")
        sys.path.insert(0, CUPY_DIR)
        try:
            spec = importlib.util.spec_from_file_location("optimalrcs_cupy", os.path.join(CUPY_DIR, "optimalrcs.py"))
            self.orc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.orc)
            import cut_profiles as cupy_cut_profiles
            self.cut_profiles = cupy_cut_profiles
        finally:
            sys.path.remove(CUPY_DIR)
        self.cp = cp

    def test_excursions_with_traps_match_continuous(self):
        cp = self.cp
        r, b0, b1 = random_walk()
        rows, i_traj = excursions(b0, b1)
        full = self.orc.Committor(boundary0=b0, boundary1=b1, i_traj=np.ones(len(r), int))
        full.r_traj = cp.asarray(r)
        cut = self.orc.Committor(boundary0=b0[rows], boundary1=b1[rows], i_traj=i_traj)
        cut.r_traj = cp.asarray(r[rows])

        def zq(m, dt):
            e, z = self.cut_profiles.comp_zq(m.r_traj, m.b_traj, m.i_traj, m.future_boundary, m.past_boundary, dt=dt)
            return e, cp.asnumpy(z)

        full.set_boundary_traps()
        cut.set_boundary_traps()
        for dt in LDT:
            npt.assert_allclose(zq_values(zq, cut, dt), zq_values(zq, full, dt), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
