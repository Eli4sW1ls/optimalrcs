"""The exact committor of a 2D overdamped system as an order parameter (a test of the samplers without the error of a
learned lambda): lambda = s (the state CV) outside (lambda_A, lambda_B), and inside

    lambda = lambda_A + (lambda_B - lambda_A) * clip((log10 q + decades) / decades, eps, 1 - eps)

with q the grid committor of toy2d.exact_committor_brownian (bilinear in (y, x)), so the interfaces are committor
surfaces, the states are those of the state CV, and lambda resolves q down to 10^-decades in the 6 decimals of
order.txt. Used by infretis_orderp.ExactLambda (in the sampler) and by loop.seed (relabelling seed paths).

    python exact_lambda.py build <system> <out.npz> [--h 0.001] [--decades 8]
"""
import argparse
import os
import sys

import numpy as np


class ExactLambdaModel:
    def __init__(self, path):
        z = np.load(path)
        self.path = os.path.abspath(path)
        self.y, self.x, self.q = z['y'], z['x'], z['q']
        self.lambda_a, self.lambda_b = float(z['lambda_a']), float(z['lambda_b'])
        self.decades = float(z['decades'])
        self.cv_names = ['y', 'x']
        from scipy.interpolate import RegularGridInterpolator
        self._f = RegularGridInterpolator((self.y, self.x), self.q, bounds_error=False, fill_value=None)
        self._eps = 1e-6

    def lam_of_q(self, q):
        u = (np.log10(np.maximum(np.asarray(q, float), 10 ** (-self.decades - 1))) + self.decades) / self.decades
        return self.lambda_a + (self.lambda_b - self.lambda_a) * np.clip(u, self._eps, 1 - self._eps)

    def q_of_lam(self, lam):
        u = (np.asarray(lam, float) - self.lambda_a) / (self.lambda_b - self.lambda_a)
        return 10 ** (u * self.decades - self.decades)

    def __call__(self, cv):
        cv = np.atleast_2d(np.asarray(cv, float))
        y, x = cv[:, 0], cv[:, 1]
        lam = self.lam_of_q(np.clip(self._f(np.c_[y, x]), 0.0, 1.0))
        lam = np.where(y <= self.lambda_a, y, lam)
        return np.where(y >= self.lambda_b, y, lam)

    def one(self, cv):
        """lambda of one frame: the bilinear interpolation of __call__ (RegularGridInterpolator, linear) written out
        for a single point, about 100x faster per call (the sampler evaluates one frame at a time)."""
        y, x = float(cv[0]), float(cv[1])
        if y <= self.lambda_a or y >= self.lambda_b:
            return y
        ys, xs, q = self.y, self.x, self.q
        fy = (y - ys[0]) / (ys[1] - ys[0]); fx = (x - xs[0]) / (xs[1] - xs[0])
        iy = min(max(int(fy), 0), len(ys) - 2); ix = min(max(int(fx), 0), len(xs) - 2)
        ty, tx = fy - iy, fx - ix                       # outside the grid: linear extrapolation, as fill_value=None
        v = ((1 - ty) * (1 - tx) * q[iy, ix] + (1 - ty) * tx * q[iy, ix + 1]
             + ty * (1 - tx) * q[iy + 1, ix] + ty * tx * q[iy + 1, ix + 1])
        return float(self.lam_of_q(min(max(v, 0.0), 1.0)))


def build(system_name, out, h=0.001, decades=8.0):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here); sys.path.insert(0, os.path.dirname(here))
    from loop import env  # noqa: F401
    from loop import ase_systems
    from loop.ase_calcs import load_toytis_potential
    import toy2d
    s = ase_systems.get(system_name)
    pot = load_toytis_potential(s['potential'], s['potential_kwargs'])
    ref = toy2d.exact_committor_brownian(pot, s['kT'], s['gamma'], s['dt'], s['lambda_a'], s['lambda_b'], h=h,
                                         y_range=(-0.03, 0.93), x_range=(-0.03, 0.33))
    np.savez(out, y=ref['y'], x=ref['x'], q=ref['q'], lambda_a=s['lambda_a'], lambda_b=s['lambda_b'], decades=decades,
             h=h, k_ab=ref['k_ab'], system=system_name)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('system'); b.add_argument('out'); b.add_argument('--h', type=float, default=0.001)
    b.add_argument('--decades', type=float, default=8.0)
    a = ap.parse_args()
    print(build(a.system, a.out, a.h, a.decades))
