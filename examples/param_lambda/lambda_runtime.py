"""Exported parametric order parameter lambda(x): numpy only, no state, one frame or many.

    model = LambdaModel('exported/z_br_m3.npz')
    lam = model(cv)            # cv: (n, d) array of the frame CVs, in the order of model.cv_names
    lam = model.one(cv_row)    # one frame (fast path for the sampler); same value as model(cv_row[None])[0]
    q = model.q(cv)            # calibrated committor estimate g(z(x))
    bad = model.ood(cv)        # True where a frame is outside the training data (range or Mahalanobis)

What lambda is, step by step (all of it depends on the current frame only):

1. features f(x): monomials of linear forms of the CVs (`Features`), standardized with the training mean/sd;
2. raw score z(f): linear, additive cubic splines (GAM) or an MLP, trained on logit q_np;
3. calibration m = g(z): a monotone cubic (PCHIP) through the isotonic regression of q_np on z, so m is a
   committor estimate in [g_min, g_max] with 0 < g_min, g_max < 1;
4. states: the frame's state coordinate s (the CV the states A: s <= lambda_A and B: s >= lambda_B are defined
   on) decides A and B, not the model. Outside (lambda_A, lambda_B), lambda = s. Inside,
       h = (1 - a(s)) (1 - b(s)) m + b(s),
       a(s) = 1 - S((s - lambda_A) / w),   b(s) = 1 - S((lambda_B - s) / w),   S(u) = 3u^2 - 2u^3 on [0, 1],
   with ramp width w = ramp * (lambda_B - lambda_A) (a fixed fraction of the state-to-state range, not tied to
   any interface). h = 0 at s = lambda_A and 1 at s = lambda_B, so lambda is continuous there; between, h is
   strictly between 0 and 1, so s < lambda_A <=> lambda < lambda_A and s > lambda_B <=> lambda > lambda_B,
   whatever the model does (lambda is also kept at least one float away from lambda_A and lambda_B, so the
   comparisons hold after rounding). h = m except within w of A or B.
4b. (export format 2, optional keys) instead of the multiplicative ramps, the smooth minimum of
   docs/param_lambda_interface_spacing.md on the lambda scale (`ramp_kind = "smin"`), applied after step 5:
       lambda = lambda_A + smin(lambda - lambda_A, kappa_A (s - lambda_A))      near A
       lambda = lambda_B - smin(lambda_B - lambda, kappa_B (lambda_B - s))      near B
   with smin the smooth minimum in log space (`_smin`, band smin_k, default ln 2: exactly min(a, b) when the two
   differ by more than a factor e^k): the isosurface of lambda_k runs parallel to A's boundary at least
   (lambda_k - lambda_A) / kappa_A from it. kappa is floored at e^k (and kappa_B at the front's level), so each cap
   is exactly inactive at the opposite state. With a provisional front (`front_s` < lambda_B, the state coordinate the data reached, `front_h` its
   committor level on the output scale) the model's h is scaled to front_h, the cap at B is applied at the front
   and beyond it lambda rises linearly in s to lambda_B.
   Trust-region blend (`blend_*` keys): lambda = (1 - t) lambda + t lambda_seed(s), with lambda_seed a monotone
   table of the seed coordinate s on the lambda scale (fitted on in-distribution frames) and
   t = S((d2 - blend_thr) / blend_width) of the Mahalanobis distance d2 of the frame's features: outside the
   training data the order parameter falls back to the seed. All of it keeps lambda strictly inside
   (lambda_A, lambda_B) for s strictly inside, so s < lambda_A <=> lambda < lambda_A exactly (and at B).
5. output scale: lambda = lambda_A + (lambda_B - lambda_A) * log1p(h / eps) / log1p(1 / eps), a strictly
   increasing map that spreads small committor values (eps = scale_eps); or, if the export has a scale table
   (scale_h, scale_u, scale_d: natural.py), lambda = lambda_A + (lambda_B - lambda_A) * u(h) with u the monotone
   PCHIP through it (the natural coordinate of the committor). Crossings and turns depend only on the order of
   lambda and the interfaces, so this only sets where interfaces land: `interface_for_q`.
"""
import bisect
import math

import numpy as np


# ---------------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------------
class Features:
    """Monomials of linear forms of the CVs, e.g. 'y', 'x^2', 'y*x', 'u^2' with u = (y + x)/sqrt2, v = (y - x)/sqrt2."""

    def __init__(self, cv_names, specs):
        self.cv_names, self.specs = list(cv_names), list(specs)
        base = {name: np.eye(len(cv_names))[i] for i, name in enumerate(cv_names)}
        if len(cv_names) >= 2:
            base['u'] = (np.eye(len(cv_names))[0] + np.eye(len(cv_names))[1]) / np.sqrt(2)
            base['v'] = (np.eye(len(cv_names))[0] - np.eye(len(cv_names))[1]) / np.sqrt(2)
        names = list(base)
        self.L = np.array([base[n] for n in names])                       # (n_base, d)
        E = np.zeros((len(specs), len(names)))
        for k, spec in enumerate(specs):
            for factor in spec.split('*'):
                sym, _, power = factor.partition('^')
                E[k, names.index(sym)] += int(power or 1)
        keep = E.any(0)
        self.L, self.E = self.L[keep], E[:, keep]
        self._index()

    def _index(self):
        """Each monomial as padded (base index, exponent) arrays: features = prod_j b[K[i, j]] ** P[i, j]."""
        self.E_int = self.E.astype(int)
        terms = [[(k, e) for k, e in enumerate(row) if e] for row in self.E_int]
        width = max(1, max(len(t) for t in terms))
        self._K = np.zeros((len(terms), width), int)
        self._P = np.zeros((len(terms), width))                             # padding: b ** 0 = 1
        for i, t in enumerate(terms):
            for j, (k, e) in enumerate(t):
                self._K[i, j], self._P[i, j] = k, e

    def __call__(self, cv):
        b = np.asarray(cv, float) @ self.L.T                               # (n, n_base)
        return np.prod(b[:, self._K] ** self._P, axis=2)

    def one(self, cv):
        """Features of one frame (1D array cv)."""
        b = self.L @ cv
        return np.prod(b[self._K] ** self._P, axis=1)

    def to_dict(self):
        return dict(feat_L=self.L, feat_E=self.E, feat_specs=np.array(self.specs), cv_names=np.array(self.cv_names))

    @classmethod
    def from_arrays(cls, L, E, specs, cv_names):
        self = cls.__new__(cls)
        self.cv_names, self.specs, self.L, self.E = list(cv_names), list(specs), np.asarray(L), np.asarray(E)
        self._index()
        return self


# ---------------------------------------------------------------------------------------------------
# uniform cubic B-splines (the GAM terms)
# ---------------------------------------------------------------------------------------------------
def bspline_basis(x, lo, hi, n_intervals):
    """(n, n_intervals + 3) uniform cubic B-spline basis on [lo, hi]; x is clipped to [lo, hi]."""
    h = (hi - lo) / n_intervals
    t = (np.clip(x, lo, hi) - lo) / h
    i = np.clip(np.floor(t).astype(int), 0, n_intervals - 1)
    u = t - i
    B = np.zeros((len(x), n_intervals + 3))
    rows = np.arange(len(x))
    B[rows, i] = (1 - u) ** 3 / 6
    B[rows, i + 1] = (3 * u ** 3 - 6 * u ** 2 + 4) / 6
    B[rows, i + 2] = (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6
    B[rows, i + 3] = u ** 3 / 6
    return B


# ---------------------------------------------------------------------------------------------------
# calibration and output map
# ---------------------------------------------------------------------------------------------------
def pchip_eval(zk, qk, dk, z):
    """Monotone cubic Hermite interpolant through (zk, qk) with slopes dk; constant beyond the knots."""
    z = np.clip(z, zk[0], zk[-1])
    i = np.clip(np.searchsorted(zk, z, side='right') - 1, 0, len(zk) - 2)
    hk = zk[i + 1] - zk[i]
    t = (z - zk[i]) / hk
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * qk[i] + h10 * hk * dk[i] + h01 * qk[i + 1] + h11 * hk * dk[i + 1]


def _pchip_one(zk, qk, dk, z):
    """pchip_eval for one float and plain-Python knot lists."""
    if z <= zk[0]:
        return qk[0]
    if z >= zk[-1]:
        return qk[-1]
    i = bisect.bisect_right(zk, z) - 1
    hk = zk[i + 1] - zk[i]
    t = (z - zk[i]) / hk
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * qk[i] + (t3 - 2 * t2 + t) * hk * dk[i]
            + (-2 * t3 + 3 * t2) * qk[i + 1] + (t3 - t2) * hk * dk[i + 1])


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


_SMIN_FLOOR = 1e-300


def _smin(a, b, k):
    """Smooth minimum of positive arrays in log space: min(a, b) exactly when the ratio a/b lies outside
    [e^-k, e^k], otherwise exp(min(ln a, ln b) - k t^2 / 4) with t = 1 - |ln a - ln b| / k. It is C1, never above
    min(a, b), positive and increasing in both arguments."""
    la = np.log(np.maximum(np.asarray(a, float), _SMIN_FLOOR))
    lb = np.log(np.maximum(np.asarray(b, float), _SMIN_FLOOR))
    t = np.maximum(k - np.abs(la - lb), 0.0) / k
    return np.exp(np.minimum(la, lb) - k * t * t / 4)


def _smin_one(a, b, k):
    la, lb = math.log(max(a, _SMIN_FLOOR)), math.log(max(b, _SMIN_FLOOR))
    t = max(k - abs(la - lb), 0.0) / k
    return math.exp(min(la, lb) - k * t * t / 4)


def _smoothstep_one(u):
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------------
class ScoreModel:
    """Features, standardization and raw score z(x) (steps 1-2); what the training code exports."""

    def __init__(self, path_or_dict):
        p = dict(np.load(path_or_dict, allow_pickle=False)) if isinstance(path_or_dict, str) else dict(path_or_dict)
        self.params = p
        self.kind = str(p['kind'])
        self.feat = Features.from_arrays(p['feat_L'], p['feat_E'], p['feat_specs'], p['cv_names'])
        self.cv_names = self.feat.cv_names
        self.mu, self.sd = np.asarray(p['mu'], float), np.asarray(p['sd'], float)
        if self.kind == 'linear':
            self.w, self.b = np.asarray(p['w'], float), float(p['b'])
        elif self.kind == 'gam':
            self.coef, self.b = np.asarray(p['coef'], float), float(p['b'])
            self.lo, self.hi, self.n_int = p['spl_lo'], p['spl_hi'], int(p['spl_n'])
        elif self.kind == 'mlp':
            n = int(p['n_layers'])
            self.layers = [(np.asarray(p[f'W{i}'], float), np.asarray(p[f'b{i}'], float)) for i in range(n)]
        elif self.kind != 'feature':
            raise ValueError(f'unknown model kind {self.kind}')

    def features(self, cv):
        return (self.feat(np.atleast_2d(cv)) - self.mu) / self.sd

    def raw(self, cv, chunk=262144):
        """z(x): the model's score, on the logit-q scale it was trained on (in chunks: bounded memory)."""
        cv = np.atleast_2d(cv)
        if len(cv) > chunk:
            return np.concatenate([self.raw(cv[s:s + chunk], chunk) for s in range(0, len(cv), chunk)])
        f = self.features(cv)
        if self.kind == 'linear':
            return f @ self.w + self.b
        if self.kind == 'gam':
            z = np.full(len(f), self.b)
            for j in range(f.shape[1]):
                z += bspline_basis(f[:, j], self.lo[j], self.hi[j], self.n_int) @ self.coef[j]
            return z
        if self.kind == 'feature':
            return f[:, 0]
        h = f
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < len(self.layers) - 1:
                h = h / (1 + np.exp(-h))                                    # SiLU
        return h[:, 0]


class LambdaModel(ScoreModel):
    """ScoreModel plus calibration, the A/B handling, the output scale and the OOD flag (steps 3-5)."""

    def __init__(self, path_or_dict):
        super().__init__(path_or_dict)
        p = self.params
        self.zk, self.qk, self.dk = p['cal_z'], p['cal_q'], p['cal_d']
        self.lambda_a, self.lambda_b = float(p['lambda_a']), float(p['lambda_b'])
        self.state_cv = int(p['state_cv'])
        self.ramp_w = float(p['ramp']) * (self.lambda_b - self.lambda_a)
        self.eps = float(p['scale_eps'])
        self._log1p_inv_eps = math.log1p(1 / self.eps)
        self.scale_table = 'scale_h' in p                     # natural-coordinate scale (natural.py)
        if self.scale_table:
            self._sh, self._su, self._sd = (np.asarray(p[k], float) for k in ('scale_h', 'scale_u', 'scale_d'))
            self._sh_l, self._su_l, self._sd_l = self._sh.tolist(), self._su.tolist(), self._sd.tolist()
        # inside (lambda_A, lambda_B) lambda stays strictly inside, also after rounding: then <, <=, >, >= against
        # lambda_A and lambda_B all agree with the state CV
        self._in_lo, self._in_hi = math.nextafter(self.lambda_a, math.inf), math.nextafter(self.lambda_b, -math.inf)
        self.cv_min, self.cv_max = p['cv_min'], p['cv_max']
        self.maha_mean, self.maha_icov, self.maha_thr = p['maha_mean'], p['maha_icov'], float(p['maha_thr'])
        # format 2: smooth-minimum ramps, provisional front, trust-region blend (absent in older exports)
        self.ramp_kind = str(p['ramp_kind']) if 'ramp_kind' in p else 'mult'
        self.kappa_a, self.kappa_b = float(p.get('kappa_a', 1.0)), float(p.get('kappa_b', 1.0))
        self.smin_k = float(p.get('smin_k', math.log(2.0)))
        self.front_s = float(p['front_s']) if 'front_s' in p else None
        self.front_h = float(p.get('front_h', 0.5))
        # the smooth minimum is exact beyond a ratio e^k: these floors keep each cap exactly inactive at the
        # opposite state (no cross-talk between the A cap, the B cap and the front)
        ek = math.exp(self.smin_k)
        self.kappa_a = max(self.kappa_a, ek)
        if self.front_s is None:
            self.kappa_b = max(self.kappa_b, ek)
        else:
            self._lam_front = float(self.scale(np.array([self.front_h]))[0])
            self.kappa_b = max(self.kappa_b, ek * (self._lam_front - self.lambda_a) / (self.front_s - self.lambda_a))
        self.has_blend = 'blend_s_knots' in p
        if self.has_blend:
            self._bs, self._bh, self._bd = (np.asarray(p[k], float) for k in ('blend_s_knots', 'blend_h_knots', 'blend_d_knots'))
            self._bs_l, self._bh_l, self._bd_l = self._bs.tolist(), self._bh.tolist(), self._bd.tolist()
            self.blend_thr, self.blend_width = float(p['blend_thr']), float(p['blend_width'])
        self._maha_mean_l, self._maha_icov_l = np.asarray(self.maha_mean, float).tolist(), np.asarray(self.maha_icov, float).tolist()
        # plain-Python copies for the one-frame path
        self._zk, self._qk, self._dk = self.zk.tolist(), self.qk.tolist(), self.dk.tolist()
        self._L = self.feat.L.tolist()
        self._numpy_features = self.feat.L.shape[1] > 8
        self._E = [[(k, int(e)) for k, e in enumerate(row) if e] for row in self.feat.E_int.tolist()]
        self._mu, self._isd = self.mu.tolist(), (1 / self.sd).tolist()
        if self.kind == 'linear':
            self._w = self.w.tolist()
        elif self.kind == 'gam':
            self._coef = self.coef.tolist()
            self._lo, self._hi = np.asarray(self.lo, float).tolist(), np.asarray(self.hi, float).tolist()
        self._cv_min, self._cv_max = np.asarray(self.cv_min, float).tolist(), np.asarray(self.cv_max, float).tolist()

    def q(self, cv):
        return pchip_eval(self.zk, self.qk, self.dk, self.raw(cv))

    def _lambda_inside(self, s, m, cv=None):
        """lambda (arrays) of frames strictly between the states from their calibrated committor m: steps 4, 5
        and 4b. `cv` is needed for the trust-region blend (skipped without it)."""
        s, m = np.asarray(s, float), np.asarray(m, float)
        la, lb = self.lambda_a, self.lambda_b
        if self.ramp_kind == 'mult':
            a = 1 - _smoothstep((s - la) / self.ramp_w)
            b = 1 - _smoothstep((lb - s) / self.ramp_w)
            lam = self.scale((1 - a) * (1 - b) * m + b)
        else:
            p = self.smin_k
            if self.front_s is None:
                lam = self.scale(m)
            else:
                lf = self._lam_front
                below = lf - _smin(lf - self.scale(self.front_h * m), self.kappa_b * (self.front_s - s), p)
                above = lf + (lb - lf) * (s - self.front_s) / (lb - self.front_s)
                lam = np.where(s < self.front_s, below, above)
            lam = la + _smin(lam - la, self.kappa_a * (s - la), p)
            if self.front_s is None:
                lam = lb - _smin(lb - lam, self.kappa_b * (lb - s), p)
        if self.has_blend and cv is not None:
            t = self.trust(cv)
            lam = (1 - t) * lam + t * pchip_eval(self._bs, self._bh, self._bd, s)
        return lam

    def blend(self, s, m, cv=None):
        """h(s, m): the committor level on the output scale after the state handling (arrays; the inverse of
        the output scale applied to `_lambda_inside`)."""
        return self.h_of_lambda(self._lambda_inside(s, m, cv))

    h_of = blend

    def trust(self, cv):
        """t in [0, 1] of the trust-region blend per frame (0 without a blend)."""
        cv = np.atleast_2d(np.asarray(cv, float))
        if not self.has_blend:
            return np.zeros(len(cv))
        d = self.features(cv) - self.maha_mean
        d2 = np.einsum('ni,ij,nj->n', d, self.maha_icov, d)
        return _smoothstep((d2 - self.blend_thr) / self.blend_width)

    def scale(self, h):
        if self.scale_table:
            return self.lambda_a + (self.lambda_b - self.lambda_a) * pchip_eval(self._sh, self._su, self._sd, h)
        return self.lambda_a + (self.lambda_b - self.lambda_a) * np.log1p(h / self.eps) / self._log1p_inv_eps

    def h_of_lambda(self, lam):
        """Inverse of scale (arrays, lambda between lambda_A and lambda_B)."""
        f = (np.asarray(lam, float) - self.lambda_a) / (self.lambda_b - self.lambda_a)
        if self.scale_table:
            hh = np.linspace(0, 1, 20001) ** 3                  # dense in h near 0, where u changes fastest
            return np.interp(f, pchip_eval(self._sh, self._su, self._sd, hh), hh)
        return self.eps * np.expm1(f * self._log1p_inv_eps)

    def interface_for_q(self, q):
        """lambda value of committor level q away from the A/B ramps (where h = m = q)."""
        return float(self.scale(np.asarray(q, float)))

    def __call__(self, cv):
        cv = np.atleast_2d(np.asarray(cv, float))
        s = cv[:, self.state_cv]
        inside = (s > self.lambda_a) & (s < self.lambda_b)
        lam = s.copy()
        if inside.any():
            lam[inside] = np.clip(self._lambda_inside(s[inside], self.q(cv[inside]), cv[inside]), self._in_lo, self._in_hi)
        return lam

    def ood(self, cv):
        """True for frames between the states that lie outside the training data (per-CV range or Mahalanobis
        distance of the features); frames in A or B never use the model and are never flagged."""
        cv = np.atleast_2d(np.asarray(cv, float))
        s = cv[:, self.state_cv]
        out = np.any((cv < self.cv_min) | (cv > self.cv_max), axis=1)
        d = self.features(cv) - self.maha_mean
        out |= np.einsum('ni,ij,nj->n', d, self.maha_icov, d) > self.maha_thr
        return out & (s > self.lambda_a) & (s < self.lambda_b)

    # -- one frame ------------------------------------------------------------------------------------
    def _features_one(self, cv):
        if self._numpy_features:                                           # many CVs: O(d^2) in Python is slow
            return ((self.feat.one(np.asarray(cv, float)) - self.mu) / self.sd).tolist()
        base = [sum(l * c for l, c in zip(row, cv)) for row in self._L]
        out = []
        for row, mu, isd in zip(self._E, self._mu, self._isd):
            v = 1.0
            for k, e in row:
                v *= base[k] ** e
            out.append((v - mu) * isd)
        return out

    def raw_one(self, cv):
        """z of one frame (pure Python except the MLP layers)."""
        return self._raw_from_features(self._features_one(cv))

    def _raw_from_features(self, f):
        if self.kind == 'linear':
            return sum(a * b for a, b in zip(self._w, f)) + self.b
        if self.kind == 'feature':
            return f[0]
        if self.kind == 'gam':
            z = self.b
            n = self.n_int
            for j, x in enumerate(f):
                lo, hi = self._lo[j], self._hi[j]
                t = (min(max(x, lo), hi) - lo) / (hi - lo) * n
                i = min(int(t), n - 1)
                u = t - i
                u2, u3 = u * u, u * u * u
                c = self._coef[j]
                z += (c[i] * (1 - u) ** 3 + c[i + 1] * (3 * u3 - 6 * u2 + 4)
                      + c[i + 2] * (-3 * u3 + 3 * u2 + 3 * u + 1) + c[i + 3] * u3) / 6
            return z
        h = np.array(f)
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < last:
                h = h / (1 + np.exp(-h))
        return float(h[0])

    def _scale_one(self, h):
        la, lb = self.lambda_a, self.lambda_b
        if self.scale_table:
            return la + (lb - la) * _pchip_one(self._sh_l, self._su_l, self._sd_l, h)
        return la + (lb - la) * math.log1p(h / self.eps) / self._log1p_inv_eps

    def ood_one(self, cv):
        """Range check of one frame (cheap; the Mahalanobis test is in `ood`)."""
        return any(c < lo or c > hi for c, lo, hi in zip(cv, self._cv_min, self._cv_max))

    def one(self, cv):
        """lambda of one frame, as a Python float (the toytis adapter calls this). cv: a sequence of floats."""
        s = float(cv[self.state_cv])
        la, lb = self.lambda_a, self.lambda_b
        if s <= la or s >= lb:
            return s
        f = self._features_one(cv)
        m = _pchip_one(self._zk, self._qk, self._dk, self._raw_from_features(f))
        if self.ramp_kind == 'mult':
            w = self.ramp_w
            u = (s - la) / w
            a = 0.0 if u >= 1 else 1 - u * u * (3 - 2 * u)
            u = (lb - s) / w
            b = 0.0 if u >= 1 else 1 - u * u * (3 - 2 * u)
            lam = self._scale_one((1 - a) * (1 - b) * m + b)
        else:
            p = self.smin_k
            if self.front_s is None:
                lam = self._scale_one(m)
            else:
                lf = self._lam_front
                if s < self.front_s:
                    lam = lf - _smin_one(lf - self._scale_one(self.front_h * m), self.kappa_b * (self.front_s - s), p)
                else:
                    lam = lf + (lb - lf) * (s - self.front_s) / (lb - self.front_s)
            lam = la + _smin_one(lam - la, self.kappa_a * (s - la), p)
            if self.front_s is None:
                lam = lb - _smin_one(lb - lam, self.kappa_b * (lb - s), p)
        if self.has_blend:
            d = [fi - mu for fi, mu in zip(f, self._maha_mean_l)]
            d2 = sum(di * sum(cij * dj for cij, dj in zip(row, d)) for di, row in zip(d, self._maha_icov_l))
            t = _smoothstep_one((d2 - self.blend_thr) / self.blend_width)
            if t > 0:
                lam = (1 - t) * lam + t * _pchip_one(self._bs_l, self._bh_l, self._bd_l, s)
        return min(max(lam, self._in_lo), self._in_hi)
