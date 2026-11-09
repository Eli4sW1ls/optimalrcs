"""Helpers for 2D toytis potentials, shared by the toy-model notebooks
(`toy2d_md.ipynb`: MD on a potential; `toy2d_staple_analysis.ipynb`: committor of a staple run).

- `load_potential`, `read_full_settings`, `potential_from_settings`: the potential and the
  engine settings of a toytis run (from its `logging.log`).
- `Integrator`, `run_md`: many independent walkers at once, with exactly the toytis update
  (Brownian: x += dt/(m gamma) F + N(0, sqrt(2 kT dt/(m gamma)));
  Langevin: the BBK scheme of `ndLangevinEngine`, with its correlated position/velocity noise).
- `next_outcome`, `count_transitions`: the model-free committor of MD frames (which state the
  trajectory reaches next) and the A<->B transitions.
- `exact_committor_brownian`: the overdamped committor on a grid (finite volumes), with the
  boundary shift for absorption checked once per saved frame, and the TPT rate.
- `shoot`: committor of given phase points by running many short trajectories (both dynamics).
- `lag_update`: the stopped-window lag-Delta NPNEq update (notes section 10.1) for CommittorNE.

Coordinates follow toytis: position index 0 is the order parameter y, index 1 the
transverse coordinate x. A potential is used vectorized when it has
`potential_and_force_xy(y, x, xp)` (e.g. `curved_channel.CurvedChannelDoubleWell`), otherwise
point by point through `potential_and_force(ph)` (slow, numpy only).
"""
import ast
import importlib
import os
import re
import sys

import numpy as np

# toytis location: $TOYTIS, else the first of the known checkouts that exists
_TOYTIS_CANDIDATES = [os.environ.get('TOYTIS', ''),
                      '/home/staff7/fbsskr/Notebooks/Students/elias/toytis',
                      '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis']
TOYTIS = next((p for p in _TOYTIS_CANDIDATES if p and os.path.isdir(p)), _TOYTIS_CANDIDATES[-1])


# ----------------------------------------------------------------------------------------------
# potential and run settings
# ----------------------------------------------------------------------------------------------
def _toytis_on_path(toytis=TOYTIS):
    if toytis not in sys.path:
        sys.path.insert(0, toytis)


def load_potential(spec, kwargs=None, toytis=TOYTIS):
    """Potential from 'module:Class' (module looked up in toytis/potentials first), as the
    toytis launcher's --potential does."""
    _toytis_on_path(toytis)
    module_name, _, class_name = spec.partition(':')
    try:
        module = importlib.import_module('potentials.' + module_name)
    except ModuleNotFoundError:
        module = importlib.import_module(module_name)
    return getattr(module, class_name)(**(kwargs or {}))


def read_full_settings(tis_dir):
    """The settings dict the toytis code logs ('Full settings:' in logging.log), as a dict.
    The potential entry is kept as its text (`'potential'`), since it is an object."""
    with open(os.path.join(tis_dir, 'logging.log')) as f:
        text = f.read()
    m = re.search(r'Full settings:\s*\n(\{.*?\})\s*\n', text, re.S)
    if m is None:
        raise ValueError(f'no "Full settings" in {tis_dir}/logging.log')
    raw = m.group(1)
    pot = re.search(r"'potential':\s*(.*?)(?:,\s*'\w+':|\}$)", raw, re.S)
    pot_text = pot.group(1).strip() if pot else None
    clean = re.sub(r"'potential':\s*(<[^>]*>|\w+\([^)]*\))", "'potential': None", raw)
    settings = ast.literal_eval(clean)
    settings['potential'] = pot_text
    return settings


def potential_from_settings(settings, spec=None, kwargs=None, toytis=TOYTIS):
    """Rebuild the run's potential. From an explicit `spec` ('module:Class') and `kwargs`, or
    from the logged text: 'Class(k: v, ...)' (potentials with a __repr__, e.g. CurvedChannelDoubleWell)
    or '<potentials.module.Class object at ...>' (default constructor arguments: pass `kwargs`
    if the run used others)."""
    if spec is not None:
        return load_potential(spec, kwargs, toytis)
    text = settings.get('potential')
    if not text:
        return None
    _toytis_on_path(toytis)
    m = re.match(r'<([\w.]+)\.(\w+) object', text)
    if m:
        module = importlib.import_module(m.group(1))
        return getattr(module, m.group(2))(**(kwargs or {}))
    m = re.match(r'(\w+)\((.*)\)$', text, re.S)
    if m:
        cls_name = m.group(1)
        kw = {k: ast.literal_eval(v.strip()) for k, v in re.findall(r'(\w+)\s*[=:]\s*([^,]+)', m.group(2))}
        kw.update(kwargs or {})
        folder = os.path.join(toytis, 'potentials')
        for fname in sorted(os.listdir(folder)):      # find the module by its source: some do not import cleanly
            if fname.endswith('.py'):
                with open(os.path.join(folder, fname)) as f:
                    if re.search(rf'^class {cls_name}\b', f.read(), re.M):
                        return getattr(importlib.import_module('potentials.' + fname[:-3]), cls_name)(**kw)
    raise ValueError(f'cannot rebuild the potential from {text!r}; pass spec= and kwargs=')


def force_function(pot, xp=np):
    """(y, x) arrays -> (V, F_y, F_x)."""
    if hasattr(pot, 'potential_and_force_xy'):
        return lambda y, x: pot.potential_and_force_xy(y, x, xp)
    if xp is not np:
        raise ValueError('this potential has no vectorized potential_and_force_xy; use xp=numpy')

    def f(y, x):
        out = [pot.potential_and_force((np.array([a, b]), np.zeros(2))) for a, b in zip(np.ravel(y), np.ravel(x))]
        v = np.array([o[0] for o in out]).reshape(np.shape(y))
        fy = np.array([o[1][0] for o in out]).reshape(np.shape(y))
        fx = np.array([o[1][1] for o in out]).reshape(np.shape(y))
        return v, fy, fx
    return f


def potential_grid(pot, ys, xs):
    """V[iy, ix] on the grid ys x xs: the potential whose gradient is the engine's force. Potentials
    with a force term outside `potential` (the Z potential's gravity) provide `effective_potential`."""
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    if not hasattr(pot, 'potential_and_force_xy') and hasattr(pot, 'effective_potential'):
        return np.array([pot.effective_potential((np.array([a, b]), np.zeros(2)))
                         for a, b in zip(Y.ravel(), X.ravel())]).reshape(Y.shape)
    return force_function(pot)(Y, X)[0]


# ----------------------------------------------------------------------------------------------
# dynamics
# ----------------------------------------------------------------------------------------------
class Integrator:
    """The toytis update for many walkers at once (arrays of shape (n,) per coordinate)."""

    def __init__(self, pot, dt, kT, gamma, mass=1.0, high_friction=False, xp=np, seed=None):
        self.f = force_function(pot, xp)
        self.dt, self.kT, self.gamma, self.mass, self.high_friction, self.xp = dt, kT, gamma, mass, high_friction, xp
        self.rng = xp.random.RandomState(seed) if xp is not np else np.random.default_rng(seed)
        if high_friction:
            self.sigma = np.sqrt(2.0 * dt * kT / (mass * gamma))
            self.bddt = dt / (mass * gamma)
        else:
            gdt = gamma * dt
            c0 = np.exp(-gdt); c1 = (1 - c0) / gdt; c2 = (1 - c1) / gdt
            self.c0, self.a1, self.a2 = c0, c1 * dt, c2 * dt ** 2 / mass
            self.b1, self.b2 = (c1 - c2) * dt / mass, c2 * dt / mass
            s_r2 = (dt * kT / (gamma * mass)) * (2. - (3. - 4. * c0 + c0 ** 2) / gdt)
            s_v2 = (1. - c0 ** 2) * kT / mass
            cov = (kT / (mass * gamma)) * (1. - c0) ** 2
            self.l11 = np.sqrt(s_r2); self.l21 = cov / self.l11; self.l22 = np.sqrt(s_v2 - self.l21 ** 2)

    def normal(self, n):
        return self.rng.standard_normal(n) if self.xp is not np else self.rng.standard_normal(n)

    def draw_velocities(self, n):
        s = np.sqrt(self.kT / self.mass)
        return s * self.normal(n), s * self.normal(n)

    def step(self, y, x, vy, vx):
        n = y.shape[0]
        _, fy, fx = self.f(y, x)
        if self.high_friction:
            ry, rx = self.sigma * self.normal(n), self.sigma * self.normal(n)
            return y + self.bddt * fy + ry, x + self.bddt * fx + rx, ry, rx      # toytis stores the kick as v
        z1y, z2y, z1x, z2x = self.normal(n), self.normal(n), self.normal(n), self.normal(n)
        y_new = y + self.a1 * vy + self.a2 * fy + self.l11 * z1y
        x_new = x + self.a1 * vx + self.a2 * fx + self.l11 * z1x
        vy2 = self.c0 * vy + self.b1 * fy + self.l21 * z1y + self.l22 * z2y
        vx2 = self.c0 * vx + self.b1 * fx + self.l21 * z1x + self.l22 * z2x
        _, fy_n, fx_n = self.f(y_new, x_new)
        return y_new, x_new, vy2 + self.b2 * fy_n, vx2 + self.b2 * fx_n


def run_md(pot, n_walkers, n_steps, stride, dt, kT, gamma, mass=1.0, high_friction=False, burn_in=0,
           start=None, seed=0, xp=np, progress=True):
    """Independent long trajectories. start: (y, x) arrays or points to start from (default: half of
    the walkers in each well, from `pot.stationary_points()`). Saves every `stride` steps after
    `burn_in` steps. Returns dict of arrays (n_walkers, n_saved): y, x, vy, vx, and the settings."""
    eng = Integrator(pot, dt, kT, gamma, mass, high_friction, xp, seed)
    if start is None:
        p = pot.stationary_points()
        ya, xa = p['A'][:2]; yb, xb = p['B'][:2]
        half = np.arange(n_walkers) % 2 == 0
        start = (np.where(half, ya, yb), np.where(half, xa, xb))
    y = xp.asarray(np.asarray(start[0], float) * np.ones(n_walkers))
    x = xp.asarray(np.asarray(start[1], float) * np.ones(n_walkers))
    vy, vx = eng.draw_velocities(n_walkers)
    n_save = n_steps // stride
    out = {k: np.empty((n_walkers, n_save), np.float32) for k in ('y', 'x', 'vy', 'vx')}
    for _ in range(burn_in):
        y, x, vy, vx = eng.step(y, x, vy, vx)
    to_np = (lambda a: a.get()) if xp is not np else (lambda a: a)
    for i in range(n_steps):
        y, x, vy, vx = eng.step(y, x, vy, vx)
        if (i + 1) % stride == 0:
            j = (i + 1) // stride - 1
            out['y'][:, j], out['x'][:, j], out['vy'][:, j], out['vx'][:, j] = map(to_np, (y, x, vy, vx))
            if progress and (j + 1) % max(1, n_save // 10) == 0:
                print(f'  {100 * (j + 1) // n_save}% ({i + 1} steps per walker)', flush=True)
    out.update(dt=dt, kT=kT, gamma=gamma, mass=mass, high_friction=high_friction, stride=stride,
               burn_in=burn_in, seed=seed)
    return out


def next_outcome(y, lambda_a, lambda_b):
    """For every frame of every trajectory (rows of y): 1 if the next frame in A or B (at or after
    it) is in B, 0 if in A, NaN if the trajectory ends before reaching either."""
    y = np.atleast_2d(y)
    out = np.full(y.shape, np.nan, np.float32)
    for w in range(y.shape[0]):
        state = np.where(y[w] <= lambda_a, 0.0, np.where(y[w] >= lambda_b, 1.0, np.nan))
        # backward fill: the state of the next boundary frame
        idx = np.where(~np.isnan(state), np.arange(len(state)), len(state))
        nxt = np.minimum.accumulate(idx[::-1])[::-1]
        ok = nxt < len(state)
        out[w, ok] = state[nxt[ok]]
    return out


def count_transitions(y, lambda_a, lambda_b):
    """Number of A->B and B->A transitions (first entries into the other state) per trajectory."""
    y = np.atleast_2d(y)
    n_ab, n_ba = np.zeros(y.shape[0], int), np.zeros(y.shape[0], int)
    for w in range(y.shape[0]):
        s = np.where(y[w] <= lambda_a, 0, np.where(y[w] >= lambda_b, 1, -1))
        s = s[s >= 0]
        if s.size > 1:
            ch = np.diff(s)
            n_ab[w], n_ba[w] = (ch == 1).sum(), (ch == -1).sum()
    return n_ab, n_ba


# ----------------------------------------------------------------------------------------------
# exact committor (overdamped) and shooting
# ----------------------------------------------------------------------------------------------
def exact_committor_brownian(pot, kT, gamma, dt, lambda_a, lambda_b, stride=1, h=0.004,
                             y_range=None, x_range=None, mass=1.0):
    """Committor of overdamped dynamics: div(exp(-V/kT) grad q) = 0 by finite volumes on a grid,
    q = 0 for y <= lambda_A - s and 1 for y >= lambda_B + s, s = 0.5826 sigma sqrt(stride) the mean
    overshoot when absorption is checked every `stride` steps (sigma = sqrt(2 kT dt/(m gamma))).
    Also the TPT rate. Returns dict(y, x, q[iy, ix], V, k_ab, steps_per_ab, shift)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    if y_range is None or x_range is None:
        ys0, xs0, V0 = (pot.grid() if hasattr(pot, 'grid') else (None, None, None))
        y_range = y_range or (ys0[0], ys0[-1])
        x_range = x_range or (xs0[0], xs0[-1])
    ys = np.arange(y_range[0] + h / 2, y_range[1], h)
    xs = np.arange(x_range[0] + h / 2, x_range[1], h)
    V = potential_grid(pot, ys, xs)
    bv = (V - V.min()) / kT
    ny, nx = V.shape
    sigma = np.sqrt(2 * kT * dt / (mass * gamma))
    shift = 0.5826 * sigma * np.sqrt(stride)
    idx = np.arange(ny * nx).reshape(ny, nx)
    fixed0 = np.repeat((ys <= lambda_a - shift)[:, None], nx, 1)
    fixed1 = np.repeat((ys >= lambda_b + shift)[:, None], nx, 1)
    w_y = np.exp(-(bv[1:, :] + bv[:-1, :]) / 2)
    w_x = np.exp(-(bv[:, 1:] + bv[:, :-1]) / 2)
    rows = np.concatenate([idx[:-1].ravel(), idx[1:].ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel()])
    cols = np.concatenate([idx[1:].ravel(), idx[:-1].ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel()])
    vals = np.concatenate([w_y.ravel(), w_y.ravel(), w_x.ravel(), w_x.ravel()])
    L = sp.coo_matrix((vals, (rows, cols)), shape=(ny * nx, ny * nx)).tocsr()
    L = L - sp.diags(np.asarray(L.sum(axis=1)).ravel())
    qfix = fixed1.ravel().astype(float)
    free = ~(fixed0 | fixed1).ravel()
    F, C = np.flatnonzero(free), np.flatnonzero(~free)
    q = qfix.copy()
    q[F] = spla.spsolve(L[F][:, F].tocsc(), -L[F][:, C] @ qfix[C])
    Q = q.reshape(ny, nx)
    # TPT rate: k_AB = D sum_edges w (dq)^2 / h^2 / sum rho (1 - q), D = kT/(m gamma)
    flux = kT / (mass * gamma) * ((w_y * np.diff(Q, axis=0) ** 2).sum() + (w_x * np.diff(Q, axis=1) ** 2).sum()) / h ** 2
    rho = np.exp(-bv)
    k_ab = flux / (rho * (1 - Q)).sum()
    return dict(y=ys, x=xs, q=Q, V=V, k_ab=k_ab, steps_per_ab=1 / (k_ab * dt), shift=shift,
                lambda_a=lambda_a, lambda_b=lambda_b, stride=stride, h=h)


def interp_grid(ref, y, x, key='q'):
    """Value of a grid quantity (default the committor) at points (y, x), clipped to [0, 1] for q."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((ref['y'], ref['x']), ref[key], bounds_error=False, fill_value=None)
    v = f(np.c_[np.ravel(y), np.ravel(x)]).reshape(np.shape(y))
    return np.clip(v, 0, 1) if key == 'q' else v


def shoot(pot, y0, x0, vy0, vx0, n_shots, dt, kT, gamma, lambda_a, lambda_b, mass=1.0, high_friction=False,
          xp=np, seed=0, max_steps=2_000_000, check_every=1, chunk=2_000_000, progress=True):
    """Committor of the phase points (y0, x0, vy0, vx0) by shooting: `n_shots` trajectories from
    each, with the toytis update, until y <= lambda_A or y >= lambda_B (checked every `check_every`
    steps). For Langevin the velocities are the given ones (the trajectory continues the frame's
    motion); for Brownian they do not matter. Returns (q, standard error) per point."""
    y0, x0 = np.ravel(y0).astype(float), np.ravel(x0).astype(float)
    vy0 = np.zeros_like(y0) if vy0 is None else np.ravel(vy0).astype(float)
    vx0 = np.zeros_like(x0) if vx0 is None else np.ravel(vx0).astype(float)
    n_pts = len(y0)
    eng = Integrator(pot, dt, kT, gamma, mass, high_friction, xp, seed)
    result = np.full(n_pts * n_shots, -1.0)
    point = np.repeat(np.arange(n_pts), n_shots)
    for c0 in range(0, n_pts * n_shots, chunk):
        sel = np.arange(c0, min(c0 + chunk, n_pts * n_shots))
        p = point[sel]
        y, x = xp.asarray(y0[p]), xp.asarray(x0[p])
        vy, vx = xp.asarray(vy0[p]), xp.asarray(vx0[p])
        alive = xp.arange(sel.size)
        out = xp.full(sel.size, -1.0)
        for step in range(max_steps):
            y[alive], x[alive], vy[alive], vx[alive] = eng.step(y[alive], x[alive], vy[alive], vx[alive])
            if (step + 1) % check_every == 0:
                ya = y[alive]
                out[alive[ya <= lambda_a]] = 0.0
                out[alive[ya >= lambda_b]] = 1.0
                alive = alive[out[alive] < 0]
                if alive.size == 0:
                    break
        result[sel] = out.get() if xp is not np else out
        if progress:
            print(f'  shot {sel[-1] + 1} of {n_pts * n_shots} trajectories ({step + 1} steps)', flush=True)
    r = result.reshape(n_pts, n_shots)
    done = r >= 0
    q = np.where(done.any(1), (r * done).sum(1) / np.maximum(done.sum(1), 1), np.nan)
    se = np.sqrt(q * (1 - q) / np.maximum(done.sum(1), 1))
    return q, se


# ----------------------------------------------------------------------------------------------
# lag-Delta NPNEq update
# ----------------------------------------------------------------------------------------------
class LazyRY:
    """basis_poly_ry evaluated on chunks of frames, normalized over all frames as basis_poly_ry does."""

    def __init__(self, orc, r, y, n, fenv):
        import cupy as cp
        self.orc = orc
        self.r, self.y, self.n = r / cp.max(cp.abs(r)), y / cp.max(cp.abs(y)), n
        self.fenv = cp.ones_like(r) if fenv is None else fenv

    def cols(self, idx):
        return self.orc._poly_ry(self.r[idx], self.y[idx], self.n, self.fenv[idx])


def lag_update(orc, i_traj, boundary, weights, lag):
    """Install the update of the fit: lag 1 = the library's NPNEq step; lag > 1 = the stopped-window
    update sum_t w_t f_k(t) [q(min(t + lag, tau_t)) - q(t)] = 0, tau_t the first A/B frame or the
    piece's last frame (notes section 10.1). Returns the extra fit_transform arguments (the basis)."""
    import cupy as cp
    nonpar = orc.nonparametrics
    if not hasattr(nonpar, 'npneq_lag1'):
        nonpar.npneq_lag1 = nonpar.npneq                 # keep the library's update
    nonpar.npneq = nonpar.npneq_lag1
    if lag == 1:
        return dict(lazy_basis=True)
    import metrics_tis
    b = np.asarray(boundary, bool)
    n = len(b)
    tau = cp.asnumpy(metrics_tis.stop_index(cp.asarray(i_traj), cp.asarray(b)))
    u_np = np.minimum(np.arange(n) + lag, tau)
    U = cp.asarray(u_np.astype(np.int64))
    starts = cp.asarray(np.flatnonzero(~b & (u_np > np.arange(n))).astype(np.int64))
    W = cp.asarray(np.ones(n) if weights is None else weights, dtype=cp.float64)
    del u_np, tau

    def npneq_lag(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk=None, w_traj=None, **kw):
        akj, bb = 0, 0
        for s0 in range(0, starts.size, 400_000):
            st = starts[s0:s0 + 400_000]
            fa, fb, wt = fk.cols(st), fk.cols(U[st]), W[st]
            akj = akj + fa @ (fb * wt - fa * (wt + gamma)).T
            bb = bb + fa @ (-(r_traj[U[st]] - r_traj[st]) * wt)
        al = cp.linalg.lstsq(akj, bb, rcond=None)[0]
        new = r_traj.copy()
        for s0 in range(0, n, 1_000_000):
            idx = cp.arange(s0, min(s0 + 1_000_000, n))
            new[idx] = r_traj[idx] + al @ fk.cols(idx)
        return cp.clip(new, 0, 1)

    nonpar.npneq = npneq_lag
    return dict(basis_functions=lambda r_, y_, n_, f_: LazyRY(orc, r_, y_, n_, f_))


def z_per_lag(r, boundary, i_traj, weights, lags, group=None):
    """Worst |stopped Z_q / standard error| over q at each lag (path data)."""
    import cupy as cp
    import metrics_tis
    b = cp.asarray(np.asarray(boundary, bool))
    it = cp.asarray(i_traj)
    tau = metrics_tis.stop_index(it, b)
    out = []
    for lag in lags:
        _, z = metrics_tis.comp_zq_stopped_zscore(cp.asarray(r), b, it, None if weights is None else cp.asarray(weights),
                                                  lag, tau=tau, g_traj=None if group is None else cp.asarray(group))
        out.append(float(cp.max(cp.abs(z[5:]))))
    return out
