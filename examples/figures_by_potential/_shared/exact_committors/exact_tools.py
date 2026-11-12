"""Exact committors of the discrete-time toytis dynamics, and high-statistics shooting to check them.

"Exact" here means the committor of the Markov chain the engine actually runs (its own one-step
update with its own random numbers), with A and B checked on every saved frame, as in the data:

    q(s) = P(the chain started in s reaches B before A) = E[ Q(s') ],   Q = 0 in A, 1 in B, q elsewhere,

with s' the state one saved frame later. This is solved on a grid (no continuous-time limit, no
boundary-shift approximation), and checked independently by GPU shooting with millions of
trajectories per point (binomial error bars).

Engines (toytis engine.py):
  * Brownian (high_friction): x' = x + dt/(m gamma) F(x) + N(0, 2 kT dt/(m gamma)) per coordinate.
  * BBK Langevin: x' = x + a1 v + a2 F(x) + xi_x;  v' = c0 v + b1 F(x) + xi_v + b2 F(x'),
    (xi_x, xi_v) correlated Gaussian (coefficients as in LangevinEngine).
  * Euler-Maruyama (the user's script): v' = v + (-gamma v + F/m) dt + sqrt(2 gamma kT dt / m) xi;  x' = x + v' dt.
"""
import numpy as np

try:
    import cupy as cp
except ImportError:                       # the grid solvers below need CuPy; the 1D Brownian one does not
    cp = None


# ------------------------------------------------------------------------------------------------
# potentials (vectorized; xp = numpy or cupy). Checked against the toytis classes in check_forces().
# ------------------------------------------------------------------------------------------------
COSWELL = dict(wall_dx=0.1, k=100., bleft=-0.25, bright=0.25, deltaf=1.2, height=0.15, width=0.1)


def coswell_force(x, xp=np):
    p = COSWELL
    L = p['bright'] - p['bleft']
    xl, xr = p['bleft'] - p['wall_dx'] - p['width'], p['bright'] + p['wall_dx'] + p['width']
    w, h = p['width'], p['height']
    f = xp.where(x < xl, -p['k'] * (x - xl),
        xp.where(x > xr, -p['k'] * (x - xr),
        xp.where((x < p['bleft'] - w) | (x > p['bright'] + w), 0.0,
        xp.where(x < p['bleft'], -h * np.pi / (2 * w) * xp.sin(np.pi * (x - p['bleft'] + w) / w),
        xp.where(x > p['bright'], -h * np.pi / (2 * w) * xp.sin(np.pi * (x - p['bright'] - w) / w),
                 np.pi * p['deltaf'] / L * xp.sin(2 * np.pi * (x - p['bleft']) / L))))))
    return f


def coswell_potential(x, xp=np):
    p = COSWELL
    L = p['bright'] - p['bleft']
    xl, xr = p['bleft'] - p['wall_dx'] - p['width'], p['bright'] + p['wall_dx'] + p['width']
    w, h = p['width'], p['height']
    return xp.where(x < xl, p['k'] * (x - xl) ** 2 / 2,
           xp.where(x > xr, p['k'] * (x - xr) ** 2 / 2,
           xp.where((x < p['bleft'] - w) | (x > p['bright'] + w), 0.0,
           xp.where(x < p['bleft'], h / 2 * (1 - xp.cos(np.pi * (x - p['bleft'] + w) / w)),
           xp.where(x > p['bright'], h / 2 * (1 - xp.cos(np.pi * (x - p['bright'] - w) / w)),
                    h + p['deltaf'] / 2 * (-1 + xp.cos(2 * np.pi * (x - p['bleft']) / L)))))))


def two_bumps_potential(x, xp=np):
    """Long-MD 'two bumps' potential (V = 0 outside [-0.1, 0.3]; walls beyond play no role between A and B)."""
    m = (x >= -0.1) & (x <= 0.3)
    return xp.where(m, 0.5 * (1 - xp.cos(2 * np.pi * (x + 0.1) / 0.2)), 0.0)


def two_bumps_force(x, xp=np):
    m = (x >= -0.1) & (x <= 0.3)
    return xp.where(m, -0.5 * 2 * np.pi / 0.2 * xp.sin(2 * np.pi * (x + 0.1) / 0.2), 0.0)


def flat_force(x, xp=np):
    return 0.0 * x


def zpot_force(y, x, xp=np, Lx=0.3, Ly=0.9, k=2000., A=0.3, L=0.2):
    """Force (fy, fx) of sjoelbak_angle.RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45), including
    the constant gravity -0.2 along y. y is the order parameter (first phase-point coordinate)."""
    fx = xp.where(x < 0, -k * x, xp.where(x > Lx, -k * (x - Lx), 0.0))
    fy = xp.where(y < 0, -k * y, xp.where(y > Ly, -k * (y - Ly), 0.0))
    a, b = -1.0, 0.45 - (-1.0) * 0.15                       # barrier line y = a x + b through (x, y) = (0.15, 0.45)
    norm = np.sqrt(a ** 2 + 1)
    D = xp.abs(a * x - y + b) / norm
    sgn = xp.sign(y - a * x - b)
    on = D < L / 4
    dVdD = -2 * np.pi * A / L * xp.sin(2 * np.pi * D / L)
    fx = fx + xp.where(on, sgn * dVdD * (a / norm), 0.0)
    fy = fy + xp.where(on, sgn * dVdD * (-1 / norm), 0.0)
    for yc in (0.025, 0.875):                              # state wells
        Dw = xp.abs(y - yc)
        onw = Dw < 0.2 / 4
        fy = fy + xp.where(onw, -0.15 * 2 * np.pi / 0.2 * xp.sin(2 * np.pi * Dw / 0.2) * xp.sign(y - yc), 0.0)
    fy = fy - 0.2                                          # gravity (force only)
    fy = fy + xp.where((y >= 0.1) & (y <= 0.825), -0.2 / (0.825 - 0.1), 0.0)   # slope
    return fy, fx


# ------------------------------------------------------------------------------------------------
# engines: one step, vectorized
# ------------------------------------------------------------------------------------------------
def bbk_coefficients(dt, gamma, kT=1.0, m=1.0):
    gdt = gamma * dt
    c0 = np.exp(-gdt); c1 = (1 - c0) / gdt; c2 = (1 - c1) / gdt
    beta = 1.0 / kT
    s_rr = (dt / (beta * gamma * m)) * (2. - (3. - 4. * c0 + c0 ** 2) / gdt)
    s_vv = (1.0 - c0 ** 2) / (beta * m)
    s_rv = (1. / (m * beta * gamma)) * (1.0 - c0) ** 2
    Lc = np.linalg.cholesky(np.array([[s_rr, s_rv], [s_rv, s_vv]]))
    return dict(c0=c0, a1=c1 * dt, a2=c2 * dt ** 2 / m, b1=(c1 - c2) * dt / m, b2=c2 * dt / m,
                L11=Lc[0, 0], L21=Lc[1, 0], L22=Lc[1, 1])


def bbk_step(x, v, z1, z2, force, c, xp):
    """toytis LangevinEngine.step with standard normals z1, z2 (xi = Cholesky(cov) @ z)."""
    f = force(x, xp)
    xn = x + c['a1'] * v + c['a2'] * f + c['L11'] * z1
    vn = c['c0'] * v + c['b1'] * f + c['L21'] * z1 + c['L22'] * z2 + c['b2'] * force(xn, xp)
    return xn, vn


def em_step(x, v, z1, z2, force, c, xp):
    """Euler-Maruyama as in the user's script (z2 unused)."""
    vn = v + (-c['gamma'] * v + force(x, xp) / c['m']) * c['dt'] + c['s'] * z1
    return x + vn * c['dt'], vn


def em_coefficients(dt, gamma, kT=1.0, m=1.0):
    return dict(dt=dt, gamma=gamma, m=m, s=np.sqrt(2 * gamma * kT * dt / m))


def gauss_hermite(n):
    """Nodes and weights for E[f(Z)], Z ~ N(0, 1)."""
    z, w = np.polynomial.hermite.hermgauss(n)
    return z * np.sqrt(2), w / np.sqrt(np.pi)


# ------------------------------------------------------------------------------------------------
# grid solvers: q(s) = E[Q(s')] with the jump of Q at the absorbing boundary integrated exactly
# ------------------------------------------------------------------------------------------------
ZMAX = 8.0                                                  # standard-normal range kept (P(|z| > 8) ~ 1e-15)


def _split_nodes(zlo, zhi, n, xp, force_split=False):
    """Quadrature in z ~ N(0, 1) restricted to [zlo, zhi] (per point). Where the interval covers [-ZMAX, ZMAX]
    (no boundary within reach): Gauss-Hermite. Otherwise (interval cut by an absorbing boundary): Gauss-Legendre
    in the probability variable u = Phi(z) on [Phi(zlo), Phi(zhi)], z = Phi^-1(u); the weights then sum to the
    exact inside probability and the integrand is smooth. Returns n node arrays and n weight arrays (the weights
    include the Gaussian density)."""
    if xp is np:
        from scipy.special import ndtr, ndtri
    else:
        from cupyx.scipy.special import ndtr, ndtri
    gz, gw = gauss_hermite(n)
    lz, lw = np.polynomial.legendre.leggauss(n)
    full = (zlo <= -ZMAX) & (zhi >= ZMAX)
    if force_split:
        full = xp.zeros_like(full)
    # in u, symmetric around 1/2 for accuracy: use the upper tail when the interval lies in it
    upper = (zlo + zhi) > 0
    zl, zh = xp.where(upper, -zhi, zlo), xp.where(upper, -zlo, zhi)      # mirrored to the lower side
    ul, uh = ndtr(xp.clip(zl, -38, 38)), ndtr(xp.clip(zh, -38, 38))
    mid, half = (uh + ul) / 2, xp.maximum(uh - ul, 0) / 2
    nodes, weights = [], []
    for k in range(n):
        u = xp.clip(mid + half * lz[k], 1e-300, 1 - 1e-16)
        z = ndtri(u)
        z = xp.where(upper, -z, z)
        nodes.append(xp.where(full, gz[k], z))
        weights.append(xp.where(full, gw[k], half * lw[k]))
    return nodes, weights


def _norm_sf(z, xp):
    """P(Z > z) for Z ~ N(0, 1)."""
    from cupyx.scipy.special import erfc
    return 0.5 * erfc(z / np.sqrt(2))


def exit_probabilities(step, coef, force, A, B, x, v, xp):
    """Exact probabilities that the next step from (x, v) ends in A and in B (x' = x_det + c_x z1 is linear in z1)."""
    zero = 0.0 * x
    x_det = step(x, v, zero, zero, force, coef, xp)[0]
    c_x = step(x, v, zero + 1, zero, force, coef, xp)[0] - x_det
    if xp is np:
        from scipy.special import erfc
    else:
        from cupyx.scipy.special import erfc
    sf = lambda z: 0.5 * erfc(z / np.sqrt(2))
    return sf((x_det - A) / c_x), sf((B - x_det) / c_x), x_det, c_x


def _xmap(A, B, c):
    """Stretched x grid: x = A + (B - A) g(xi), g(xi) = xi - c/(2 pi) sin(2 pi xi), xi uniform in [0, 1].
    Spacing at the boundaries is (1 - c) times, in the middle (1 + c) times the uniform spacing."""
    def x_of(xi):
        return A + (B - A) * (xi - c / (2 * np.pi) * np.sin(2 * np.pi * xi))

    def xi_of(x, xp):
        t = xp.clip((x - A) / (B - A), 0.0, 1.0)
        xi = t.copy()
        for _ in range(30):                                   # Newton; g' >= 1 - c > 0
            xi = xi - (xi - c / (2 * np.pi) * xp.sin(2 * np.pi * xi) - t) / (1 - c * xp.cos(2 * np.pi * xi))
        return xp.clip(xi, 0.0, 1.0)
    return x_of, xi_of


def solve_underdamped_1d(step, coef, force, A, B, nx, nv, vmax=5.0, n1=12, n2=7, c_map=0.8, tol=1e-12,
                         maxiter=4000, verbose=True):
    """Committor q(x, v) of the 1D Langevin chain (BBK or Euler-Maruyama), A: x <= A, B: x >= B, every step.

    Next to A and B the committor has real fine structure: whether a state leaves in 1, 2, 3, ... steps
    depends on x + k v dt passing the boundary, giving thin sloped layers (widths ~1e-4 to 1e-3 within
    ~0.02 of the boundaries). The first one is taken out analytically,
        q(s) = pB(s) + (1 - pA(s) - pB(s)) m(s),
    pA, pB = exact probabilities to land in A, B at the next step (error functions), m(s) = E[q(s') | s' inside];
    the later ones are resolved by a grid that is dense near A and B (stretching c_map, see _xmap).
    m is solved on the grid (cubic splines in grid-index space) with q(s') replaced by the same decomposition:
        m(s) = sum_k w_k [pB(s'_k) + (1 - pA(s'_k) - pB(s'_k)) m(s'_k)] / sum_k w_k.
    The z1-integral over the inside part uses nodes on [zA, zB] (Gauss-Legendre where a boundary cuts it),
    z2 (BBK only) Gauss-Hermite. Returns a dict for eval_q_underdamped and the residual."""
    from cupyx.scipy import ndimage
    from cupyx.scipy.sparse.linalg import gmres, LinearOperator
    x_of, xi_of = _xmap(A, B, c_map)
    xs, vs = x_of(np.linspace(0, 1, nx)), np.linspace(-vmax, vmax, nv)
    hv = vs[1] - vs[0]
    X, V = cp.meshgrid(cp.asarray(xs), cp.asarray(vs), indexing='ij')
    X, V = X.ravel(), V.ravel()
    M = X.size
    zero = cp.zeros(M)
    pA, pB, x_det, c_x = exit_probabilities(step, coef, force, A, B, X, V, cp)
    z1n, w1n = _split_nodes((A - x_det) / c_x, (B - x_det) / c_x, n1, cp)
    z2n, w2n = ([0.0], [1.0]) if step is em_step else gauss_hermite(n2)
    # conditional (inside) weights, normalized by the quadrature itself (no cancellation in 1 - pA - pB);
    # a state that surely leaves gets uniform weights on nodes at the boundary edge: a smooth continuation of m
    S1 = sum(w1n)
    sure_exit = S1 <= 0
    coords, W = [], []
    cvec = cp.zeros(M)
    for z1, w1 in zip(z1n, w1n):
        w1c = cp.where(sure_exit, 1.0 / n1, w1 / cp.where(sure_exit, 1.0, S1))
        for z2, w2 in zip(z2n, w2n):
            xn, vn = step(X, V, z1, zero + z2, force, coef, cp)
            xn = cp.clip(xn, A, B)
            qA, qB = exit_probabilities(step, coef, force, A, B, xn, vn, cp)[:2]
            wk = w1c * w2
            cvec += wk * qB
            coords.append(cp.stack([xi_of(xn, cp) * (nx - 1), (vn + vmax) / hv]).astype(cp.float32))
            W.append(wk * (1 - qA - qB))
    coords = cp.concatenate(coords, axis=1)
    W = cp.concatenate(W)
    n_nodes = W.size // M

    block = max(1, int(4e7 // M))                             # quadrature nodes per block (GPU memory)

    def K(m):
        cf = ndimage.spline_filter(m.reshape(nx, nv), order=3, mode='mirror')
        out = cp.zeros(M)
        for k0 in range(0, n_nodes, block):
            sl = slice(k0 * M, min(k0 + block, n_nodes) * M)
            vals = ndimage.map_coordinates(cf, coords[:, sl], order=3, mode='mirror', prefilter=False)
            out += (W[sl] * vals).reshape(-1, M).sum(0)
        return out

    Aop = LinearOperator((M, M), matvec=lambda m: m - K(m), dtype=cp.float64)
    m0 = cp.clip((X - A) / (B - A) + 0.3 * V, 0, 1)
    restart = int(min(200, max(30, 1.2e9 / (8 * M))))           # Krylov basis M x restart doubles (GPU memory)
    m, info = gmres(Aop, cvec, x0=m0, tol=tol, restart=restart, maxiter=maxiter)
    res = float(cp.linalg.norm(m - K(m) - cvec) / cp.linalg.norm(cvec))
    if verbose:
        print(f'  grid {nx} x {nv} (x spacing {xs[1] - xs[0]:.1e} at A/B, {np.diff(xs).max():.1e} in the middle), '
              f'{n_nodes} quadrature nodes: gmres info {info}, relative residual {res:.2e}', flush=True)
    sol = dict(xs=xs, vs=vs, m=cp.asnumpy(m.reshape(nx, nv)), A=A, B=B, c_map=c_map)
    del coords, W
    cp.get_default_memory_pool().free_all_blocks()
    return sol, res


def eval_q_underdamped(step, coef, force, sol, x, v):
    """q(x, v) = pB + (1 - pA - pB) m(x, v) from a solved grid (numpy, CPU). x must lie in (A, B)."""
    from scipy import ndimage as nd
    x, v = np.asarray(x, dtype=float), np.asarray(v, dtype=float)
    A, B = sol['A'], sol['B']
    pA, pB = exit_probabilities(step, coef, force, A, B, x, v, np)[:2]
    xi_of = _xmap(A, B, sol['c_map'])[1]
    nx = len(sol['xs']); vs = sol['vs']
    cf = nd.spline_filter(sol['m'], order=3, mode='mirror')
    iv = (np.clip(v, vs[0], vs[-1]) - vs[0]) / (vs[1] - vs[0])
    mval = nd.map_coordinates(cf, np.stack([xi_of(x, np) * (nx - 1), iv]), order=3, mode='mirror', prefilter=False)
    return pB + (1 - pA - pB) * mval


ZPIECE = 6.0                                                # nodes placed within |z| <= 6; tail mass added to the end pieces
BASE_CUTS = (-3.0, -1.5, 0.0, 1.5, 3.0)


def _pieces(zlo, zhi, zbreaks, n, xp, n_per_piece=5):
    """Quadrature in z ~ N(0, 1) over [zlo, zhi] (per point; -inf/inf allowed), split at the integrand's jumps
    (zbreaks: per-point arrays; the two nearest inside the interval are used) and at fixed cuts BASE_CUTS, so every
    piece is short; Gauss-Legendre in z with the Gaussian density on each piece, rescaled to the piece's exact
    probability. Mass and moments are then both accurate (Gauss-Legendre in u = Phi(z) gets the mass exactly but
    the variance a few per cent low, which biases the diffusion every step). `n` is ignored (kept for the call
    signature). Returns (len(BASE_CUTS) + 3) * n_per_piece node arrays and weights."""
    if xp is np:
        from scipy.special import ndtr
    else:
        from cupyx.scipy.special import ndtr
    M = zlo.shape[0]
    lo_t, hi_t = xp.maximum(zlo, -38.0), xp.minimum(zhi, 38.0)        # true limits (for the masses)
    inner = []
    if zbreaks:
        zb = xp.stack([xp.where((b > lo_t) & (b < hi_t), b, xp.inf) for b in zbreaks])
        zb = xp.sort(zb, axis=0)
        inner += [zb[0], zb[1] if zb.shape[0] > 1 else xp.full(M, xp.inf)]
    else:
        inner += [xp.full(M, xp.inf)] * 2
    inner += [xp.where((c > lo_t) & (c < hi_t), c, xp.inf) * xp.ones(M) for c in BASE_CUTS]
    # points with no jump and no absorbing boundary within reach: plain Gauss-Hermite (exact moments)
    no_cut = (lo_t <= -ZPIECE) & (hi_t >= ZPIECE)
    for b in inner[:2]:
        no_cut &= ~((b > -ZPIECE) & (b < ZPIECE))
    cuts = xp.sort(xp.stack(inner), axis=0)                           # inf = unused
    cuts = xp.minimum(cuts, hi_t)
    edges = [lo_t] + [cuts[k] for k in range(cuts.shape[0])] + [hi_t]
    lz, lw = np.polynomial.legendre.leggauss(n_per_piece)
    n_total = (len(edges) - 1) * n_per_piece
    ghz, ghw = gauss_hermite(n_total)
    nodes, weights = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        b = xp.maximum(b, a)
        mass = xp.clip(ndtr(b) - ndtr(a), 0, 1)                        # exact probability of the piece
        mass = xp.where((a > 0), xp.clip(ndtr(-a) - ndtr(-b), 0, 1), mass)   # upper tail: no cancellation
        ap, bp = xp.clip(a, -ZPIECE, ZPIECE), xp.clip(b, -ZPIECE, ZPIECE)
        mid, half = (ap + bp) / 2, (bp - ap) / 2
        zs = [mid + half * lz[k] for k in range(n_per_piece)]
        ws = [half * lw[k] * xp.exp(-zs[k] ** 2 / 2) for k in range(n_per_piece)]
        S = sum(ws)
        scale = xp.where(S > 0, mass / xp.where(S > 0, S, 1.0), 0.0)
        # a piece entirely in the far tail (|z| > 6): its mass sits on one node at the clipped end
        tail_only = (S <= 0) & (mass > 0)
        for k in range(n_per_piece):
            wk = ws[k] * scale
            if k == 0:
                wk = xp.where(tail_only, mass, wk)
            nodes.append(zs[k]); weights.append(wk)
    nodes = [xp.where(no_cut, ghz[j], z) for j, z in enumerate(nodes)]
    weights = [xp.where(no_cut, ghw[j], w) for j, w in enumerate(weights)]
    return nodes, weights


def solve_brownian_2d(force2d, dt, gamma, kT, A, B, h, x_range=(-0.03, 0.33), y_box=(-0.03, 0.93), stride=1,
                      n1=8, n2=5, y_jumps=(), ridge_jumps=None, tol=1e-11, maxiter=4000, m=1.0, verbose=True):
    """Committor of the 2D Brownian chain s' = s + mu F(s) + sigma z (mu = dt/(m gamma), sigma^2 = 2 kT mu, z
    standard normal per coordinate), A: y <= A, B: y >= B, checked every `stride` steps; y = first coordinate.

    Where F jumps (the Z potential's ridge and well edges) the mean of the next step jumps by mu dF, and so does
    the exact committor of the chain. So it is not solved for directly. With the smoothed field
        w(r) = E_z[ Q(r + sigma z) ]          (Gaussian convolution of Q: smooth),
    one step from s gives q(s) = w(s + mu F(s)) exactly (force at s). The unknown is w on a grid (cubic splines);
    Q(s') = 1 in B, 0 in A, q(s') = w(s' + mu F(s')) inside. The noise integral over s' is split exactly where its
    integrand jumps: at A and B (absorption; absorbed probability by erfc) and at the force discontinuities, given as
    y_jumps (lines y = const) and ridge_jumps = (normal (ny, nx), offsets c): lines n . s = c. Points within reach
    (8 sigma) of a ridge line integrate along its normal, all others along y; each piece by Gauss-Legendre in the
    probability variable, the other noise direction by Gauss-Hermite.
        stride 1:  w(r) = E_z[ 1_B(s') + 1_inside(s') w(s' + mu F(s')) ],  s' = r + sigma z;
        stride k:  w_k(r) = E_z[Q(r + sigma z)] (Q from w_1), then k - 1 free steps backwards,
                   w_j(r) = E_z[ w_{j+1}(s' + mu F(s')) ], on the whole box; solve w_1 = (these steps)(w_1).
    Returns a dict for eval_q_brownian_2d and the residual."""
    from cupyx.scipy import ndimage
    from cupyx.scipy.sparse.linalg import gmres, LinearOperator
    sg = np.sqrt(2 * kT * dt / (m * gamma)); mu = dt / (m * gamma)
    xs = np.arange(x_range[0], x_range[1] + h / 2, h)
    ys = np.arange(A - 0.01, B + 0.01 + h / 2, h) if stride == 1 else np.arange(y_box[0], y_box[1] + h / 2, h)
    shape = (len(ys), len(xs))
    Y, X = cp.meshgrid(cp.asarray(ys), cp.asarray(xs), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    M = Y.size
    idx = lambda yy, xx: cp.stack([(yy - ys[0]) / h, (xx - xs[0]) / h]).astype(cp.float32)

    def mapped(yy, xx):
        fy, fx = force2d(yy, xx, cp)
        return yy + mu * fy, xx + mu * fx

    if ridge_jumps is not None:
        (nY, nX), offs = ridge_jumps
        nn = np.hypot(nY, nX); nY, nX = nY / nn, nX / nn
        cR = nY * Y + nX * X                                      # coordinate along the ridge normal
        ridge = cp.zeros(M, dtype=bool)
        for c0 in offs:
            ridge |= cp.abs(cR - c0 / nn) < ZMAX * sg
    else:
        ridge = cp.zeros(M, dtype=bool)
    gz, gw = gauss_hermite(n2)

    def step_nodes(absorbing):
        """Nodes s' = r + sigma z and weights of one step from every grid point r."""
        # along y: absorption at A, B (if absorbing) and the y jumps
        zlo = (A - Y) / sg if absorbing else cp.full(M, -np.inf)
        zhi = (B - Y) / sg if absorbing else cp.full(M, np.inf)
        z1y, w1y = _pieces(zlo, zhi, [(yj - Y) / sg for yj in y_jumps], n1, cp)
        if ridge_jumps is not None:                               # along the ridge normal, no absorption there
            z1r, w1r = _pieces(cp.full(M, -np.inf), cp.full(M, np.inf), [(c0 / nn - cR) / sg for c0 in offs], n1, cp)
        coords, W = [], []
        for k in range(len(z1y)):
            for z2, w2 in zip(gz, gw):
                yp, xp_ = Y + sg * z1y[k], X + sg * z2
                wk = w1y[k] * w2
                if ridge_jumps is not None:                       # normal (nY, nX), tangent (-nX, nY)
                    ypr = Y + sg * (z1r[k] * nY - z2 * nX)
                    xpr = X + sg * (z1r[k] * nX + z2 * nY)
                    yp, xp_ = cp.where(ridge, ypr, yp), cp.where(ridge, xpr, xp_)
                    wk = cp.where(ridge, w1r[k] * w2, wk)
                if absorbing:
                    yp = cp.clip(yp, A, B)
                coords.append(idx(*mapped(yp, xp_))); W.append(wk)
        cB = _norm_sf((B - Y) / sg, cp) if absorbing else None
        if absorbing and ridge_jumps is not None:
            cB = cp.where(ridge, 0.0, cB)                         # ridge points are far from A and B
        return cp.concatenate(coords, axis=1), cp.concatenate(W), cB

    cL, WL, cB = step_nodes(True)
    if stride > 1:
        cF, WF, _ = step_nodes(False)

    def apply(coords, W, field):
        cf = ndimage.spline_filter(field.reshape(shape), order=3, mode='mirror')
        n_nodes = W.size // M
        block = max(1, int(4e7 // M))
        out = cp.zeros(M)
        for k0 in range(0, n_nodes, block):
            sl = slice(k0 * M, min(k0 + block, n_nodes) * M)
            vals = ndimage.map_coordinates(cf, coords[:, sl], order=3, mode='mirror', prefilter=False)
            out += (W[sl] * vals).reshape(-1, M).sum(0)
        return out

    def T(w, with_c):
        u = apply(cL, WL, w) + (cB if with_c else 0.0)
        for _ in range(stride - 1):
            u = apply(cF, WF, u)
        return u

    c = T(cp.zeros(M), True)
    restart = int(min(150, max(30, 1.2e9 / (8 * M))))
    Aop = LinearOperator((M, M), matvec=lambda w: w - T(w, False), dtype=cp.float64)
    w0 = cp.clip((Y - A) / (B - A), 0, 1)
    w, info = gmres(Aop, c, x0=w0, tol=tol, restart=restart, maxiter=maxiter)
    res = float(cp.linalg.norm(w - T(w, False) - c) / cp.linalg.norm(c))
    if verbose:
        print(f'  2D grid {shape[0]} x {shape[1]} (h {h}), stride {stride}: gmres info {info}, residual {res:.2e}', flush=True)
    sol = dict(ys=ys, xs=xs, w=cp.asnumpy(w.reshape(shape)), A=A, B=B, mu=mu, stride=stride, h=h)
    cp.get_default_memory_pool().free_all_blocks()
    return sol, res


# discontinuities of the Z potential's force (sjoelbak_angle, Lx 0.3, 45 deg): state-well edges at
# y = 0.025 +- 0.05 and 0.875 +- 0.05, slope ends at y = 0.1 and 0.825, ridge edges x + y = 0.6 +- sqrt(2) L/4
ZPOT_Y_JUMPS = (-0.025, 0.075, 0.1, 0.825, 0.925)
ZPOT_RIDGE_JUMPS = ((1.0, 1.0), (0.6 - np.sqrt(2) * 0.05, 0.6 + np.sqrt(2) * 0.05))


def eval_q_brownian_2d(force2d, sol, y, x):
    """q(y, x) = w(s + mu F(s)) from a solved 2D Brownian grid (numpy, CPU); (y, x) between A and B."""
    from scipy import ndimage as nd
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    fy, fx = force2d(y, x, np)
    ry, rx = y + sol['mu'] * fy, x + sol['mu'] * fx
    ys, xs = sol['ys'], sol['xs']
    cf = nd.spline_filter(sol['w'], order=3, mode='mirror')
    iy = (np.clip(ry, ys[0], ys[-1]) - ys[0]) / (ys[1] - ys[0])
    ix = (np.clip(rx, xs[0], xs[-1]) - xs[0]) / (xs[1] - xs[0])
    return nd.map_coordinates(cf, np.stack([iy, ix]), order=3, mode='mirror', prefilter=False)


def interp_grid(xs, vs, q, x, v):
    """Cubic-spline interpolation of a solved grid q(xs, vs) at points (x, v) (numpy, CPU); points outside the
    grid in the second coordinate are clamped to its edge."""
    from scipy import ndimage as nd
    cf = nd.spline_filter(q, order=3, mode='mirror')
    ix = (np.clip(x, xs[0], xs[-1]) - xs[0]) / (xs[1] - xs[0])
    iv = (np.clip(v, vs[0], vs[-1]) - vs[0]) / (vs[1] - vs[0])
    return nd.map_coordinates(cf, np.stack([ix, iv]), order=3, mode='mirror', prefilter=False)


# ------------------------------------------------------------------------------------------------
# shooting on the GPU (absorbed trajectories are removed at once, so their outcome cannot be overwritten)
# ------------------------------------------------------------------------------------------------
def shoot_underdamped_1d(step, coef, force, A, B, points, n_per_point, seed=0, max_steps=10 ** 7):
    """Fraction of trajectories from each (x0, v0) that reach B before A (checked every step), its binomial
    standard error, and the mean number of steps to absorption."""
    rs = cp.random.RandomState(seed)
    P = len(points)
    x = cp.repeat(cp.asarray([p[0] for p in points], dtype=cp.float64), n_per_point)
    v = cp.repeat(cp.asarray([p[1] for p in points], dtype=cp.float64), n_per_point)
    idx = cp.arange(x.size)
    out = cp.full(x.size, -1.0)
    steps = cp.zeros(x.size)
    for s_ in range(1, max_steps + 1):
        x, v = step(x, v, rs.standard_normal(idx.size), rs.standard_normal(idx.size), force, coef, cp)
        a, b = x <= A, x >= B
        done = a | b
        out[idx[a]] = 0.0; out[idx[b]] = 1.0; steps[idx[done]] = s_
        keep = ~done
        x, v, idx = x[keep], v[keep], idx[keep]
        if idx.size == 0:
            break
    o = cp.asnumpy(out).reshape(P, n_per_point)
    if (o < 0).any():
        print(f'  warning: {(o < 0).sum()} trajectories not absorbed after {max_steps} steps')
    p = np.array([r[r >= 0].mean() for r in o])
    return p, np.sqrt(p * (1 - p) / n_per_point), float(steps.mean())


def shoot_brownian_2d(force2d, dt, gamma, kT, A, B, points, n_per_point, stride=1, seed=0, m=1.0,
                      max_steps=10 ** 9, report_every=None):
    """2D Brownian shooting (y = order parameter); A/B checked every `stride` steps."""
    rs = cp.random.RandomState(seed)
    s = np.sqrt(2 * kT * dt / (m * gamma)); mu = dt / (m * gamma)
    y = cp.repeat(cp.asarray([p[0] for p in points], dtype=cp.float64), n_per_point)
    x = cp.repeat(cp.asarray([p[1] for p in points], dtype=cp.float64), n_per_point)
    idx = cp.arange(y.size); out = cp.full(y.size, -1.0)
    for s_ in range(1, max_steps + 1):
        fy, fx = force2d(y, x, cp)
        y = y + mu * fy + s * rs.standard_normal(idx.size)
        x = x + mu * fx + s * rs.standard_normal(idx.size)
        if s_ % stride == 0:
            a, b = y <= A, y >= B
            out[idx[a]] = 0.0; out[idx[b]] = 1.0
            keep = ~(a | b)
            y, x, idx = y[keep], x[keep], idx[keep]
            if idx.size == 0:
                break
            if report_every and (s_ // stride) % report_every == 0:
                print(f'    step {s_}: {idx.size} trajectories left', flush=True)
    o = cp.asnumpy(out).reshape(len(points), n_per_point)
    if (o < 0).any():
        print(f'  warning: {(o < 0).sum()} trajectories not absorbed')
    p = np.array([r[r >= 0].mean() for r in o])
    return p, np.sqrt(p * (1 - p) / n_per_point)


def check_forces():
    """Compare the vectorized forces with the toytis classes at random points."""
    import sys
    sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis')
    from potentials.cos_well import CosWellWalls
    from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
    rng = np.random.default_rng(0)
    cw = CosWellWalls()
    xs = rng.uniform(-0.6, 0.6, 2000)
    d1 = max(abs(coswell_force(x) - cw.force((x, 0.0))) for x in xs)
    d1p = max(abs(coswell_potential(x) - cw.potential((np.array([x]), 0.0))) for x in xs)
    zp = RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)
    pts = np.c_[rng.uniform(-0.02, 0.92, 2000), rng.uniform(-0.02, 0.32, 2000)]
    d2 = max(np.abs(np.array(zpot_force(y, x)) - zp.potential_and_force((np.array([y, x]), np.zeros(2)))[1]).max()
             for y, x in pts)
    return d1, d1p, d2


def solve_brownian_1d(force, dt, gamma, kT, A, B, n, m=1.0):
    """1D Brownian chain x' = x + dt/(m gamma) F(x) + N(0, 2 kT dt/(m gamma)), A: x <= A, B: x >= B every step.
    q piecewise linear on n nodes over [A, B] (one-sided limits at A and B); the Gaussian is integrated
    against it exactly (error functions), so the only error is the O(h^2) of the piecewise-linear q."""
    from scipy.special import erf
    sigma = np.sqrt(2 * kT * dt / (m * gamma))
    xs = np.linspace(A, B, n); h = xs[1] - xs[0]
    mean = xs + dt / (m * gamma) * force(xs)
    Phi = lambda t: 0.5 * (1 + erf((t[None, :] - mean[:, None]) / (sigma * np.sqrt(2))))
    dens = lambda t: np.exp(-0.5 * ((t[None, :] - mean[:, None]) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    H = lambda t: mean[:, None] * Phi(t) - sigma ** 2 * dens(t)
    I0 = Phi(xs[1:]) - Phi(xs[:-1]); I1 = H(xs[1:]) - H(xs[:-1])
    Wm = np.zeros((n, n))
    Wm[:, :-1] += (xs[1:][None, :] * I0 - I1) / h
    Wm[:, 1:] += (I1 - xs[:-1][None, :] * I0) / h
    b = 1 - 0.5 * (1 + erf((B - mean) / (sigma * np.sqrt(2))))
    return xs, np.linalg.solve(np.eye(n) - Wm, b)
