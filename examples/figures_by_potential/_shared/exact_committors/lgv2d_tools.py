"""Reference committors of the 2D Langevin (BBK) toytis runs by GPU shooting.

The committor of an underdamped run depends on the phase point (y, x, v_y, v_x): a grid solve as for the overdamped
runs would be 4D and does not fit the GPU at a useful resolution. Instead every reference value is the fraction of
N trajectories from that phase point that reach B before A, with the run's own integrator (the BBK step of toytis'
ndLangevinEngine, as in toy2d.Integrator) and absorption checked every `check_every` steps, as in the data. This is
exact in expectation; its standard error sqrt(q(1 - q)/N) is given with every value. Absorbed trajectories are removed
at once (their outcome cannot be overwritten).

Coordinates: y = the order parameter (toytis position index 0 for curved channel and staggered channels; index 1 for
the original sjoelbak Z potential, which takes (x, y)), x = the transverse coordinate.

Systems (`SYSTEMS`): the curved channel (γ 2, `sim_istar_cc_lgv` / `test_cc_lgv`), the Z potential (γ 2,
`sim_istarz_2603`, which uses `sjoelbak.RectangularGridWithBarrierPotential()`: identical to `exact_tools.zpot_force`,
checked to 0) and the staggered channels with the defaults of `notebooks/staggered_channels_run.py` (kT 0.25,
friction 0.1, dt 0.02, A: x <= -1.6, B: x >= 1.6).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOYTIS = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis"


# ------------------------------------------------------------------------------------------------
# forces (y = order parameter, x = transverse), vectorized for numpy / cupy
# ------------------------------------------------------------------------------------------------
def cc_force(y, x, xp=np):
    """Curved channel (CurvedChannelDoubleWell defaults)."""
    s = (y - 0.5) / 0.4
    c = -0.15 + 0.3 * (s + 1) / 2 + 0.2 * (1 - s * s)
    dc_ds = 0.15 - 0.4 * s
    n = x - c
    dv_ds = 4 * 0.2 * s * (s * s - 1) - 15.0 * n * dc_ds
    return -dv_ds / 0.4, -15.0 * n


def cc_potential(y, x, xp=np):
    s = (y - 0.5) / 0.4
    c = -0.15 + 0.3 * (s + 1) / 2 + 0.2 * (1 - s * s)
    return 0.2 * (s * s - 1) ** 2 + 7.5 * (x - c) ** 2


STAG = dict(xol=-3.0, xor=3.0, yob=-2.0, yot=2.0, xl=-1.0, xr=1.0, yb=-0.6, yt=0.6,
            wall_a=100.0, wall_b=0.0, wall_c=0.15, x_kink=0.4,
            V_peak=0.5 / 0.07 * 0.25, V_dip=3.0 * 0.25)


def _stag_parts(y, x, xp, p=STAG):
    """Potential and force (F_op, F_tr) of PotentialStaggeredChannels; op = toytis x (index 0), tr = toytis y."""
    X, Y = y, x                                                      # toytis names
    a, b, c = p["wall_a"], p["wall_b"], p["wall_c"]
    gauss = lambda d: a * xp.exp(-((d - b) ** 2) / (2 * c * c))
    dgauss = lambda d: -a / (c * c) * (d - b) * xp.exp(-((d - b) ** 2) / (2 * c * c))   # dV/dd
    V = xp.zeros_like(X); FX = xp.zeros_like(X); FY = xp.zeros_like(X)
    # background: half-cosines in x inside the channels, 0 in the strips and the forbidden rectangle
    xl, xr, yb, yt, k_ = p["xl"], p["xr"], p["yb"], p["yt"], p["x_kink"]
    inch = (X > xl) & (X < xr) & ~((Y > yb) & (Y < yt))
    upper = Y >= yt
    nodes = np.array([xl, xl + k_, xr - k_, xr])
    up_vals = np.array([0.0, p["V_peak"], -p["V_dip"], 0.0])
    lo_vals = np.array([0.0, -p["V_dip"], p["V_peak"], 0.0])
    k = xp.clip(xp.searchsorted(xp.asarray(nodes), X), 1, 3)
    n0, n1 = xp.asarray(nodes)[k - 1], xp.asarray(nodes)[k]
    v0 = xp.where(upper, xp.asarray(up_vals)[k - 1], xp.asarray(lo_vals)[k - 1])
    v1 = xp.where(upper, xp.asarray(up_vals)[k], xp.asarray(lo_vals)[k])
    width, dv = n1 - n0, v1 - v0
    t = np.pi * (X - n0) / width
    V += xp.where(inch, v0 + 0.5 * dv * (1 - xp.cos(t)), 0.0)
    FX += xp.where(inch, -0.5 * dv * np.pi / width * xp.sin(t), 0.0)
    # outer walls
    for d, gx, gy in ((X - p["xol"], 1.0, 0.0), (p["xor"] - X, -1.0, 0.0),
                      (Y - p["yob"], 0.0, 1.0), (p["yot"] - Y, 0.0, -1.0)):
        V += gauss(d); g = dgauss(d); FX -= g * gx; FY -= g * gy
    # inner rectangle EFGH: signed distance to the nearest segment (EF, FH, HG, GE, ties to the first)
    segs = (((xl, yt), (xr, yt), (0.0, 1.0)), ((xr, yt), (xr, yb), (1.0, 0.0)),
            ((xr, yb), (xl, yb), (0.0, -1.0)), ((xl, yb), (xl, yt), (-1.0, 0.0)))
    best_d = xp.full(X.shape, np.inf); best_dx = xp.zeros_like(X); best_dy = xp.zeros_like(X)
    best_nx = xp.zeros_like(X); best_ny = xp.zeros_like(X)
    for (ax, ay), (bx, by), (nx_, ny_) in segs:
        abx, aby = bx - ax, by - ay
        tt = xp.clip(((X - ax) * abx + (Y - ay) * aby) / (abx * abx + aby * aby), 0.0, 1.0)
        dx_, dy_ = X - (ax + tt * abx), Y - (ay + tt * aby)
        dist = xp.sqrt(dx_ * dx_ + dy_ * dy_)
        better = dist < best_d
        best_d = xp.where(better, dist, best_d)
        best_dx, best_dy = xp.where(better, dx_, best_dx), xp.where(better, dy_, best_dy)
        best_nx, best_ny = xp.where(better, nx_, best_nx), xp.where(better, ny_, best_ny)
    inside = (X > xl) & (X < xr) & (Y > yb) & (Y < yt)
    safe = xp.where(best_d > 0, best_d, 1.0)
    gdx = xp.where(inside | (best_d == 0), best_nx, best_dx / safe)
    gdy = xp.where(inside | (best_d == 0), best_ny, best_dy / safe)
    sd = xp.where(inside, -best_d, best_d)
    V += gauss(sd); g = dgauss(sd); FX -= g * gdx; FY -= g * gdy
    return V, FX, FY


def stag_force(y, x, xp=np):
    _, fx, fy = _stag_parts(y, x, xp)
    return fx, fy


def stag_potential(y, x, xp=np):
    return _stag_parts(y, x, xp)[0]


def zpot_potential(y, x, xp=np):
    """Effective Z potential (potential + gravity y): the one the dynamics integrates (numpy, slow)."""
    sys.path.insert(0, TOYTIS)
    from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
    pot = RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)
    yy, xx = np.broadcast_arrays(np.asarray(y, float), np.asarray(x, float))
    return np.array([pot.effective_potential((np.array([a, b]), np.zeros(2)))
                     for a, b in zip(yy.ravel(), xx.ravel())]).reshape(yy.shape)


def _zpot_force(y, x, xp=np):
    import exact_tools as E
    return E.zpot_force(y, x, xp)


SYSTEMS = {
    "cc": dict(name="curved channel", force=cc_force, potential=cc_potential, kT=0.05, gamma=2.0, dt=0.005, m=1.0,
               A=0.15, B=0.85, y_range=(0.15, 0.85), x_range=(-0.35, 0.5), folder="2d_curved_channel",
               runs="sim_istar_cc_lgv, test_cc_lgv (staple); MD of toy2d_md-lgv.ipynb checks every 50 steps"),
    "zpot": dict(name="Z potential", force=_zpot_force, potential=zpot_potential, kT=0.05, gamma=2.0, dt=0.002, m=1.0,
                 A=0.05, B=0.85, y_range=(0.05, 0.85), x_range=(0.0, 0.3), folder="2d_zpot",
                 runs="sim_istarz_2603 (staple; positions only)"),
    "stag": dict(name="staggered channels", force=stag_force, potential=stag_potential, kT=0.25, gamma=0.1, dt=0.02,
                 m=1.0, A=-1.6, B=1.6, y_range=(-1.6, 1.6), x_range=(-1.9, 1.9), folder="2d_staggered_channels",
                 runs="staggered_channels_run.py defaults (istar, friction 0.1, dt 0.02)"),
}


def check_stag_force(n=20000, seed=0):
    """Max |difference| of the vectorized staggered-channel potential and force against the toytis class."""
    sys.path.insert(0, TOYTIS)
    from potentials.potential_staggered_channels import PotentialStaggeredChannels
    pot = PotentialStaggeredChannels(V_peak=STAG["V_peak"], V_dip=STAG["V_dip"], V_block=30 * 0.25)
    rng = np.random.default_rng(seed)
    X = np.r_[rng.uniform(-3.1, 3.1, n), rng.choice([-1, 1, -0.6, 0.6, -0.2, 0.2], 2000) + rng.normal(0, 0.02, 2000)]
    Y = np.r_[rng.uniform(-2.1, 2.1, n), rng.uniform(-2.1, 2.1, 2000)]
    ref = [pot.potential_and_force((np.array([a, b]), np.zeros(2))) for a, b in zip(X, Y)]
    Vr = np.array([r[0] for r in ref]); Fr = np.array([r[1] for r in ref])
    V, FX, FY = _stag_parts(X, Y, np)
    return np.abs(V - Vr).max(), np.abs(np.c_[FX, FY] - Fr).max()


# ------------------------------------------------------------------------------------------------
# BBK (toytis ndLangevinEngine / toy2d.Integrator) and shooting
# ------------------------------------------------------------------------------------------------
def bbk_coefficients(dt, gamma, kT, m=1.0):
    gdt = gamma * dt
    c0 = np.exp(-gdt); c1 = (1 - c0) / gdt; c2 = (1 - c1) / gdt
    s_r2 = (dt * kT / (gamma * m)) * (2. - (3. - 4. * c0 + c0 ** 2) / gdt)
    s_v2 = (1. - c0 ** 2) * kT / m
    cov = (kT / (m * gamma)) * (1. - c0) ** 2
    l11 = np.sqrt(s_r2); l21 = cov / l11; l22 = np.sqrt(s_v2 - l21 ** 2)
    return dict(c0=c0, a1=c1 * dt, a2=c2 * dt ** 2 / m, b1=(c1 - c2) * dt / m, b2=c2 * dt / m,
                l11=l11, l21=l21, l22=l22)


def shoot(sysd, y0, x0, vy0=None, vx0=None, n_shots=10000, check_every=1, seed=0, max_batch=4_000_000,
          max_steps=10 ** 7, progress=False):
    """Committor of the phase points by shooting, on the GPU.

    vy0, vx0 None: every trajectory draws its own Maxwell velocity, which gives the velocity-averaged committor
    qbar(y, x) = E_v[q(y, x, v)] (the committor of a position under equilibrium velocities). Returns
    (q, standard error, mean number of steps) per point."""
    import cupy as cp
    c = bbk_coefficients(sysd["dt"], sysd["gamma"], sysd["kT"], sysd["m"])
    force, A, B = sysd["force"], sysd["A"], sysd["B"]
    y0, x0 = np.ravel(y0).astype(float), np.ravel(x0).astype(float)
    maxwell = vy0 is None
    if not maxwell:
        vy0, vx0 = np.ravel(vy0).astype(float), np.ravel(vx0).astype(float)
    P = len(y0)
    rs = cp.random.RandomState(seed)
    hits = np.zeros(P); steps_sum = np.zeros(P)
    per = max(1, max_batch // n_shots)                      # points per batch
    sv = np.sqrt(sysd["kT"] / sysd["m"])
    for p0 in range(0, P, per):
        sl = slice(p0, min(P, p0 + per)); npnt = sl.stop - sl.start
        pid = cp.repeat(cp.arange(npnt), n_shots)
        y = cp.asarray(y0[sl])[pid]; x = cp.asarray(x0[sl])[pid]
        if maxwell:
            vy = sv * rs.standard_normal(pid.size); vx = sv * rs.standard_normal(pid.size)
        else:
            vy = cp.asarray(vy0[sl])[pid]; vx = cp.asarray(vx0[sl])[pid]
        out = cp.full(pid.size, -1.0); nst = cp.zeros(pid.size); idx = cp.arange(pid.size)
        fy, fx = force(y, x, cp)
        for s in range(1, max_steps + 1):
            n = idx.size
            z1y, z2y, z1x, z2x = (rs.standard_normal(n) for _ in range(4))
            y = y + c["a1"] * vy + c["a2"] * fy + c["l11"] * z1y
            x = x + c["a1"] * vx + c["a2"] * fx + c["l11"] * z1x
            vy = c["c0"] * vy + c["b1"] * fy + c["l21"] * z1y + c["l22"] * z2y
            vx = c["c0"] * vx + c["b1"] * fx + c["l21"] * z1x + c["l22"] * z2x
            fy, fx = force(y, x, cp)
            vy = vy + c["b2"] * fy; vx = vx + c["b2"] * fx
            if s % check_every == 0:
                a, b = y <= A, y >= B
                done = a | b
                out[idx[b]] = 1.0; out[idx[a]] = 0.0; nst[idx[done]] = s
                keep = ~done
                y, x, vy, vx, fy, fx, idx = y[keep], x[keep], vy[keep], vx[keep], fy[keep], fx[keep], idx[keep]
                if idx.size == 0:
                    break
                if progress and s % 20000 == 0:
                    print(f'    step {s}: {idx.size} of {pid.size} trajectories left', flush=True)
        o = cp.asnumpy(out).reshape(npnt, n_shots)
        if (o < 0).any():
            print(f'  warning: {(o < 0).sum()} trajectories not absorbed after {max_steps} steps (counted as missing)')
        ok = o >= 0
        hits[sl] = (o * ok).sum(1) / np.maximum(ok.sum(1), 1)
        steps_sum[sl] = cp.asnumpy(nst).reshape(npnt, n_shots).mean(1)
        del y, x, vy, vx, fy, fx, out, nst, idx, pid
        cp.get_default_memory_pool().free_all_blocks()
    se = np.sqrt(hits * (1 - hits) / n_shots)
    return hits, se, steps_sum
