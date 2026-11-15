"""GPU Brownian dynamics of toytis potentials/zpot_w.ZChannelsW (two mirrored Z channels selected by w).

The same chain as the toytis engine in the overdamped limit (ndLangevinEngine, high_friction):
    s' = s + mu F(s) + sigma z,   mu = dt / (m gamma),  sigma^2 = 2 kT mu,  z standard normal per coordinate,
with A: y <= lambda_A and B: y >= lambda_B checked on every step, as on the saved frames of a staple run.
Random numbers are counter-based (splitmix64 + Box-Muller), not those of the engine.

    python zpotw_gpu.py check                       force of the kernel vs the toytis class
    python zpotw_gpu.py scan  [--dV 2 3 4 5]        brute force from A at several w barriers (in kT)
    python zpotw_gpu.py shoot --dV 3                committor at points (y, x, +-w0) by shooting

`scan` gives, per w barrier: the time between channel hops in A; the A -> B rate; the transition paths
(last exit from A to first entry into B): their length in steps and the fraction that switch channel on the way.
That is what fixes dV_w for the staple run: hops must be rare within a path, but not so rare that the sampler
cannot equilibrate the channels.
"""
import argparse
import json
import os
import sys

import numpy as np
import cupy as cp

HERE = os.path.dirname(os.path.abspath(__file__))
TOYTIS = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis'

# Z runs (sim_istarz_br_g10): overdamped, gamma 10, dt 0.002, kT 0.05, interfaces 0.05 ... 0.85
RUN = dict(dt=0.002, gamma=10.0, kT=0.05, lam_a=0.05, lam_b=0.85)

_SRC = r'''
#define LX 0.3
#define LY 0.9
#define KW 2000.0
#define RA 0.3
#define RL 0.2
#define PI 3.141592653589793

struct P { double w0, dV, delta; };

__device__ void ridge(double x, double y, double* pot, double* fx, double* fy) {
    // Z ridge: line y = a x + b, a = -1, b = 0.6 (internal (x, y), as sjoelbak_angle)
    const double a = -1.0, b = 0.6, norm = 1.4142135623730951;
    double D = fabs(a * x - y + b) / norm;
    *pot = 0.0; *fx = 0.0; *fy = 0.0;
    if (D >= RL / 4) return;
    double arg = 2 * PI * D / RL;
    *pot = RA * cos(arg);
    double dVdD = -2 * PI * RA / RL * sin(arg);
    double t = y - a * x - b;
    double sg = (t > 0) - (t < 0);
    *fx = sg * dVdD * a / norm;
    *fy = -sg * dVdD / norm;
}

__device__ void force(double y, double x, double w, P p, double* fy, double* fx, double* fw) {
    double Fx = 0, Fy = 0;
    if (x < 0) Fx += -KW * x; else if (x > LX) Fx += -KW * (x - LX);
    if (y < 0) Fy += -KW * y; else if (y > LY) Fy += -KW * (y - LY);
    const double Ys[2] = {0.025, 0.875};
    for (int i = 0; i < 2; i++) {
        double D = fabs(y - Ys[i]);
        if (D < 0.05) {
            double sg = (y > Ys[i]) - (y < Ys[i]);
            Fy += -0.15 * 2 * PI / 0.2 * sin(2 * PI * D / 0.2) * sg;
        }
    }
    Fy -= 0.2;                                               // gravity
    if (y >= 0.1 && y <= 0.825) Fy += -0.2 / (0.825 - 0.1);  // slope
    double th = tanh(w / p.delta);
    double s = 0.5 * (1 - th), ds = -0.5 * (1 - th * th) / p.delta;
    double p1, f1x, f1y, p2, f2x, f2y;
    ridge(x, y, &p1, &f1x, &f1y);
    ridge(LX - x, y, &p2, &f2x, &f2y);
    Fx += s * f1x - (1 - s) * f2x;
    Fy += s * f1y + (1 - s) * f2y;
    double r = w / p.w0;
    *fw = -ds * (p1 - p2) - 4 * p.dV * r * (r * r - 1) / p.w0;
    *fy = Fy; *fx = Fx;
}

__device__ unsigned long long splitmix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ void normals(unsigned long long seed, unsigned long long id, unsigned long long step, double* z) {
    // 4 normals from 2 counter-based uniform pairs (Box-Muller)
    for (int k = 0; k < 2; k++) {
        unsigned long long h1 = splitmix(seed ^ splitmix(id * 0x100000001B3ULL + splitmix(step * 4 + 2 * k)));
        unsigned long long h2 = splitmix(h1 + 0x632BE59BD9B4E019ULL);
        double u1 = ((h1 >> 11) + 1.0) * (1.0 / 9007199254740993.0);
        double u2 = (h2 >> 11) * (1.0 / 9007199254740992.0);
        double rr = sqrt(-2 * log(u1));
        z[2 * k] = rr * cos(2 * PI * u2); z[2 * k + 1] = rr * sin(2 * PI * u2);
    }
}

extern "C" __global__ void force_at(const double* y, const double* x, const double* w, int n,
                                    double w0, double dV, double delta, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    P p = {w0, dV, delta};
    double fy, fx, fw;
    force(y[i], x[i], w[i], p, &fy, &fx, &fw);
    out[3 * i] = fy; out[3 * i + 1] = fx; out[3 * i + 2] = fw;
}

// brute force: walkers move freely; per walker counters of hops and A <-> B transitions
extern "C" __global__ void run(double* Y, double* X, double* W, int* st, long long* cnt, int n, long long nsteps,
                               long long step0, unsigned long long seed, double mu, double sg,
                               double lamA, double lamB, double w0, double dV, double delta) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    P p = {w0, dV, delta};
    double y = Y[i], x = X[i], w = W[i];
    // st: last state (0 A, 1 B), channel (+-1, hysteresis at +-w0/2), channel when the last state was left,
    //     steps since the last state was left
    int last = st[4 * i], ch = st[4 * i + 1], ch0 = st[4 * i + 2], since = st[4 * i + 3];
    long long* c = cnt + 12 * i;
    // c: 0 hops in A, 1 steps in A, 2 nAB, 3 nAB switched, 4 TP steps AB, 5 nBA, 6 nBA switched, 7 TP steps BA,
    //    8 hops total, 9 steps total, 10 TP hops AB (hop count summed over AB transition paths), 11 unused
    double z[4];
    for (long long t = 0; t < nsteps; t++) {
        double fy, fx, fw;
        force(y, x, w, p, &fy, &fx, &fw);
        normals(seed, (unsigned long long)i, (unsigned long long)(step0 + t), z);
        y += mu * fy + sg * z[0]; x += mu * fx + sg * z[1]; w += mu * fw + sg * z[2];
        int nch = ch;
        if (w > 0.5 * w0) nch = 1; else if (w < -0.5 * w0) nch = -1;
        int hop = (nch != ch); ch = nch;
        c[8] += hop; c[9] += 1;
        int inA = y <= lamA, inB = y >= lamB;
        if (inA) { c[0] += hop; c[1] += 1; }
        if (inA || inB) {
            int now = inA ? 0 : 1;
            if (now != last) {
                if (last == 0) { c[2] += 1; c[3] += (ch != ch0); c[4] += since; }
                else           { c[5] += 1; c[6] += (ch != ch0); c[7] += since; }
            }
            last = now; ch0 = ch; since = 0;
        } else {
            since += 1;
        }
    }
    Y[i] = y; X[i] = x; W[i] = w;
    st[4 * i] = last; st[4 * i + 1] = ch; st[4 * i + 2] = ch0; st[4 * i + 3] = since;
}

// shooting: absorbed at A (0) or B (1); out = -1 while running
extern "C" __global__ void shoot(double* Y, double* X, double* W, double* out, int n, long long nsteps,
                                 long long step0, unsigned long long seed, double mu, double sg,
                                 double lamA, double lamB, double w0, double dV, double delta) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n || out[i] >= 0) return;
    P p = {w0, dV, delta};
    double y = Y[i], x = X[i], w = W[i];
    double z[4];
    for (long long t = 0; t < nsteps; t++) {
        double fy, fx, fw;
        force(y, x, w, p, &fy, &fx, &fw);
        normals(seed, (unsigned long long)i, (unsigned long long)(step0 + t), z);
        y += mu * fy + sg * z[0]; x += mu * fx + sg * z[1]; w += mu * fw + sg * z[2];
        if (y <= lamA) { out[i] = 0.0; break; }
        if (y >= lamB) { out[i] = 1.0; break; }
    }
    Y[i] = y; X[i] = x; W[i] = w;
}
'''
_mod = cp.RawModule(code=_SRC, options=('-std=c++11',))
_force_at, _run, _shoot = (_mod.get_function(k) for k in ('force_at', 'run', 'shoot'))


def _grid(n, block=128):
    return ((n + block - 1) // block,), (block,)


def coefs(run=RUN):
    mu = run['dt'] / run['gamma']
    return mu, np.sqrt(2 * run['kT'] * mu)


def check(pars):
    sys.path.insert(0, TOYTIS)
    from potentials.zpot_w import ZChannelsW
    pot = ZChannelsW(**pars)
    rng = np.random.default_rng(1)
    n = 20000
    P = np.c_[rng.uniform(-0.05, 0.95, n), rng.uniform(-0.05, 0.35, n), rng.uniform(-0.25, 0.25, n)]
    ref = np.array([pot.force((q, None)) for q in P])
    y, x, w = (cp.asarray(P[:, k].copy()) for k in range(3))
    out = cp.zeros(3 * n)
    _force_at(*_grid(n), (y, x, w, np.int32(n), np.float64(pars['w0']), np.float64(pars['dV_w']),
                          np.float64(pars['delta']), out))
    err = np.abs(cp.asnumpy(out).reshape(n, 3) - ref).max()
    print(f'kernel vs toytis ZChannelsW: max |dF| = {err:.2e} over {n} points')
    return err


def scan(pars, n_walk, n_steps, chunk, seed, run=RUN):
    """Brute force from A (both channels); returns the counters summed over walkers."""
    mu, sg = coefs(run)
    rng = np.random.default_rng(seed)
    Y = cp.asarray(np.full(n_walk, 0.025))
    X = cp.asarray(rng.uniform(0, 0.3, n_walk))
    sign = rng.choice([-1.0, 1.0], n_walk)
    W = cp.asarray(sign * pars['w0'])
    st = cp.asarray(np.stack([np.zeros(n_walk), sign, sign, np.zeros(n_walk)], 1).astype(np.int32).ravel())
    cnt = cp.zeros(12 * n_walk, dtype=cp.int64)
    args = lambda s0: (Y, X, W, st, cnt, np.int32(n_walk), np.int64(chunk), np.int64(s0), np.uint64(seed),
                       np.float64(mu), np.float64(sg), np.float64(run['lam_a']), np.float64(run['lam_b']),
                       np.float64(pars['w0']), np.float64(pars['dV_w']), np.float64(pars['delta']))
    for s0 in range(0, n_steps, chunk):
        _run(*_grid(n_walk), args(s0))
        cp.cuda.Device().synchronize()
    c = cp.asnumpy(cnt).reshape(n_walk, 12).sum(0)
    out = dict(walker_steps=int(c[9]), hops=int(c[8]), steps_in_A=int(c[1]), hops_in_A=int(c[0]),
               nAB=int(c[2]), nAB_switched=int(c[3]), tp_steps_AB=int(c[4]),
               nBA=int(c[5]), nBA_switched=int(c[6]), tp_steps_BA=int(c[7]))
    out['steps_between_hops_in_A'] = out['steps_in_A'] / max(out['hops_in_A'], 1)
    out['steps_between_hops'] = out['walker_steps'] / max(out['hops'], 1)
    n_tr = out['nAB'] + out['nBA']
    out['transitions_per_1e9_steps'] = n_tr / out['walker_steps'] * 1e9
    out['frac_tp_switched'] = (out['nAB_switched'] + out['nBA_switched']) / max(n_tr, 1)
    out['mean_tp_steps'] = (out['tp_steps_AB'] + out['tp_steps_BA']) / max(n_tr, 1)
    return out


def shoot(pars, points, n_per, seed, max_steps=2_000_000, chunk=20000, run=RUN):
    mu, sg = coefs(run)
    pts = np.repeat(np.asarray(points, float), n_per, axis=0)
    n = len(pts)
    Y, X, W = (cp.asarray(pts[:, k].copy()) for k in range(3))
    out = cp.full(n, -1.0)
    for s0 in range(0, max_steps, chunk):
        _shoot(*_grid(n), (Y, X, W, out, np.int32(n), np.int64(chunk), np.int64(s0), np.uint64(seed),
                           np.float64(mu), np.float64(sg), np.float64(run['lam_a']), np.float64(run['lam_b']),
                           np.float64(pars['w0']), np.float64(pars['dV_w']), np.float64(pars['delta'])))
        if not bool((out < 0).any()):
            break
    o = cp.asnumpy(out).reshape(len(points), n_per)
    p = np.array([r[r >= 0].mean() for r in o])
    return p, np.sqrt(p * (1 - p) / n_per), int((o < 0).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['check', 'scan', 'shoot'])
    ap.add_argument('--dV', type=float, nargs='+', default=[2, 3, 4, 5], help='w barrier in kT')
    ap.add_argument('--w0', type=float, default=0.1)
    ap.add_argument('--delta', type=float, default=0.02)
    ap.add_argument('--walkers', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=2_000_000)
    ap.add_argument('--chunk', type=int, default=50_000)
    ap.add_argument('--shots', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, 'explore'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kT = RUN['kT']
    if a.cmd == 'check':
        for dV in a.dV:
            check(dict(w0=a.w0, dV_w=dV * kT, delta=a.delta))
    elif a.cmd == 'scan':
        res = {}
        for dV in a.dV:
            pars = dict(w0=a.w0, dV_w=dV * kT, delta=a.delta)
            r = scan(pars, a.walkers, a.steps, a.chunk, a.seed)
            res[f'{dV:g}'] = dict(pars=pars, **r)
            print(f'dV_w {dV:g} kT: steps between hops in A {r["steps_between_hops_in_A"]:9.0f}, '
                  f'overall {r["steps_between_hops"]:9.0f}; transitions {r["nAB"]} + {r["nBA"]} '
                  f'({r["transitions_per_1e9_steps"]:.0f} per 1e9 steps), switched {r["frac_tp_switched"]:.3f}, '
                  f'mean TP {r["mean_tp_steps"]:.0f} steps', flush=True)
        with open(os.path.join(a.out, f'scan_w0{a.w0:g}_d{a.delta:g}.json'), 'w') as f:
            json.dump(dict(run=RUN, walkers=a.walkers, steps=a.steps, seed=a.seed, results=res), f, indent=1)
    else:
        ys = np.round(np.arange(0.25, 0.651, 0.05), 3)
        xs = [0.05, 0.15, 0.25]
        rows = []
        for dV in a.dV:
            pars = dict(w0=a.w0, dV_w=dV * kT, delta=a.delta)
            pts = [(y, x, s * a.w0) for x in xs for y in ys for s in (-1, 1)]
            p, se, left = shoot(pars, pts, a.shots, a.seed)
            print(f'dV_w {dV:g} kT ({left} not absorbed)\n   x     y     q(w=-w0)          q(w=+w0)')
            for k in range(0, len(pts), 2):
                print(f'  {pts[k][1]:.2f}  {pts[k][0]:.2f}  {p[k]:.4f} +- {se[k]:.4f}   {p[k+1]:.4f} +- {se[k+1]:.4f}')
                rows.append(dict(dV_kT=dV, y=pts[k][0], x=pts[k][1], q_minus=p[k], se_minus=se[k],
                                 q_plus=p[k + 1], se_plus=se[k + 1]))
        with open(os.path.join(a.out, f'shoot_w0{a.w0:g}_d{a.delta:g}.json'), 'w') as f:
            json.dump(dict(run=RUN, shots=a.shots, seed=a.seed, rows=rows), f, indent=1)


if __name__ == '__main__':
    main()
