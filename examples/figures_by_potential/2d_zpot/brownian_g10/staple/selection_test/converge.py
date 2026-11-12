"""Data-selection fits of the 2D Brownian Z staple run (sim_istarz_br_g10), run to convergence.

Versions (see selection_test.py and blind_starts.py for the 11000-iteration runs):
  base           1500 paths per ensemble (seed 0), forward + mirrored pieces, rejection weights
  select_paths   the same, whole paths kept with probability min(1, K / N_rarest) by the q_n bins they
                 visit (K: 25% quantile of the bin counts); weights not corrected
  flat_qn        the same frames, frame weights x g(q_n(x_t)) flattening the weighted density over 30 q_n
                 bins (capped at 20x); q_n from the converged base fit (OUT/base/r.npy: run base first)
  blind_uniform  pool of 3000 paths per ensemble (seed 1); BUDGET window starts with equal probability;
                 each run of flagged frames kept as its own short segment (lag-1 transitions)
  blind_flat_y   the same pool and budget, start probability flat over 40 bins of y (no committor used)
select_paths uses the q_n of the converged base fit as well (the 11000-iteration runs used an earlier fit).

Fit: CVs (y, x); warm-up 1000 iterations at envelope 1, then chunks of CHUNK iterations at envelope 0.01;
gamma 0.1, ny 6, lazy basis, seed 0. After every chunk: the fit at 4 plateau points against engine
shooting, and RMS(fit - exact) per (y, x) bin. Stops when the mean plateau value has changed by less than
TOL in each of the last two chunks (or at MAX_ITER). Saves OUT/<version>/r.npy and history.csv after
every chunk (resume with --resume) and the metrics/TIS/map plots.

    python converge.py <version> --sim /path/to/sim_istarz_br_g10 [--out DIR] [--max-iter 60000]
                       [--chunk 2500] [--tol 0.001] [--resume]
    python converge.py summary [--out DIR]          plateau and RMS against iteration for all versions
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'optimalrcs-cupy'))
sys.path.insert(0, os.path.join(REPO, 'examples'))

VERSIONS = ('base', 'select_paths', 'flat_qn', 'blind_uniform', 'blind_flat_y')
SHOOT = [((0.50, 0.25), 0.639), ((0.55, 0.10), 0.664), ((0.62, 0.15), 0.698), ((0.70, 0.15), 0.747)]
LAMBDA_A, LAMBDA_B = 0.05, 0.85
DT = 0.002

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('version', choices=VERSIONS + ('summary',))
p.add_argument('--sim', default=os.environ.get('SIM_ISTARZ'), help='the staple run folder (or $SIM_ISTARZ)')
p.add_argument('--out', default=os.path.join(HERE, 'converged'), help='output folder (default: ./converged)')
p.add_argument('--max-iter', type=int, default=60000, help='main iterations at most (after the warm-up)')
p.add_argument('--chunk', type=int, default=2500)
p.add_argument('--tol', type=float, default=0.001, help='plateau change per chunk counted as converged')
p.add_argument('--budget', type=float, default=1e7, help='window starts of the blind versions')
p.add_argument('--resume', action='store_true', help='continue from OUT/<version>/r.npy')
p.add_argument('--ppe', type=int, help='paths per ensemble (default: 1500, blind versions 3000)')
args = p.parse_args()


def summary():
    import csv
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    for v in VERSIONS:
        f = os.path.join(args.out, v, 'history.csv')
        if not os.path.exists(f):
            continue
        rows = list(csv.DictReader(open(f)))
        it = [int(r['iteration']) for r in rows]
        a1.plot(it, [float(r['plateau_offset']) for r in rows], 'o-', ms=3, label=v)
        a2.plot(it, [float(r['rms_per_bin']) for r in rows], 'o-', ms=3, label=v)
        last = rows[-1]
        print(f'{v:14s} {last["iteration"]:>6s} iterations: plateau offset {float(last["plateau_offset"]):+.4f}, '
              f'RMS per bin {float(last["rms_per_bin"]):.4f}, converged: {last["converged"]}')
    a1.axhline(0, color='k', lw=0.8)
    a1.set(xlabel='iteration', ylabel='mean (fit - shooting), 4 plateau points', title='plateau level')
    a2.set(xlabel='iteration', ylabel='RMS(fit - exact) per (y, x) bin', yscale='log', title='whole map')
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'summary.png'), dpi=110)
    print('saved', os.path.join(args.out, 'summary.png'))


if args.version == 'summary':
    summary()
    sys.exit()
if not args.sim or not os.path.isdir(args.sim):
    sys.exit('give the staple run folder with --sim (or $SIM_ISTARZ)')

import cupy as cp
import optimalrcs
import plots
import seed_runs
import tis_data
import toy2d
plt.show = lambda *a, **k: None
out = os.path.join(args.out, args.version)
os.makedirs(out, exist_ok=True)
t_start = time.time()


def log(msg):
    print(f'[{time.time() - t_start:7.0f} s] {msg}', flush=True)


# ------------------------------------------------------------------------------------------------
# exact committor (reference grid; computed with toy2d if the file is not there)
# ------------------------------------------------------------------------------------------------
ref_file = os.path.join(REPO, 'examples', 'figures_by_potential', '2d_zpot', 'brownian_g10', 'reference_and_analyses', 'committor_reference_h0.001.npz')
if os.path.exists(ref_file):
    ref = dict(np.load(ref_file))
else:
    log('no reference file: solving the backward equation on a grid (h = 0.002, toy2d)')
    pot = toy2d.load_potential('sjoelbak_angle:RectangularGridWithAngledBarrierPotential', {'Lx': 0.3, 'angle': 45})
    ref = toy2d.exact_committor_brownian(pot, 0.05, 10.0, DT, LAMBDA_A, LAMBDA_B, h=0.002,
                                         y_range=(-0.04, 0.94), x_range=(-0.04, 0.34))
    log(f'reference: {ref["steps_per_ab"]:.3g} steps per A->B (expected about 4.4e7)')


def exact_at(y, x):
    return toy2d.interp_grid(ref, y, x)


# ------------------------------------------------------------------------------------------------
# data versions
# ------------------------------------------------------------------------------------------------
def load_pool(ppe, seed):
    parser = argparse.ArgumentParser()
    tis_data.add_data_arguments(parser)
    with contextlib.redirect_stdout(io.StringIO()):
        data = tis_data.prepare_data(parser.parse_args(
            [args.sim, '--reversed', '--paths-per-ensemble', str(ppe), '--seed', str(seed)]))
    return dict(y=data.X_all[:, 0].astype(float), x=data.X_all[:, 1].astype(float), A=data.boundary0,
                B=data.boundary1, i=data.i_traj, t=data.t_traj.astype(float), w=data.path_weights,
                g=data.group_traj)


def qn_bins_from_base(d, n_bins):
    f = os.path.join(args.out, 'base', 'r.npy')
    if not os.path.exists(f):
        sys.exit(f'{args.version} needs the base fit first ({f})')
    r = np.load(f).astype(float)
    assert len(r) == len(d['y']), 'base fit and data differ'
    qn = plots.transform_q2qn(cp.asarray(r), i_traj=cp.asarray(d['i']), t_traj=cp.asarray(d['t']),
                              w_traj=cp.asarray(d['w']), dt_sim=DT)
    qn = np.asarray(qn)
    return np.clip((qn / qn.max() * n_bins).astype(int), 0, n_bins - 1)


def build(version):
    """Frames, weights and segments of a version, as a dict like load_pool's."""
    if not version.startswith('blind_'):
        d = load_pool(args.ppe or 1500, 0)
        if version == 'base':
            return d
        qn_bin = qn_bins_from_base(d, 30)
        if version == 'flat_qn':
            rho = np.bincount(qn_bin, weights=d['w'], minlength=30)
            g = np.clip(rho.mean() / np.maximum(rho, 1e-12), 0, 20)
            w = d['w'] * g[qn_bin]
            d['w'] = w / w.mean()
            return d
        _, g_idx = np.unique(d['g'], return_inverse=True)            # select_paths
        visits = np.zeros((g_idx.max() + 1, 30), bool)
        visits[g_idx, qn_bin] = True
        n_bin = visits.sum(0)
        rarest = np.where(visits, n_bin[None, :], np.iinfo(int).max).min(1)
        keep_g = np.random.default_rng(0).random(len(rarest)) < np.minimum(1, np.quantile(n_bin, 0.25) / rarest)
        keep = keep_g[g_idx]
        log(f'select_paths: {keep_g.mean():.0%} of the paths kept')
        return {k: v[keep] for k, v in d.items()}
    d = load_pool(args.ppe or 3000, 1)
    y, n = d['y'], len(d['y'])
    can = ~(d['A'] | d['B'])
    can[:-1] &= d['i'][1:] == d['i'][:-1]
    can[-1] = False
    edges = np.linspace(LAMBDA_A, LAMBDA_B, 41)
    k = np.clip(np.digitize(y, edges) - 1, 0, 39)
    n_b = np.bincount(k[can], minlength=40).astype(float)
    rho = np.bincount(k[can], weights=d['w'][can], minlength=40)
    if version == 'blind_flat_y':
        lo, hi = 0.0, rho.max()
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.sum(n_b * np.minimum(1, mid / np.maximum(rho, 1e-12))) < args.budget else (lo, mid)
        p_b = np.minimum(1, hi / np.maximum(rho, 1e-12))
    else:
        p_b = np.full(40, min(1.0, args.budget / can.sum()))
    flag = can & (np.random.default_rng(0).random(n) < p_b[k])
    keep = flag.copy()
    keep[1:] |= flag[:-1]
    new_seg = keep.copy()
    new_seg[1:] &= ~flag[:-1]
    seg = np.cumsum(new_seg) - 1
    ix = np.flatnonzero(keep)
    d = {k_: v[ix] for k_, v in d.items()}
    d['i'] = seg[ix]
    log(f'{version}: {flag.sum():,} window starts, {len(ix):,} frames in {d["i"].max() + 1:,} segments')
    return d


log(f'{args.version}: loading {args.sim}')
d = build(args.version)
y, x, A, B = d['y'], d['x'], d['A'], d['B']
free = ~(A | B)
blind = args.version.startswith('blind_')
log(f'{len(y):,} frames, {len(np.unique(d["i"])):,} segments, {len(np.unique(d["g"])):,} paths')

q_exact = exact_at(y, x)
q_exact[A], q_exact[B] = 0.0, 1.0
near = [np.flatnonzero(free & ((y - y0) ** 2 + (x - x0) ** 2 < 0.01 ** 2)) for (y0, x0), _ in SHOOT]
ey, ex = np.linspace(LAMBDA_A, LAMBDA_B, 81), np.linspace(-0.01, 0.31, 33)
bm = seed_runs.BinnedMean((y, x), (ey, ex), free, 20)
Qe = bm(q_exact)
trans = (Qe > 0.05) & (Qe < 0.95)


def evaluate(r):
    at_points = [float(r[m].mean()) for m in near]
    offset = float(np.mean([a - qs for a, (_, qs) in zip(at_points, SHOOT)]))
    rms = float(np.sqrt(np.nanmean((bm(r) - Qe)[trans] ** 2)))
    return at_points, offset, rms


# ------------------------------------------------------------------------------------------------
# fit in chunks until the plateau stops moving
# ------------------------------------------------------------------------------------------------
np.random.seed(0)
cp.random.seed(0)
r_file, hist_file = os.path.join(out, 'r.npy'), os.path.join(out, 'history.csv')
seed_r, done = None, 0
if args.resume and os.path.exists(r_file):
    seed_r = np.load(r_file).astype(float)
    assert len(seed_r) == len(y), 'saved fit and data differ'
    rows = open(hist_file).read().strip().split('\n')
    done = int(rows[-1].split(',')[0])
    log(f'resuming at {done} iterations')
else:
    with open(hist_file, 'w') as f:
        f.write('iteration,envelope,plateau_offset,q1,q2,q3,q4,rms_per_bin,converged\n')

q = optimalrcs.CommittorNE(boundary0=A, boundary1=B, i_traj=d['i'], t_traj=d['t'], path_weights=d['w'],
                           group_traj=d['g'], seed_r=seed_r)
if not blind:
    q.set_fixed_traj_length_trap(A | B, 2 ** 16)
del d
X_cv = cp.asarray(np.column_stack([y, x]))
metrics = ('iter', 'delta_r2', 'delta_x', 'time_elapsed') if blind else \
          ('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped', 'delta_x', 'time_elapsed')
fit_kw = dict(ny=6, print_step=500, lazy_basis=True, metrics_print=metrics,
              save_min_metric='delta_r2' if blind else 'max_z_zq_stopped')
q_map = seed_runs.committor_map([seed_runs.plane(y, ey, 'y', x, ex, 'x')], mask=free, q_ref=q_exact)
stage_plots = ('metrics',) if blind else ('metrics', 'tis', 'feps')
plot_metrics = ['delta_r2', 'delta_x'] if blind else ['delta_r2', 'max_z_zq_stopped', 'delta_x']

stages = [] if done else [(1000, 100000)]
n_main = args.max_iter - max(0, done - 1000)
stages += [(args.chunk, 0.01)] * max(0, -(-n_main // args.chunk))
previous, calm = None, 0
for n_iter, env in stages:
    optimalrcs.envelope_scale = env
    q.fit_transform(lambda: X_cv[:, np.random.randint(2)], gamma=0.1, max_iter=n_iter, **fit_kw)
    done += n_iter
    r = cp.asnumpy(q.r_traj)
    at_points, offset, rms = evaluate(r)
    if env < 1:
        calm = calm + 1 if previous is not None and abs(np.mean(at_points) - previous) < args.tol else 0
        previous = np.mean(at_points)
    converged = calm >= 2
    np.save(r_file, r.astype(np.float32))
    with open(hist_file, 'a') as f:
        f.write(f'{done},{env},{offset:.5f},' + ','.join(f'{a:.5f}' for a in at_points) + f',{rms:.5f},{converged}\n')
    seed_runs.save_and_show(q, out, f'it{done}', stage_plots, dict(metrics=plot_metrics), {'q_map': q_map})
    log(f'{done} iterations (envelope {env}): plateau offset {offset:+.4f}, RMS per bin {rms:.4f}, '
        f'fit/shooting ' + ' '.join(f'{a:.3f}/{qs:.3f}' for a, (_, qs) in zip(at_points, SHOOT))
        + (' -> converged' if converged else ''))
    if converged:
        break
log(f'done: {args.version}, {done} iterations')
