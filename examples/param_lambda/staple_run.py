#!/usr/bin/env python3
"""A toytis staple (i*) run with an exported learned lambda as the order parameter (toytis_adapter).

Settings are those of the run the lambda was trained on (z_br: sim_istarz_br_g10, cc_br: test_cc_br); only the order
parameter and the interfaces change. The states stay y <= lambda_A, y >= lambda_B (lambda = y outside them).

    python staple_run.py z_br  --model m3-fast --placement max --check   # interface diagnostics, no run
    python staple_run.py z_br  --model m3-fast --seed 1           # -> toytis/simulations/sim_istarz_br_g10_lam-m3-fast_s1
    python staple_run.py cc_br --model m3 --seed 1 --max-cycles 2000
    python staple_run.py z_br  --model m3-fast --seed 1 --restart --max-cycles 150000
    python staple_run.py z_br  --model none --seed 1             # control: plain y, the training run's interfaces

Interfaces (lambda_A and lambda_B always first and last). All are lambda isosurfaces; --placement sets their levels:
  lam   (default) n equally spaced in lambda. lambda is log1p(q/eps)/log1p(1/eps)-scaled (eps = 1e-3), so this is
        roughly equal spacing in ln q above 1e-3 (11 interfaces: q ~ 0.001 ... 0.5).
  y     equally spaced in Y(lambda), the weighted median y of the training-run frames on that isosurface: staple path
        lengths per ensemble like those of the y run. Fails where many isosurfaces share one y (the Z ridge).
  max   equally spaced in the distance that counts each step with the larger of its Y and lambda increments
        (both scaled to [0, 1]): lambda-like across a barrier, y-like on a committor plateau.
  blend (1 - alpha) Y + alpha lambda.
y, max and blend need the frame cache of the training run; --check prints the values for --interfaces elsewhere.
A natural-coordinate model (exported/<system>_<family>-nat.npz, `run.py natural`) with the default `lam` placement
spaces the interfaces equally in that coordinate; add --max-ratio R to bound the crossing probabilities.
  --q-levels q1 q2 ...  the interior interfaces at these committor levels instead.
--check prints, for each interface on the held-out test paths of the training run (frames.npz): its q level, how
often an upward crossing of it also passes the next one in the same step, and how far (in RMS MD steps of y) its
first crossings lie from lambda_A. Interfaces within ~1-2 steps of A, or skipped by many steps, are too close.

Long runs in their own unit, e.g.
    systemd-run --user --unit=lam-z1 --collect -p MemoryMax=8G -p MemorySwapMax=0 --working-directory=$PWD \\
        python staple_run.py z_br --model m3-fast --seed 1
"""
import os

# single-threaded BLAS: the one-frame MLP is ~5x slower with threads (per-call overhead)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import toytis_adapter  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402

TOYTIS = Path(common.TOYTIS)

COMMON_SETTINGS = dict(simtype='i*', method='load', max_len=200000, temperature=0.05, friction=10.0, high_friction=True,
                       p_shoot=0.9, include_stateB=False, prime_both_starts=True, snake_Lmax=0, max_paths=5, dim=2,
                       mass=1, permeability=False, zero_left=0.1, v_ord=False)
SEED_INTERFACES = {'z_br': [0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85],
                   'cc_br': [0.15, 0.22, 0.29, 0.36, 0.43, 0.5, 0.57, 0.64, 0.71, 0.78, 0.85]}
SYSTEMS = {
    # name: (run directory prefix, dt, max_cycles of the training run, potential factory)
    'z_br': ('sim_istarz_br_g10', 0.002, 100000,
             lambda: __import__('potentials.sjoelbak_angle', fromlist=['x']).RectangularGridWithAngledBarrierPotential(
                 Lx=0.3, angle=45)),
    'cc_br': ('sim_istar_cc_br', 0.005, 2000,
              lambda: __import__('potentials.curved_channel', fromlist=['x']).CurvedChannelDoubleWell()),
}


def model_path(system, model):
    p = os.path.join(HERE, 'exported', f'{system}_{model}.npz')
    if not os.path.isfile(p):
        raise SystemExit(f'no exported model {p} (run.py export {system})')
    return p


def q_of_lambda(m, lam):
    """Committor level of an interior lambda away from the ramps (inverse of LambdaModel.scale)."""
    return m.h_of_lambda(lam)


def cap_ratio(m, intfs, R):
    """Insert interfaces until neighbouring committor levels differ by at most a factor R: h_k+1 / h_k <= R below
    h = 0.5 and (1 - h_k) / (1 - h_k+1) <= R above. For a martingale q the probability to reach the next level
    before falling back to the previous one is set by the levels alone (about 1/R at small q)."""
    h = list(m.h_of_lambda(intfs[1:-1]))
    changed = True
    while changed:
        changed = False
        for k in range(len(h) - 1):
            a, b = h[k], h[k + 1]
            if b <= 0.5 and b / a > R:
                h.insert(k + 1, float(np.sqrt(a * b))); changed = True; break
            if a >= 0.5 and (1 - a) / (1 - b) > R:
                h.insert(k + 1, float(1 - np.sqrt((1 - a) * (1 - b)))); changed = True; break
    return [m.lambda_a] + [float(v) for v in m.scale(np.array(h))] + [m.lambda_b]


def _frames(system, path):
    """(lambda, y, weight) of the valid training-run frames of the train and test splits."""
    import data
    cfg = common.load_config(system)
    f = data.Frames(cfg)
    idx = np.flatnonzero(f.valid & (f.mask('train') | f.mask('test')))
    idx = idx[::max(1, len(idx) // 1_000_000)]
    return LambdaModel(path)(f.cv[idx]), f.cv[idx, cfg['data']['state_cv']], f.w[idx].astype(float)


def make_interfaces(system, path, m, n, q_levels=None, placement='lam', alpha=0.5):
    """Interfaces lambda_A, ..., lambda_B (see the module docstring)."""
    if q_levels:
        return [m.lambda_a] + [m.interface_for_q(q) for q in sorted(q_levels)] + [m.lambda_b]
    if placement == 'lam':
        return [float(v) for v in np.linspace(m.lambda_a, m.lambda_b, n)]
    # where each lambda isosurface lies in y: weighted median of y over the frames in narrow lambda bins
    lam, y, w = _frames(system, path)
    edges = np.unique(np.quantile(lam, np.linspace(0, 1, 401)))
    k = np.clip(np.searchsorted(edges, lam) - 1, 0, len(edges) - 2)
    lam_c, y_med = [], []
    for b in range(len(edges) - 1):
        sel = k == b
        if sel.sum() < 20:
            continue
        o = np.argsort(y[sel]); cw = np.cumsum(w[sel][o])
        lam_c.append(np.median(lam[sel])); y_med.append(y[sel][o][np.searchsorted(cw, cw[-1] / 2)])
    lam_c = np.r_[m.lambda_a, lam_c, m.lambda_b]
    y_med = np.maximum.accumulate(np.r_[m.lambda_a, np.clip(y_med, m.lambda_a, m.lambda_b), m.lambda_b])
    span = m.lambda_b - m.lambda_a
    Y, L = (y_med - m.lambda_a) / span, (lam_c - m.lambda_a) / span
    if placement == 'max':          # each step counts with the larger of its y and lambda increments
        c = np.r_[0, np.cumsum(np.maximum(np.diff(Y), np.diff(L)))]
    else:
        a = 0.0 if placement == 'y' else alpha
        c = (1 - a) * Y + a * L
    c = c + 1e-9 * np.arange(len(c))                        # strictly increasing for the inversion
    c = (c - c[0]) / (c[-1] - c[0])
    inner = np.interp(np.linspace(0, 1, n)[1:-1], c, lam_c)
    return [m.lambda_a] + [float(v) for v in inner] + [m.lambda_b]


def check(system, path, intfs):
    """Interface diagnostics on the test pieces of the training run (see the module docstring)."""
    import data
    cfg = common.load_config(system)
    f = data.Frames(cfg)
    idx = np.flatnonzero(f.take_pieces('test'))
    m = LambdaModel(path)
    cv, piece, w = f.cv[idx], f.piece[idx], f.w[idx].astype(float)
    s = cv[:, cfg['data']['state_cv']]
    lam = m(cv)
    i0 = np.flatnonzero(piece[1:] == piece[:-1]); i1 = i0 + 1
    step = np.sqrt(np.average((s[i1] - s[i0]) ** 2, weights=w[i0]))
    a, b = lam[i0], lam[i1]
    print(f'{system} {os.path.basename(path)}: test paths, {len(idx)} frames; RMS step of y {step:.4f}')
    print('interfaces (for --interfaces):', ' '.join(f'{v:.6f}' for v in intfs))
    print(' k  lambda_k   q_k       up-crossings  skip next   y - lambda_A at crossing: 5% / 50% (RMS steps)   median y')
    for k in range(1, len(intfs) - 1):
        lo, hi = intfs[k], intfs[k + 1]
        up = (a < lo) & (b >= lo)
        skip = np.sum(w[i0][up & (b >= hi)]) / max(np.sum(w[i0][up]), 1e-300)
        d = s[i1[up]] - m.lambda_a
        qq = np.quantile(d, [0.05, 0.5]) / step if up.any() else [np.nan, np.nan]
        ym = np.median(s[i1[up]]) if up.any() else np.nan
        print(f'{k:2d}  {lo:.5f}  {q_of_lambda(m, lo):.2e}  {up.sum():12d}  {skip:9.3f}   {qq[0]:6.1f} / {qq[1]:6.1f}'
              f'                    {ym:.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('system', choices=sorted(SYSTEMS))
    ap.add_argument('--model', default='m3-fast', help='exported/<system>_<model>.npz (m3, m3-fast, m1-gam, ...); '
                    '"none": the plain order parameter y with the training run\'s interfaces (control run)')
    ap.add_argument('--n-interfaces', type=int, default=11, help='including lambda_A and lambda_B (default 11)')
    ap.add_argument('--q-levels', type=float, nargs='+', help='interior interfaces at these committor levels')
    ap.add_argument('--placement', choices=['lam', 'y', 'max', 'blend'], default='lam',
                    help='interface placement (see the module docstring); y, max, blend need the frame cache')
    ap.add_argument('--alpha', type=float, default=0.5, help='blend: weight of lambda against y (default 0.5)')
    ap.add_argument('--max-ratio', type=float, help='insert interfaces until neighbouring committor levels differ by '
                    'at most this factor (see cap_ratio; e.g. 10 with a natural-coordinate model)')
    ap.add_argument('--interfaces', type=float, nargs='+', help='explicit interfaces in lambda units, lambda_A and '
                    'lambda_B included (e.g. printed by --check on a machine with the frame cache)')
    ap.add_argument('--check', action='store_true', help='interface diagnostics only, no run')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max-cycles', type=int)
    ap.add_argument('--work-dir', type=Path, help='relative to toytis/ (default simulations/<run>_lam-<model>_s<seed>); '
                                                  'a fresh run DELETES it first')
    ap.add_argument('--restart', action='store_true', help='continue from the work dir restart.pkl')
    a = ap.parse_args()

    plain = a.model == 'none'
    if plain:
        if a.check or a.q_levels:
            raise SystemExit('--model none uses the training run\'s interfaces on y (no --check, no --q-levels)')
        path, m, intfs = None, None, SEED_INTERFACES[a.system]
    else:
        path = model_path(a.system, a.model)
        m = LambdaModel(path)
        if a.interfaces:
            intfs = list(a.interfaces)
            if not (np.isclose(intfs[0], m.lambda_a) and np.isclose(intfs[-1], m.lambda_b) and np.all(np.diff(intfs) > 0)):
                raise SystemExit(f'--interfaces must increase from lambda_A = {m.lambda_a} to lambda_B = {m.lambda_b}')
            intfs[0], intfs[-1] = m.lambda_a, m.lambda_b
        else:
            intfs = make_interfaces(a.system, path, m, a.n_interfaces, a.q_levels, a.placement, a.alpha)
            if a.max_ratio:
                intfs = cap_ratio(m, intfs, a.max_ratio)
    if a.check:
        check(a.system, path, intfs)
        return

    prefix, dt, cycles, potential = SYSTEMS[a.system]
    sys.path[:0] = [str(TOYTIS), str(TOYTIS / 'notebooks')]
    from staple_test import open_run_directory, configure_logging
    work = TOYTIS / (a.work_dir or Path('simulations') / f'{prefix}_lam-{a.model}_s{a.seed}')
    work = open_run_directory(TOYTIS, work, a.restart)
    os.chdir(work)
    if not plain:
        local = work / os.path.basename(path)       # the run keeps its own copy of the model
        if not a.restart:
            shutil.copy2(path, local)
        toytis_adapter.install(str(local))          # before Simulation and before loading restart.pkl
    from simulation import Simulation

    np.random.seed(a.seed)
    settings = dict(COMMON_SETTINGS, interfaces=intfs, dt=dt, max_cycles=a.max_cycles or cycles,
                    potential=potential())
    if a.restart:
        settings['method'] = 'restart'
    logger = configure_logging(work / 'logging.log')
    logger.info(f'\ninterfaces = {intfs}\ntimestep = {dt}\n')
    if plain:
        logger.info(f'order parameter y (control run, no learned lambda); seed {a.seed}')
        print(work, '\ninterfaces', intfs)
    else:
        md5 = hashlib.md5(open(local, 'rb').read()).hexdigest()
        logger.info(f'learned lambda: {path} (copy {local.name}, md5 {md5}); interface q levels '
                    f'{[float(f"{q:.3g}") for q in q_of_lambda(m, intfs[1:-1])]}; seed {a.seed}')
        print(work, '\ninterfaces', np.round(intfs, 5), '\nq levels', [f'{q:.2e}' for q in q_of_lambda(m, intfs[1:-1])])
    sim = Simulation(settings)
    logger.info('Full settings:\n{}'.format(sim.settings))
    import logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    sim.run()
    if not plain:
        n_ood = sum(getattr(e.orderparameter, 'n_ood', 0) for e in sim.ensembles)
        logger.info(f'frames outside the training range (interior, all ensembles; this session): {n_ood}')


if __name__ == '__main__':
    main()
