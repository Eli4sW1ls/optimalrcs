"""Hidden slow coordinate: can NPNEq histories of (y, x) recover the committor's dependence on an unobserved w?

System: the gate of the loop (`loop_zw`, ase_systems 'zw_br', toytis zpot_w.ZChannelsW): 2D Brownian Z potential in
(y, x) plus a double well in w (wells w = +-0.1, 3 kT barrier) whose well moves the ridge, so the exact committor
q(y, x, w) depends on w (well difference about 0.1 in q). Data: the 20 000-cycle StapleTIS production of loop_zw
(frozen learned lambda, infinite swap), read from its archive; pieces cut at M, forward + mirrored (overdamped, so
mirrored pieces are valid), path_f weights ('rejections', the unbiased one for committor pieces).

The fit never sees w (or lambda, which is a function of y, x and w). Variants:

    full        (y, x, w), no history            upper bound
    yx          (y, x), no history               baseline: the best function of the current (y, x)
    yx_h8       (y, x), history 0 1 2 4 8
    yx_h64      (y, x), history 0 1 2 4 8 16 32 64
    yx_h256     (y, x), history 0 1 2 ... 256 (log-spaced)
    yx_h64_2t   (y, x), history as yx_h64, two history types drawn at random

Scores, path-weighted over the frames outside A and B, against the exact grid committor (loop/exact_zw.py, h 0.004):

    rms          RMS(q - q_exact)
    rms_proj     RMS(q_proj - q_exact), q_proj = the weighted mean of q_exact in (y, x) bins: the floor of any
                 function of the current (y, x) on these frames
    r2_w         1 - E[(q_exact - q)^2] / E[(q_exact - q_proj)^2]: the fraction of the w-driven part of the committor
                 that the fit recovers (about 0 for a function of (y, x) alone, 1 for the exact committor)
    corr_w       correlation of (q - q_proj) with (q_exact - q_proj)
    zqs_z        the stopped Z_q criterion of the fit (final and minimum over the print steps)

Usage:
    python hidden_w_history.py prepare [--paths-per-ensemble 250]
    python hidden_w_history.py fit yx_h64 [--max-iter 10000] [--shift r(t)]
    python hidden_w_history.py report
    python hidden_w_history.py all              prepare (if needed), every variant, report
Outputs next to this script: data.npz, results/<variant>.npz and .json, summary.md, summary.png.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
REPO = os.path.dirname(EXAMPLES)
RUN = os.path.join(EXAMPLES, 'figures_by_potential', '_shared', 'param_lambda', 'op_loop', 'loop_zw', 'production')
DATA = os.path.join(HERE, 'data.npz')
RESULTS = os.path.join(HERE, 'results')
sys.path.insert(0, EXAMPLES)
sys.path.insert(0, os.path.join(EXAMPLES, 'param_lambda'))

LOG_LAGS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
VARIANTS = {
    'full': dict(cvs=('y', 'x', 'w'), history=None),
    'yx': dict(cvs=('y', 'x'), history=None),
    'yx_h8': dict(cvs=('y', 'x'), history=LOG_LAGS[:5]),
    'yx_h64': dict(cvs=('y', 'x'), history=LOG_LAGS[:8]),
    'yx_h256': dict(cvs=('y', 'x'), history=LOG_LAGS),
    'yx_h64_2t': dict(cvs=('y', 'x'), history=LOG_LAGS[:8], history_type=['y(t-d),r(t-d)', 'y(t-d),y(t)']),
}
CV_COLS = {'lam': 0, 'y': 1, 'x': 2, 'w': 3}       # columns of the run's order.txt after time


def prepare(paths_per_ensemble, burn_in_fraction):
    import infretis_data
    from loop import ase_systems
    cvs, order, time_, weights, paths, settings = infretis_data.load_infretis_data(
        RUN, weighting='rejections', h5_file=os.path.join(RUN, 'archive.h5'), burn_in_fraction=burn_in_fraction,
        paths_per_ensemble=paths_per_ensemble, reversed_pieces=True)
    X = np.concatenate(cvs).astype(np.float64)
    pw = np.concatenate([np.full(len(o), w, float) for o, w in zip(order, weights)])
    pw /= pw.mean()
    i_traj = np.concatenate([np.full(len(o), k, np.int64) for k, o in enumerate(order)])
    t_traj = np.concatenate(time_).astype(np.float64)
    group = np.concatenate([np.full(len(o), n, np.int64) for o, n in zip(order, paths)])
    la, lb = settings['interfaces'][0], settings['interfaces'][-1]
    A, B = X[:, 0] <= la, X[:, 0] >= lb
    s = ase_systems.get('zw_br')
    q_exact = ase_systems.exact_committor(s)(X[:, 1:4])
    q_exact[A], q_exact[B] = 0.0, 1.0
    np.savez(DATA, X=X, path_weights=pw, i_traj=i_traj, t_traj=t_traj, group_traj=group, A=A, B=B,
             q_exact=q_exact, interfaces=np.array(settings['interfaces']),
             paths_per_ensemble=paths_per_ensemble, burn_in_fraction=burn_in_fraction)
    print(f'{len(order)} pieces, {len(X)} frames ({(~A & ~B).sum()} outside A and B) -> {DATA}')


def load_data():
    return dict(np.load(DATA))


def projection(d, n_bins=80):
    """Weighted mean of q_exact in (y, x) bins over the interior frames: the best function of (y, x) on these data."""
    inner = ~d['A'] & ~d['B']
    y, x, w, qe = d['X'][:, 1], d['X'][:, 2], d['path_weights'], d['q_exact']
    ey = np.quantile(y[inner], np.linspace(0, 1, n_bins + 1))
    ex = np.quantile(x[inner], np.linspace(0, 1, n_bins + 1))
    iy = np.clip(np.searchsorted(ey, y, side='right') - 1, 0, n_bins - 1)
    ix = np.clip(np.searchsorted(ex, x, side='right') - 1, 0, n_bins - 1)
    cell = iy * n_bins + ix
    num = np.bincount(cell[inner], weights=(w * qe)[inner], minlength=n_bins ** 2)
    den = np.bincount(cell[inner], weights=w[inner], minlength=n_bins ** 2)
    qp = np.where(den[cell] > 0, num[cell] / np.maximum(den[cell], 1e-300), qe)
    return qp


def scores(d, q, qp):
    inner = ~d['A'] & ~d['B']
    w, qe = d['path_weights'][inner], d['q_exact'][inner]
    q, qp = q[inner], qp[inner]
    wm = lambda a: float(np.sum(w * a) / np.sum(w))
    e_fit, e_proj = wm((q - qe) ** 2), wm((qp - qe) ** 2)
    a, b = q - qp, qe - qp
    ma, mb = wm(a), wm(b)
    corr = wm((a - ma) * (b - mb)) / np.sqrt(wm((a - ma) ** 2) * wm((b - mb) ** 2) + 1e-300)
    return dict(rms=np.sqrt(e_fit), rms_proj=np.sqrt(e_proj), r2_w=1 - e_fit / e_proj, corr_w=float(corr))


def _optimalrcs_cupy():
    cdir = os.path.join(REPO, 'optimalrcs-cupy')
    sys.path.insert(0, cdir)
    spec = importlib.util.spec_from_file_location('optimalrcs_cupy', os.path.join(cdir, 'optimalrcs.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit(name, max_iter, print_step, shift, gamma, ny, seed, lazy_basis):
    import cupy as cp
    orc = _optimalrcs_cupy()
    v = VARIANTS[name]
    d = load_data()
    qp = projection(d)
    np.random.seed(seed)
    cp.random.seed(seed)
    q = orc.CommittorNE(boundary0=d['A'], boundary1=d['B'], i_traj=d['i_traj'], t_traj=d['t_traj'],
                        path_weights=d['path_weights'], group_traj=d['group_traj'])
    q.set_fixed_traj_length_trap(d['A'] | d['B'], 2 ** 16)        # as load_tis_data_cupy.py
    X = cp.asarray(d['X'][:, [CV_COLS[c] for c in v['cvs']]])

    def comp_y():
        return X[:, np.random.randint(X.shape[1])]

    t0 = time.time()
    q.fit_transform(comp_y, history_delta_t=v['history'], history_type=v.get('history_type'),
                    history_shift_type=shift if v['history'] else None, gamma=gamma, ny=ny, max_iter=max_iter,
                    print_step=print_step,
                    metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped', 'delta_x',
                                   'time_elapsed'),
                    save_min_metric='max_z_zq_stopped', lazy_basis=lazy_basis)
    minutes = (time.time() - t0) / 60
    r = cp.asnumpy(q.r_traj)
    r_min = cp.asnumpy(q.r_traj_min_sd_zq)
    zq = q.metrics_history.get('max_z_zq_stopped', [np.nan])
    out = dict(variant=name, cvs=list(v['cvs']), history=v['history'], history_type=v.get('history_type'),
               shift=shift if v['history'] else None, max_iter=max_iter, gamma=gamma, ny=ny, seed=seed,
               minutes=minutes, frames=int(len(r)), zqs_z_final=float(zq[-1]), zqs_z_min=float(np.min(zq)),
               final=scores(d, r, qp), at_min_zqs=scores(d, r_min, qp))
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, f'{name}.npz'), r=r.astype(np.float32), r_min=r_min.astype(np.float32),
             **{f'metric_{k}': np.asarray(vals) for k, vals in q.metrics_history.items()})
    with open(os.path.join(RESULTS, f'{name}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = load_data()
    qp = projection(d)
    rows = []
    for name in VARIANTS:
        fn = os.path.join(RESULTS, f'{name}.json')
        if os.path.isfile(fn):
            rows.append(json.load(open(fn)))
    if not rows:
        print('no results yet')
        return
    lines = ['| variant | CVs | history | r2_w | corr_w | RMS vs exact | floor of (y, x) | zqs_z final / min | minutes |',
             '|---|---|---|---|---|---|---|---|---|']
    for o in rows:
        s = o['final']
        lines.append(f"| {o['variant']} | {', '.join(o['cvs'])} | {o['history'] or '-'} | {s['r2_w']:.3f} | "
                     f"{s['corr_w']:.3f} | {s['rms']:.4f} | {s['rms_proj']:.4f} | "
                     f"{o['zqs_z_final']:.2f} / {o['zqs_z_min']:.2f} | {o['minutes']:.0f} |")
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(HERE, 'summary.md'), 'w') as f:
        f.write(f"Data: {len(d['X'])} frames, {int(d['paths_per_ensemble'])} paths per ensemble, "
                f"burn-in {float(d['burn_in_fraction'])}.\n\n{text}\n")

    # the w-driven part of the committor, (q_exact - q_proj) against (q - q_proj), per variant
    inner = ~d['A'] & ~d['B']
    rng = np.random.default_rng(0)
    pick = rng.choice(np.flatnonzero(inner), size=min(20000, int(inner.sum())), replace=False,
                      p=d['path_weights'][inner] / d['path_weights'][inner].sum())
    fig, axes = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3.2), sharex=True, sharey=True, squeeze=False)
    for ax, o in zip(axes[0], rows):
        r = np.load(os.path.join(RESULTS, f"{o['variant']}.npz"))['r']
        ax.scatter((d['q_exact'] - qp)[pick], (r - qp)[pick], s=1, alpha=0.3)
        ax.plot([-0.3, 0.3], [-0.3, 0.3], 'k--', lw=0.8)
        ax.set_title(f"{o['variant']}: r2_w {o['final']['r2_w']:.2f}", fontsize=9)
        ax.set_xlabel('q_exact - q_proj')
    axes[0][0].set_ylabel('q_fit - q_proj')
    axes[0][0].set_xlim(-0.3, 0.3)
    axes[0][0].set_ylim(-0.3, 0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'summary.png'), dpi=130)
    print(f"-> {os.path.join(HERE, 'summary.md')}, summary.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('stage', choices=['prepare', 'fit', 'report', 'all'])
    ap.add_argument('variant', nargs='?', choices=list(VARIANTS))
    ap.add_argument('--paths-per-ensemble', type=int, default=250)
    ap.add_argument('--burn-in-fraction', type=float, default=0.1)
    ap.add_argument('--max-iter', type=int, default=10000)
    ap.add_argument('--print-step', type=int, default=500)
    ap.add_argument('--shift', default='r(t)', help="history_shift_type for frames with t - d before the piece start")
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--ny', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lazy-basis', action='store_true')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), help="with 'all': the variants to fit")
    a = ap.parse_args()
    fit_args = (a.max_iter, a.print_step, a.shift, a.gamma, a.ny, a.seed, a.lazy_basis)
    if a.stage == 'prepare':
        prepare(a.paths_per_ensemble, a.burn_in_fraction)
    elif a.stage == 'fit':
        if a.variant is None:
            ap.error('fit needs a variant')
        fit(a.variant, *fit_args)
    elif a.stage == 'report':
        report()
    else:
        if not os.path.isfile(DATA):
            prepare(a.paths_per_ensemble, a.burn_in_fraction)
        for name in a.variants:
            if os.path.isfile(os.path.join(RESULTS, f'{name}.json')):
                print(f'{name}: done, skipped')
                continue
            fit(name, *fit_args)
        report()


if __name__ == '__main__':
    main()
