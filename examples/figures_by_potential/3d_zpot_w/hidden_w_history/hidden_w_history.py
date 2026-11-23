"""Hidden slow coordinate: can NPNEq histories of (y, x) recover the committor's dependence on an unobserved w?

System: the gate of the loop (`loop_zw`, ase_systems 'zw_br', toytis zpot_w.ZChannelsW): 2D Brownian Z potential in
(y, x) plus a double well in w (wells w = +-0.1, 3 kT barrier) whose well moves the ridge, so the exact committor
q(y, x, w) depends on w (well difference about 0.1 in q). Data: the 20 000-cycle StapleTIS production of loop_zw
(frozen learned lambda, infinite swap), read from its archive; pieces cut at M, forward + mirrored (overdamped, so
mirrored pieces are valid), path_f weights ('rejections', the unbiased one for committor pieces).

The fit never sees w (or lambda, which is a function of y, x and w). Variants:

    full        (y, x, u, v, w), no history      upper bound (u, v = (y +- x)/sqrt2, the loop teacher's ridge-normal
                                                 features; functions of (y, x), so allowed in every variant)
    yx          (y, x, u, v), no history         baseline: the best function of the current (y, x)
    yx_h8       (y, x, u, v), history 0 1 2 4 8
    yx_h64      (y, x, u, v), history 0 1 2 4 8 16 32 64
    yx_h256     (y, x, u, v), history 0 1 2 ... 256 (log-spaced)
    yx_h64_2t   (y, x, u, v), history as yx_h64, two history types drawn at random
    full_no_uv  (y, x, w): the first run, without u, v (plateau bias -0.16; kept for reference)

Scores are also computed against the full fit (report): r2_w / corr_w of the w-dependent part relative to it.

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
# u, v = (y +- x)/sqrt2: the ridge-normal coordinates the loop's teacher offers on Z and the gate (teacher.features);
# without them the polynomial basis cannot build the diagonal ridge and the fit carries a plateau bias of about -0.16
# at exact q 0.2-0.4 (results/full_no_uv: RMS vs exact 0.109), larger than the w effect. Functions of (y, x) only.
YX = ('y', 'x', 'u', 'v')
VARIANTS = {
    'full': dict(cvs=YX + ('w',), history=None),
    'yx': dict(cvs=YX, history=None),
    'yx_h8': dict(cvs=YX, history=LOG_LAGS[:5]),
    'yx_h64': dict(cvs=YX, history=LOG_LAGS[:8]),
    'yx_h256': dict(cvs=YX, history=LOG_LAGS),
    'yx_h64_2t': dict(cvs=YX, history=LOG_LAGS[:8], history_type=['y(t-d),r(t-d)', 'y(t-d),y(t)']),
    'full_no_uv': dict(cvs=('y', 'x', 'w'), history=None),            # the first run, kept for reference
    'full_nowarm': dict(cvs=YX + ('w',), history=None),               # with u, v, no warm-up, kept for reference
}
CV_COLS = {'lam': 0, 'y': 1, 'x': 2, 'w': 3}       # columns of the run's order.txt after time


def cv_matrix(X, names):
    cols = []
    for c in names:
        if c == 'u':
            cols.append((X[:, 1] + X[:, 2]) / np.sqrt(2))
        elif c == 'v':
            cols.append((X[:, 1] - X[:, 2]) / np.sqrt(2))
        else:
            cols.append(X[:, CV_COLS[c]])
    return np.column_stack(cols)


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


def projection(d, n_bins=80, ref=None):
    """Weighted mean of the reference (default q_exact) in (y, x) bins over the interior frames: the best function of
    (y, x) on these data."""
    inner = ~d['A'] & ~d['B']
    y, x, w = d['X'][:, 1], d['X'][:, 2], d['path_weights']
    qe = d['q_exact'] if ref is None else ref
    ey = np.quantile(y[inner], np.linspace(0, 1, n_bins + 1))
    ex = np.quantile(x[inner], np.linspace(0, 1, n_bins + 1))
    iy = np.clip(np.searchsorted(ey, y, side='right') - 1, 0, n_bins - 1)
    ix = np.clip(np.searchsorted(ex, x, side='right') - 1, 0, n_bins - 1)
    cell = iy * n_bins + ix
    num = np.bincount(cell[inner], weights=(w * qe)[inner], minlength=n_bins ** 2)
    den = np.bincount(cell[inner], weights=w[inner], minlength=n_bins ** 2)
    qp = np.where(den[cell] > 0, num[cell] / np.maximum(den[cell], 1e-300), qe)
    return qp


def scores(d, q, qp, ref=None):
    """Against q_exact, or against `ref` (the w-aware fit) with qp its (y, x) projection."""
    inner = ~d['A'] & ~d['B']
    w = d['path_weights'][inner]
    qe = (d['q_exact'] if ref is None else ref)[inner]
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


def fit(name, max_iter, print_step, shift, gamma, ny, seed, lazy_basis, warmup_iter=2000, warmup_envelope=1e6,
        envelope=0.01, report_every=2000, tag=''):
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
    X = cp.asarray(cv_matrix(d['X'], v['cvs']))

    def comp_y():
        return X[:, np.random.randint(X.shape[1])]

    t0 = time.time()
    common_kw = dict(history_delta_t=v['history'], history_type=v.get('history_type'),
                     history_shift_type=shift if v['history'] else None, gamma=gamma, ny=ny, print_step=print_step,
                     metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped', 'delta_x',
                                    'time_elapsed'),
                     save_min_metric='max_z_zq_stopped', lazy_basis=lazy_basis)
    # warm-up with global updates (envelope 1e6), as the staple notebooks and qq_-Copy1: from q = 1/2 the local
    # envelope alone (0.01) cannot move the whole plateau; without it the full fit stalls at zqs_z 6.2 and a plateau
    # bias of -0.15, while the exact committor scores 2.67 on the same data (results/full_nowarm)
    if warmup_iter > 0:
        orc.envelope_scale = warmup_envelope
        q.fit_transform(comp_y, max_iter=warmup_iter, **common_kw)
        print(f'warm-up done ({warmup_iter} iterations, envelope {warmup_envelope})', flush=True)
    orc.envelope_scale = envelope
    # main stage in chunks, each scored against the exact committor (and the plateau bias, weighted Dirichlet
    # energy sum w (q(t+1) - q(t))^2: the exact committor has 127.5 on these data)
    inner = ~d['A'] & ~d['B']
    same = d['i_traj'][1:] == d['i_traj'][:-1]
    plateau = inner & (d['q_exact'] > 0.2) & (d['q_exact'] < 0.4)
    trace = []
    done = 0
    while done < max_iter:
        n = min(report_every, max_iter - done)
        q.fit_transform(comp_y, max_iter=n, **common_kw)
        done += n
        r_now = cp.asnumpy(q.r_traj).astype(float)
        sc = scores(d, r_now, qp)
        dirichlet = float(np.sum(d['path_weights'][:-1] * np.where(same, np.diff(r_now), 0) ** 2))
        bias = float(np.average((r_now - d['q_exact'])[plateau], weights=d['path_weights'][plateau]))
        zq = q.metrics_history.get('max_z_zq_stopped', [np.nan])[-1]
        trace.append(dict(iter=done, rms=sc['rms'], r2_w=sc['r2_w'], plateau_bias=bias, dirichlet=dirichlet,
                          zqs_z=float(zq)))
        print(f'CHECK {name} main iter {done}: rms vs exact {sc["rms"]:.4f}, plateau bias {bias:+.3f}, '
              f'dirichlet {dirichlet:.1f} (exact 127.5), zqs_z {zq:.2f}, r2_w {sc["r2_w"]:.3f}', flush=True)
    minutes = (time.time() - t0) / 60
    r = cp.asnumpy(q.r_traj)
    r_min = cp.asnumpy(q.r_traj_min_sd_zq)
    zq = q.metrics_history.get('max_z_zq_stopped', [np.nan])
    out = dict(variant=name + tag, cvs=list(v['cvs']), history=v['history'], history_type=v.get('history_type'),
               shift=shift if v['history'] else None, max_iter=max_iter, warmup_iter=warmup_iter,
               warmup_envelope=warmup_envelope, envelope=envelope, gamma=gamma, ny=ny, seed=seed,
               minutes=minutes, frames=int(len(r)), trace=trace, zqs_z_final=float(zq[-1]), zqs_z_min=float(np.min(zq)),
               final=scores(d, r, qp), at_min_zqs=scores(d, r_min, qp))
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, f'{name}{tag}.npz'), r=r.astype(np.float32), r_min=r_min.astype(np.float32),
             **{f'metric_{k}': np.asarray(vals) for k, vals in q.metrics_history.items()})
    with open(os.path.join(RESULTS, f'{name}{tag}.json'), 'w') as f:
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
    # reference 2: the w-aware fit itself. The NPNEq fit on these data carries a plateau bias (about -0.16 at exact q
    # 0.2-0.4) much larger than the w effect, so against q_exact every variant is dominated by it; against the full
    # fit, r2_w_full measures how much of the w-dependence the w-aware fit finds that a fit without w recovers.
    full_fn = os.path.join(RESULTS, 'full.npz')
    r_full = np.load(full_fn)['r'].astype(float) if os.path.isfile(full_fn) else None
    qp_full = projection(d, ref=r_full) if r_full is not None else None
    lines = ['| variant | CVs | history | r2_w vs full fit | corr_w vs full fit | r2_w vs exact | RMS vs exact | '
             'floor of (y, x) | zqs_z final / min | minutes |', '|---|---|---|---|---|---|---|---|---|---|']
    for o in rows:
        s = o['final']
        if r_full is not None:
            r = np.load(os.path.join(RESULTS, f"{o['variant']}.npz"))['r'].astype(float)
            sf = scores(d, r, qp_full, ref=r_full)
            o['vs_full'] = sf
            vf = f"{sf['r2_w']:.3f} | {sf['corr_w']:.3f}"
        else:
            vf = '- | -'
        lines.append(f"| {o['variant']} | {', '.join(o['cvs'])} | {o['history'] or '-'} | {vf} | {s['r2_w']:.3f} | "
                     f"{s['rms']:.4f} | {s['rms_proj']:.4f} | "
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
    ap.add_argument('--warmup-iter', type=int, default=2000, help='iterations with global updates first')
    ap.add_argument('--warmup-envelope', type=float, default=1e6)
    ap.add_argument('--envelope', type=float, default=0.01, help='envelope scale of the main stage')
    ap.add_argument('--report-every', type=int, default=2000, help='main-stage iterations between checks vs exact')
    ap.add_argument('--tag', default='', help='suffix of the result files (test runs of one variant)')
    ap.add_argument('--variants', nargs='+', default=[v for v in VARIANTS if v not in ('full_no_uv', 'full_nowarm')],
                    help="with 'all': the variants to fit")
    a = ap.parse_args()
    fit_args = (a.max_iter, a.print_step, a.shift, a.gamma, a.ny, a.seed, a.lazy_basis, a.warmup_iter,
                a.warmup_envelope, a.envelope, a.report_every, a.tag)
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
