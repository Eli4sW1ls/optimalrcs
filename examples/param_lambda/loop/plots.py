#!/usr/bin/env python3
"""Figures of a loop run (2D systems): <loop_dir>/figures/

    python -m loop.plots <loop name> [--no-tis]

- interfaces.png       one panel per pass: the potential (grey, 1 kT contours), the interfaces of that pass
                       (pass 0: the last infinit iteration's interfaces in y; pass k: the lambda_k isosurfaces, blue)
                       and the exact isocommittors at the same committor levels (dashed)
- maps.png             one row per pass: teacher q (bin means of the pass's labels on the pooled frames), the
                       model's committor m(x), the exact committor, and logit m - logit q_exact
- calibration.png      bin means of logit m (and logit q_teacher) against logit q_exact, per pass
- convergence.png      teacher RMS, isosurface spread and logit RMSE against the exact committor, per pass
- tis_pass<k>.png, feps_pass<k>.png   optimalrcs TIS diagnostics (stopped Z_q, obs-vs-pred) and path-weighted
                       free-energy profiles of model k's committor on the pass's own sampled pieces
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from loop import env, controller, pool as pool_mod, ase_systems  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402
import common  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
BLUE, RED, GRAY = '#2a78d6', '#e34948', '#f0efec'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div_blue_red', ['#1c5cab', '#86b6ef', GRAY, '#ef9a99', '#b42a2a'])
PASS_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']        # ordinal: pass index

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 9, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'legend.frameon': False})


def _logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q) - np.log1p(-q)


class Grid:
    def __init__(self, cfg, pool, n=(240, 140)):
        import toy2d
        from loop.ase_calcs import load_toytis_potential
        s = cfg['system_spec']
        self.s = s
        xs_all = np.concatenate([pool.load(k)[0]['cv'][::20, 2] for k in pool.passes()])
        self.ys = np.linspace(s['lambda_a'] - 0.06, s['lambda_b'] + 0.06, n[0])
        self.xs = np.linspace(*np.quantile(xs_all, [0.002, 0.998]), n[1])
        pot = load_toytis_potential(s['potential'], s['potential_kwargs'])
        V = toy2d.potential_grid(pot, self.ys, self.xs) / s['kT']
        self.V = V - np.nanmin(V)
        Y, X = np.meshgrid(self.ys, self.xs, indexing='ij')
        self.cv = np.c_[Y.ravel(), X.ravel()]
        self.shape = Y.shape
        self.mask = self.V > float(cfg.get('plots', {}).get('vmax_kT', 12))
        self.q_exact = ase_systems.exact_committor(s)(self.cv).reshape(self.shape)

    def field(self, values):
        return np.ma.array(np.asarray(values, float).reshape(self.shape), mask=self.mask)

    def background(self, ax):
        ax.contourf(self.ys, self.xs, np.clip(self.V, 0, 14).T, levels=np.arange(0, 14.5, 1), cmap='Greys', alpha=0.45)
        ax.contour(self.ys, self.xs, self.V.T, levels=np.arange(1, 14, 1), colors='0.6', linewidths=0.35)
        for v in (self.s['lambda_a'], self.s['lambda_b']):
            ax.axvline(v, color=INK, lw=1.2)
        ax.set_xlim(self.ys[0], self.ys[-1]); ax.set_ylim(self.xs[0], self.xs[-1])


def _mark_unvisited(ax, grid):
    """Hatch the region above the map threshold (V > vmax_kT): (almost) no sampled frame lies there, so lambda
    there is an extrapolation that the sampler never evaluates."""
    ax.contourf(grid.ys, grid.xs, grid.mask.T.astype(float), levels=[0.5, 1.5], colors='none', hatches=['////'])
    ax.contour(grid.ys, grid.xs, grid.mask.T.astype(float), levels=[0.5], colors=INK2, linewidths=0.6)


def _passes(cfg):
    with open(os.path.join(cfg['loop_dir'], 'state.json')) as f:
        return json.load(f)


def fig_interfaces(cfg, st, grid, out):
    recs = list(st['passes'])
    fz = os.path.join(cfg['loop_dir'], 'freeze.json')
    if os.path.isfile(fz):
        with open(fz) as f:
            fr = json.load(f)
        if fr.get('ok') and fr.get('source', '').startswith('verification'):
            recs.append(dict(k=f'production ({fr["source"]})', model=fr['model'], interfaces=fr['interfaces'],
                             recheck=dict(ok=True)))
    n = len(recs) + 1
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.0 * rows), sharex=True, sharey=True, squeeze=False)
    with open(os.path.join(cfg['loop_dir'], 'pass_0', 'pass0.json')) as f:
        p0 = json.load(f)
    ax = axes.flat[0]
    grid.background(ax)
    intf0 = p0['iterations'][-1]['interfaces']
    for v in intf0[1:-1]:
        ax.axvline(v, color=BLUE, lw=1.1)
    ax.set_title(f'pass 0 (infinit, lambda = y): {len(intf0)} interfaces\n'
                 f'y = {", ".join(f"{v:.3g}" for v in intf0[1:-1])}', loc='left')
    for i, rec in enumerate(recs):
        ax = axes.flat[i + 1]
        grid.background(ax)
        m = LambdaModel(rec['model'])
        L = m(grid.cv).reshape(grid.shape)                     # unmasked: lambda is defined everywhere
        inner = rec['interfaces'][1:-1]
        ax.contour(grid.ys, grid.xs, L.T, levels=inner, colors=BLUE, linewidths=1.2)
        h = m.h_of_lambda(np.array(inner))
        ax.contour(grid.ys, grid.xs, grid.q_exact.T, levels=np.sort(np.unique(np.clip(h, 1e-6, 1 - 1e-6))),
                   colors=RED, linewidths=0.8, linestyles='--')
        _mark_unvisited(ax, grid)
        rc = 'all checks pass' if rec['recheck']['ok'] else 'checks fail'
        ax.set_title(f'{"pass " if isinstance(rec["k"], int) else ""}{rec["k"]}: {len(rec["interfaces"])} interfaces, {rc} on its own paths\n'
                     f'q = {", ".join(f"{v:.2g}" for v in h)}', loc='left')
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('y (state CV; A left of the first black line, B right of the second)')
    for ax in axes[:, 0]:
        ax.set_ylabel('x')
    fig.suptitle(f'{cfg["name"]}: interfaces per pass (blue: lambda isosurfaces; red dashed: exact isocommittors at '
                 f'the same levels; grey: potential, 1 kT contours; hatched: V > {cfg.get("plots", {}).get("vmax_kT", 12)} kT, '
                 f'not visited by the sampled paths)', x=0.01, ha='left', y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'interfaces.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def _teacher_frames(cfg, pool, k):
    """(cv, q, w) of the pooled frames labelled by the teacher of pass k, or None."""
    p = os.path.join(cfg['loop_dir'], f'pass_{k}', 'teacher_q.npz')
    if not os.path.isfile(p):
        return None
    lab = np.load(p)
    cvs, qs, ws = [], [], []
    for key in lab.files:
        j = int(key.split('_')[1])
        if not pool.has(j):
            continue
        z, _ = pool.load(j)
        q = lab[key]
        ok = np.isfinite(q)
        w = np.repeat(z['piece_w'], z['piece_n'])
        cvs.append(z['cv'][ok, 1:]); qs.append(q[ok]); ws.append(w[ok])
    if not cvs:
        return None
    return np.concatenate(cvs).astype(float), np.concatenate(qs).astype(float), np.concatenate(ws)


def _binned(grid, cv, val, w, min_w=1e-9, step=4):
    iy = np.clip(np.searchsorted(grid.ys, cv[:, 0]) - 1, 0, len(grid.ys) - 2)
    ix = np.clip(np.searchsorted(grid.xs, cv[:, 1]) - 1, 0, len(grid.xs) - 2)
    # coarser bins for the data: step x step grid cells
    by, bx = iy // step, ix // step
    ny, nx = (len(grid.ys) + step - 1) // step, (len(grid.xs) + step - 1) // step
    idx = by * nx + bx
    sw = np.bincount(idx, w, ny * nx)
    sv = np.bincount(idx, w * val, ny * nx)
    out = np.where(sw > min_w, sv / np.maximum(sw, min_w), np.nan).reshape(ny, nx)
    ys = grid.ys[::step][:ny]
    xs = grid.xs[::step][:nx]
    return ys, xs, np.ma.masked_invalid(out)


def fig_maps(cfg, st, grid, pool, out):
    """Row 0: exact committor | teacher labels of the latest pass with kept labels, as logit q_teacher - logit q_exact
    (bin means). Rows 1..K: model committor m_k(x) | logit m_k - logit q_exact. Left column committor scale, right
    column logit-difference scale (one colorbar each)."""
    recs = st['passes']
    sysd = cfg['system_spec']
    fig, axes = plt.subplots(len(recs) + 1, 2, figsize=(11, 2.55 * (len(recs) + 1)), sharex=True, sharey=True,
                             squeeze=False, layout='constrained')
    lq = grid.field(grid.q_exact)
    inside = ((grid.cv[:, 0] > sysd['lambda_a']) & (grid.cv[:, 0] < sysd['lambda_b'])).reshape(grid.shape)
    ax = axes[0, 0]
    im = ax.pcolormesh(grid.ys, grid.xs, lq.T, cmap=SEQ, vmin=0, vmax=1, shading='auto')
    ax.contour(grid.ys, grid.xs, lq.T, levels=[0.01, 0.1, 0.5, 0.9, 0.99], colors='w', linewidths=0.6)
    ax.set_title('exact committor (white: q = 0.01, 0.1, 0.5, 0.9, 0.99)', loc='left')
    ax = axes[0, 1]
    kt = next((r['k'] for r in recs[::-1] if _teacher_frames(cfg, pool, r['k']) is not None), None)
    if kt is not None:
        cvt, qt, wt = _teacher_frames(cfg, pool, kt)
        ins = (cvt[:, 0] > sysd['lambda_a']) & (cvt[:, 0] < sysd['lambda_b'])
        d = np.clip(_logit(qt[ins]) - _logit(ase_systems.exact_committor(sysd)(cvt[ins])), -3, 3)
        ys, xs, B = _binned(grid, cvt[ins], d, wt[ins], step=2)
        imd = ax.pcolormesh(ys, xs, B.T, cmap=DIV, norm=TwoSlopeNorm(0, -3, 3), shading='nearest')
        ax.set_title(f'teacher of pass {kt}: logit q_np - logit q_exact (bin means over the pooled frames)', loc='left')
    for i, rec in enumerate(recs):
        m = LambdaModel(rec['model'])
        mq = grid.field(m.q(grid.cv))
        ax = axes[i + 1, 0]
        im = ax.pcolormesh(grid.ys, grid.xs, mq.T, cmap=SEQ, vmin=0, vmax=1, shading='auto')
        ax.contour(grid.ys, grid.xs, mq.T, levels=[0.01, 0.1, 0.5, 0.9, 0.99], colors='w', linewidths=0.6)
        ax.set_title(f'pass {rec["k"]}: model committor m(x)', loc='left')
        ax = axes[i + 1, 1]
        d = np.clip(_logit(np.ma.filled(mq, 0.5)) - _logit(np.ma.filled(lq, 0.5)), -3, 3)
        imd = ax.pcolormesh(grid.ys, grid.xs, np.ma.array(d, mask=grid.mask | ~inside).T, cmap=DIV,
                            norm=TwoSlopeNorm(0, -3, 3), shading='auto')
        ex = rec['diagnostics'].get('exact') or {}
        ax.set_title(f'pass {rec["k"]}: logit m - logit q_exact (spread {ex.get("spread_mean", np.nan):.3f}, '
                     f'logit RMSE {ex.get("logit_rmse", np.nan):.2f})', loc='left')
    for a in axes.flat:
        for v in (sysd['lambda_a'], sysd['lambda_b']):
            a.axvline(v, color=INK, lw=0.8)
    fig.colorbar(im, ax=axes[:, 0], location='bottom', shrink=0.8, aspect=40, label='committor')
    fig.colorbar(imd, ax=axes[:, 1], location='bottom', shrink=0.8, aspect=40,
                 label='logit difference (clipped at +-3; red: fit too high, blue: too low)')
    for a in axes[:, 0]:
        a.set_ylabel('x')
    for a in axes[-1]:
        a.set_xlabel('y')
    fig.suptitle(f'{cfg["name"]}: fits per pass (regions above {cfg.get("plots", {}).get("vmax_kT", 12)} kT blank; '
                 f'black lines: A and B)', x=0.01, ha='left', fontsize=10)
    fig.savefig(os.path.join(out, 'maps.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


def fig_calibration(cfg, st, pool, out):
    recs = st['passes']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    qx_fn = ase_systems.exact_committor(cfg['system_spec'])
    edges = np.linspace(-9, 9, 37)
    for i, rec in enumerate(recs):
        color = PASS_RAMP[min(i, len(PASS_RAMP) - 1)] if len(recs) <= len(PASS_RAMP) else plt.cm.Blues(0.35 + 0.6 * i / len(recs))
        z, _ = pool.load(rec['pool_pass'])
        cv = z['cv'][:, 1:].astype(float)
        w = np.repeat(z['piece_w'], z['piece_n'])
        s = cv[:, 0]
        inside = (s > cfg['system_spec']['lambda_a']) & (s < cfg['system_spec']['lambda_b'])
        sub = np.flatnonzero(inside)[::max(1, inside.sum() // 300_000)]
        lx = _logit(qx_fn(cv[sub]))
        m = LambdaModel(rec['model'])
        for ax, val, label in ((axes[0], _logit(m.q(cv[sub])), 'model m(x)'),):
            b = np.clip(np.searchsorted(edges, lx) - 1, 0, len(edges) - 2)
            sw = np.bincount(b, w[sub], len(edges) - 1)
            mean = np.bincount(b, w[sub] * val, len(edges) - 1) / np.where(sw > 0, sw, 1)
            ok = sw > 0
            c = 0.5 * (edges[1:] + edges[:-1])
            ax.plot(c[ok], mean[ok], '-', color=color, lw=2, label=f'pass {rec["k"]}')
        tf = _teacher_frames(cfg, pool, rec['k'])
        if tf is not None:
            cvt, qt, wt = tf
            st_ = np.arange(len(qt))[::max(1, len(qt) // 300_000)]
            inside_t = (cvt[st_, 0] > cfg['system_spec']['lambda_a']) & (cvt[st_, 0] < cfg['system_spec']['lambda_b'])
            st_ = st_[inside_t]
            lxt = _logit(qx_fn(cvt[st_]))
            b = np.clip(np.searchsorted(edges, lxt) - 1, 0, len(edges) - 2)
            sw = np.bincount(b, wt[st_], len(edges) - 1)
            mean = np.bincount(b, wt[st_] * _logit(qt[st_]), len(edges) - 1) / np.where(sw > 0, sw, 1)
            ok = sw > 0
            c = 0.5 * (edges[1:] + edges[:-1])
            axes[1].plot(c[ok], mean[ok], '-', color=color, lw=2, label=f'pass {rec["k"]}')
    for ax, title in zip(axes, ('model m(x) on the pass\'s own sampled frames', 'teacher q on the pooled frames')):
        ax.plot(edges, edges, color=MUTED, lw=1, ls='--', label='exact')
        ax.set(xlabel='logit q_exact (bins)', title=title, xlim=(-9, 9), ylim=(-9, 9))
        ax.grid(color='#e6e5e1', lw=0.5)
        ax.legend(fontsize=8, loc='upper left')
    axes[0].set_ylabel('weighted mean logit of the fit in the bin')
    fig.suptitle(f'{cfg["name"]}: calibration against the exact committor', x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'calibration.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_convergence(cfg, st, out):
    recs = st['passes']
    k = [r['k'] for r in recs]
    md = np.cumsum([r['cost']['md_steps'] for r in recs]) + st['pass0']['md_steps']
    series = [('teacher RMS of q vs exact', [r['teacher'].get('rms_vs_exact', np.nan) for r in recs]),
              ('isosurface spread vs exact (floor 0.013)', [(r['diagnostics'].get('exact') or {}).get('spread_mean', np.nan) for r in recs]),
              ('logit RMSE vs exact (binned)', [(r['diagnostics'].get('exact') or {}).get('logit_rmse', np.nan) for r in recs])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (title, v) in zip(axes, series):
        ax.plot(k, v, '-o', color=BLUE, lw=2, ms=6, mec='#fcfcfb', mew=1.5)
        ax.set(title=title, xlabel='pass', xticks=k)
        ax.grid(color='#e6e5e1', lw=0.5)
        ax.set_ylim(0, None)
    axes[1].axhline(0.013, color=MUTED, ls='--', lw=1)
    for ax in axes:
        sec = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
        sec.set_xticks(k)
        sec.set_xticklabels([f'{m / 1e6:.1f}M' for m in md], fontsize=7, color=MUTED)
    fig.suptitle(f'{cfg["name"]}: accuracy per pass (top axis: cumulative MD steps including pass 0)', x=0.01, ha='left', fontsize=10, y=1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'convergence.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_tis(cfg, st, pool, out):
    """optimalrcs TIS diagnostics and path-weighted F(q) of model k on the pieces of pass k (cupy)."""
    import cupy as cp
    sys.path.insert(0, os.path.join(env.REPO, 'optimalrcs-cupy'))
    from loop.teacher import _optimalrcs_cupy
    orc = _optimalrcs_cupy()
    plt.show = lambda *a, **k: None
    s = cfg['system_spec']
    qx_fn = ase_systems.exact_committor(s)
    for rec in st['passes']:
        z, _ = pool.load(rec['pool_pass'])
        cv = z['cv'][:, 1:].astype(float)
        sy = cv[:, 0]
        A, B = sy <= s['lambda_a'], sy >= s['lambda_b']
        w = np.repeat(z['piece_w'], z['piece_n']).astype(float)
        w /= w.mean()
        q = orc.CommittorNE(boundary0=A, boundary1=B, i_traj=z['piece'], t_traj=z['t'].astype(float), path_weights=w,
                            group_traj=np.repeat(z['piece_pnr'], z['piece_n']))
        q.set_fixed_traj_length_trap(A | B, 2 ** 16)
        m = LambdaModel(rec['model'])
        for tag, r in (('model', m.q(cv)), ('exact', qx_fn(cv))):
            if tag == 'exact' and rec is not st['passes'][-1]:
                continue
            r = np.asarray(r, float)
            r[A], r[B] = 0.0, 1.0
            q.r_traj = cp.asarray(r, dtype=cp.float64)
            name = f'pass{rec["k"]}' if tag == 'model' else f'pass{rec["k"]}_exact'
            q.plots_tis()
            plt.gcf().suptitle(f'{cfg["name"]} pass {rec["k"]}: {"model " + str(rec["k"]) if tag == "model" else "exact committor"} '
                               f'on the pass\'s own pieces', y=1.02)
            plt.gcf().savefig(os.path.join(out, f'tis_{name}.png'), dpi=100, bbox_inches='tight')
            plt.close('all')
            q.plots_feps()
            plt.gcf().suptitle(f'{cfg["name"]} pass {rec["k"]}: {"model " + str(rec["k"]) if tag == "model" else "exact committor"} '
                               f'(path-weighted F(q), Z_q)', y=1.02)
            plt.gcf().savefig(os.path.join(out, f'feps_{name}.png'), dpi=100, bbox_inches='tight')
            plt.close('all')
        del q
        cp.get_default_memory_pool().free_all_blocks()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--no-tis', action='store_true')
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    st = _passes(cfg)
    pool = pool_mod.Pool(cfg['loop_dir'])
    out = os.path.join(cfg['loop_dir'], 'figures')
    os.makedirs(out, exist_ok=True)
    grid = Grid(cfg, pool)
    fig_interfaces(cfg, st, grid, out); print('interfaces.png')
    fig_maps(cfg, st, grid, pool, out); print('maps.png')
    fig_calibration(cfg, st, pool, out); print('calibration.png')
    fig_convergence(cfg, st, out); print('convergence.png')
    if not a.no_tis:
        fig_tis(cfg, st, pool, out); print('tis / feps')
    print(out)


if __name__ == '__main__':
    main()
