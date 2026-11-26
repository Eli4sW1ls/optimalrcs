#!/usr/bin/env python3
"""Figures of a loop run: <loop_dir>/figures/ (3D systems: interfaces / maps per slice w = -w0, +w0, file suffix _w<value>)

    python -m loop.plots <loop name> [--no-tis]

- interfaces.png       one panel per pass: the potential (grey, 1 kT contours), the interfaces of that pass
                       (pass 0: the last infinit iteration's interfaces in y; pass k: the lambda_k isosurfaces, blue)
                       and the exact isocommittors at the same committor levels (dashed)
- maps.png             one row per pass: teacher q (bin means of the pass's labels on the pooled frames), the
                       model's committor m(x), the exact committor, and logit m - logit q_exact
- paths.png            production: the frozen lambda's isosurfaces, the exact isocommittors and a few accepted
                       paths (two A -> B, two crossing staples at a low and a high interface); also `--paths-only`
- calibration.png      bin means of logit m (and logit q_teacher) against logit q_exact, per pass
- convergence.png      teacher RMS, isosurface spread and logit RMSE against the exact committor, per pass
- pcross_convergence.png   production runs (learned lambda, refilled interfaces, seed-lambda control): running
                       P_cross and rate / exact rate against MD steps with +-1 block error, and the block-error curves
                       (inftools' `error_analysis_staple`, run by `python -m loop.pcross_errors`); also `--pcross-only`
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


_FRAMES = {}


def _pool_frames(pool, step=5):
    """(cv, w) of the pooled frames (every `step`-th; cv without the recorded lambda column, in the system's CV
    order; w the piece weight), cached per pool directory."""
    key = (pool.dir, step)
    if key not in _FRAMES:
        cvs, ws = [], []
        for k in pool.passes():
            z, _ = pool.load(k)
            cvs.append(np.asarray(z['cv'][::step, 1:], float))
            ws.append(np.asarray(z['piece_w'], float)[z['piece'][::step]])
        _FRAMES.clear()
        _FRAMES[key] = (np.concatenate(cvs), np.concatenate(ws))
    return _FRAMES[key]


def _wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / np.sum(w)
    return np.interp(q, c, x[o])


def _axes(cfg):
    """Indices (state CV, second plotted CV, the CVs held fixed) in the system's CV order. The second CV is
    [plots] x_cv (a CV name) or the first CV after the state CV."""
    s = cfg['system_spec']
    names, a = list(s['cv_names']), int(s.get('state_cv', 0))
    xc = cfg.get('plots', {}).get('x_cv')
    b = names.index(xc) if xc else next(i for i in range(len(names)) if i != a)
    return a, b, [i for i in range(len(names)) if i not in (a, b)]


class Grid:
    """(state CV, second CV) grid. With more CVs, a slice: the first further CV at `w` (one grid per slice value,
    `_grids`), any others at their weighted medians in the pool. Background: the analytic potential where the system
    has one (toytis potentials, the known 3D systems), otherwise -ln of the weighted density of the pooled frames
    that lie nearest this slice (`slice_values`: all slice values of the figure). The model is evaluated on the
    full CV vector."""

    def __init__(self, cfg, pool, n=(240, 140), w=None, slice_values=None):
        s = cfg['system_spec']
        names = list(s['cv_names'])
        self.s, self.w = s, w
        self.ia, self.ib, self.rest = _axes(cfg)
        self.names = names
        self.slice_name = names[self.rest[0]] if self.rest else None
        self.slice_values = list(slice_values) if slice_values is not None else ([w] if w is not None else [])
        pc = cfg.get('plots', {})
        pad = 0.075 * (s['lambda_b'] - s['lambda_a'])
        self.ys = np.linspace(s['lambda_a'] - pad, s['lambda_b'] + pad, n[0])
        xr = pc.get('x_range')                                 # default: the data's range (0.2% quantiles)
        cvp, wp = _pool_frames(pool)
        if not xr:
            xr = np.quantile(cvp[:, self.ib], [0.002, 0.998])
        self.xs = np.linspace(*xr, n[1])
        Y, X = np.meshgrid(self.ys, self.xs, indexing='ij')
        self.fixed = {}
        if self.rest:
            self.fixed[self.rest[0]] = float(w) if w is not None else float(_wquantile(cvp[:, self.rest[0]], wp, 0.5))
            for r in self.rest[1:]:
                self.fixed[r] = float(_wquantile(cvp[:, r], wp, 0.5))
            if w is None:
                self.w = self.fixed[self.rest[0]]
        self.cv = np.zeros((Y.size, len(names)))
        self.cv[:, self.ia], self.cv[:, self.ib] = Y.ravel(), X.ravel()
        for r, v in self.fixed.items():
            self.cv[:, r] = v
        self.shape = Y.shape
        V = None
        try:
            V = self._analytic(s, Y, X)
        except Exception:                                      # noqa: BLE001  no analytic potential: from the data
            V = None
        if V is not None:
            V = V / s['kT']
            self.bg_note = 'potential, 1 kT contours'
        else:
            V = self._density(cvp, wp, n)
            self.bg_note = ('-ln sampled density of the pooled frames' + (f' nearest this {self.slice_name} slice'
                            if self.rest else '') + ', unit contours')
        self.V = V - np.nanmin(V)
        self.mask = ~np.isfinite(self.V) | (self.V > float(pc.get('vmax_kT', 12)))
        self.V = np.where(np.isfinite(self.V), self.V, 99.0)
        qx = ase_systems.exact_committor(s)
        try:
            self.q_exact = qx(self.cv).reshape(self.shape) if qx is not None else None     # None: no exact reference
        except Exception:                                      # noqa: BLE001
            self.q_exact = None

    def _analytic(self, s, Y, X):
        """The potential on the grid for the known toy systems, or None."""
        import toy2d
        from loop.ase_calcs import load_toytis_potential
        if s['dim'] == 1 and s.get('velocity_cvs'):            # underdamped 1D (x, v): the energy V(x) + v^2 / 2
            sys.path.insert(0, os.path.join(env.EXAMPLES, 'figures_by_potential', '_shared', 'exact_committors'))
            import exact_tools
            return exact_tools.coswell_potential(Y) + 0.5 * X ** 2
        if (self.ia, self.ib) != (0, 1):
            return None
        if not self.rest and s.get('potential'):
            return toy2d.potential_grid(load_toytis_potential(s['potential'], s['potential_kwargs']), self.ys, self.xs)
        c = s.get('calculator') or {}
        if len(self.rest) == 1 and self.rest[0] == 2 and c.get('spec') and c.get('params') is not None:
            if c.get('base_dim', s['dim']) < s['dim']:          # uncoupled w: the 2D potential (+ a constant)
                return toy2d.potential_grid(load_toytis_potential(c['spec'], c['params']), self.ys, self.xs)
            pot = load_toytis_potential(c['spec'], c['params'])  # coupled (zpot_w): the effective potential at w
            return np.array([pot.effective_potential((np.array([a, b, self.w]), np.zeros(3)))
                             for a, b in zip(Y.ravel(), X.ravel())]).reshape(Y.shape)
        return None

    def _density(self, cvp, wp, n, nb=(60, 40), empty=1e-5):
        """-ln of the weighted 2D histogram of the pooled frames on (state CV, second CV), smoothed by one bin and
        interpolated onto the grid; with slices, only the frames whose fixed CV lies nearest this slice's value.
        Where the smoothed density is below `empty` x its maximum: inf (shown as not visited)."""
        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import gaussian_filter
        sel = np.ones(len(cvp), bool)
        if self.rest and len(self.slice_values) > 1:
            r0 = self.rest[0]
            vals = np.asarray(self.slice_values, float)
            near = vals[np.argmin(np.abs(cvp[:, r0][:, None] - vals[None, :]), axis=1)]
            sel = near == self.w
        ey = np.linspace(self.ys[0], self.ys[-1], nb[0] + 1)
        ex = np.linspace(self.xs[0], self.xs[-1], nb[1] + 1)
        H, _, _ = np.histogram2d(cvp[sel, self.ia], cvp[sel, self.ib], bins=(ey, ex), weights=wp[sel])
        H = gaussian_filter(H / max(H.sum(), 1e-300), 1.0, mode='nearest')
        cy, cx = 0.5 * (ey[1:] + ey[:-1]), 0.5 * (ex[1:] + ex[:-1])
        Y, X = np.meshgrid(np.clip(self.ys, cy[0], cy[-1]), np.clip(self.xs, cx[0], cx[-1]), indexing='ij')
        Hg = RegularGridInterpolator((cy, cx), H)(np.c_[Y.ravel(), X.ravel()]).reshape(Y.shape)
        with np.errstate(divide='ignore'):
            return np.where(Hg > empty * H.max(), -np.log(np.maximum(Hg, 1e-300)), np.inf)

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


def _sfx(grid):
    return '' if not grid.rest else f'_{grid.slice_name}{grid.w:+.3g}'


def _slice_note(grid):
    if not grid.rest:
        return ''
    more = ', ' + ', '.join(f'{grid.names[r]} = {grid.fixed[r]:.3g}' for r in grid.rest[1:]) + ' (medians)' if len(grid.rest) > 1 else ''
    return f' (slice {grid.slice_name} = {grid.w:+.3g}{more})'


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
    ax.set_title(f'pass 0 (infinit, lambda = {cfg["system_spec"]["cv_names"][0]}): {len(intf0)} interfaces\n'
                 f'{cfg["system_spec"]["cv_names"][0]} = {", ".join(f"{v:.3g}" for v in intf0[1:-1])}', loc='left')
    for i, rec in enumerate(recs):
        ax = axes.flat[i + 1]
        grid.background(ax)
        m = LambdaModel(rec['model'])
        L = m(grid.cv).reshape(grid.shape)                     # unmasked: lambda is defined everywhere
        inner = rec['interfaces'][1:-1]
        ax.contour(grid.ys, grid.xs, L.T, levels=inner, colors=BLUE, linewidths=1.2)
        h = m.h_of_lambda(np.array(inner))
        if grid.q_exact is not None:
            ax.contour(grid.ys, grid.xs, grid.q_exact.T, levels=np.sort(np.unique(np.clip(h, 1e-6, 1 - 1e-6))),
                       colors=RED, linewidths=0.8, linestyles='--')
        _mark_unvisited(ax, grid)
        rc ='all checks pass' if rec['recheck']['ok'] else 'checks fail'
        ax.set_title(f'{"pass " if isinstance(rec["k"], int) else ""}{rec["k"]}: {len(rec["interfaces"])} interfaces, {rc} on its own paths\n'
                     f'q = {", ".join(f"{v:.2g}" for v in h)}', loc='left')
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f'{cfg["system_spec"]["cv_names"][0]} (state CV; A left of the first black line, B right of the second)')
    for ax in axes[:, 0]:
        ax.set_ylabel(cfg['system_spec']['cv_names'][1])
    fig.suptitle(f'{cfg["name"]}{_slice_note(grid)}: interfaces per pass (blue: lambda isosurfaces; red dashed: exact isocommittors at '
                 f'the same levels; grey: potential, 1 kT contours; hatched: V > {cfg.get("plots", {}).get("vmax_kT", 12)} kT, '
                 f'not visited by the sampled paths)', x=0.01, ha='left', y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'interfaces{_sfx(grid)}.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


PATH_COLS = ['#eb6834', '#1baf7a', '#e87ba4', '#c98500']     # categorical slots 2, 3, 5, 4 (blue / red taken)


def _pick_paths(cfg, sub, near=None, n_reactive=2, n_total=4, data_file='infretis_data.txt', max_rows=4000):
    """A few accepted paths of a run (newest rows first, at most `max_rows` read): the shortest reactive ones (a
    frame in A and one in B) among the newest, then crossing staples (LMR, the shortest per interface) spread over
    the interfaces up to `n_total` paths. `near` = (CV index, slice values, this value): only paths whose median of
    that CV is nearest this slice. Returns
    [(label, cv array (frames x n_cv))]; needs the paths' load dirs."""
    s = cfg['system_spec']
    run = os.path.join(cfg['loop_dir'], sub)
    rows = [ln.split() for ln in open(os.path.join(run, data_file)) if ln.strip() and not ln.startswith('#')]
    reactive, staples = [], {}
    for r in rows[::-1][:max_rows]:
        d = os.path.join(run, 'load', r[0], 'order.txt')
        if len(r) < 5 or not os.path.isfile(d):
            continue
        ptype, n = r[4], int(r[1])
        lmr = 'LMR' in ptype
        if len(reactive) >= 4 * n_reactive and not lmr:
            continue
        if lmr and len(reactive) >= 4 * n_reactive:
            k = int(ptype.split('LMR')[0])
            if k in staples and staples[k][2] <= n:
                continue
        cv = np.loadtxt(d, comments='#', ndmin=2)[:, 2:]
        if cv.shape[1] < 2:
            continue
        if near is not None:
            col, vals, v = near
            if cv.shape[1] <= col or vals[int(np.argmin(np.abs(np.asarray(vals) - np.median(cv[:, col]))))] != v:
                continue
        sc = cv[:, int(s.get('state_cv', 0))]
        hit_a, hit_b = sc.min() <= s['lambda_a'], sc.max() >= s['lambda_b']
        if hit_a and hit_b and len(reactive) < 4 * n_reactive:
            reactive.append((f'path {r[0]}: {"A -> B" if sc[0] < sc[-1] else "B -> A"}, {len(cv)} frames', cv))
        elif lmr and not hit_b:
            a, b = ptype.split('LMR')
            k = int(a)
            if k not in staples or len(cv) < staples[k][2]:
                staples[k] = (f'path {r[0]}: staple LMR, turns at interfaces {a} and {b}, {len(cv)} frames', cv, len(cv))
    reactive = sorted(reactive, key=lambda t: len(t[1]))[:n_reactive]      # the shortest of the newest 4 n
    n_st = max(n_total - len(reactive), 0)
    ks = sorted(staples)
    pick = list(dict.fromkeys(ks[int(i)] for i in np.linspace(0, len(ks) - 1, min(n_st, len(ks))).round())) if ks and n_st else []
    return reactive + [staples[k][:2] for k in pick]


def fig_interface_paths(cfg, grid, out, sub, model_path, interfaces, title, fname, prev=None, data_file='infretis_data.txt'):
    """One panel: an interface set (blue; lambda isosurfaces, or vertical lines for the seed lambda when
    model_path is None), optionally a previous set (grey dotted) and the exact isocommittors at the same q levels
    (red dashed, where the system has an exact committor), and a few accepted paths of the run in <loop>/<sub>
    (`_pick_paths`), projected on (state CV, second CV)."""
    run = os.path.join(cfg['loop_dir'], sub)
    near = (grid.rest[0], grid.slice_values, grid.w) if grid.rest and len(grid.slice_values) > 1 else None
    paths = _pick_paths(cfg, sub, near, data_file=data_file) \
        if os.path.isfile(os.path.join(run, data_file)) and os.path.isdir(os.path.join(run, 'load')) else []
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    grid.background(ax)
    inner = [float(v) for v in interfaces[1:-1]]

    def draw(mp, lv, **kw):
        if mp is None:
            for v in lv:
                ax.axvline(v, **kw)
        else:
            L = LambdaModel(mp)(grid.cv).reshape(grid.shape)
            ax.contour(grid.ys, grid.xs, L.T, levels=lv, colors=kw['color'], linewidths=kw['lw'],
                       linestyles=kw.get('ls', 'solid'))
    if prev is not None and len(prev['interfaces']) > 2:
        draw(prev['model'], [float(v) for v in prev['interfaces'][1:-1]], color='0.45', lw=0.8, ls=':')
        ax.plot([], [], color='0.45', lw=0.8, ls=':', label=f'previous set: {prev["label"]} ({len(prev["interfaces"])} interfaces)')
    draw(model_path, inner, color=BLUE, lw=1.1)
    ax.plot([], [], color=BLUE, lw=1.1, label=f'this set: {len(interfaces)} interfaces')
    h = None
    if model_path is not None:
        h = LambdaModel(model_path).h_of_lambda(np.array(inner))
        if grid.q_exact is not None:
            ax.contour(grid.ys, grid.xs, grid.q_exact.T, levels=np.sort(np.unique(np.clip(h, 1e-6, 1 - 1e-6))),
                       colors=RED, linewidths=0.7, linestyles='--')
            ax.plot([], [], color=RED, lw=0.7, ls='--', label='exact isocommittors, same q levels')
    _mark_unvisited(ax, grid)
    for (lab, cv), col in zip(paths, PATH_COLS):
        ax.plot(cv[:, grid.ia], cv[:, grid.ib], color=col, lw=0.8, alpha=0.9, label=lab)
        ax.plot(cv[0, grid.ia], cv[0, grid.ib], 'o', ms=5, color=col, mec='#fcfcfb', mew=1.0)
    if not paths:
        ax.plot([], [], ' ', label='(no paths of the run found)')
    ax.plot([], [], ' ', label=f'grey: {grid.bg_note}')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2, handlelength=1.8, columnspacing=1.5)
    ax.set_xlabel(f'{grid.names[grid.ia]} (A left of the first black line, B right of the second)')
    ax.set_ylabel(grid.names[grid.ib])
    note = ('' if not grid.rest else f'; paths whose median {grid.slice_name} is nearest this slice, projected on '
            f'({grid.names[grid.ia]}, {grid.names[grid.ib]})')
    qs = '' if h is None else 'q = ' + ', '.join(f'{v:.2g}' for v in h)
    import textwrap
    head = textwrap.fill(f'{cfg["name"]}{_slice_note(grid)}: {title} (dot = first frame){note}', 120)
    ax.set_title(head + '\n' + textwrap.fill(qs, 120, subsequent_indent='    ') if qs else head, loc='left', fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, f'{fname}{_sfx(grid)}.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_paths(cfg, grid, out, sub='production'):
    """The frozen production lambda (blue isosurfaces at its interfaces), the exact isocommittors at the same
    levels (red dashed) and a few accepted production paths projected on (state CV, second CV)."""
    fz = os.path.join(cfg['loop_dir'], 'freeze.json')
    if not (os.path.isfile(fz) and os.path.isdir(os.path.join(cfg['loop_dir'], sub, 'load'))):
        return None
    with open(fz) as f:
        fr = json.load(f)
    return fig_interface_paths(cfg, grid, out, sub, fr['model'], fr['interfaces'],
                               f'production paths, lambda of pass {fr["pass_index"]}', 'paths')


def fig_pass(cfg, pool, k, model_path, interfaces, note, prev=None):
    """pass_<k>/interfaces_paths[_w<value>].png after pass k (pass 0: infinit's set, seed lambda): the set, the
    previous one, the exact isocommittors and a few paths of the pass's own run. Written by the loop after every
    pass ([run] pass_figures, default true)."""
    out = os.path.join(cfg['loop_dir'], f'pass_{k}')
    sub, data_file = f'pass_{k}/run', 'infretis_data.txt'
    if k == 0:                         # infinit: one data file per iteration, the last one sampled these interfaces
        sub = 'pass_0'
        with open(os.path.join(cfg['loop_dir'], 'pass_0', 'pass0.json')) as f:
            data_file = os.path.basename(json.load(f)['iterations'][-1]['data_file'])
    return [fig_interface_paths(cfg, grid, out, sub, model_path, interfaces, f'pass {k}: {note}', 'interfaces_paths',
                                prev, data_file=data_file) for grid in _grids(cfg, pool)]


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
        if grid.w is not None:                                  # frames near the slice
            ins &= np.abs(cvt[:, 2] - grid.w) < 0.03
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
        a.set_ylabel(sysd['cv_names'][1])
    for a in axes[-1]:
        a.set_xlabel(sysd['cv_names'][0])
    fig.suptitle(f'{cfg["name"]}{_slice_note(grid)}: fits per pass (regions above {cfg.get("plots", {}).get("vmax_kT", 12)} kT blank; '
                 f'black lines: A and B)', x=0.01, ha='left', fontsize=10)
    fig.savefig(os.path.join(out, f'maps{_sfx(grid)}.png'), dpi=110, bbox_inches='tight')
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

PROD_RUNS = [('production', 'learned lambda', BLUE), ('production_refill3', 'learned lambda, refilled', '#0d366b'),
             ('production_nsh2', 'learned lambda, ridge split (n_sh 2, max_ratio 4)', '#1baf7a'),
             ('production_dp_nsh2', 'learned lambda, dp placement (n_sh 2)', '#e87ba4'),
             ('production_refill', 'learned lambda, dp + even regions (p floor 0.2)', '#e8590c'),
             ('production_measured', 'learned lambda, dp on measured levels (margin 1.5)', '#7048e8'),
             ('production_measured_rep/run', 'same set, other seed', '#b197fc'),
             ('production_balanced', 'learned lambda, refined for acceptance (19 interfaces)', '#0ca678'),
             ('production_var', 'learned lambda, refined for pair variance (16 interfaces)', '#f59f00'),
             ('production_seed', 'seed lambda', RED)]


def _read_runav(path):
    """inftools *_runav.txt: a header line, then one fixed-width row per snapshot (each column starts where its
    header name starts). Wide values fill their column with no separator (MFPT > 1e6 runs into the next one), so the
    rows are cut at the header offsets; a row that does not parse that way is taken by number pattern."""
    import re
    num = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
    with open(path) as f:
        hline = f.readline()
        head = hline.split()
        starts = [m.start() for m in re.finditer(r'\S+', hline)]
        rows = []
        for line in f:
            if not line.strip():
                continue
            cuts = starts[1:] + [None]
            try:
                rows.append([float(line[a:b]) for a, b in zip(starts, cuts)])
            except ValueError:
                r = [float(v) for v in num.findall(line)]
                if len(r) == len(head):
                    rows.append(r)
    return head, np.array(rows)


def _read_block(path):
    best = rel = None
    blocks = []
    for line in open(path):
        if line.startswith('# Best estimate'):
            best = np.array(line.split()[3:], float)
        elif line.startswith('# Avg rel error'):
            rel = np.array(line.split()[4:], float)
        elif line.strip() and not line.startswith(('#', '-')):
            blocks.append([float(v) for v in line.split()])
    return best, rel, np.array(blocks)


def fig_pcross_convergence(cfg, out):
    """Running P_cross(lambda_B) and rate / exact rate of the production runs against accepted paths (the legend gives
    MD steps per accepted path), with a
    +-1 block-error band (the plateau relative error at the end, scaled by sqrt(N / n) back in time), and the
    relative block error of P_cross(lambda_B) against block length (inftools error_analysis_staple)."""
    runs = []
    for sub, label, col in PROD_RUNS:
        ea = os.path.join(cfg['loop_dir'], sub, 'error_analysis')
        if not os.path.isfile(os.path.join(ea, 'pcross_runav.txt')):
            continue
        with open(os.path.join(cfg['loop_dir'], sub, 'rates.json')) as f:
            rj = json.load(f)
        _, pc = _read_runav(os.path.join(ea, 'pcross_runav.txt'))
        rh, rt = _read_runav(os.path.join(ea, 'rate_runav.txt'))
        bfile = [x for x in os.listdir(ea) if x.startswith('pcross_block_errors_') and x.endswith('.txt')][0]
        best, rel, blocks = _read_block(os.path.join(ea, bfile))
        rbfile = [x for x in os.listdir(ea) if x.startswith('rate_block_errors_') and x.endswith('.txt')][0]
        _, rrel, _ = _read_block(os.path.join(ea, rbfile))
        n_rows = sum(1 for line in open(os.path.join(cfg['loop_dir'], sub, 'infretis_data.txt'))
                     if line.strip() and not line.startswith('#'))
        n_skip = n_rows - int(pc[-1, 0]) if pc[-1, 0] <= n_rows else 0
        to_n = lambda rows: n_skip + rows                                     # rows -> accepted paths of the production
        rate = rt[:, rh.index('Rate')] * ase_systems.TIME_FS                 # engine time (fs) -> reduced time units
        runs.append(dict(label=label, col=col, md=to_n(pc[:, 0]), p=pc[:, -1], md_r=to_n(rt[:, 0]), mdpp=rj['cost']['md_steps'] / n_rows, rate=rate, rel=rel[-1], rrel=rrel[rh.index('Rate') - 1],
                         blocks=blocks, init=0 if sub == 'production_seed' else _init_md(cfg, sub), flux=rj['Flux'], tpf=rj['time_per_frame'], ex=(rj.get('exact') or {}).get('rate')))
    if not runs:
        return None
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    ex = next((r['ex'] for r in runs if r['ex']), None)
    for r in runs:
        band = r['rel'] * np.sqrt(r['md'][-1] / r['md'])
        ok = r['p'] > 0
        axs[0].plot(r['md'][ok], r['p'][ok], color=r['col'], lw=1.4,
                    label=f"{r['label']} ({r['mdpp']:.0f} MD steps/accepted path; init. {r['init'] / 1e6:.1f}M MD steps)")
        axs[0].fill_between(r['md'][ok], r['p'][ok] / (1 + band[ok]), r['p'][ok] * (1 + band[ok]), color=r['col'], alpha=0.15, lw=0)
        if ex:
            okr = r['rate'] > 0
            bandr = r['rrel'] * np.sqrt(r['md_r'][-1] / r['md_r'])
            y = r['rate'][okr] / ex
            axs[1].plot(r['md_r'][okr], y, color=r['col'], lw=1.4, label=r['label'])
            axs[1].fill_between(r['md_r'][okr], y / (1 + bandr[okr]), y * (1 + bandr[okr]), color=r['col'], alpha=0.15, lw=0)
        b = r['blocks']
        axs[2].plot(b[:, 0], b[:, -1], color=r['col'], lw=1.2, label=f"{r['label']}: plateau {r['rel']:.2f}")
        axs[2].axhline(r['rel'], color=r['col'], lw=0.8, ls=':')
    if ex:                                     # exact P_cross = exact rate x time per frame / flux (flux per frame)
        for r in runs:
            axs[0].axhline(ex * r['tpf'] / r['flux'], color=r['col'], lw=0.8, ls='--')
        axs[0].plot([], [], color=INK2, lw=0.8, ls='--', label='exact rate / the run\'s flux')
    pmax = max(float(np.max(r['p'])) for r in runs)
    pfin = min(float(r['p'][-1]) for r in runs)
    axs[0].set_ylim(pfin / 30, pmax * 3)
    axs[0].set(yscale='log', xlabel='accepted paths', ylabel='P_cross(lambda_B)', title='running P_cross, +-1 block error')
    if ex:
        axs[1].axhline(1, color=INK2, lw=0.8, ls='--')
        axs[1].set(yscale='log', xlabel='accepted paths', ylabel='rate / exact rate',
                   title=f'running rate / exact ({ex:.3g} per time unit)')
    axs[2].set(xlabel='block length [paths]', ylabel='relative error of P_cross(lambda_B)', title='block error (inftools)')
    for ax in axs:                             # legends below the panels: the run labels are long
        ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False)
    fig.suptitle(f"{cfg['name']}: production convergence (lambda frozen)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, 'pcross_convergence.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _init_md(cfg, sub):
    """MD steps of the initialization behind a production run: its manifest (written at freeze), else state.json."""
    m = os.path.join(cfg['loop_dir'], sub, 'manifest.json')
    if os.path.isfile(m):
        with open(m) as f:
            v = json.load(f).get('initialization_md_steps')
        if v:
            return v
    with open(os.path.join(cfg['loop_dir'], 'state.json')) as f:
        return json.load(f).get('md_steps', 0)


def _slice_values(cfg, pool):
    """Values of the first CV beyond the two plotted ones, one figure panel each: [plots] slices if given; for the
    known w systems the two well centres -w0, +w0; otherwise the 25% and 75% weighted quantiles of that CV over
    the pooled frames (for a double well about the well centres; one value, the median, if they coincide)."""
    s, pc = cfg['system_spec'], cfg.get('plots', {})
    _, _, rest = _axes(cfg)
    if not rest:
        return []
    if pc.get('slices'):
        return [float(v) for v in pc['slices']]
    c = s.get('calculator') or {}
    w0 = ((c.get('w_well') or {}).get('w0') or (c.get('params') or {}).get('w0'))
    if w0 and s['cv_names'][rest[0]] == 'w':
        return [-float(w0), float(w0)]
    cvp, wp = _pool_frames(pool)
    lo, mid, hi = _wquantile(cvp[:, rest[0]], wp, np.array([0.25, 0.5, 0.75]))
    spread = np.subtract(*np.quantile(cvp[:, rest[0]], [0.99, 0.01]))
    return [float(mid)] if hi - lo < 1e-3 * max(spread, 1e-12) else [float(f'{lo:.3g}'), float(f'{hi:.3g}')]


def _grids(cfg, pool):
    vals = _slice_values(cfg, pool)
    if not vals:                                           # 2D, or underdamped 1D: the (x, v) plane
        return [Grid(cfg, pool)]
    return [Grid(cfg, pool, w=v, slice_values=vals) for v in vals]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('--no-tis', action='store_true')
    ap.add_argument('--pcross-only', action='store_true', help='only pcross_convergence.png')
    ap.add_argument('--paths-only', action='store_true', help='only paths*.png')
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    if a.pcross_only:
        out = os.path.join(cfg['loop_dir'], 'figures')
        os.makedirs(out, exist_ok=True)
        print(fig_pcross_convergence(cfg, out))
        return
    st = _passes(cfg)
    pool = pool_mod.Pool(cfg['loop_dir'])
    if a.paths_only:
        out = os.path.join(cfg['loop_dir'], 'figures')
        os.makedirs(out, exist_ok=True)
        for grid in _grids(cfg, pool):
            print(fig_paths(cfg, grid, out))
        return
    out = os.path.join(cfg['loop_dir'], 'figures')
    os.makedirs(out, exist_ok=True)
    for grid in _grids(cfg, pool):
        fig_interfaces(cfg, st, grid, out); print(f'interfaces{_sfx(grid)}.png')
        fig_maps(cfg, st, grid, pool, out); print(f'maps{_sfx(grid)}.png')
        print(fig_paths(cfg, grid, out))
    fig_calibration(cfg, st, pool, out); print('calibration.png')
    fig_convergence(cfg, st, out); print('convergence.png')
    if not a.no_tis:
        fig_tis(cfg, st, pool, out); print('tis / feps')
    print(out)


if __name__ == '__main__':
    main()
