"""Seed loop with an optional warm-up and periodic plots, shared by the example notebooks.

For every seed a fresh committor object is fitted:
    1. optionally `warmup_iter` iterations with their own envelope, gamma and CV choice;
    2. then `max_iter` iterations with the main envelope and gamma, in chunks of `chunk`.
`fit_transform` continues from the current RC on every call, so the stages form one run.
After the warm-up and after every chunk the chosen plots are saved to
`<out_dir>/seed<s>/it<N>_<plot>.png` and shown in the notebook, and `report(q)` is printed.

The seed sets which CV and envelope centre each iteration draws (both use np.random).

`committor_map` and `committor_profile` make `extra_plot` functions that show the fitted q
against the CVs after every chunk (mean q per bin, optionally next to the exact committor).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

PLOT_CALLS = {
    'metrics': lambda q, opt: q.plots_metrics(opt.get('metrics')),
    'feps': lambda q, opt: q.plots_feps(**opt.get('feps_kw', {})),
    'obs_pred': lambda q, opt: q.plots_obs_pred(),
    'obs_pred_log': lambda q, opt: q.plots_obs_pred(log_scale=True),
    'tis': lambda q, opt: q.plots_tis(),
}


def save_and_show(q, folder, tag, plots, plot_options=None, extra_plot=None):
    """Save the plots of `q` as <folder>/<tag>_<plot>.png, then show them in the notebook.

    The library's plot methods call plt.show() themselves; it is held off until every
    figure is saved. `extra_plot(q)` may return one more figure (saved as <tag>_extra.png);
    a dict {name: extra_plot} gives several, saved as <tag>_<name>.png.
    """
    os.makedirs(folder, exist_ok=True)
    plot_options = plot_options or {}
    extras = {} if extra_plot is None else extra_plot if isinstance(extra_plot, dict) else {'extra': extra_plot}
    show, plt.show = plt.show, (lambda *a, **k: None)
    try:
        for name in plots:
            PLOT_CALLS[name](q, plot_options)
            plt.gcf().savefig(f'{folder}/{tag}_{name}.png', dpi=100)
        for name, make_fig in extras.items():
            fig = make_fig(q)
            if fig is not None:
                fig.savefig(f'{folder}/{tag}_{name}.png', dpi=100)
    finally:
        plt.show = show
    print(f'--- {folder}/{tag} ---')
    plt.show()


def _fit(q, comp_y, gamma, n_iter, fit_kw, done):
    n_before = len(q.metrics_history.get('iter', []))
    q.fit_transform(comp_y, gamma=gamma, max_iter=n_iter, **fit_kw)
    # The equilibrium Committor restarts its iteration counter at every call (CommittorNE does not):
    # shift the new entries so the metrics plots run on.
    if type(q).__name__ == 'Committor' and 'iter' in q.metrics_history:
        it = q.metrics_history['iter']
        it[n_before:] = [i + done for i in it[n_before:]]


def run_seeds(orc, make_q, comp_y, fit_kw, seeds, max_iter, chunk, out_dir, envelope, gamma,
              warmup_iter=0, warmup_envelope=None, warmup_gamma=None, warmup_comp_y=None,
              plots=('metrics',), plot_options=None, extra_plot=None, report=None):
    """Fit one committor per seed; see the module docstring.

    orc: the optimalrcs module (its `envelope_scale` is set per stage).
    make_q(): returns a fresh committor object (called after seeding).
    comp_y(): the CV of each iteration in the main run; warmup_comp_y in the warm-up (default comp_y).
    fit_kw: other fit_transform arguments (ny, print_step, metrics_print, ...).
    extra_plot: q -> figure, or {name: q -> figure}, shown and saved after every chunk
    (e.g. `committor_map(...)`).
    Returns {seed: {'r': final RC (numpy), 'report': last report}} and the last seed's object;
    the earlier objects are freed to keep GPU memory for the next seed.
    """
    results, q = {}, None
    warmup_envelope = envelope if warmup_envelope is None else warmup_envelope
    warmup_gamma = gamma if warmup_gamma is None else warmup_gamma
    for seed in seeds:
        if q is not None:
            del q
            cp.get_default_memory_pool().free_all_blocks()
        np.random.seed(seed); cp.random.seed(seed)
        q = make_q()
        folder = f'{out_dir}/seed{seed}'
        stages = []
        if warmup_iter > 0:
            stages.append((warmup_iter, warmup_envelope, warmup_gamma, warmup_comp_y or comp_y, 'warm-up'))
        n_chunks, rest = divmod(max_iter, chunk)
        stages += [(chunk, envelope, gamma, comp_y, 'main')] * n_chunks
        if rest:
            stages.append((rest, envelope, gamma, comp_y, 'main'))
        done, text = 0, ''
        for n_iter, env, gam, cy, stage in stages:
            orc.envelope_scale = env
            _fit(q, cy, gam, n_iter, fit_kw, done)
            done += n_iter
            save_and_show(q, folder, f'it{done}', plots, plot_options, extra_plot)
            text = report(q) if report is not None else ''
            print(f'seed {seed}, {done} iterations ({stage}: envelope {env}, gamma {gam}) {text}', flush=True)
        r = cp.asnumpy(q.r_traj)
        np.save(f'{folder}/q_traj.npy', r.astype(np.float32))
        results[seed] = {'r': r, 'report': text}
    for seed, res in results.items():
        print(f'seed {seed}: {res["report"]}')
    return results, q


class BinnedMean:
    """Mean of a per-frame value in fixed bins of one or two coordinates.

    The bin of every frame is found once, so the mean of a new RC costs one bincount.
    Bins with fewer than `min_count` frames (of `mask`) are NaN.
    """

    def __init__(self, coords, edges, mask=None, min_count=20):
        n = len(coords[0])
        self.keep = np.ones(n, bool) if mask is None else np.array(mask, bool)
        flat = np.zeros(n, np.int64)
        for c, e in zip(coords, edges):
            i = np.digitize(c, e) - 1
            self.keep &= (i >= 0) & (i < len(e) - 1)
            flat = flat * (len(e) - 1) + np.clip(i, 0, len(e) - 2)
        self.idx = flat[self.keep]
        self.shape = tuple(len(e) - 1 for e in edges)
        self.count = np.bincount(self.idx, minlength=int(np.prod(self.shape))).reshape(self.shape)
        self.ok = self.count >= min_count

    def __call__(self, val):
        s = np.bincount(self.idx, weights=np.asarray(val)[self.keep], minlength=int(np.prod(self.shape)))
        return np.where(self.ok, s.reshape(self.shape) / np.maximum(self.count, 1), np.nan)


def plane(c1, edges1, label1, c2, edges2, label2, contours=None):
    """One (c1, c2) plane for `committor_map`. contours = (g1, g2, V, levels): grey contours of
    a potential V[i1, i2] on the grid (g1, g2), e.g. the Z potential."""
    return dict(c=(c1, c2), edges=(edges1, edges2), labels=(label1, label2), contours=contours)


def _rms_transition(diff, ref):
    trans = (ref > 0.05) & (ref < 0.95)
    return np.sqrt(np.nanmean(diff[trans] ** 2)) if trans.any() else np.nan


def committor_map(planes, mask=None, q_ref=None, min_count=20, diff_lim=0.15):
    """extra_plot for `run_seeds`: the fitted q against two CVs, after every chunk.

    planes: list of `plane(...)`. Each shows the mean q per bin (frames of `mask`, at least
    `min_count` per bin), with dashed q = 0.1, 0.5, 0.9. Without `q_ref` the planes are side by
    side. With `q_ref` (exact committor per frame) every plane gets a row: fit, exact, and
    fit - exact with its RMS over the bins where 0.05 < exact < 0.95.
    Returns a function q -> figure.
    """
    binned = [BinnedMean(p['c'], p['edges'], mask, min_count) for p in planes]
    refs = [None if q_ref is None else bm(q_ref) for bm in binned]
    n_rows, n_cols = (1, len(planes)) if q_ref is None else (len(planes), 3)

    def plot(q):
        r = cp.asnumpy(q.r_traj)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.5 * n_rows), squeeze=False)
        for k, (p, bm, Q_ref) in enumerate(zip(planes, binned, refs)):
            Q = bm(r)
            panels = [(Q, 'fit', 'coolwarm', (0, 1))]
            if Q_ref is not None:
                panels += [(Q_ref, 'exact', 'coolwarm', (0, 1)),
                           (Q - Q_ref, f'fit - exact, RMS {_rms_transition(Q - Q_ref, Q_ref):.4f}', 'PuOr',
                            (-diff_lim, diff_lim))]
            row = axes[k] if q_ref is not None else axes[0, k:k + 1]
            (e1, e2), (l1, l2) = p['edges'], p['labels']
            m1, m2 = (e1[1:] + e1[:-1]) / 2, (e2[1:] + e2[:-1]) / 2
            for ax, (M, title, cmap, (lo, hi)) in zip(row, panels):
                im = ax.pcolormesh(e1, e2, M.T, cmap=cmap, vmin=lo, vmax=hi)
                if p['contours'] is not None:
                    g1, g2, V, lev = p['contours']
                    ax.contour(g1, g2, V.T, levels=lev, colors='0.4', linewidths=0.5)
                if lo == 0:
                    ax.contour(m1, m2, np.nan_to_num(M, nan=0.5).T, levels=[0.1, 0.5, 0.9],
                               colors='k', linewidths=1, linestyles='--')
                ax.set(title=title, xlabel=l1, ylabel=l2, xlim=(e1[0], e1[-1]), ylim=(e2[0], e2[-1]))
                fig.colorbar(im, ax=ax, label='$q$' if lo == 0 else '')
        fig.tight_layout()
        return fig
    return plot


def committor_profile(c, edges, label, mask=None, ref=None, min_count=20):
    """extra_plot for `run_seeds`: the mean fitted q per bin of one CV, after every chunk.

    ref = (x, q_exact): the exact committor as a curve; adds it, and fit - exact with its RMS.
    Returns a function q -> figure.
    """
    bm = BinnedMean((c,), (edges,), mask, min_count)
    mid = (edges[1:] + edges[:-1]) / 2
    q_ex = None if ref is None else np.interp(mid, *ref)

    def plot(q):
        Q = bm(cp.asnumpy(q.r_traj))
        fig, axes = plt.subplots(1, 1 if ref is None else 2, figsize=(6 if ref is None else 12, 4), squeeze=False)
        ax = axes[0, 0]
        ax.plot(mid, Q, label='fit')
        if ref is not None:
            ax.plot(mid, q_ex, 'k--', label='exact')
            ax.legend()
            axes[0, 1].plot(mid, Q - q_ex)
            axes[0, 1].set(xlabel=label, ylabel='fit $-$ exact',
                           title=f'RMS {np.sqrt(np.nanmean((Q - q_ex) ** 2)):.4f}')
            axes[0, 1].grid()
        ax.set(xlabel=label, ylabel='$q$')
        ax.grid()
        fig.tight_layout()
        return fig
    return plot
