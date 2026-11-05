"""Seed loop with an optional warm-up and periodic plots, shared by the example notebooks.

For every seed a fresh committor object is fitted:
    1. optionally `warmup_iter` iterations with their own envelope, gamma and CV choice;
    2. then `max_iter` iterations with the main envelope and gamma, in chunks of `chunk`.
`fit_transform` continues from the current RC on every call, so the stages form one run.
After the warm-up and after every chunk the chosen plots are saved to
`<out_dir>/seed<s>/it<N>_<plot>.png` and shown in the notebook, and `report(q)` is printed.

The seed sets which CV and envelope centre each iteration draws (both use np.random).
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
    figure is saved. `extra_plot(q)` may return one more figure (saved as <tag>_extra.png).
    """
    os.makedirs(folder, exist_ok=True)
    plot_options = plot_options or {}
    show, plt.show = plt.show, (lambda *a, **k: None)
    try:
        for name in plots:
            PLOT_CALLS[name](q, plot_options)
            plt.gcf().savefig(f'{folder}/{tag}_{name}.png', dpi=100)
        if extra_plot is not None:
            fig = extra_plot(q)
            if fig is not None:
                fig.savefig(f'{folder}/{tag}_extra.png', dpi=100)
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
