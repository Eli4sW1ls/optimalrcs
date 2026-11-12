"""Redraw the feps plot of saved committor fits with the stopped Z_q.

Older fits were checked with the Z_q of `plots_feps`, which drops every window that runs past the end of
a piece. When the end depends on the trajectory (a turn of a staple path, A or B), that conditions on the
future and biases Z_q even for the exact committor. Here every piece end that is not in A or B is a stop
(`FutureBoundary.set_stopped_end`: a window passing it uses q there) and the A/B ends get a trap longer
than the longest piece plus the longest lag, so no window is dropped: Z_q tests q(min(t + dt, tau)) - q(t).

For every saved fit found under the given folders (default: examples/figures_by_potential) one figure is written to
<out>/<path of the fit>.png with four panels: F(q), F(q_n), the stopped Z_q of plots_feps (not normalised)
and the same test divided by its standard error (metrics_tis, about 2 or less for the committor).
A summary table goes to <out>/summary.csv. Path weights are not used unless --weights is given.

What is found:
- npz with q and the frames: q in r / q_fit / q_traj / q_np / q_npq / q_exact (each one present is drawn),
  segments in i_traj (or piece), A/B in boundary0/boundary1, b0/b1 or A/B, or from the order parameter
  (lam, y, X_all[:, 0] or cv[:, 0]) and lambda_a/lambda_b stored in the file.
- q alone (q_traj.npy, q_*.npy, r_*.npy, or an npz holding only r): the frames are taken from the nearest
  frames file (any npz above with segments and A/B) with the same number of frames, else from re-reading
  the paths for staple_compare_p<P>_s<S>/<config>/ and toy2d staple[_test]/<fit tag>/ runs, else
  from a frames file anywhere under the folders, if exactly one has that number of frames.
  A match by frame count does not prove the frames are in the same order: check that the F(q) looks
  like the original plot.
- MANUAL below: (q file, frames file, extra) for anything else.

Usage (on the GPU machine, in the rctis environment):
    python rerun_stopped_zq.py                       # everything under examples/figures_by_potential
    python rerun_stopped_zq.py figures_by_potential/2d_zpot --dry-run
    python rerun_stopped_zq.py --only zpot --weights
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

EXAMPLES = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(EXAMPLES)
sys.path.insert(0, f'{REPO}/optimalrcs-cupy')
sys.path.insert(0, EXAMPLES)

FIGURES = os.path.join(EXAMPLES, 'figures_by_potential')
Q_KEYS = ('r', 'q_fit', 'q_traj', 'q_np', 'q_npq', 'q_exact')
SKIP_DIRS = ('exact_committors', 'exact', 'stopped_zq', '__pycache__')
SIMS = '/home/staff7/fbsskr/Notebooks/Students/elias/data/simulations'
LAMBDAS = {'sim_istar0br10intf': (-0.1, 0.8), 'sim_istarz_br_g10': (0.05, 0.85)}   # staple_compare presets

# Fits that saved q only, with the tis_data arguments of the script that made them (in the same folder):
# (pattern of the q file below figures_by_potential/, simulation folder name, tis_data arguments, half). half: None, or
# 'even'/'odd' for fits on the paths with an even/odd group number (half_fits.py, lag_halves.py).
# The frames are checked by their number; older runs drew --seed-dependent subsets, which tis_data no longer
# does, so those fail the check instead of giving wrong frames.
_LGV = ['--seed', '0', '--fd-velocity']
_ZREV = ['--reversed', '--seed', '0']
KNOWN = [
    ('1d_flat/langevin_staple/lgv_check/fit_basis_*.npz', 'sim_istar0lgv15intf', ['--paths-per-ensemble', '30'] + _LGV, None),
    ('1d_flat/langevin_staple/lgv_check/fit_lag1.npz', 'sim_istar0lgv15intf', ['--paths-per-ensemble', '30'] + _LGV, None),
    ('1d_flat/langevin_staple/lgv_check/fit_lag4.npz', 'sim_istar0lgv15intf', ['--paths-per-ensemble', '30'] + _LGV, None),
    ('1d_flat/langevin_staple/lgv_check/fit_lag16.npz', 'sim_istar0lgv15intf', ['--paths-per-ensemble', '30'] + _LGV, None),
    ('1d_flat/langevin_staple/lgv_check/fit_g0.1_p100*.npz', 'sim_istar0lgv15intf', ['--paths-per-ensemble', '100'] + _LGV, None),
    ('1d_flat/langevin_staple/lgv_check/fit_lag4_p100_s0.npz', 'sim_istar0lgv15intf', ['--paths-per-ensemble', '100'] + _LGV, None),
    ('1d_flat/langevin_staple/lgv_check/fit_lag4_p30_s1.npz', 'sim_istar0lgv15intf', ['--paths-per-ensemble', '30', '--seed', '1', '--fd-velocity'], None),
    ('1d_flat/langevin_staple/lgv_check/fit_lag4_p30_s2.npz', 'sim_istar0lgv15intf', ['--paths-per-ensemble', '30', '--seed', '2', '--fd-velocity'], None),
    ('1d_cos_well/langevin_staple/checks/fit_fd_lag*.npz', 'sim_istarwell2rv', ['--paths-per-ensemble', '2000'] + _LGV, None),
    ('1d_cos_well/langevin_staple/checks/fit_recorded_lag*.npz', 'sim_istarwell2rv', ['--paths-per-ensemble', '2000'] + _LGV, None),
    ('1d_cos_well/langevin_staple/checks/fit_rev_*_lag*.npz', 'sim_istarwell2rv', ['--paths-per-ensemble', '2000', '--reversed'] + _LGV, None),
    ('2d_zpot/brownian_g10/staple/selection_test/base/*/q_traj.npy', 'sim_istarz_br_g10', ['--paths-per-ensemble', '1500'] + _ZREV, None),
    ('2d_zpot/brownian_g10/staple/selection_test/flat_qn/*/q_traj.npy', 'sim_istarz_br_g10', ['--paths-per-ensemble', '1500'] + _ZREV, None),
    ('2d_zpot/brownian_g10/staple/reversed/r_half_even.npy', 'sim_istarz_br_g10', ['--paths-per-ensemble', '1500'] + _ZREV, 'even'),
    ('2d_zpot/brownian_g10/staple/reversed/r_half_odd.npy', 'sim_istarz_br_g10', ['--paths-per-ensemble', '1500'] + _ZREV, 'odd'),
    ('2d_zpot/brownian_g10/staple/reversed/r_lag50_even_ppe1500.npy', 'sim_istarz_br_g10', ['--paths-per-ensemble', '1500'] + _ZREV, 'even'),
    ('2d_zpot/brownian_g10/staple/reversed/r_lag50_odd.npy', 'sim_istarz_br_g10', ['--paths-per-ensemble', '1500'] + _ZREV, 'odd'),
    ('2d_zpot/brownian_g10/staple/reversed/r_lag50_all_ppe2250.npy', 'sim_istarz_br_g10', ['--paths-per-ensemble', '2250'] + _ZREV, None),
]

# Fits on the first N frames of one long MD run (md_path.npy of the simulation folder: columns position,
# velocity): (pattern, simulation folder name, N, lambda_A, lambda_B). One trajectory, so the stopped and
# the dropped-window Z_q differ only at its end.
KNOWN_MD = [
    ('1d_cos_well/langevin_md/fits/*.npy', 'md_istarwell2_100M', 6_000_000, -0.35, 0.35),
    ('1d_cos_well/langevin_md/fits/*/q_*.npy', 'md_istarwell2_100M', 6_000_000, -0.35, 0.35),
    ('1d_cos_well/langevin_md/seeds/*/q_traj.npy', 'md_istarwell2_100M', 6_000_000, -0.35, 0.35),
]

# (q file, frames file, extra) for anything else, paths relative to examples/; extra may hold
# lambda_a/lambda_b for frames files without A/B, e.g.
# ('figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/npq_md/npq_md.npz', 'figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/npq_md/npq_md.npz', dict(lambda_a=0.1, lambda_b=0.85)),
MANUAL = []


# ------------------------------------------------------------------------------------------------------
# reading saved files without loading them
# ------------------------------------------------------------------------------------------------------
def npz_arrays(path):
    """{name: shape} of the arrays in an npz, read from the headers only."""
    out = {}
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.filename.endswith('.npy'):
                    continue
                with z.open(info) as f:
                    version = np.lib.format.read_magic(f)
                    shape = np.lib.format._read_array_header(f, version)[0]
                out[info.filename[:-4]] = shape
    except Exception:
        return None
    return out


def frame_count(arrays):
    """Number of frames of a frames file (length of its segment array), or None if it has none."""
    for k in ('i_traj', 'piece'):
        if k in arrays and len(arrays[k]) == 1:
            return arrays[k][0]
    return None


def has_boundaries(arrays):
    if {'boundary0', 'boundary1'} <= arrays.keys() or {'b0', 'b1'} <= arrays.keys() or {'A', 'B'} <= arrays.keys():
        return True
    has_order = any(k in arrays for k in ('lam', 'y', 'X_all', 'cv'))
    return has_order and {'lambda_a', 'lambda_b'} <= arrays.keys()


def load_frames(path, extra=None):
    """Segments, A/B, times, groups and weights of a frames file."""
    extra = extra or {}
    d = np.load(path, allow_pickle=False)
    k = 'i_traj' if 'i_traj' in d.files else 'piece'
    i_traj = np.asarray(d[k])
    for b0, b1 in (('boundary0', 'boundary1'), ('b0', 'b1'), ('A', 'B')):
        if b0 in d.files and b1 in d.files:
            A, B = np.asarray(d[b0], bool), np.asarray(d[b1], bool)
            break
    else:
        if 'lam' in d.files:
            lam = d['lam']
        elif 'y' in d.files and d['y'].ndim == 1:
            lam = d['y']
        elif 'X_all' in d.files:
            lam = d['X_all'][:, 0]
        else:
            lam = d['cv'][:, 0]
        la = float(extra.get('lambda_a', d['lambda_a'] if 'lambda_a' in d.files else np.nan))
        lb = float(extra.get('lambda_b', d['lambda_b'] if 'lambda_b' in d.files else np.nan))
        A, B = lam <= la, lam >= lb
    t = next((np.asarray(d[k], float) for k in ('t_traj', 't') if k in d.files), None)
    g = next((np.asarray(d[k]) for k in ('group_traj', 'path') if k in d.files), None)
    w = next((np.asarray(d[k], float) for k in ('path_weights', 'w') if k in d.files and d[k].shape == i_traj.shape), None)
    return dict(i_traj=i_traj, A=A, B=B, t=t, group=g, w=w)


# ------------------------------------------------------------------------------------------------------
# frames of runs that saved q only: re-read the paths with tis_data
# ------------------------------------------------------------------------------------------------------
def sim_dir(name):
    """The simulation folder `name`, in toytis' simulations or in the data folder, or None."""
    return next((d for d in (f'{toytis()}/simulations/{name}', f'{SIMS}/{name}') if os.path.isdir(d)), None)


def known_source(q_path):
    """('reread', argv, half) of a fit listed in KNOWN, ('md', md_path.npy, (N, lambda_A, lambda_B)) of one
    in KNOWN_MD, or None."""
    import fnmatch
    rel = os.path.relpath(q_path, FIGURES)
    for pattern, sim, args, half in KNOWN:
        if fnmatch.fnmatch(rel, pattern):
            d = sim_dir(sim)
            return ('reread', [d] + args, half) if d else None
    for pattern, sim, n, la, lb in KNOWN_MD:
        if fnmatch.fnmatch(rel, pattern):
            d = sim_dir(sim)
            return ('md', f'{d}/md_path.npy', (n, la, lb)) if d and os.path.exists(f'{d}/md_path.npy') else None
    return None


def md_frames(path, n, la, lb):
    """First n frames of one long MD run as a single trajectory."""
    lam = np.array(np.load(path, mmap_mode='r')[:n, 0], float)
    return dict(i_traj=np.zeros(len(lam), np.int64), A=lam <= la, B=lam >= lb, t=np.arange(len(lam), dtype=float),
                group=None, w=None)


# where the staple_compare_* folders and the toy2d staple fits of each simulation are (below figures_by_potential/)
STAPLE_COMPARE_SIMS = {'1d_flat/brownian_staple': 'sim_istar0br10intf', '2d_zpot/brownian_g10/staple': 'sim_istarz_br_g10',
                       '2d_curved_channel/brownian_g10': 'sim_istar_cc_br'}
TOY2D_STAPLE_RUNS = {'2d_curved_channel/brownian_g10/staple': 'sim_istar_cc_br', '2d_curved_channel/brownian_g10/staple_test': 'test_cc_br',
                     '2d_curved_channel/langevin_g2/staple': 'sim_istar_cc_lgv', '2d_curved_channel/langevin_g2/staple_test': 'test_cc_lgv'}


def reread_argv(q_path):
    """tis_data arguments that rebuild the frames of a staple_compare or toy2d staple fit, or None."""
    rel = os.path.relpath(q_path, FIGURES)
    m = re.match(r'(.+)/staple_compare_p(\d+|None)_s(\d+)/([^/]+)/', rel)
    if m and m.group(1) in STAPLE_COMPARE_SIMS:
        sim, P, S = STAPLE_COMPARE_SIMS[m.group(1)], m.group(2), m.group(3)
        summary = os.path.join(FIGURES, m.group(1), f'staple_compare_p{P}_s{S}', m.group(4), 'summary.npz')
        if not os.path.exists(summary):
            return None
        cfg = json.loads(str(np.load(summary)['cfg']))
        tis_dir = sim_dir(sim)
        if tis_dir is None:
            return None
        argv = [tis_dir, '--weights', cfg['weights'], '--long-pieces', cfg.get('long_pieces', 'truncate')]
        if sim in LAMBDAS:
            argv += ['--lambda-a', str(LAMBDAS[sim][0]), '--lambda-b', str(LAMBDAS[sim][1])]
        if P != 'None':
            argv += ['--paths-per-ensemble', P, '--seed', S]
        argv += ['--reversed'] if cfg.get('reversed') else []
        argv += [] if cfg.get('cut', True) else ['--no-cut']
        argv += ['--max-length', str(cfg['max_length'])] if cfg.get('max_length') else []
        return argv
    for folder, run in TOY2D_STAPLE_RUNS.items():
        if rel.startswith(folder + '/'):
            tag = rel[len(folder) + 1:].split('/')[0]
            m = re.match(r'(rev|fwd)_(rejections|staple|none)_p(\d+|None)_s(\d+)(?:_len(\d+))?_([a-z\-]+)_lag', tag)
            tis_dir = sim_dir(run)
            if not m or tis_dir is None:
                return None
            rev, weights, P, S, length, cvs = m.groups()
            argv = [tis_dir, '--weights', weights]
            argv += ['--paths-per-ensemble', P, '--seed', S] if P != 'None' else []
            argv += ['--reversed'] if rev == 'rev' else []
            argv += ['--max-length', length] if length else []
            argv += ['--fd-velocity'] if ('dy' in cvs.split('-') or 'dx' in cvs.split('-')) else []
            return argv
    return None


def toytis():
    import toy2d
    return toy2d.TOYTIS


_reread_cache = {}


def reread_frames(argv, half=None):
    """Frames re-read with tis_data; half 'even'/'odd': only the paths with an even/odd group number."""
    key = tuple(argv)
    if key not in _reread_cache:
        import contextlib
        import io
        import tis_data
        parser = argparse.ArgumentParser()
        tis_data.add_data_arguments(parser)
        with contextlib.redirect_stdout(io.StringIO()):
            data = tis_data.prepare_data(parser.parse_args(argv))
        _reread_cache.clear()                        # keep one data set in memory
        _reread_cache[key] = dict(i_traj=data.i_traj, A=data.boundary0, B=data.boundary1, t=data.t_traj,
                                  group=data.group_traj, w=data.path_weights)
    fr = _reread_cache[key]
    if half is None:
        return fr
    keep = (fr['group'] % 2) == (0 if half == 'even' else 1)
    sub = {k: (None if v is None else v[keep]) for k, v in fr.items()}
    if sub['w'] is not None:
        sub['w'] = sub['w'] / sub['w'].mean()
    return sub


# ------------------------------------------------------------------------------------------------------
# finding the fits
# ------------------------------------------------------------------------------------------------------
def find_jobs(roots, only=None):
    """[(label, q file, q key or None, frames source)]; frames source: ('file', path, extra) or ('reread', argv)."""
    files = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            files += [os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(('.npz', '.npy'))]
    frames_files, q_only, jobs, skipped = {}, [], [], []
    for f in files:
        if f.endswith('.npz'):
            arrays = npz_arrays(f)
            if not arrays:
                continue
            n = frame_count(arrays)
            if n is not None and has_boundaries(arrays):
                frames_files[f] = n
                for k in Q_KEYS:
                    if k in arrays and arrays[k] == (n,):
                        jobs.append((f'{f}:{k}', f, k, ('file', f, None)))
            elif n is None and set(arrays) == {'r'} and len(arrays['r']) == 1:
                q_only.append((f, 'r', arrays['r'][0]))
        else:
            name = os.path.basename(f)
            if name == 'q_traj.npy' or re.match(r'(q|r)_.*\.npy$', name):
                arr = np.load(f, mmap_mode='r')
                if arr.ndim == 1:
                    q_only.append((f, None, arr.shape[0]))
    by_count = {}
    for f, n in frames_files.items():
        by_count.setdefault(n, []).append(f)
    for f, key, n in q_only:
        label = f if key is None else f'{f}:{key}'
        src = known_source(os.path.abspath(f))
        d = os.path.dirname(f)
        while src is None and any(os.path.commonpath([os.path.abspath(d), r]) == r for r in roots):
            near = [g for g, m in frames_files.items() if os.path.dirname(g) == d and m == n]
            if len(near) == 1:
                src = ('file', near[0], None)
            elif len(near) > 1:                       # several frames files with this many frames: the one named alike
                stem = os.path.splitext(os.path.basename(f))[0]
                named = [g for g in near if os.path.splitext(os.path.basename(g))[0].split('_', 1)[-1] in stem]
                if len(named) == 1:
                    src = ('file', named[0], None)
            if d == os.path.dirname(d):
                break
            d = os.path.dirname(d)
        if src is None:
            argv = reread_argv(os.path.abspath(f))
            src = ('reread', argv, None) if argv else None
        if src is None and len(by_count.get(n, [])) == 1:
            src = ('file', by_count[n][0], 'matched by frame count only')
        (jobs.append((label, f, key, src)) if src else skipped.append((label, n)))
    for q_file, frames_file, extra in MANUAL:
        q_file, frames_file = (p if os.path.isabs(p) else os.path.join(EXAMPLES, p) for p in (q_file, frames_file))
        jobs.append((q_file, q_file, None, ('file', frames_file, extra)))
    if only:
        jobs = [j for j in jobs if only in j[0]]
        skipped = [s for s in skipped if only in s[0]]
    return jobs, skipped


def load_q(q_file, key):
    if key is None:
        return np.asarray(np.load(q_file) if q_file.endswith('.npy') else np.load(q_file)['r'], float)
    return np.asarray(np.load(q_file)[key], float)


# ------------------------------------------------------------------------------------------------------
# the stopped Z_q
# ------------------------------------------------------------------------------------------------------
def stopped_zq(r, fr, ldt, use_weights, nbins_z=200):
    """Stopped Z_q per lag (plots_feps form) and its z-score (metrics_tis), as host arrays."""
    import cupy as cp
    import boundaries as bd
    import cut_profiles
    import metrics_tis
    i_traj, A, B = fr['i_traj'], fr['A'], fr['B']
    n = len(r)
    b = A | B
    ends = np.r_[np.flatnonzero(i_traj[1:] != i_traj[:-1]), n - 1]
    stop = np.zeros(n, bool)
    stop[ends] = ~b[ends]
    longest = int(np.max(np.diff(np.r_[-1, ends])))
    trap = longest + max(ldt) + 1
    r_g, b_g, i_g = cp.asarray(r), cp.asarray(b, dtype=cp.float64), cp.asarray(i_traj)
    t_g = None if fr['t'] is None else cp.asarray(fr['t'])
    w_g = cp.asarray(fr['w']) if (use_weights and fr['w'] is not None) else None
    fb = bd.FutureBoundary(r_g, b_g, t_g, i_g)
    fb.set_distance_to_end_fixed_traj_length_trap(i_g, b_g, trap)
    fb.set_stopped_end(i_g, stop)
    pb = bd.PastBoundary(r_g, b_g, t_g, i_g)
    tau = metrics_tis.stop_index(i_g, b_g)
    zq, zs = [], []
    for dt in ldt:
        lx, z = cut_profiles.comp_zq(r_g, b_g, i_g, fb, pb, w_traj=w_g, dt=dt)
        zq.append((cp.asnumpy(lx[:-2]), cp.asnumpy(z[:-1])))
        lz, s = metrics_tis.comp_zq_stopped_zscore(r_g, b_g, i_g, w_g, dt, nbins_z, tau, fr['group'])
        zs.append((cp.asnumpy(lz), cp.asnumpy(s)))
    info = dict(frames=n, pieces=len(ends), stopped_ends=int(stop.sum()), ab_ends=int(len(ends) - stop.sum()),
                longest=longest, trap=trap, weights=w_g is not None)
    del fb, pb, tau, r_g, b_g, i_g, t_g, w_g
    cp.get_default_memory_pool().free_all_blocks()
    return zq, zs, info


def draw(r, fr, zq, zs, ldt, title, out_png, use_weights, dt_sim, skip_bins=5):
    import cupy as cp
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plots
    w = cp.asarray(fr['w']) if (use_weights and fr['w'] is not None) else None
    i_g, t_g = cp.asarray(fr['i_traj']), None if fr['t'] is None else cp.asarray(fr['t'])
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.2))
    try:
        plots.plot_fep(axes[0], cp.asarray(r), i_traj=i_g, t_traj=t_g, w_traj=w)
        plots.plot_fep(axes[1], cp.asarray(r), i_traj=i_g, t_traj=t_g, w_traj=w, natural=True, dt_sim=dt_sim)
        axes[1].set_xlabel('$q_n$')
    except Exception as e:                           # e.g. too few distinct q values for the spline
        axes[0].set_title(f'F(q) failed: {e}', fontsize=9)
    for (lx, z), dt in zip(zq, ldt):
        axes[2].plot(lx, z, lw=1)
    axes[2].set(xlabel='$q$', ylabel='$Z_q$ (stopped)', title='stopped $Z_q$, lags ' + f'{ldt[0]}..{ldt[-1]}')
    axes[2].grid()
    for (lz, s), dt in zip(zs, ldt):
        axes[3].plot(lz[skip_bins:], s[skip_bins:], lw=1, label=str(dt))
    for v in (-2, 2):
        axes[3].axhline(v, color='k', ls=':', lw=0.8)
    axes[3].set(xlabel='$q$', ylabel='$Z_q$ / standard error', title='stopped $Z_q$ z-score')
    axes[3].legend(ncol=2, fontsize=7, title='lag', title_fontsize=7)
    axes[3].grid()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=100, bbox_inches='tight')
    plt.close(fig)


def describe(src):
    """Where a fit's frames come from, in words."""
    if src[0] == 'file':
        return os.path.relpath(src[1], EXAMPLES) + (f' ({src[2]})' if isinstance(src[2], str) else '')
    if src[0] == 'md':
        n, la, lb = src[2]
        return f'MD {os.path.basename(os.path.dirname(src[1]))}, first {n:,} frames, A <= {la}, B >= {lb}'
    return f'tis_data {os.path.basename(src[1][0])} ' + ' '.join(src[1][1:]) + (f' ({src[2]} half)' if src[2] else '')


def get_frames(src, cache):
    """Frames of a source; the last file or MD run read is kept in `cache`."""
    if src[0] == 'reread':
        return reread_frames(src[1], src[2])
    key = (src[0], src[1], str(src[2]))
    if cache.get('key') != key:
        cache.clear()
        cache['key'] = key
        cache['frames'] = (md_frames(src[1], *src[2]) if src[0] == 'md'
                           else load_frames(src[1], src[2] if isinstance(src[2], dict) else None))
    return cache['frames']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('roots', nargs='*', default=[FIGURES], help='folders to search')
    p.add_argument('--out', default=os.path.join(FIGURES, '_shared', 'stopped_zq'), help='output folder')
    p.add_argument('--only', help='only fits whose path contains this text')
    p.add_argument('--weights', action='store_true', help='use the path weights of the frames (default: none)')
    p.add_argument('--max-lag-exp', type=int, default=15, help='lags 2^0 .. 2^N (default 15, as plots_feps)')
    p.add_argument('--dt-sim', type=float, default=1.0, help='time step for the natural coordinate axis')
    p.add_argument('--no-plot', action='store_true', help='only the summary table')
    p.add_argument('--dry-run', action='store_true', help='list what would be done')
    a = p.parse_args()
    roots = [os.path.abspath(r) for r in a.roots]
    ldt = [2 ** i for i in range(a.max_lag_exp + 1)]
    jobs, skipped = find_jobs(roots, a.only)
    print(f'{len(jobs)} fits to redraw, {len(skipped)} without frames')
    for label, _, _, src in jobs:
        print(f'  {os.path.relpath(label, EXAMPLES)}  <-  {describe(src)}')
    for label, n in skipped:
        print(f'  no frames for {os.path.relpath(label, EXAMPLES)} ({n} frames): add it to MANUAL')
    if a.dry_run:
        return
    rows = []
    cache = {}
    for label, q_file, key, src in sorted(jobs, key=lambda j: (str(j[3][1]), str(j[3][2]))):   # one read per source
        rel = os.path.relpath(label, EXAMPLES)
        t0 = time.time()
        try:
            fr = get_frames(src, cache)
            r = load_q(q_file, key)
            if len(r) != len(fr['i_traj']):
                raise ValueError(f'{len(r)} q values but {len(fr["i_traj"])} frames')
            zq, zs, info = stopped_zq(r, fr, ldt, a.weights)
            worst = [np.max(np.abs(s[5:])) for _, s in zs]
            k = int(np.argmax(worst))
            row = dict(fit=rel, frames_from=describe(src),
                       **info, max_abs_z=round(float(worst[k]), 2), at_lag=ldt[k],
                       max_abs_z_per_lag=' '.join(f'{v:.1f}' for v in worst))
            if not a.no_plot:
                out_png = os.path.join(a.out, rel.replace(':', '_').replace(os.sep, '__') + ('_w' if a.weights else '') + '.png')
                draw(r, fr, zq, zs, ldt, f'{rel}: {info["pieces"]} pieces ({info["stopped_ends"]} stopped ends), '
                     f'worst |z| {worst[k]:.1f} at lag {ldt[k]}' + (' (weighted)' if info['weights'] else ' (no weights)'),
                     out_png, a.weights, a.dt_sim)
                row['figure'] = os.path.relpath(out_png, EXAMPLES)
            print(f'{rel}: {info["frames"]:,} frames, worst |z| {worst[k]:.1f} at lag {ldt[k]} ({time.time() - t0:.0f} s)', flush=True)
        except Exception as e:
            row = dict(fit=rel, error=f'{type(e).__name__}: {e}')
            print(f'{rel}: FAILED {row["error"]}', flush=True)
        rows.append(row)
    os.makedirs(a.out, exist_ok=True)
    fields = sorted({k for row in rows for k in row}, key=lambda k: (k != 'fit', k))
    with open(os.path.join(a.out, 'summary' + ('_w' if a.weights else '') + '.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    print('summary:', os.path.join(a.out, 'summary' + ('_w' if a.weights else '') + '.csv'))


if __name__ == '__main__':
    main()
