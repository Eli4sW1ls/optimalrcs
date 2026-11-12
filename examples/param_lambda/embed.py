"""Embedding test: the 2D staple frames of a base system plus extra coordinates that do not interact with (y, x).

    python run.py embed z_emb               frame cache of the embedded system (configs/z_emb.toml, [embed])
    python run.py fit-qnp-embedded cc_emb_np   NPNEq fit of q_np on all embedded CVs (labels for cc_emb_np)

The extra coordinates are stationary Ornstein-Uhlenbeck processes (AR(1) per saved frame) that do not interact
with (y, x). Then the joint path ensemble factorizes: appending independent realizations to the existing staple
paths samples the N-D path ensemble exactly (the extra coordinates are time-reversible, so mirrored pieces are
fine), and the exact committor stays q_2D(y, x). The CVs:

- y, x                the relevant coordinates (the base CVs)
- c<cv><a>            noisy copies: <cv> + a * xi, xi a fast unit OU process of its own (copy_tau frames);
                      <cv> is y, x, u = (y + x)/sqrt2 or v = (y - x)/sqrt2; a in units of 0.01
- f01 ...             fast irrelevant coordinates, OU with relaxation times fast_taus (cycled), SD fast_sd
- w                   one slow coordinate, OU with relaxation time slow_tau >> path length: nearly constant
                      along a path, so it identifies the path (the q_np labels are path-correlated noise)

Each coordinate depends on the physical frame (path, frame index), so the forward and mirrored pieces of a path
see the same values.
"""
import os
import runpy
import sys

import numpy as np
from scipy.signal import lfilter

import common
import data


def _physical_index(fz):
    """Index of every cached frame in the concatenation of all paths (frame 0 .. len - 1 of each path)."""
    path_len = fz['path_len'].astype(np.int64)
    offset = np.concatenate([[0], np.cumsum(path_len)[:-1]])
    pos = np.searchsorted(fz['path_id'], fz['path'])
    mirrored = fz['piece_mirrored'][fz['piece']]
    t = fz['t'].astype(np.int64)
    # a forward piece holds frames t* .. len - 1 at their own index; a mirrored piece holds frames t_last .. 0 of
    # the path in reverse order (its t counts from 0), so its frame t is the path's frame t_last - t
    t_phys = np.where(mirrored, fz['path_tlast'][pos] - t, t)
    assert np.all((t_phys >= 0) & (t_phys < path_len[pos]))
    return offset[pos] + t_phys, path_len, offset


def _ou(rng, path_len, offset, tau, sd):
    """Stationary AR(1) series of every path, concatenated: x_0 ~ N(0, sd^2), x_t = rho x_{t-1} + sqrt(1-rho^2) sd e_t,
    rho = exp(-1/tau). Paths are independent (the filter's carry-over between paths is removed)."""
    n = int(path_len.sum())
    rho = np.exp(-1.0 / tau)
    e = rng.standard_normal(n) * np.sqrt(1 - rho ** 2)
    e[offset] = 0.0                                          # each path starts from its own x_0 below
    y = lfilter([1.0], [1.0, -rho], e)
    path_of = np.repeat(np.arange(len(path_len)), path_len)
    t_loc = np.arange(n) - offset[path_of]
    end_prev = np.concatenate([[0.0], y[offset[1:] - 1]])  # filter state carried into each path
    x0 = rng.standard_normal(len(path_len))
    decay = rho ** (t_loc + 1.0)
    return sd * (x0[path_of] * rho ** t_loc.astype(float) + y - decay * end_prev[path_of])


def coordinates(cfg, fz, keep=None):
    """(names, array) of the embedded CVs for the frames of the base cache `fz` (those in `keep`, default all);
    the columns are filled one at a time (memory: one (n, d) array)."""
    e = cfg['embed']
    rng = np.random.default_rng(e['seed'])
    phys, path_len, offset = _physical_index(fz)
    base = fz['cv']
    if keep is not None:
        phys, base = phys[keep], base[keep]
    y, x = base[:, 0], base[:, 1]
    rel = {'y': y, 'x': x, 'u': (y + x) / np.sqrt(2), 'v': (y - x) / np.sqrt(2)}
    names = ['y', 'x'] + [f'c{cv}{round(a * 100):g}' for cv, a in e['copies']]
    names += [f'f{k + 1:02d}' for k in range(e['n_fast'])] + ['w']
    X = np.empty((len(y), len(names)))
    X[:, 0], X[:, 1] = y, x
    j = 2
    for cv, a in e['copies']:
        X[:, j] = rel[cv] + a * _ou(rng, path_len, offset, e['copy_tau'], 1.0)[phys]
        j += 1
    taus = e['fast_taus']
    for k in range(e['n_fast']):
        X[:, j] = _ou(rng, path_len, offset, taus[k % len(taus)], e['fast_sd'])[phys]
        j += 1
    X[:, j] = _ou(rng, path_len, offset, e['slow_tau'], e['slow_sd'])[phys]
    return names, X


def groups(names):
    """CV group of each embedded CV name (for plots and tables)."""
    out = []
    for n in names:
        if n in ('y', 'x'):
            out.append('relevant (y, x)')
        elif n.startswith('c'):
            out.append(f'copy, noise {int(n[2:]) / 100:g}')
        elif n.startswith('f'):
            out.append('fast irrelevant')
        else:
            out.append('slow w')
    return out


def _base_frames(cfg):
    return np.load(os.path.join(common.out_dir(cfg['embed']['base']), 'frames.npz'))


def _labels(cfg, fz):
    """q_np of the embedded frames: the base system's (2D fit) or an NPNEq fit on the embedded CVs."""
    lab = cfg['embed'].get('labels', 'base')
    if lab == 'base':
        return fz['q_np']
    ref = np.load(lab)
    assert len(ref['r']) == len(fz['q_np']) and np.array_equal(ref['lam'], fz['cv'][:, 0]), \
        'the NPNEq fit is not on the frames of the base cache'
    return ref['r'].astype(np.float32)


def cmd_embed(cfg, args):
    fz = _base_frames(cfg)
    keep = np.ones(len(fz['q_np']), bool)
    if cfg['embed'].get('drop_burnin', False):
        keep = data.Frames(common.load_config(cfg['embed']['base'])).frame_split != 'burnin'
    names, X = coordinates(cfg, fz, keep)
    out = {k: fz[k] for k in fz.files}
    out['q_np'] = _labels(cfg, fz)
    for k in ('q_np', 'piece', 'path', 't', 'w', 'A', 'B', 'q_exact'):
        if k in out:
            out[k] = out[k][keep]
    out['cv'] = X
    out['cv_names'] = np.array(names)
    # consistency: the same physical frame has the same base CVs in every piece (checked on a sample)
    phys = _physical_index(fz)[0]
    order = np.argsort(phys, kind='stable')
    dup = order[1:][phys[order[1:]] == phys[order[:-1]]]
    prev = order[:-1][phys[order[1:]] == phys[order[:-1]]]
    assert np.array_equal(fz['cv'][dup[:100000]], fz['cv'][prev[:100000]]), 'physical frame index is wrong'
    np.savez(os.path.join(cfg['out_dir'], 'frames.npz'), **out)
    common.say(f'{cfg["name"]}: {keep.sum()} frames x {len(names)} CVs ({", ".join(names)}); labels '
               f'{cfg["embed"].get("labels", "base")}; {len(dup)} frames shared by two pieces (checked)')


def cmd_fit_qnp_embedded(cfg, args):
    """NPNEq (load_tis_data_cupy.py, seeded as in run.py fit-qnp) with the embedded CVs as the CV set, on the base
    system's frames (the same frames, in the same order, as the base cache). Writes cfg['embed']['labels']."""
    base_cfg = common.load_config(cfg['embed']['base'])
    fz = _base_frames(cfg)
    names, X = coordinates(cfg, fz)
    d = base_cfg['data']
    fig_dir = os.path.dirname(cfg['embed']['labels'])
    os.makedirs(fig_dir, exist_ok=True)
    argv = [os.path.join(common.EXAMPLES, 'load_tis_data_cupy.py'), d['tis_dir']]
    argv += list(d['loader_args']) + list(cfg['embed']['qnp_fit_args']) + ['--figure-dir', fig_dir, '--no-potential']
    sys.path.insert(0, common.EXAMPLES)
    import cupy
    import tis_data
    prepare = tis_data.prepare_data

    def prepare_embedded(a):
        out = prepare(a)
        assert len(out.lam) == len(X) and np.array_equal(out.lam, fz['cv'][:, 0]), 'frames differ from the cache'
        out.X = X
        print(f'Embedded CVs: {X.shape[1]} ({", ".join(names)})')
        return out

    tis_data.prepare_data = prepare_embedded
    np.random.seed(cfg['seed'])
    cupy.random.seed(cfg['seed'])
    common.say('seed', cfg['seed'], ':', ' '.join(argv))
    sys.argv = argv
    cwd = os.getcwd()
    os.chdir(common.EXAMPLES)
    try:
        runpy.run_path(argv[0], run_name='__main__')
    finally:
        os.chdir(cwd)
