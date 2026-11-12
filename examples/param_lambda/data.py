"""Step 2: frame data for fitting lambda(x) to q_np.

`extract` reads the staple run exactly as `tis_data.prepare_data` does (same pieces, same order, so the
saved q_np lines up frame by frame) and also keeps what prepare_data drops: the MC cycle, ensemble and
M crossings of every path. It writes a float32 cache `<out>/frames.npz`.

`Frames` loads the cache and provides the splits (whole paths, contiguous blocks of MC cycles, a later
block kept out as the OOD test), the within-piece pairs at the training lags, and the training
standardization.
"""
import argparse
import contextlib
import glob
import importlib.util
import inspect
import io
import os
import subprocess
import sys

import numpy as np

import common
import tis_data

# ---------------------------------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------------------------------


def tis_data_module(cfg):
    """The tis_data the run's q_np was fitted with: the current one, or `data.tis_data_rev` taken from git.

    The selection of paths changed on 2026-11-10 (3393145: burn-in + evenly spaced thinning instead of a random
    subset); fits made before that line up with the frames of the old version only.
    """
    rev = cfg['data'].get('tis_data_rev')
    if rev is None:
        return tis_data
    path = os.path.join(cfg['out_dir'], f'tis_data_{rev}.py')
    if not os.path.isfile(path):
        src = subprocess.check_output(['git', 'show', f'{rev}:examples/tis_data.py'], cwd=common.REPO).decode()
        with open(path, 'w') as f:
            f.write(src)
    spec = importlib.util.spec_from_file_location(f'tis_data_{rev}', path)
    mod = importlib.util.module_from_spec(spec)
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = dont_write
    return mod


def _kept_cycles(td, order_file, path_info, args, rng):
    """Cycle numbers of the paths `td.load_order_file` keeps, in its order (its `n_path` indexes this list).

    The selection is copied from load_order_file: accepted, not 'ld', past the burn-in, nonzero weight; then
    either thinned and evenly spaced (current tis_data) or a random subset drawn with the loader's shared
    generator (old tis_data; `rng` is a second generator with the same seed, so both draw the same paths).
    `extract` checks every piece against it, and the whole frame set against the q_np fit.
    """
    cycles, accepted = [], []
    with open(order_file) as f:
        for line in f:
            m = td._CYCLE_RE.match(line)
            if m:
                cycles.append(int(m.group(1)))
                accepted.append(m.group(2) == 'ACC')
    n_cycles = max(path_info) + 1 if path_info else 0
    new_selection = 'thin' in inspect.signature(td.load_order_file).parameters
    burn_in = args.burn_in
    if new_selection:
        burn_in = max(burn_in, int(np.ceil(args.burn_in_fraction * n_cycles)))
    keep = [c for c, acc in zip(cycles, accepted)
            if acc and c >= burn_in and path_info[c][0] > 0 and path_info[c][2] != 'ld']
    max_paths = args.paths_per_ensemble
    if new_selection:
        keep = keep[::max(1, int(args.thin))]
        if max_paths is not None and len(keep) > max_paths:
            keep = [keep[j] for j in np.unique(np.round(np.linspace(0, len(keep) - 1, max_paths)).astype(int))]
    elif max_paths is not None and len(keep) > max_paths:
        keep = sorted(rng.choice(keep, max_paths, replace=False).tolist())
    return keep, n_cycles


def _exact_q(cfg, cv, A, B, lambda_a, lambda_b):
    """Exact committor at the frames (validation only), 0 in A and 1 in B."""
    d = cfg['data']
    kind = d.get('exact')
    if kind is None:
        return None
    if kind == '1d':
        ref = np.load(d['exact_file'])
        q = np.interp(cv[:, 0], ref['x'], ref['q'])
    elif kind in ('zpot', 'cc'):
        # discrete-chain grid solutions (figures_by_potential/<potential>/exact/; exact_tools in _shared/exact_committors)
        sys.path.insert(0, os.path.join(common.FIGURES, '_shared', 'exact_committors'))
        import exact_tools
        if kind == 'zpot':
            force = exact_tools.zpot_force
        else:
            import toy2d
            pot = toy2d.load_potential('curved_channel:CurvedChannelDoubleWell')

            def force(y, x, xp=np):
                return pot.potential_and_force_xy(y, x, xp)[1:]
        sol = dict(np.load(d['exact_file']))
        assert abs(float(sol['A']) - lambda_a) < 1e-12 and abs(float(sol['B']) - lambda_b) < 1e-12, \
            'exact committor made for other states'
        q = np.empty(len(cv))
        for s in range(0, len(cv), 2_000_000):
            q[s:s + 2_000_000] = exact_tools.eval_q_brownian_2d(force, sol, cv[s:s + 2_000_000, 0],
                                                               cv[s:s + 2_000_000, 1])
    else:
        raise ValueError(f'unknown exact committor kind {kind!r}')
    q = np.clip(q, 0, 1)
    q[A], q[B] = 0, 1
    return q


def read_run(cfg, loader_args=None):
    """The frames of the run as `tis_data.prepare_data([tis_dir] + loader_args)` makes them (same pieces, order,
    weights, states), with the MC cycle, ensemble and M crossings of every path."""
    d = cfg['data']
    td = tis_data_module(cfg)
    parser = argparse.ArgumentParser()
    td.add_data_arguments(parser)
    args = parser.parse_args([d['tis_dir']] + list(d['loader_args'] if loader_args is None else loader_args))
    with contextlib.redirect_stdout(io.StringIO()):
        interfaces = td.read_interfaces(d['tis_dir'], args.interfaces_file)
    lambda_a = interfaces[0] if args.lambda_a is None else args.lambda_a
    lambda_b = interfaces[-1] if args.lambda_b is None else args.lambda_b
    dim, v_ord = td.read_run_settings(d['tis_dir'])
    odd_columns = range(1 + dim, 1 + 2 * dim) if (v_ord and dim) else ()
    accepts = inspect.signature(td.load_order_file).parameters
    seed = getattr(args, 'seed', None)
    rng_lib, rng_mine = np.random.default_rng(seed), np.random.default_rng(seed)   # as load_tis_data: one per run

    folders = sorted(glob.glob(os.path.join(d['tis_dir'], '0[0-9][0-9]')))
    n_ens = len(folders)
    cvs, times, weights, paths = [], [], [], []
    piece_mirrored, path_cycle, path_ens, path_tstar, path_tlast, path_len = [], {}, {}, {}, {}, {}
    n_cycles_all = 0
    for folder in folders:
        k = int(os.path.basename(folder))
        if k == 0 and not args.include_zero_minus:
            continue
        order_file = os.path.join(folder, 'order.txt')
        lambda_m = interfaces[max(k - 1, 0)]
        with contextlib.redirect_stdout(io.StringIO()):
            path_info = td.read_path_info(folder, k, n_ens, args.weights)
        kept, n_cycles = _kept_cycles(td, order_file, path_info, args, rng_mine)
        n_cycles_all = max(n_cycles_all, n_cycles)
        kw = dict(lambda_m=lambda_m, burn_in=args.burn_in, burn_in_fraction=getattr(args, 'burn_in_fraction', 0.0),
                  thin=getattr(args, 'thin', 1), max_paths=args.paths_per_ensemble, rng=rng_lib,
                  reversed_pieces=args.reversed, odd_columns=odd_columns)
        c, o, t, w, pth, _, _ = td.load_order_file(order_file, path_info,
                                                   **{key: v for key, v in kw.items() if key in accepts})
        offset = paths[-1] + 1 if paths else 0                     # as tis_data.load_tis_data
        seen = set()
        for piece, tt, ww, n_path in zip(c, t, w, pth):
            cycle = kept[n_path]
            weight, length, _, _, _ = path_info[cycle]
            assert ww == weight
            g = offset + n_path
            # A forward piece starts at its first frame past M (t* >= 1); a mirrored piece starts at time 0.
            mirrored = tt[0] == 0
            if mirrored:
                assert len(tt) <= length and np.all(np.diff(tt) == 1)
                path_tlast[g] = len(tt) - 1
            else:
                assert tt[-1] == length - 1 and np.all(np.diff(tt) == 1), 'Time column is not the frame index'
                path_tstar[g] = int(tt[0])
            assert (g, mirrored) not in seen
            seen.add((g, mirrored))
            path_cycle[g], path_ens[g], path_len[g] = cycle, k, length
            cvs.append(np.asarray(piece, dtype=float))        # float64: A/B below must match prepare_data
            times.append(tt.astype(np.int32))
            weights.append(ww)
            paths.append(g)
            piece_mirrored.append(mirrored)
        common.say(f'  {folder}: {len(c)} pieces, {sum(len(p) for p in o)} frames, '
                   f'cycles {min(kept)}..{max(kept)} of {n_cycles}')

    n = np.array([len(p) for p in cvs])
    cv = np.concatenate(cvs)
    piece = np.repeat(np.arange(len(cvs), dtype=np.int32), n)
    path = np.repeat(np.array(paths, dtype=np.int32), n)
    t = np.concatenate(times)
    w = np.repeat(np.array(weights, dtype=float), n)
    w /= w.mean()
    lam = cv[:, 0]
    A, B = lam <= lambda_a, lam >= lambda_b

    g_ids = np.array(sorted(path_cycle))
    return dict(
        cv=cv, piece=piece, path=path, t=t, w=w.astype(np.float32), A=A, B=B,
        piece_path=np.array(paths, dtype=np.int32), piece_mirrored=np.array(piece_mirrored),
        path_id=g_ids, path_cycle=np.array([path_cycle[g] for g in g_ids]),
        path_ens=np.array([path_ens[g] for g in g_ids]),
        path_tstar=np.array([path_tstar.get(g, -1) for g in g_ids]),
        path_tlast=np.array([path_tlast.get(g, -1) for g in g_ids]),
        path_len=np.array([path_len[g] for g in g_ids]),
        n_cycles=n_cycles_all, lambda_a=lambda_a, lambda_b=lambda_b, interfaces=np.array(interfaces),
        cv_names=np.array(d['cv_names']), burn_in_fraction=getattr(args, 'burn_in_fraction', 0.0))


def extract(cfg):
    """read_run with the config's loader arguments, plus q_np (checked to be of these frames) and the exact q."""
    out = read_run(cfg)
    cv, piece, A, B = out['cv'], out['piece'], out['A'], out['B']
    out['q_np'] = _load_q_np(cfg, len(cv), cv, piece, A, B).astype(np.float32)
    q_exact = _exact_q(cfg, cv, A, B, out['lambda_a'], out['lambda_b'])
    if q_exact is not None:
        out['q_exact'] = q_exact.astype(np.float32)
    path_out = os.path.join(cfg['out_dir'], 'frames.npz')
    np.savez(path_out, **out)
    common.say(f'{len(cv)} frames, {len(out["piece_path"])} pieces, {len(out["path_id"])} paths; '
               f'A {A.sum()}, B {B.sum()} frames; saved {path_out}')
    return out


def _load_q_np(cfg, n, cv, piece, A, B):
    """q_np of every frame, checked to belong to exactly these frames (order parameter, pieces, states)."""
    d = cfg['data']
    src = d['q_np']
    ref = np.load(src if src.endswith('.npz') else d['check_against'])
    q = ref['r'] if src.endswith('.npz') else np.load(src)
    assert len(q) == n == len(ref['lam']), f'q_np has {len(q)} frames, the data {n}'
    assert np.array_equal(ref['i_traj'], piece), 'piece order differs from the q_np fit'
    assert np.array_equal(ref['lam'], cv[:, 0]), 'order parameter differs from the q_np fit'
    assert np.array_equal(ref['boundary0'], A) and np.array_equal(ref['boundary1'], B), 'A/B differ from the q_np fit'
    return q


# ---------------------------------------------------------------------------------------------------
# loading, splits, pairs
# ---------------------------------------------------------------------------------------------------


class Frames:
    """The frame cache with splits. Splits are by whole path (both pieces of a path together), by MC cycle:

    - OOD: cycles >= ood_from * n_cycles (the later sampling block);
    - in-distribution: the cycles from id_from (default the loader's burn-in) to ood_from, cut into contiguous
      blocks of `block` * n_cycles; val_blocks and test_blocks are held out, the rest is training;
    - cycles before id_from ('burnin') are not used.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        z = np.load(os.path.join(cfg['out_dir'], 'frames.npz'))
        for key in z.files:
            setattr(self, key, z[key])
        self.n = len(self.q_np)
        self.dim = self.cv.shape[1]
        self.n_cycles, self.lambda_a, self.lambda_b = int(self.n_cycles), float(self.lambda_a), float(self.lambda_b)
        self.valid = ~self.A & ~self.B & np.isfinite(self.q_np)
        self.q = np.clip(self.q_np.astype(float), common.Q_CLIP, 1 - common.Q_CLIP)
        self.z = common.logit(self.q)
        self.has_exact = hasattr(self, 'q_exact')
        self._split_paths()

    def _split_paths(self):
        s = self.cfg['split']
        frac = self.path_cycle / self.n_cycles
        start = s.get('id_from', float(self.burn_in_fraction) or 0.4)
        block = np.floor((frac - start) / s['block']).astype(int)
        lab = np.full(len(self.path_id), 'train', dtype=object)
        lab[frac < start] = 'burnin'                 # early MC cycles: not used (the q_np fit may have used them)
        lab[np.isin(block, s['val_blocks'])] = 'val'
        lab[np.isin(block, s['test_blocks'])] = 'test'
        lab[frac >= s['ood_from']] = 'ood'
        self.path_split = lab
        pos = np.searchsorted(self.path_id, self.path)
        assert np.array_equal(self.path_id[pos], self.path)
        self.frame_split = lab[pos]
        self.block_of_path = block

    def mask(self, split):
        return self.frame_split == split

    def summary(self):
        rows = []
        for sp in ('burnin', 'train', 'val', 'test', 'ood'):
            m = self.mask(sp)
            pc = self.path_cycle[self.path_split == sp]
            rows.append(f'{sp:5s}: {np.sum(self.path_split == sp):6d} paths, {m.sum():9d} frames, '
                        f'cycles {pc.min() if pc.size else -1}..{pc.max() if pc.size else -1}')
        return '\n'.join(rows)

    def pairs(self, frame_mask, lag, stride=1):
        """Index pairs (i, i + lag) inside one piece, both frames valid (not in A/B) and in frame_mask."""
        i = np.flatnonzero(frame_mask[:-lag] & self.valid[:-lag])
        i = i[(self.piece[i + lag] == self.piece[i]) & self.valid[i + lag] & frame_mask[i + lag]]
        return i[::stride], i[::stride] + lag

    def take_pieces(self, split, max_pieces=None, seed=0):
        """Frame mask of the pieces of a split, at most `max_pieces` whole pieces (drawn with a fixed seed)."""
        m = self.mask(split)
        if max_pieces is None:
            return m
        pieces = np.unique(self.piece[m])
        if len(pieces) > max_pieces:
            pieces = np.sort(np.random.default_rng(seed).choice(pieces, max_pieces, replace=False))
        return np.isin(self.piece, pieces)


def standardization(x, w):
    """Weighted mean and standard deviation of the training features."""
    w = w / w.sum()
    mu = w @ x
    sd = np.sqrt(w @ (x - mu) ** 2)
    return mu, np.where(sd > 0, sd, 1.0)
