"""Append-only frame pool of the loop: one store per pass, selection into a Frames-like view.

    pool/pass_<k>.npz    pieces cut with the lambda and interfaces of pass k (never re-cut), the whole paths,
                         per-piece ensemble / row / mirrored / lineage flags, per-frame labels q (after teaching)
    pool/pass_<k>.json   meta: lambda version (md5 of the export, or "seed"), interfaces, mode, cost

`Pool.select` builds a `View` with the attributes `models.TrainSet`, `evaluate.evaluate` and `natural.Increments`
read from `data.Frames` (cv, piece, path, t, w, A, B, valid, q, z, q_np, n, dim, cv_names, lambda_a, lambda_b,
mask(split), pairs(), take_pieces(), has_exact, q_exact), so the student and the metrics run on the pool unchanged.
Here `cv` holds the CV pool only (positions), and `lam` the order parameter the frame was sampled with.
"""
import json
import os

import numpy as np

import common


class Pool:
    def __init__(self, loop_dir):
        self.dir = os.path.join(loop_dir, 'pool')
        os.makedirs(self.dir, exist_ok=True)

    # -- storage --------------------------------------------------------------------------------
    def path(self, k):
        return os.path.join(self.dir, f'pass_{k}.npz')

    def meta_path(self, k):
        return os.path.join(self.dir, f'pass_{k}.json')

    def passes(self):
        ks = []
        for f in os.listdir(self.dir):
            if f.startswith('pass_') and f.endswith('.npz'):
                ks.append(int(f[5:-4]))
        return sorted(ks)

    def has(self, k):
        return os.path.isfile(self.path(k))

    def store(self, k, pieces, raw, meta):
        """Store pass k: `pieces` from backends.infretis.load_pieces, `raw` from raw_paths, `meta` a dict."""
        rawn = np.array([len(o) for o in raw['ops']], int)
        arrays = dict(
            cv=pieces['cv'].astype(np.float32), t=pieces['t'], piece=pieces['piece'], piece_n=pieces['piece_n'],
            piece_w=pieces['piece_w'], piece_pnr=pieces['piece_pnr'], piece_ens=pieces['piece_ens'],
            piece_row=pieces['piece_row'], piece_mirrored=pieces['piece_mirrored'], piece_core=pieces['piece_core'],
            piece_tainted=pieces['piece_tainted'],
            raw_cv=(np.concatenate(raw['ops']) if raw['ops'] else np.zeros((0, pieces['cv'].shape[1]))).astype(np.float32),
            raw_n=rawn, raw_pnr=np.array(raw['pnr'], np.int64), raw_row=np.array(raw['rows'], np.int32),
            raw_W=np.asarray(raw['W'], float), raw_ptype=np.array(raw['ptype'], dtype=str),
            raw_gen_ens=np.array(raw['gen_ens'], np.int32), raw_tainted=np.array(raw['tainted'], bool),
            raw_move=np.array(raw['move'], dtype=str),
        )
        if 'q_exact' in pieces:
            arrays['q_exact'] = np.asarray(pieces['q_exact'], np.float32)
        np.savez(self.path(k), **arrays)
        m = dict(meta, pass_index=int(k), n_pieces=int(len(pieces['piece_n'])), n_frames=int(len(pieces['cv'])),
                 n_paths=int(len(raw['pnr'])), weighting=pieces['weighting'], reversed_pieces=bool(pieces['reversed_pieces']),
                 n_rows=int(meta.get('n_rows', 0)))
        with open(self.meta_path(k), 'w') as f:
            json.dump(m, f, indent=1)
        return m

    def set_labels(self, k, q):
        """Per-frame labels of pass k (the teacher's q, in the order of `cv`)."""
        z = dict(np.load(self.path(k), allow_pickle=False))
        assert len(q) == len(z['cv'])
        z['q_np'] = np.asarray(q, np.float32)
        np.savez(self.path(k), **z)

    def set_labels_from_view(self, view, q):
        """Write per-frame labels of a view back into the pass stores (frames the view does not hold keep
        their old labels; NaN labels are not written)."""
        q = np.asarray(q, float)
        for k in np.unique(view.pass_of_frame):
            sel = (view.pass_of_frame == k) & np.isfinite(q)
            z = dict(np.load(self.path(k), allow_pickle=False))
            arr = z['q_np'].astype(np.float32) if 'q_np' in z else np.full(len(z['cv']), np.nan, np.float32)
            arr[view.src_idx[sel]] = q[sel]
            z['q_np'] = arr
            np.savez(self.path(k), **z)

    def meta(self, k):
        with open(self.meta_path(k)) as f:
            return json.load(f)

    def load(self, k):
        return dict(np.load(self.path(k), allow_pickle=False)), self.meta(k)

    # -- selection ------------------------------------------------------------------------------
    def select(self, passes='all', lineage='exclude', burn_in=0.0, cap=None, block=0.1, val_blocks=(1,),
               test_blocks=(3,), need_labels=False, seed=0):
        """A View of the selected pieces.

        passes: 'all', 'last', 'last2' or a list of pass indices.
        lineage: 'exclude' drops tainted paths (constraint 7), 'include' keeps them.
        burn_in: fraction of each pass's path-table rows dropped at its start (after a lambda switch).
        cap: at most this many paths per ensemble and pass (evenly spaced over the rows), the ensembles near A
             hold far more data than needed.
        Splits (by whole path): the rows of a pass after its burn-in are cut into contiguous blocks of `block`
        of the rows; `val_blocks` / `test_blocks` are held out, the rest is training.
        """
        ks = self.passes()
        if passes == 'last':
            ks = ks[-1:]
        elif passes == 'last2':
            ks = ks[-2:]
        elif passes != 'all':
            ks = [k for k in ks if k in set(passes)]
        parts = []
        for k in ks:
            z, m = self.load(k)
            if need_labels and 'q_np' not in z:
                continue
            n_rows = max(int(m.get('n_rows', 0)), int(z['piece_row'].max()) + 1 if len(z['piece_row']) else 1)
            keep = z['piece_row'] >= burn_in * n_rows
            if lineage == 'exclude':
                keep &= ~z['piece_tainted']
            if cap is not None:
                for e in np.unique(z['piece_ens']):
                    sel = np.flatnonzero(keep & (z['piece_ens'] == e) & ~z['piece_mirrored'])
                    if len(sel) > cap:
                        rows = np.unique(z['piece_row'][sel])
                        rows = rows[np.unique(np.round(np.linspace(0, len(rows) - 1, cap)).astype(int))]
                        drop = sel[~np.isin(z['piece_row'][sel], rows)]
                        keep[drop] = False
                        # the mirrored pieces follow their forward piece
                        mir = np.flatnonzero(keep & (z['piece_ens'] == e) & z['piece_mirrored'])
                        keep[mir[~np.isin(z['piece_row'][mir], rows)]] = False
            parts.append((k, z, m, keep, n_rows))
        return View(parts, block=block, val_blocks=val_blocks, test_blocks=test_blocks, burn_in=burn_in)


class View:
    """Frames-like view of selected pieces (see the module docstring)."""

    def __init__(self, parts, block=0.1, val_blocks=(1,), test_blocks=(3,), burn_in=0.0):
        cvs, lams, ts, pieces, paths, ws, qs, qx, ens, pass_f, piece_meta = [], [], [], [], [], [], [], [], [], [], []
        src_idx = []
        piece_off = 0
        self.lambda_a = self.lambda_b = None
        self.cv_names = None
        self.interfaces_of_pass = {}
        for k, z, m, keep, n_rows in parts:
            idx_pieces = np.flatnonzero(keep)
            if not len(idx_pieces):
                continue
            starts = np.r_[0, np.cumsum(z['piece_n'])[:-1]]
            frame_idx = np.concatenate([np.arange(starts[i], starts[i] + z['piece_n'][i]) for i in idx_pieces])
            src_idx.append(frame_idx.astype(np.int64))
            cv = z['cv'][frame_idx].astype(float)
            cvs.append(cv[:, 1:])
            lams.append(cv[:, 0])
            ts.append(z['t'][frame_idx])
            n = z['piece_n'][idx_pieces]
            pieces.append(np.repeat(piece_off + np.arange(len(idx_pieces)), n))
            uid = k * 10_000_000 + z['piece_pnr'][idx_pieces].astype(np.int64)
            paths.append(np.repeat(uid, n))
            ws.append(np.repeat(z['piece_w'][idx_pieces], n))
            ens.append(np.repeat(z['piece_ens'][idx_pieces], n))
            pass_f.append(np.full(len(frame_idx), k, np.int32))
            qs.append(z['q_np'][frame_idx].astype(float) if 'q_np' in z else np.full(len(frame_idx), np.nan))
            qx.append(z['q_exact'][frame_idx].astype(float) if 'q_exact' in z else None)
            frac = (z['piece_row'][idx_pieces] - burn_in * n_rows) / max(n_rows * (1 - burn_in), 1)
            blk = np.floor(frac / block).astype(int)
            lab = np.full(len(idx_pieces), 'train', dtype=object)
            lab[np.isin(blk, list(val_blocks))] = 'val'
            lab[np.isin(blk, list(test_blocks))] = 'test'
            piece_meta.append(dict(pass_index=k, uid=uid, split=lab, mirrored=z['piece_mirrored'][idx_pieces],
                                   row=z['piece_row'][idx_pieces], ens=z['piece_ens'][idx_pieces], w=z['piece_w'][idx_pieces]))
            piece_off += len(idx_pieces)
            self.lambda_a, self.lambda_b = float(m['lambda_a']), float(m['lambda_b'])
            self.cv_names = np.array(m['cv_names'])
            self.interfaces_of_pass[k] = list(m['interfaces'])
        if not cvs:
            raise ValueError('the selection holds no pieces')
        self.cv = np.concatenate(cvs)
        self.lam = np.concatenate(lams)
        self.t = np.concatenate(ts)
        self.piece = np.concatenate(pieces).astype(np.int32)
        self.path = np.concatenate(paths)
        w = np.concatenate(ws)
        self.w = (w / w.mean()).astype(np.float32)
        self.ens = np.concatenate(ens)
        self.pass_of_frame = np.concatenate(pass_f)
        self.src_idx = np.concatenate(src_idx)                 # frame index inside the pass store
        self.q_np = np.concatenate(qs).astype(np.float32)
        if all(q is not None for q in qx):
            self.q_exact = np.concatenate(qx).astype(np.float32)
        self.state_cv = 0
        s = self.cv[:, self.state_cv]
        self.A, self.B = s <= self.lambda_a, s >= self.lambda_b
        self.n, self.dim = len(self.cv), self.cv.shape[1]
        self.inside = ~self.A & ~self.B
        self.has_labels = bool(np.isfinite(self.q_np).any())
        # as data.Frames: valid = inside with a label; before the first teacher pass every inside frame is valid
        self.valid = self.inside & np.isfinite(self.q_np) if self.has_labels else self.inside.copy()
        self.q = np.clip(np.nan_to_num(self.q_np.astype(float), nan=0.5), common.Q_CLIP, 1 - common.Q_CLIP)
        self.z = common.logit(self.q)
        self.has_exact = hasattr(self, 'q_exact')
        # per-piece tables and per-frame split
        self.piece_pass = np.concatenate([np.full(len(pm['uid']), pm['pass_index']) for pm in piece_meta])
        self.piece_path = np.concatenate([pm['uid'] for pm in piece_meta])
        self.piece_split = np.concatenate([pm['split'] for pm in piece_meta])
        self.piece_mirrored = np.concatenate([pm['mirrored'] for pm in piece_meta])
        self.piece_row = np.concatenate([pm['row'] for pm in piece_meta])
        self.piece_ens = np.concatenate([pm['ens'] for pm in piece_meta])
        self.piece_w = np.concatenate([pm['w'] for pm in piece_meta])
        self.frame_split = self.piece_split[self.piece]
        self.path_id = np.unique(self.path)

    # -- the Frames interface -----------------------------------------------------------------------
    def mask(self, split):
        return self.frame_split == split

    def pairs(self, frame_mask, lag, stride=1):
        i = np.flatnonzero(frame_mask[:-lag] & self.valid[:-lag])
        i = i[(self.piece[i + lag] == self.piece[i]) & self.valid[i + lag] & frame_mask[i + lag]]
        return i[::stride], i[::stride] + lag

    def take_pieces(self, split, max_pieces=None, seed=0):
        m = self.mask(split)
        if max_pieces is None:
            return m
        pieces = np.unique(self.piece[m])
        if len(pieces) > max_pieces:
            pieces = np.sort(np.random.default_rng(seed).choice(pieces, max_pieces, replace=False))
        return np.isin(self.piece, pieces)

    def summary(self):
        rows = []
        for sp in ('train', 'val', 'test'):
            m = self.mask(sp)
            rows.append(f'{sp:5s}: {np.sum(self.piece_split == sp):6d} pieces, {m.sum():9d} frames')
        rows.append(f'passes {sorted(set(self.piece_pass.tolist()))}; labels for {np.isfinite(self.q_np).sum()} frames')
        return '\n'.join(rows)
