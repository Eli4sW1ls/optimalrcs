"""Find the reaction channels of a system from committor-labelled path data, and a channel map c(x) for new frames.

Standalone: it reads data the loop or the staple notebooks wrote, but nothing in the loop calls it.

    python -m channels.find_channels --npz <fit npz> [--cv-cols 0 1] [--q-key q_fit] --name stag_staple
    python -m channels.find_channels --loop loop_stag --name loop_stag
    python -m channels.find_channels --infretis <run dir> --system stag_br --cv-cols 1 2 --name ...   (exact q labels)

Idea. A channel is a separate piece of the transition region: with one channel the steps that cross committor 1/2
form one connected sheet in CV space, with two channels two sheets that no short step connects.

 1. Band steps. The transitions t -> t+1 inside a piece whose q interval overlaps the band (default 0.3-0.7), placed
    at their midpoint. Steps, not frames: a narrow barrier is crossed in one or two long steps that jump over the
    band and leave no band frame (the upper stag barrier).
 2. Scale. Each CV is divided by its RMS step on the band steps, so distances count typical steps.
 3. Components. A k-nearest-neighbour graph on (a uniform sample of) the band steps: an edge from each step to its k
    nearest ones, but no edge longer than r_max steps; connected components = pieces of the band contiguous at the
    data's sampling resolution (single linkage). Without the cutoff every point links to its k neighbours however far
    away, and a sparsely sampled channel is attached to a big one. `--method dbscan` is the same with an epsilon graph
    and a core-point rule; `hdbscan` needs no scale but can split one channel at a density minimum along the band.
 4. Channels. A component is a channel if steps of >= min_paths independent paths fall in it (forward and mirrored
    piece of a path count once). Not by flux share: path_f weights compare frames only within one ensemble, and the
    barriers of two channels can sit in different ensembles. Smaller components within r_merge steps of a channel
    are stray steps of it and join it; farther ones stay unassigned and are reported as candidate channels (crossed
    too rarely to resolve). Flux shares (Dirichlet energy sum w (q(t+1) - q(t))^2 of the band steps per channel,
    also with q_exact when present) are reported, as an indication only.
 5. Path labels. Each path takes the channel carrying most of its band energy; paths without band steps stay
    unlabelled.
 6. Channel map. The frames of labelled paths vote on a uniform sample S of all frames between the states (k_map
    nearest, within d_cover steps): one channel with >= purity_common of the vote -> its label, mixed -> 0 (common).
    These frames are sources; every other frame of S takes the label of its geodesically nearest source on a kNN graph
    of S (edges <= r_graph steps), so channel labels run along each channel into its entrance, which pieces that
    cross the band rarely visit (staple pieces start at M). Frames the graph does not reach: -1 (unknown).
    ChannelMap (a kNN vote over S) labels any new point: the classifier a channel-aware order parameter needs, since
    the clustering itself labels only the steps it saw.
 7. Unexplored branches. Frames of unlabelled paths farther than d_cover from every labelled frame, clustered
    (DBSCAN): regions the sampler visits but no path crosses the band from. A channel the data enter but never cross
    shows up here (loop_stag's lower channel), but so can a basin-side region no reactive path passes: read them with
    their q range and position.

Outputs in <out>/<name>/: summary.json, channels.png, channel_map.npz (ChannelMap.load).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM_LAMBDA = os.path.dirname(HERE)
EXAMPLES = os.path.dirname(PARAM_LAMBDA)
REPO = os.path.dirname(EXAMPLES)
OUT = os.path.join(EXAMPLES, 'figures_by_potential', '_shared', 'param_lambda', 'channels')


# -- data -------------------------------------------------------------------------------------------------------
def from_npz(path, cv_cols=(0, 1), q_key='q_fit', cv_names=None):
    """A staple-notebook fit npz (X_all, i_traj, group_traj, path_weights, boundary0/1, q_fit, q_exact)."""
    z = np.load(path)
    d = dict(cv=z['X_all'][:, list(cv_cols)].astype(np.float64), piece=z['i_traj'], group=z['group_traj'],
             w=z['path_weights'].astype(np.float64), A=z['boundary0'], B=z['boundary1'], q=z[q_key].astype(np.float64),
             cv_names=list(cv_names or [f'cv{c}' for c in cv_cols]))
    if 'q_exact' in z.files:
        d['q_exact'] = z['q_exact'].astype(np.float64)
    return d


def from_loop(name, max_frames=6_000_000):
    """The labelled pool of a loop directory (all passes; the teacher's q_np of the last fit)."""
    sys.path.insert(0, PARAM_LAMBDA)
    from loop import env  # noqa: F401  (paths of the samplers)
    from loop import pool as pool_mod
    import common
    v = pool_mod.Pool(os.path.join(common.OUT_ROOT, 'op_loop', name)).select(need_labels=True, max_frames=max_frames)
    ok = np.isfinite(v.q_np) | v.A | v.B
    q = np.where(v.A, 0.0, np.where(v.B, 1.0, v.q_np.astype(np.float64)))
    d = dict(cv=v.cv.astype(np.float64), piece=v.piece, group=v.path, w=v.w.astype(np.float64), A=v.A, B=v.B, q=q,
             cv_names=list(v.cv_names))
    if v.has_exact:
        d['q_exact'] = v.q_exact.astype(np.float64)
    if not ok.all():
        for k in ('cv', 'piece', 'group', 'w', 'A', 'B', 'q', 'q_exact'):
            if k in d:
                d[k] = d[k][ok]
    return d


def from_infretis(run_dir, system, cv_cols=(1, 2), paths_per_ensemble=1000, burn_in_fraction=0.1):
    """An infretis StapleTIS run (order.txt: time, lambda, CVs ...), labelled with the exact committor of an
    ase_systems system (positive controls without a fit). cv_cols index the CVs after time (0 = lambda)."""
    sys.path.insert(0, EXAMPLES)
    sys.path.insert(0, PARAM_LAMBDA)
    from loop import env  # noqa: F401
    from loop import ase_systems
    import infretis_data
    h5 = os.path.join(run_dir, 'archive.h5')
    cvs, order, _, weights, paths, settings = infretis_data.load_infretis_data(
        run_dir, weighting='rejections', h5_file=h5 if os.path.isfile(h5) else None,
        burn_in_fraction=burn_in_fraction, paths_per_ensemble=paths_per_ensemble, reversed_pieces=True)
    X = np.concatenate(cvs)
    s = ase_systems.get(system)
    pos = X[:, 1:1 + s['dim']]
    la, lb = settings['interfaces'][0], settings['interfaces'][-1]
    A, B = X[:, 0] <= la, X[:, 0] >= lb
    q = ase_systems.exact_committor(s)(pos)
    q[A], q[B] = 0.0, 1.0
    return dict(cv=X[:, list(cv_cols)].astype(np.float64),
                piece=np.concatenate([np.full(len(o), k) for k, o in enumerate(order)]),
                group=np.concatenate([np.full(len(o), n) for o, n in zip(order, paths)]),
                w=np.concatenate([np.full(len(o), wt, float) for o, wt in zip(order, weights)]),
                A=A, B=B, q=q, q_exact=q.copy(), cv_names=[s['cv_names'][c - 1] for c in cv_cols])


# -- steps ------------------------------------------------------------------------------------------------------
def same_piece_next(piece):
    """Boolean per frame: frame t+1 belongs to the same piece (the transition t -> t+1 exists)."""
    nxt = np.zeros(len(piece), bool)
    nxt[:-1] = piece[1:] == piece[:-1]
    return nxt


def cluster_band(xs, method, k, min_cluster, r_max):
    """Component labels of the scaled band points; -1 = noise (DBSCAN / HDBSCAN)."""
    if method == 'knn':
        # k nearest neighbours, but no edge longer than r_max (in RMS steps): without the cutoff every point links to
        # its k nearest neighbours however far away they are, and a small group (a sparsely sampled channel) always
        # gets attached to a big one
        g = kneighbors_graph(xs, n_neighbors=min(k, len(xs) - 1), mode='distance', include_self=False)
        g.data[g.data > r_max] = 0
        g.eliminate_zeros()
        _, lab = connected_components(g + g.T, directed=False)
    elif method == 'dbscan':
        lab = DBSCAN(eps=r_max, min_samples=3).fit_predict(xs)
    elif method == 'hdbscan':
        lab = HDBSCAN(min_cluster_size=min_cluster).fit_predict(xs)
    else:
        raise ValueError(method)
    return lab


def dirichlet_energy(q, w, piece):
    """Per frame t: w_t (q(t+1) - q(t))^2 for transitions inside a piece, else 0."""
    e = np.zeros(len(q))
    nxt = same_piece_next(piece)
    i = np.flatnonzero(nxt)
    e[i] = w[i] * (q[i + 1] - q[i]) ** 2
    return e


def find_channels(d, band=(0.3, 0.7), n_band=40000, method='knn', k=10, r_max=6.0, r_merge=20.0, min_paths=5, n_map=120000, n_ref=60000,
                  k_map=25, purity_common=0.9, d_cover=8.0, r_graph=15.0, seed=0, rank_opts=None):
    rng = np.random.default_rng(seed)
    cv, q, w, piece, A, B = d['cv'], d['q'], d['w'], d['piece'], d['A'], d['B']
    inside = ~A & ~B

    # band transitions t -> t+1 (inside a piece) whose q interval overlaps the band: a narrow barrier is crossed in
    # one or two frames that jump over the band, so band *frames* miss it, band *steps* do not
    nxt = same_piece_next(piece)
    it = np.flatnonzero(nxt)
    lo, hi = np.minimum(q[it], q[it + 1]), np.maximum(q[it], q[it + 1])
    tb = it[(hi >= band[0]) & (lo <= band[1])]
    scale = np.sqrt(np.mean((cv[tb + 1] - cv[tb]) ** 2, axis=0))
    xs_all = cv / scale
    mid = 0.5 * (xs_all[tb] + xs_all[tb + 1])                 # step midpoints, scaled
    e_tb = w[tb] * (q[tb + 1] - q[tb]) ** 2                   # Dirichlet energy of each band step

    # 2-3: band sample (uniform over the steps: weights are not comparable across ensembles, and a sparsely sampled
    # channel must not be thinned further) and its components
    pick = rng.choice(len(tb), size=min(n_band, len(tb)), replace=False)
    lab = cluster_band(mid[pick], method, k, max(30, len(pick) // 100), r_max)
    nn_band = NearestNeighbors(n_neighbors=1).fit(mid[pick])
    band_lab = lab[nn_band.kneighbors(mid, return_distance=False)[:, 0]]

    # 4: which components are channels. Not by flux share: path_f weights compare frames only within one ensemble,
    # and the barriers of two channels can sit in different ensembles (lambda = x on stag), so a component's share
    # of the Dirichlet energy can be far off. A component is a channel if steps of >= min_paths independent paths
    # (groups: forward and mirrored piece of one path count once) fall in it.
    group_tb = d['group'][tb]
    comps = np.unique(band_lab)
    n_paths = {int(c): int(len(np.unique(group_tb[band_lab == c]))) for c in comps}
    raw = sorted(((n_paths[int(c)], float(e_tb[band_lab == c].sum() / e_tb.sum())) for c in comps), reverse=True)[:10]
    big = [c for c in comps if c >= 0 and n_paths[int(c)] >= min_paths] or [max(n_paths, key=n_paths.get)]
    # Smaller components: within r_merge (RMS steps) of a channel they are stray steps of it (a narrow barrier
    # crossed in one or two long steps) and join it; farther away they are not merged: their steps stay unassigned
    # and are reported as candidate channels (a channel the data cross too rarely to resolve). DBSCAN / HDBSCAN
    # noise (-1) is treated like a small component.
    keep = np.isin(lab, big)
    nn_big = NearestNeighbors(n_neighbors=1).fit(mid[pick][keep])
    for c in comps:
        if c in big:
            continue
        m = band_lab == c
        dist, idx = nn_big.kneighbors(mid[m])
        if c < 0:                                              # noise: point by point
            near = dist[:, 0] <= r_merge
            sub = np.flatnonzero(m)
            band_lab[sub[near]] = lab[keep][idx[near, 0]]
        elif np.median(dist[:, 0]) <= r_merge:
            band_lab[m] = lab[keep][idx[np.argmin(dist[:, 0]), 0]]
    weak = []
    for c in np.unique(band_lab):
        if c in big:
            continue
        m = band_lab == c
        if c >= 0:
            mc = 0.5 * (cv[tb[m]] + cv[tb[m] + 1])
            weak.append(dict(paths=int(len(np.unique(group_tb[m]))), steps=int(m.sum()), centre=[float(v) for v in mc.mean(0)],
                             extent_lo=[float(v) for v in mc.min(0)], extent_hi=[float(v) for v in mc.max(0)]))
    weak.sort(key=lambda x: -x['paths'])
    order = sorted(big, key=lambda c: -len(np.unique(group_tb[band_lab == c])))
    remap = {int(c): i + 1 for i, c in enumerate(order)}
    band_lab = np.array([remap.get(int(c), 0) for c in band_lab])          # 0: weak component or noise
    n_ch = len(order)
    share = [float(e_tb[band_lab == c].sum() / e_tb.sum()) for c in range(0, n_ch + 1)]   # [unassigned, 1, ..]
    share_exact = None
    if 'q_exact' in d:
        qe = d['q_exact']
        ex = w[tb] * (qe[tb + 1] - qe[tb]) ** 2
        share_exact = [float(ex[band_lab == c].sum() / ex.sum()) for c in range(0, n_ch + 1)]

    # 5: piece and path labels (by band energy)
    n_pieces = int(piece.max()) + 1
    votes = np.zeros((n_pieces, n_ch + 1))
    np.add.at(votes, (piece[tb], band_lab), e_tb + 1e-12)
    votes[:, 0] = 0                                                          # unassigned steps do not vote
    g_ids, g_inv = np.unique(d['group'], return_inverse=True)
    piece_group = np.zeros(n_pieces, np.int64)
    piece_group[piece] = g_inv
    gvotes = np.zeros((len(g_ids), n_ch + 1))
    np.add.at(gvotes, piece_group, votes)
    has = gvotes[:, 1:].sum(1) > 0
    g_lab = np.where(has, gvotes[:, 1:].argmax(1) + 1, 0)
    g_purity = np.where(has, gvotes[:, 1:].max(1) / np.maximum(gvotes[:, 1:].sum(1), 1e-300), 0)
    piece_lab = g_lab[piece_group]
    lab_of_frame = piece_lab[piece]

    # 6: channel map. (a) Coverage: a uniform sample S of the frames between the states; the frames of labelled pieces
    # vote (k_map nearest, within d_cover RMS steps) and make the confident frames of S sources: label c where one
    # channel has >= purity_common of the vote, 0 (common) where channels mix. (b) Spreading: every other frame of S
    # takes the label of its geodesically nearest source on a kNN graph of S (edges <= r_graph steps), so channel
    # labels run along each channel into its entrance, which pieces that reach the band rarely visit (staple pieces
    # start at M), and stop where they meet a common region or the other channel. Frames of S the graph does not
    # reach stay -1 (unknown).
    pool = np.flatnonzero(inside & (lab_of_frame > 0))
    ref = rng.choice(pool, size=min(n_ref, len(pool)), replace=False)
    nn_ref = NearestNeighbors(n_neighbors=k_map).fit(xs_all[ref])
    ins = np.flatnonzero(inside)
    S = rng.choice(ins, size=min(n_map, len(ins)), replace=False)
    dist, idx = nn_ref.kneighbors(xs_all[S])
    votes = np.stack([(lab_of_frame[ref][idx] == c).mean(1) for c in range(1, n_ch + 1)], axis=1)
    pur = votes.max(1)
    src_lab = np.where(pur >= purity_common, votes.argmax(1) + 1, 0)
    is_src = dist[:, -1] <= d_cover
    g = kneighbors_graph(xs_all[S], n_neighbors=10, mode='distance', include_self=False)
    g.data[g.data > r_graph] = 0
    g.eliminate_zeros()
    from scipy.sparse.csgraph import dijkstra
    src_idx = np.flatnonzero(is_src)
    gd, _, nearest = dijkstra(g + g.T, directed=False, indices=src_idx, min_only=True, return_predecessors=True)
    labS = np.where(np.isfinite(gd) & (nearest >= 0), src_lab[np.maximum(nearest, 0)], -1)
    cmap = ChannelMap(xs_all[S], labS, scale, n_ch, k=5, d_unknown=d_cover)

    # 7: unexplored branches: frames of unlabelled pieces far (> d_cover steps) from every frame of a labelled piece
    unl = np.flatnonzero(inside & (lab_of_frame == 0))
    branches, far, bl = [], np.zeros(0, np.int64), np.zeros(0, np.int64)
    if len(unl):
        sub = rng.choice(unl, size=min(40000, len(unl)), replace=False)
        far = sub[nn_ref.kneighbors(xs_all[sub], n_neighbors=1)[0][:, 0] > d_cover]
        if len(far) > 50:
            bl = DBSCAN(eps=3.0, min_samples=10).fit_predict(xs_all[far])
            for c in np.unique(bl[bl >= 0]):
                m = far[bl == c]
                branches.append(dict(cluster=int(c), frames_in_sample=int(len(m)), weight_share=float(w[m].sum() / w[inside].sum()),
                                     q_max=float(q[m].max()), q_median=float(np.median(q[m])),
                                     centre=[float(v) for v in np.average(cv[m], axis=0, weights=w[m])],
                                     extent_lo=[float(v) for v in cv[m].min(0)],
                                     extent_hi=[float(v) for v in cv[m].max(0)]))
            branches.sort(key=lambda b: -b['weight_share'])
    if branches:
        rank_branches(branches, d, xs_all, S, g + g.T, labS, far, bl, nn_ref, mid, band_lab, n_ch, d_cover, rank_opts)

    summary = dict(band=list(band), method=method, k=k, n_band_steps=int(len(tb)), n_band_sample=int(len(pick)),
                   n_components_raw=int(len(comps)), raw_components_paths_share=raw, r_max=r_max, r_merge=r_merge, min_paths=min_paths,
                   band_paths_per_channel=[int(len(np.unique(group_tb[band_lab == c]))) for c in range(1, n_ch + 1)],
                   n_channels=n_ch, candidate_channels=weak,
                   flux_share_unassigned_then_channels=share, flux_share_exact=share_exact,
                   scale=[float(s) for s in scale], cv_names=d['cv_names'],
                   pieces_labelled=int((piece_lab > 0).sum()), pieces=int(n_pieces),
                   paths_per_channel=[int((g_lab == c).sum()) for c in range(1, n_ch + 1)],
                   path_purity_median=float(np.median(g_purity[has])) if has.any() else None,
                   unexplored_branches=branches)
    return summary, dict(tb=tb, mid_cv=0.5 * (cv[tb] + cv[tb + 1]), band_lab=band_lab, cmap=cmap,
                         lab_of_frame=lab_of_frame, branches_far=far, branches_cluster=bl)


RANK_DEFAULTS = dict(min_weight=1e-4, min_depth_ratio=0.5, a_contact=3.0)


def rank_branches(branches, d, xs_all, S, graph, labS, far, bl, nn_ref, mid, band_lab, n_ch, d_cover, opts=None):
    """Stage 1 of the branch check: features that separate a channel the data enter but never cross from a basin-side
    corner no reactive path passes, from the data alone. Adds to each branch dict (distances in RMS steps):

      depth_A, depth_B  95% quantile of the geodesic distance from A (B) over the branch's frames: Dijkstra on the
                   kNN graph of the map sample S, sources = frames of S within a_contact steps of an A (B) frame;
                   None if the branch is not connected to that state in the data graph
      depth_ratio_A/B  depth / the median depth of the band steps of the shallowest channel, from the same state;
                   depth_ratio = the one from the nearer state (deep_from), the state the branch hangs off: about 1
                   or more means it reaches as far from that state as a known channel reaches before its barrier. Both states: a channel the data never cross shows up next to
                   A (loop_stag's lower entrance) or, if only its far side was sampled, next to B (the upper stag
                   channel in the toytis staple data, not connected to A at all)
      dead_end     share of the branch's deepest frames (top 10%) with no graph neighbour deeper by more than one step
      dist_to_labelled  smallest distance to a frame of a labelled path (> d_cover by construction)
      exit_share   weighted share of the runs between the states that touch A (staple pieces end in A, so a run that
                   ends in A is an excursion read backwards) and enter the branch; exit_share_channels: the same for
                   reaching each known channel to half its band depth. Reported only (see below).
      probe        True if weight_share >= min_weight and depth_ratio >= min_depth_ratio; `fails` lists the failed
                   criteria. Branches with probe = True go to stage 2 (shooting from their deepest frames).
    Sorted by probe, then score = weight_share x depth_ratio."""
    o = dict(RANK_DEFAULTS, **(opts or {}))
    cv_s = xs_all                                     # scaled CVs of all frames
    A, B, piece, w = d['A'], d['B'], d['piece'], d['w']
    inside = ~A & ~B
    # depth from A and from B on the map graph (geodesic distance in RMS steps)
    from scipy.sparse.csgraph import dijkstra
    rng = np.random.default_rng(1)
    nn_S = NearestNeighbors(n_neighbors=1).fit(cv_s[S])
    sidx = lambda X: nn_S.kneighbors(X, return_distance=False)[:, 0]

    def depth_from(state):
        ist = np.flatnonzero(state)
        if len(ist) == 0:
            return np.full(len(S), np.inf)
        ist = rng.choice(ist, size=min(50000, len(ist)), replace=False)
        src = np.flatnonzero(NearestNeighbors(n_neighbors=1).fit(cv_s[ist]).kneighbors(cv_s[S])[0][:, 0]
                             <= o['a_contact'])
        return dijkstra(graph, directed=False, indices=src, min_only=True) if len(src) else np.full(len(S), np.inf)

    depth = {'A': depth_from(A), 'B': depth_from(B)}

    def band_depths(dS):
        out = []
        for c in range(1, n_ch + 1):
            v = dS[sidx(mid[band_lab == c])] if (band_lab == c).any() else np.array([np.inf])
            v = v[np.isfinite(v)]
            out.append(float(np.median(v)) if len(v) else np.nan)
        return out

    band_depth = {k: band_depths(v) for k, v in depth.items()}
    ref_depth = {k: (np.nanmin(v) if np.isfinite(v).any() else np.nan) for k, v in band_depth.items()}
    depthS = depth['A']
    # excursions from A: runs of inside frames that touch an A frame of the same piece at either end. Staple pieces
    # start at M and end in A or B, so A frames sit only at piece ends: a run that ends in A is an excursion from A
    # read backwards, which for time-reversible dynamics samples the same thing.
    n = len(piece)
    same_prev = np.zeros(n, bool)
    same_prev[1:] = piece[1:] == piece[:-1]
    prevA, previn, nextA = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    prevA[1:], previn[1:], nextA[:-1] = A[:-1], inside[:-1], A[1:]
    same_next = np.zeros(n, bool)
    same_next[:-1] = same_prev[1:]
    newrun = inside & ~(same_prev & previn)
    runid = np.cumsum(newrun) - 1
    touch = inside & ((same_prev & prevA) | (same_next & nextA))
    exc_runs = np.unique(runid[touch])
    run_w = np.zeros(int(runid.max()) + 1 if n else 0)
    run_w[runid[inside]] = w[inside]                  # the run's weight (constant within a piece)
    if len(exc_runs) == 0:
        for b in branches:
            b.update(probe=None, fails=['no excursions from A in the data'], score=0.0)
        return
    fr = np.flatnonzero(inside & np.isin(runid, exc_runs))
    tot = run_w[exc_runs].sum()
    # branch membership of excursion frames (nearest clustered far frame within 3 steps)
    nn_far = NearestNeighbors(n_neighbors=1).fit(cv_s[far])
    dfar, ifar = nn_far.kneighbors(cv_s[fr])
    memb = np.where(dfar[:, 0] <= 3.0, bl[ifar[:, 0]], -1)
    # known channels reached to half their band depth
    si = sidx(cv_s[fr])
    lab_fr, dep_fr = labS[si], depthS[si]
    exit_ch = []
    for c in range(1, n_ch + 1):
        hit = np.unique(runid[fr[(lab_fr == c) & (dep_fr >= 0.5 * band_depth['A'][c - 1])]])
        exit_ch.append(float(run_w[hit].sum() / tot) if tot > 0 else 0.0)
    max_ch = max(exit_ch) if exit_ch else 0.0
    for b in branches:
        m = far[bl == b['cluster']]
        si_m = sidx(cv_s[m])
        for k in ('A', 'B'):
            dm = depth[k][si_m]
            dm = dm[np.isfinite(dm)]
            b[f'depth_{k}'] = float(np.quantile(dm, 0.95)) if len(dm) else None
            b[f'depth_ratio_{k}'] = (float(b[f'depth_{k}'] / ref_depth[k])
                                     if b[f'depth_{k}'] is not None and np.isfinite(ref_depth[k]) else None)
            b[f'connected_to_{k}'] = bool(len(dm))
        # measured from the state the branch hangs off (the nearer one): anything next to A is far from B anyway
        near = [k for k in ('A', 'B') if b[f'depth_{k}'] is not None]
        b['deep_from'] = min(near, key=lambda k: b[f'depth_{k}']) if near else None
        b['depth_ratio'] = b[f"depth_ratio_{b['deep_from']}"] if near else None
        # dead end: the branch's deepest frames (from the state it is deep from) have no graph neighbour deeper by
        # more than one step: the data stop there
        if b['deep_from'] is not None:
            dS = depth[b['deep_from']]
            dm = dS[si_m]
            ok = np.isfinite(dm)
            top = np.unique(si_m[ok][dm[ok] >= np.quantile(dm[ok], 0.9)])
            deeper = [np.any(dS[graph.indices[graph.indptr[i]:graph.indptr[i + 1]]] > dS[i] + 1.0) for i in top]
            b['dead_end'] = float(1 - np.mean(deeper)) if len(top) else None
        else:
            b['dead_end'] = None
        b['dist_to_labelled'] = float(nn_ref.kneighbors(cv_s[m], n_neighbors=1)[0][:, 0].min())
        hit = np.unique(runid[fr[memb == b['cluster']]])
        b['exit_share'] = float(run_w[hit].sum() / tot) if tot > 0 else 0.0
        b['exit_share_channels'] = exit_ch
        b['channel_band_depth'] = band_depth
        # the exit share is reported, not used: exits measured on path data carry the sampler's lineage bias (all
        # loop_stag paths descend from one upper-channel path: the missed entrance gets 0.001 of the exits against
        # 0.35 for the known channel, while plain dynamics from A enters both channels about equally)
        fails = []
        if b['weight_share'] < o['min_weight']:
            fails.append('weight')
        if b['depth_ratio'] is None or b['depth_ratio'] < o['min_depth_ratio']:
            fails.append('depth')
        b['probe'] = not fails
        b['fails'] = fails
        b['score'] = float(b['weight_share'] * (b['depth_ratio'] or 0.0))
    branches.sort(key=lambda b: (not b['probe'], -b['score']))


class ChannelMap:
    """c(x): label (1..n channel, 0 common, -1 unknown) and vote share of CV points: a k-nearest-neighbour vote over
    the labelled map points (frames of the data). This is the classifier the channel-aware order parameter needs:
    the clustering labels only the frames it saw."""

    def __init__(self, ref_scaled, ref_lab, scale, n_ch, k=5, d_unknown=8.0):
        self.ref, self.lab, self.scale, self.n_ch = ref_scaled, np.asarray(ref_lab), np.asarray(scale), n_ch
        self.k, self.d_unknown = k, d_unknown
        self.nn = NearestNeighbors(n_neighbors=k).fit(ref_scaled)

    def distance(self, cv):
        return self.nn.kneighbors(np.atleast_2d(cv) / self.scale, n_neighbors=1)[0][:, 0]

    def __call__(self, cv):
        dist, idx = self.nn.kneighbors(np.atleast_2d(cv) / self.scale)
        labs = self.lab[idx]
        cand = np.arange(-1, self.n_ch + 1)
        votes = np.stack([(labs == c).mean(1) for c in cand], axis=1)
        lab = cand[votes.argmax(1)]
        lab = np.where(dist[:, 0] > self.d_unknown, -1, lab)
        return lab, votes.max(1)

    def save(self, path):
        np.savez(path, ref=self.ref.astype(np.float32), lab=self.lab.astype(np.int8), scale=self.scale,
                 n_ch=self.n_ch, k=self.k, d_unknown=self.d_unknown)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z['ref'], z['lab'], z['scale'], int(z['n_ch']), int(z['k']), float(z['d_unknown']))


# -- figure -----------------------------------------------------------------------------------------------------
def plot(d, summary, aux, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cv = d['cv']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True, sharey=True)
    rng = np.random.default_rng(1)
    bg = rng.choice(len(cv), size=min(60000, len(cv)), replace=False)
    for a in ax:
        a.scatter(cv[bg, 0], cv[bg, 1], s=0.3, c='0.85', rasterized=True)
    mc, bl = aux['mid_cv'], aux['band_lab']
    sb = rng.choice(len(mc), size=min(30000, len(mc)), replace=False)
    ax[0].scatter(mc[sb, 0], mc[sb, 1], s=2, c=bl[sb], cmap='tab10', vmin=0, vmax=9, rasterized=True)
    for cand in summary['candidate_channels']:
        ax[0].annotate(f"candidate ({cand['paths']} path(s))", cand['centre'][:2], color='red', fontsize=8)
        ax[0].scatter(*cand['centre'][:2], marker='x', c='red', s=40)
    sh = summary['flux_share_unassigned_then_channels']
    shares = ', '.join(f'{c + 1}: {s:.2f} ({n} paths)' for c, (s, n) in
                       enumerate(zip(sh[1:], summary['band_paths_per_channel'])))
    shares += f"; {len(summary['candidate_channels'])} candidate(s)"
    ex = summary['flux_share_exact']
    ax[0].set_title(f"steps overlapping q in {summary['band']}: {summary['n_channels']} channel(s)\nflux share {shares}"
                    + (f"\n(exact q, unassigned first: {', '.join(f'{s:.2f}' for s in ex)})" if ex else ''), fontsize=9)
    lo, hi = np.quantile(cv[:, :2], 0.001, axis=0), np.quantile(cv[:, :2], 0.999, axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 220), np.linspace(lo[1], hi[1], 160))
    pts = np.column_stack([gx.ravel(), gy.ravel()] + [np.full(gx.size, np.median(cv[:, j])) for j in range(2, cv.shape[1])])
    lab, pur = aux['cmap'](pts)
    ax[1].pcolormesh(gx, gy, np.where(lab.reshape(gx.shape) < 0, np.nan, lab.reshape(gx.shape)), cmap='tab10',
                     vmin=0, vmax=9, shading='auto', alpha=0.8)
    ax[1].set_title('channel map c(x): 0 common (blue), 1 orange, 2 green, ...; blank = unknown', fontsize=9)
    ax[2].scatter(cv[bg, 0], cv[bg, 1], s=0.3, c=aux['lab_of_frame'][bg], cmap='tab10', vmin=0, vmax=9, rasterized=True)
    far, bl = aux.get('branches_far'), aux.get('branches_cluster')
    if far is not None and len(far):
        probe = {b['cluster'] for b in summary['unexplored_branches'] if b.get('probe')}
        isp = np.isin(bl, list(probe)) if bl is not None else np.zeros(len(far), bool)
        ax[2].scatter(cv[far[~isp], 0], cv[far[~isp], 1], s=2, c='orange', rasterized=True)
        ax[2].scatter(cv[far[isp], 0], cv[far[isp], 1], s=3, c='red', rasterized=True)
        for b in summary['unexplored_branches'][:6]:
            ax[2].annotate(f"{'PROBE ' if b.get('probe') else ''}d {b.get('depth_ratio') or 0:.2f}, "
                           f"exits {b.get('exit_share', 0):.3f}", b['centre'][:2], fontsize=7,
                           color='red' if b.get('probe') else 'darkorange')
    ax[2].set_title(f"frames by piece label (blue = no band step); branches: red = probe, orange = not "
                    f"({len(summary['unexplored_branches'])})", fontsize=9)
    for a in ax:
        a.set_xlabel(summary['cv_names'][0])
    ax[0].set_ylabel(summary['cv_names'][1])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--npz', help='staple-notebook fit npz (path relative to the repo or absolute)')
    src.add_argument('--loop', help='loop name (its pool, all passes)')
    src.add_argument('--infretis', help='infretis run directory, labelled with the exact committor (--system)')
    ap.add_argument('--system', help='ase_systems name for --infretis (exact committor)')
    ap.add_argument('--paths-per-ensemble', type=int, default=1000)
    ap.add_argument('--name', required=True)
    ap.add_argument('--cv-cols', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--cv-names', nargs='+')
    ap.add_argument('--q-key', default='q_fit')
    ap.add_argument('--band', type=float, nargs=2, default=[0.3, 0.7])
    ap.add_argument('--method', choices=['knn', 'dbscan', 'hdbscan'], default='knn')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--n-band', type=int, default=40000)
    ap.add_argument('--r-max', type=float, default=6.0, help='longest graph edge, in RMS steps')
    ap.add_argument('--r-merge', type=float, default=20.0,
                    help='small components closer than this (RMS steps) to a channel join it')
    ap.add_argument('--min-paths', type=int, default=5, help='independent paths a component needs to be a channel')
    ap.add_argument('--min-weight', type=float, default=RANK_DEFAULTS['min_weight'],
                    help='branch ranking: smallest weight share worth probing')
    ap.add_argument('--min-depth-ratio', type=float, default=RANK_DEFAULTS['min_depth_ratio'],
                    help='branch ranking: depth from A relative to a known channel\'s band depth')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    if a.npz:
        npz = a.npz if os.path.isabs(a.npz) else os.path.join(REPO, a.npz)
        d = from_npz(npz, a.cv_cols, a.q_key, a.cv_names)
        src_txt = npz
    elif a.infretis:
        d = from_infretis(a.infretis, a.system, tuple(a.cv_cols), a.paths_per_ensemble)
        src_txt = f'infretis {a.infretis} (exact committor of {a.system})'
    else:
        d = from_loop(a.loop)
        src_txt = f'loop {a.loop}'
    summary, aux = find_channels(d, band=tuple(a.band), n_band=a.n_band, method=a.method, k=a.k, r_max=a.r_max, r_merge=a.r_merge,
                                  min_paths=a.min_paths,
                                  rank_opts=dict(min_weight=a.min_weight, min_depth_ratio=a.min_depth_ratio))
    summary['source'] = src_txt
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    tag = '' if a.method == 'knn' else f'_{a.method}'
    with open(os.path.join(out, f'summary{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    aux['cmap'].save(os.path.join(out, f'channel_map{tag}.npz'))
    plot(d, summary, aux, os.path.join(out, f'channels{tag}.png'), f'{a.name} ({a.method})')
    print(json.dumps({k: v for k, v in summary.items() if k != 'unexplored_branches'}, indent=1))
    print(f"unexplored branches: {json.dumps(summary['unexplored_branches'], indent=1)}")
    print(f'-> {out}')


if __name__ == '__main__':
    main()
