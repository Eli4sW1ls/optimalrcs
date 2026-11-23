"""Do histories (or other candidate variables) carry information that a committor estimate misses?  No exact committor.

The committor is a martingale along a path: the stopped increment q(x_{t'}) - q(x_t), t' = min(t + Delta, end of the
piece), has expectation zero given EVERYTHING up to t (the whole past included). A committor estimate r built on a set
of CVs that leaves out relevant information is not a martingale in a larger set of variables: its increment becomes
predictable from them. So, held out by path:

    predict  inc_t = r(t') - r(t)  from candidate features at t  (boosted trees, weights = path weights)
    held-out R^2 > 0 (against zero, bootstrap over paths)  ->  the features hold information r does not use

Candidates: the current CVs (anything the fit failed to resolve), the displacement of (y, x) over the last 8, 32, 128
frames (history), w (oracle: only available because this is a test system). Two estimates r are tested, both from
the 10000-cluster solve at lag Delta (galerkin/galerkin_solve.py: one linear solve, no iteration):
    r_yx    clusters on (y, x), w hidden: the case histories are meant to help
    r_yxw   clusters on (y, x, w): the control; its increments must be unpredictable from everything (R^2 ~ 0)

Frames with less than 128 earlier frames in their piece are left out (pieces start at M), the same frames for every
feature set.

Usage: python martingale_check.py [--lag 50] [--stride 4]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'galerkin'))
import hidden_w_history as h  # noqa: E402
import galerkin_solve as g  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--lag', type=int, default=50)
ap.add_argument('--stride', type=int, default=4, help='use every stride-th transition (autocorrelated frames)')
ap.add_argument('--clusters', type=int, default=10000)
ap.add_argument('--hist-lags', type=int, nargs='+', default=[8, 32, 128])
a = ap.parse_args()

from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

d = h.load_data()
inner = ~d['A'] & ~d['B']
t0, t1 = g.transitions(d, a.lag)
Xall = d['X'][:, [h.CV_COLS[c] for c in ('y', 'x', 'w')]].astype(np.float64)


def cluster_solve(cols):
    S = Xall[:, cols] / Xall[inner][:, cols].std(0)
    rng = np.random.default_rng(0)
    sub = rng.choice(np.flatnonzero(inner), size=400_000, replace=False)
    km = MiniBatchKMeans(n_clusters=a.clusters, random_state=0, batch_size=8192, n_init=3).fit(S[sub])
    q, _ = g.solve_kmeans(d, km.predict(S), a.clusters, t0, t1)
    return np.clip(q, 0, 1)


r_yx = cluster_solve([0, 1])
r_yxw = cluster_solve([0, 1, 2])
print('cluster solves done', flush=True)

# features
it = d['i_traj']
n = len(it)
starts = np.flatnonzero(np.r_[True, it[1:] != it[:-1]])
pid = np.cumsum(np.r_[True, it[1:] != it[:-1]]) - 1
offset = np.arange(n) - starts[pid]
D = max(a.hist_lags)
ok = offset[t0] >= D
sel = np.flatnonzero(ok)[::a.stride]
T0, T1 = t0[sel], t1[sel]
y, x, w = Xall[:, 0], Xall[:, 1], Xall[:, 2]
cur = np.column_stack([y[T0], x[T0], (y[T0] + x[T0]) / np.sqrt(2), (y[T0] - x[T0]) / np.sqrt(2)])
hist = np.column_stack([f(T0) - f(T0 - dl) for dl in a.hist_lags for f in (lambda i: y[i], lambda i: x[i])])
FEATS = {'current (y,x,u,v)': cur,
         '+ history of (y,x)': np.column_stack([cur, hist]),
         '+ w (oracle)': np.column_stack([cur, w[T0]]),
         '+ history + w': np.column_stack([cur, hist, w[T0]])}
wt = d['path_weights'][T0]
grp = d['group_traj'][T0]
ug = np.unique(grp)
rng = np.random.default_rng(1)
test_g = set(rng.choice(ug, size=len(ug) // 2, replace=False).tolist())
test = np.array([gi in test_g for gi in grp])
print(f'{len(T0)} transitions at lag {a.lag} ({test.sum()} held out), {len(ug)} paths', flush=True)


def r2_with_se(inc, pred, wt, grp, n_boot=300):
    """Weighted held-out R^2 against ZERO (the increments should have mean zero) and a bootstrap over paths."""
    tot = wt * inc ** 2
    imp = tot - wt * (inc - pred) ** 2
    ug_, inv = np.unique(grp, return_inverse=True)
    s_imp = np.bincount(inv, weights=imp)
    s_tot = np.bincount(inv, weights=tot)
    r2 = s_imp.sum() / s_tot.sum()
    rng = np.random.default_rng(2)
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(ug_), len(ug_))
        bs.append(s_imp[k].sum() / s_tot[k].sum())
    return float(r2), float(np.std(bs))


rows = []
for name_r, r in (('r_yx (w hidden)', r_yx), ('r_yxw (control)', r_yxw)):
    inc = (r[T1] - r[T0])
    for name_f, F in FEATS.items():
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0)
        m.fit(F[~test], inc[~test], sample_weight=wt[~test])
        pred = m.predict(F[test])
        r2, se = r2_with_se(inc[test], pred, wt[test], grp[test])
        row = dict(estimate=name_r, features=name_f, lag=a.lag, r2_heldout=r2, se=se, z=r2 / se if se > 0 else None)
        rows.append(row)
        print('RESULT ' + json.dumps(row), flush=True)
json.dump(rows, open(os.path.join(HERE, f'martingale_lag{a.lag}.json'), 'w'), indent=1)
print('DONE', flush=True)
