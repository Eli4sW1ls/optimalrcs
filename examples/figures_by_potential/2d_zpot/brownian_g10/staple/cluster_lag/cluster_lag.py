"""Recipe check on the 2D Z staple data (sim_istarz_br_g10, forward + mirrored pieces, rejection weights): cluster solve
(k-means on (y, x), indicator basis = committor of the cluster transition matrix, galerkin_solve.solve_kmeans) at several
lags, scored against the finite-volume reference committor (h 0.001). Reports the plateau bias (fit - exact over frames
with exact q in 0.55-0.85, as lag_halves.py), the RMS per (y, x) bin over bins with exact q in 0.05-0.95, and RMS per frame.
    python cluster_lag.py [--ppe 400] [--lags 1 5 20 50 100] [--clusters 2000]
"""
import argparse, contextlib, io, json, os, sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO + '/examples/')
sys.path.insert(0, REPO + '/examples/figures_by_potential/3d_zpot_w/hidden_w_history/galerkin')
import tis_data
import galerkin_solve as g
ap = argparse.ArgumentParser()
ap.add_argument('--ppe', default='400'); ap.add_argument('--lags', type=int, nargs='+', default=[1, 5, 20, 50, 100])
ap.add_argument('--clusters', type=int, default=2000)
a = ap.parse_args()
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args(['/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10',
        '--reversed', '--paths-per-ensemble', a.ppe, '--seed', '0']))
ref = np.load(REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/committor_reference_h0.001.npz')
f_ref = RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)
y, x = data.X_all[:, 0].astype(float), data.X_all[:, 1].astype(float)
A, B = np.asarray(data.boundary0), np.asarray(data.boundary1)
w = np.asarray(data.path_weights, float); w = w / w.mean()
d = dict(i_traj=np.asarray(data.i_traj), A=A, B=B, path_weights=w)
free = ~(A | B); n = len(y)
q_ex = np.clip(f_ref(np.c_[y, x]), 0, 1)
print(f'{n} frames, {int(free.sum())} free', flush=True)
ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
nb = np.histogram2d(y[free], x[free], bins=(ey, ex))[0]
bm = lambda v: np.where(nb >= 20, np.histogram2d(y[free], x[free], bins=(ey, ex), weights=v[free])[0] / np.maximum(nb, 1), np.nan)
QX = bm(q_ex); trans = (QX > 0.05) & (QX < 0.95); plateau = free & (q_ex > 0.55) & (q_ex < 0.85)
from sklearn.cluster import MiniBatchKMeans
S = np.c_[y / y[free].std(), x / x[free].std()]
rng = np.random.default_rng(0); sub = rng.choice(np.flatnonzero(free), size=min(400_000, int(free.sum())), replace=False)
km = MiniBatchKMeans(n_clusters=a.clusters, random_state=0, batch_size=8192, n_init=3).fit(S[sub]); lab = km.predict(S)
rows = []
for lag in a.lags:
    t0, t1 = g.transitions(d, lag)
    q, empty = g.solve_kmeans(d, lab, a.clusters, t0, t1); q = np.clip(q, 0, 1)
    row = dict(lag=lag, plateau_bias=float(np.average((q - q_ex)[plateau], weights=w[plateau])),
               rms_bin=float(np.sqrt(np.nanmean((bm(q) - QX)[trans] ** 2))),
               rms_frame=float(np.sqrt(np.average((q - q_ex)[free] ** 2, weights=w[free]))), empty_clusters=empty)
    rows.append(row); print('RESULT ' + json.dumps(row), flush=True)
    np.save(os.path.join(HERE, f'q_ppe{a.ppe}_k{a.clusters}_lag{lag}.npy'), q.astype(np.float32))
json.dump(rows, open(os.path.join(HERE, f'cluster_lag_ppe{a.ppe}_k{a.clusters}.json'), 'w'), indent=1)
print('DONE', flush=True)
