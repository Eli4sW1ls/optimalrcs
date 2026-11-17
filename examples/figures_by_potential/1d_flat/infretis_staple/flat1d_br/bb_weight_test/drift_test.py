"""Fit-free bias test of the pieces: E[q_exact(piece end) - q_exact(piece start)] = 0 for unbiased futures.

piece end: A -> 0, B -> 1, turn -> q_exact(x_end). Per start bin and per weighting, bootstrap s.e. over paths.
Then a short NPNEq fit (rejections) per run for RMS(fit - exact).
    python drift_test.py RUN_DIR [RUN_DIR ...]
"""
import argparse, os, sys, time
import numpy as np

EX = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples'
sys.path.insert(0, f'{EX}/../optimalrcs-cupy/'); sys.path.insert(0, EX)
import tis_data

A, B = -0.1, 0.3
SIG = 0.00894
shift = 0.5826 * SIG
qx = lambda x: np.clip((x - A + shift) / (B - A + 2 * shift), 0, 1)   # exact to 3e-6 for this chain (notebook)


def load(run, w, reversed_=True):
    p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
    argv = [run, '--burn-in-fraction', '0.3', '--thin', '1', '--weights', w, '--lambda-a', str(A), '--lambda-b', str(B),
            '--paths-per-ensemble', '5000']
    if reversed_:
        argv.append('--reversed')
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return tis_data.prepare_data(p.parse_args(argv))


def drift(d, nboot=200, rng=np.random.default_rng(0)):
    i = d.i_traj
    first = np.r_[0, np.flatnonzero(np.diff(i)) + 1]
    last = np.r_[first[1:] - 1, len(i) - 1]
    qs = qx(d.lam[first])
    qe = np.where(d.boundary0[last], 0.0, np.where(d.boundary1[last], 1.0, qx(d.lam[last])))
    w = d.path_weights[first]
    g = d.group_traj[first]
    D = qe - qs
    ug, gi = np.unique(g, return_inverse=True)
    out = {}
    edges = [-0.1, 0.0, 0.1, 0.2, 0.3]
    # by the interface the piece starts at (= its M), and pooled
    for lab, m in [('all', np.ones_like(qs, bool))] + [(f'M~{e:+.1f}', np.abs(d.lam[first] - e) < 0.03) for e in edges[:-1]]:
        if m.sum() == 0:
            continue
        sw = np.bincount(gi[m], w[m], len(ug)); sd = np.bincount(gi[m], (w * D)[m], len(ug))
        est = sd.sum() / sw.sum()
        bs = []
        for _ in range(nboot):
            k = rng.integers(0, len(ug), len(ug))
            bs.append(sd[k].sum() / sw[k].sum())
        out[lab] = (est, np.std(bs), int(m.sum()), float((w * (qe < 0.5) * (d.boundary0[last] | d.boundary1[last] == False))[m].sum() / w[m].sum()))
    return out


for run in sys.argv[1:]:
    print(f'===== {run}', flush=True)
    for w in ('rejections', 'staple'):
        for rev in (False, True):
            d = load(run, w, rev)
            r = drift(d)
            print(f'  {w:10s} reversed={rev!s:5s}: ' + '  '.join(f'{k} {v[0]*1e3:+6.2f}±{v[1]*1e3:.2f}' for k, v in r.items())
                  + '   (x1e-3)', flush=True)
