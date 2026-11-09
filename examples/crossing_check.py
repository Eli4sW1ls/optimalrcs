"""Level check of a committor fitted to staple path pieces with the iSTAR turn-to-turn MSM (tistools).

The MSM of tistools (`istar_analysis`) has states "turned at lambda_j, heading right" [j>] and
"turned at lambda_j, heading left" [j<]. From the path ensembles alone (turn interfaces of the
staple paths, no committor) it estimates q_t[j][k]: the probability that a trajectory that last
turned at lambda_j and has reached lambda_{k-1} (heading right; lambda_{k+1} heading left) also
reaches lambda_k. Solving the MSM gives z[state] = P(B before A | state).

Every path piece in the committor data starts at a crossing of its ensemble's middle interface
lambda_m (upward or downward), and its partner piece (the same path read the other way) holds the
turn the trajectory came from: interface j. For such a crossing the MSM predicts

    up   (j < m): C = sum_{k=m}^{N-2} P(next turn at k | j, reached m) z[k<] + P(reach B | j, reached m)
    down (j > m): C = sum_{k=1}^{m}   P(next turn at k | j, reached m) z[k>]   (+ 0 for reaching A)

with P(next turn at k | j, reached m) = prod_{l=m+1}^{k} q_t[j][l] (1 - q_t[j][k+1]) (up), and the
mirror image down. By the strong Markov property the committor averaged over these crossing
points is exactly the probability of B before A after such a crossing; the MSM gives it under its
memory-loss assumption (the future after a turn depends on the turn interface only). So the
weighted mean of a fitted q over the start frames of each (m, direction) must match the MSM
prediction there. The MSM uses the outcomes of whole paths turn by turn, i.e. how far the paths
get, which fixes the level of q at every interface, also where the stopped Z_q cannot (plateaus).

Usage: see `run_check` at the bottom (2D Brownian Z example).
"""
import contextlib
import io
import os

import numpy as np


# ------------------------------------------------------------------------------------------------
# MSM from tistools
# ------------------------------------------------------------------------------------------------
def istar_msm(tis_dir, interfaces, weighting='staple', cycles=None):
    """q_t (conditional turn-to-turn probabilities), the MSM matrix and its committor per state.
    weighting: 'staple' (tistools default, with the high-acceptance factor) or 'rejections'.
    cycles: optional (start, stop) to use a block of MC cycles."""
    from tistools.reading import read_pathensemble, get_weights, set_flags_ACC_REJ
    from tistools import istar_analysis as ia
    folders = sorted(d for d in os.listdir(tis_dir) if d.isdigit() and os.path.isdir(os.path.join(tis_dir, d)))
    N = len(interfaces)
    acc, rej = set_flags_ACC_REJ()
    pes, weights = [], {}
    for i, d in enumerate(folders[:N]):
        pe = read_pathensemble(os.path.join(tis_dir, d, 'pathensemble.txt'))
        pe.name = d
        with contextlib.redirect_stdout(io.StringIO()):
            if weighting == 'staple':
                w = ia.get_weights_staple(i, pe.flags, pe.generation, pe.lmrs, N, acc, rej, verbose=False)
            else:
                w = get_weights(pe.flags, acc, rej, verbose=False)[0]
        w = np.asarray(w, float)
        if cycles is not None:          # weights from the whole chain, then only this block of cycles
            w = w * ((pe.cyclenumbers >= cycles[0]) & (pe.cyclenumbers < cycles[1]))
        pes.append(pe)
        weights[i] = w
    with contextlib.redirect_stdout(io.StringIO()):
        X = ia.compute_weight_matrices(pes, interfaces, N, tr=False, weights=weights)
        p, q_t = ia.get_transition_probs_weights(X)
        M = ia.construct_M_istar(p, ia.istar_n_states(N), N)
        z1, z2, y1, y2 = ia.global_pcross_msm_star(M)
    z = np.r_[0.0, np.ravel(z2), 1.0]                    # committor of states 0 .. 2N-2
    return dict(q_t=q_t, p=p, M=M, z=z, N=N, pcross=float(np.ravel(y1)[0]))


def z_turn(msm, j, heading):
    """MSM committor of 'turned at lambda_j' heading 'right' or 'left' (state layout of
    tistools.istar_analysis.istar_n_states)."""
    N, z = msm['N'], msm['z']
    if heading == 'right':
        return z[1] if j == 0 else z[1 + j]              # [0->] or [j>]
    return z[N - 1 + j]                                 # [j<]


def predicted_committor(msm, j, m, direction):
    """MSM prediction of P(B before A) at an upward (direction +1, j < m) or downward (-1, j > m)
    crossing of lambda_m by a trajectory that last turned at lambda_j."""
    N, qt = msm['N'], msm['q_t']
    total = 0.0
    if direction > 0:
        reach = 1.0                                     # P(reached lambda_k | j, reached m)
        for k in range(m, N - 1):
            if k > m:
                reach *= qt[j][k]
            total += reach * (1 - qt[j][k + 1]) * z_turn(msm, k, 'left')
        reach *= qt[j][N - 1] if N - 1 > m else 1.0
        return total + reach                            # reach B
    reach = 1.0
    for k in range(m, 0, -1):
        if k < m:
            reach *= qt[j][k]
        total += reach * (1 - qt[j][k - 1]) * z_turn(msm, k, 'right')
    return total                                        # reaching A contributes 0


# ------------------------------------------------------------------------------------------------
# crossings in the committor data
# ------------------------------------------------------------------------------------------------
def crossings(lam, i_traj, group, weights, interfaces):
    """Per piece with a partner piece (same group): start frame, crossed interface m, direction,
    turn interface j of the trajectory before the crossing (from the partner piece's extremum),
    weight and group. Pieces at lambda_A, lambda_B or coming from B are left out."""
    lam = np.asarray(lam, float)
    intf = np.asarray(interfaces, float)
    n, N = len(lam), len(intf)
    ends = np.r_[np.flatnonzero(np.diff(i_traj) != 0), n - 1]
    starts = np.r_[0, ends[:-1] + 1]
    lmin = np.minimum.reduceat(lam, starts)
    lmax = np.maximum.reduceat(lam, starts)
    g = np.asarray(group)[starts]
    lam0 = lam[starts]
    m = np.abs(lam0[:, None] - intf[None, :]).argmin(axis=1)
    direction = np.where(lam0 > intf[m], 1, -1)
    rows = []
    for a in range(len(starts) - 1):
        if g[a] != g[a + 1]:
            continue
        for me, other in ((a, a + 1), (a + 1, a)):
            mm, d = m[me], direction[me]
            if not 1 <= mm <= N - 2:
                continue
            if d > 0:                                   # came from below: turn at the partner's minimum
                if lmin[other] <= intf[0]:
                    j = 0
                else:
                    j = int(np.searchsorted(intf, lmin[other]))   # lmin in [lambda_{j-1}, lambda_j]
                if j >= mm:
                    continue
            else:                                       # came from above: turn at the partner's maximum
                if lmax[other] >= intf[-1]:
                    continue                            # came from B
                j = int(np.searchsorted(intf, lmax[other])) - 1    # lmax in [lambda_j, lambda_{j+1}]
                if j <= mm:
                    continue
            rows.append((starts[me], mm, d, j))
    rows = np.array(rows, dtype=int)
    return dict(start=rows[:, 0], m=rows[:, 1], dir=rows[:, 2], j=rows[:, 3],
                w=np.asarray(weights)[rows[:, 0]], group=np.asarray(group)[rows[:, 0]])


def compare(cr, msm, q_frames, n_boot=200, seed=0):
    """Weighted means per (m, direction) of the MSM prediction and of every q in q_frames at the
    crossing frames, with bootstrap errors over groups (paths)."""
    pred = np.array([predicted_committor(msm, j, m, d) for j, m, d in zip(cr['j'], cr['m'], cr['dir'])])
    vals = {'MSM': pred, **{k: np.asarray(v)[cr['start']] for k, v in q_frames.items()}}
    cells = sorted(set(zip(cr['m'], cr['dir'])))
    _, gi = np.unique(cr['group'], return_inverse=True)
    rng = np.random.default_rng(seed)

    def means(w):
        return {k: np.array([np.sum(w[(cr['m'] == m) & (cr['dir'] == d)] * v[(cr['m'] == m) & (cr['dir'] == d)])
                             / np.sum(w[(cr['m'] == m) & (cr['dir'] == d)]) for m, d in cells]) for k, v in vals.items()}
    est = means(cr['w'])
    boots = [means(cr['w'] * rng.poisson(1.0, gi.max() + 1)[gi]) for _ in range(n_boot)]
    err = {k: np.std([b[k] for b in boots], axis=0) for k in est}
    diff_err = {k: np.std([b[k] - b['MSM'] for b in boots], axis=0) for k in est if k != 'MSM'}
    count = np.array([np.sum((cr['m'] == m) & (cr['dir'] == d)) for m, d in cells])
    return dict(cells=cells, est=est, err=err, diff_err=diff_err, count=count, pred=pred)
