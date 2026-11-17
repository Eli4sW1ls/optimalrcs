"""CV screening by held-out dynamics (M3, experiment E7).

The labels cannot tell a slow CV that matters from one that only labels paths: NPNEq uses a path-constant CV to
fit path-level noise (docs/param_lambda_embedding.md, finding 4), so a student trained on those labels leans on it
in both cases. The test here does not use the labels for the decision: the student is trained once on all
candidate CVs and once with each candidate left out, and each model is scored by its **held-out Dirichlet energy**

    E(q) = sum_pairs w (q(x_{t+lag}) - q(x_t))^2 / sum_pairs w      over the validation pieces (path weights),

the variational objective of the committor (the exact committor minimizes it among functions with the state
values; states are exact in every export, so q = 0 / 1 there). A CV the committor depends on lowers E; a CV that
only labels paths adds per-path offsets that leave the increments along a path unchanged.

K-fold (default K = 5): the MC blocks of every pass (pool.block, 10% of its rows) are dealt into K folds
(block b -> fold b mod K, so a fold mixes early and late sampling); for each fold the student is trained on the
other folds and E is evaluated on the fold, so every path is held out once. A candidate is kept when leaving it out
raises the pooled held-out E (sum over folds) by more than `tol` AND raises E in a majority of the folds.

Why K-fold: on one validation block the statistic depends on the block. loop_zw pass 1, leaving out x / w, two
student seeds (which agree): train blocks +6% / +3%, the val block +15% / -5%, the test block +50% / +94%. Path
weights are heavy-tailed, and the energy differences between imperfect students are of second order
(E(q) = E(q*) + |q - q*|^2 in the energy norm), so a single block can flip the decision.

Why not SD(Z_q): on the gate system (loop_zw, pass 1) leaving out x or w changed the validation SD(Z_q) by -1%,
while E rose by 49% and 64% and the RMS error against the exact committor from 0.054 to 0.078 (exact E 8.6e-6, all
CVs 1.32e-5, without x 1.96e-5, without w 2.16e-5, y alone 2.15e-5; lag 1). SD(Z_q) is still reported.

    res = screen.run(view, cfg, specs, seed, log)       # res['kept'], res['rows'], res['path_constant']

[screen] config: enabled (default true for systems with more than two CVs), candidates (default: every CV
except the state CV), tol (default 0.1), lag (default 1; lags 1, 4, 16 are reported), folds (default 5). The state CV is never
dropped (states are exact in it).

Also reported, not used for the decision: the path-constant share of each CV (weighted variance of the per-path
means over the total variance; ~1 for a CV that hardly moves along a path).
"""
import numpy as np

import common
from lambda_runtime import LambdaModel

from loop import student

LAGS = (1, 4, 16)


def path_constant_share(view, columns):
    """{cv name: between-path variance / total variance} over the training frames (path weights)."""
    tr = np.flatnonzero(view.mask('train') & view.valid)
    pid = view.path[tr]
    w = view.w[tr].astype(float)
    uniq, inv = np.unique(pid, return_inverse=True)
    out = {}
    for c in columns:
        j = list(view.cv_names).index(c)
        x = view.cv[tr, j].astype(float)
        mu = np.average(x, weights=w)
        tot = np.average((x - mu) ** 2, weights=w)
        sw = np.bincount(inv, w)
        pm = np.bincount(inv, w * x) / np.where(sw > 0, sw, 1)
        between = np.average((pm[inv] - mu) ** 2, weights=w)
        out[c] = float(between / tot) if tot > 0 else float('nan')
    return out


def dirichlet(view, q, mask, lags=LAGS):
    """{lag: path-weighted mean squared increment of q over the frame pairs of `mask` (same piece)}."""
    out = {}
    for lag in lags:
        i0, i1 = view.pairs(mask, lag)
        if len(i0) == 0:
            out[lag] = float('nan')
            continue
        out[lag] = float(np.average((q[i1] - q[i0]) ** 2, weights=view.w[i0].astype(float)))
    return out


def enabled(cfg, view):
    sc = cfg.get('screen', {})
    return bool(sc.get('enabled', len(view.cv_names) > 2))


def fold_of_frame(view, k_folds):
    """Fold index per frame: the MC block of its piece modulo K (blocks of each pass, after its burn-in)."""
    return (np.asarray(view.piece_block)[view.piece] % k_folds).astype(int)


def _energy_sums(view, q, mask, lags):
    """{lag: (sum of w (dq)^2, sum of w)} over the pairs of `mask` (pooled over folds)."""
    out = {}
    for lag in lags:
        i0, i1 = view.pairs(mask, lag)
        w = view.w[i0].astype(float)
        out[lag] = (float(np.sum(w * (q[i1] - q[i0]) ** 2)), float(np.sum(w)))
    return out


def _cv_energy(view, cfg, specs, seed, folds, k_folds, lags):
    """Per-fold held-out energy sums of the student on `specs` (trained on the other folds)."""
    saved = view.frame_split
    per_fold, sdz = [], []
    try:
        for f in range(k_folds):
            hold = folds == f
            if not hold.any():
                continue
            view.frame_split = np.where(hold, 'val', 'train').astype(object)
            params, info = student.train(view, cfg, seed=seed + f, specs=specs, log=None)
            m = LambdaModel(student.build_export(view, params, cfg, blend=False))
            idx = np.flatnonzero(hold)
            q = np.zeros(view.n)
            q[idx] = m.q(view.cv[idx])
            per_fold.append(_energy_sums(view, q, hold, lags))
            sdz.append(float(info['sdzqs']))
    finally:
        view.frame_split = saved
    return per_fold, sdz


def _pooled(per_fold, lag):
    num = sum(f[lag][0] for f in per_fold)
    den = sum(f[lag][1] for f in per_fold)
    return num / den if den > 0 else float('nan')


def run(view, cfg, specs, seed=0, log=None):
    """Screen the candidates among `specs` (the current CV set) by K-fold held-out Dirichlet energy.
    Returns dict(kept, rows, path_constant, tol, lag, folds)."""
    sc = cfg.get('screen', {})
    tol, lag, k_folds = float(sc.get('tol', 0.1)), int(sc.get('lag', 1)), int(sc.get('folds', 5))
    lags = tuple(sorted(set(LAGS) | {lag}))
    state = view.cv_names[view.state_cv]
    cands = [c for c in sc.get('candidates', [c for c in specs if c != state]) if c in specs and c != state]
    folds = fold_of_frame(view, k_folds)
    pcs = path_constant_share(view, list(specs))
    base, base_sdz = _cv_energy(view, cfg, list(specs), seed, folds, k_folds, lags)
    row0 = dict(left_out=None, specs=list(specs), energy={L: _pooled(base, L) for L in lags},
                fold_energy=[f[lag][0] / f[lag][1] for f in base], sdzqs=base_sdz)
    if view.has_exact:
        qe = np.where(np.isfinite(view.q_exact), view.q_exact, 0.0).astype(float)
        ex = [_energy_sums(view, qe, folds == f, lags) for f in range(k_folds) if (folds == f).any()]
        row0['energy_exact'] = {L: _pooled(ex, L) for L in lags}
    rows = [row0]
    kept = list(specs)
    for c in cands:
        sub = [s for s in specs if s != c]
        pf, sdz = _cv_energy(view, cfg, sub, seed, folds, k_folds, lags)
        rise = _pooled(pf, lag) / row0['energy'][lag] - 1.0
        fold_rise = [a[lag][0] / a[lag][1] / (b[lag][0] / b[lag][1]) - 1.0 for a, b in zip(pf, base)]
        n_up = int(sum(r > 0 for r in fold_rise))
        keep = rise > tol and n_up > len(fold_rise) / 2
        rows.append(dict(left_out=c, specs=sub, energy={L: _pooled(pf, L) for L in lags}, rel_change=float(rise),
                         fold_rel_change=fold_rise, folds_up=n_up, keep=bool(keep), path_constant=pcs.get(c),
                         sdzqs=sdz))
        if not keep:
            kept.remove(c)
    common.say(f'screen ({len(base)}-fold Dirichlet energy, lag {lag}): all {row0["energy"][lag]:.4g}'
               + (f' (exact q {row0["energy_exact"][lag]:.4g})' if 'energy_exact' in row0 else '') + '; '
               + '; '.join(f'without {r["left_out"]} {r["energy"][lag]:.4g} ({100 * r["rel_change"]:+.0f}%, up in '
                           f'{r["folds_up"]}/{len(r["fold_rel_change"])} folds: {"keep" if r["keep"] else "drop"}, '
                           f'path-constant share {r["path_constant"]:.2f})' for r in rows[1:]) + f' -> {kept}', log=log)
    return dict(kept=kept, rows=rows, path_constant=pcs, tol=tol, lag=lag, folds=len(base), candidates=cands,
                statistic='dirichlet_kfold')
