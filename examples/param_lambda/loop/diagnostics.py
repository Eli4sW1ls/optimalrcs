"""Per-pass diagnostics (no labels needed unless stated):

- `zq_profile(model, view)`: stopped Z_q of the model's committor m on the view's pieces, SD over q per lag and
  the max |z| with MC-block groups (path groups here; the pieces of one path are one group);
- `ood_by_ensemble(model, view)`: fraction of frames outside the model's training data, per ensemble;
- `stability(m_new, m_old, ref_cv, levels)`: spread of logit m_new on the slabs of m_old and the reverse, per level;
- `exact_metrics(model, view)`: with an exact committor only (never used for fitting, selection or stopping):
  isosurface spread against the exact committor and binned logit RMSE.
"""
import numpy as np

import common
import evaluate as ev


def _r_b(model, view, mask):
    idx = np.flatnonzero(mask)
    m = np.zeros(len(idx))
    inside = view.inside[idx]
    m[inside] = model.q(view.cv[idx][inside])
    m[view.B[idx]] = 1.0
    b = (view.A[idx] | view.B[idx]).astype(float)
    return idx, m, b


def zq_profile(model, view, mask=None, lags=(1, 4, 16, 64, 256)):
    mask = np.ones(view.n, bool) if mask is None else mask
    idx, r, b = _r_b(model, view, mask)
    zmax, sd, _ = ev.zq_stats(r, b, view.piece[idx], view.w[idx].astype(float), view.path[idx], list(lags))
    return dict(lags=list(lags), sd=sd.tolist(), zmax=zmax.tolist(), sdzqs=float(sd.max()), zqs_z=float(zmax.max()))


def ood_by_ensemble(model, view, mask=None):
    mask = np.ones(view.n, bool) if mask is None else mask
    idx = np.flatnonzero(mask & view.inside)
    bad = model.ood(view.cv[idx])
    ens = view.ens[idx]
    w = view.w[idx].astype(float)
    out = {int(e): float(np.sum(w[(ens == e) & bad]) / np.sum(w[ens == e])) for e in np.unique(ens)}
    out['all'] = float(np.sum(w[bad]) / np.sum(w)) if len(w) else np.nan
    return out


def stability(m_new, m_old, ref_cv, ref_w, levels, half_logit=0.2):
    """Spread (weighted variance) of logit m_new on the slabs of m_old at `levels`, and the reverse."""
    a, b = m_new.q(ref_cv), m_old.q(ref_cv)
    za, zb = common.logit(a), common.logit(b)
    fwd, n1 = ev.iso_spread(b, za, ref_w, levels, half_logit)
    rev, n2 = ev.iso_spread(a, zb, ref_w, levels, half_logit)
    return dict(levels=list(map(float, levels)), new_on_old=fwd.tolist(), old_on_new=rev.tolist(),
                n_old=n1.tolist(), n_new=n2.tolist(), floor=half_logit ** 2 / 3)


def reference_set(view, n_per_bin=2000, n_bins=20, seed=0):
    """A fixed stratified reference set of frames (bins of the state CV), for the stability check."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(view.inside)
    s = view.cv[idx, view.state_cv]
    edges = np.linspace(view.lambda_a, view.lambda_b, n_bins + 1)
    b = np.clip(np.searchsorted(edges, s) - 1, 0, n_bins - 1)
    pick = []
    for k in range(n_bins):
        sel = idx[b == k]
        if len(sel):
            pick.append(rng.choice(sel, min(n_per_bin, len(sel)), replace=False))
    pick = np.sort(np.concatenate(pick)) if pick else idx[:0]
    return view.cv[pick].copy(), view.w[pick].astype(float)


def exact_metrics(model, view, levels, mask=None, half_logit=0.2):
    if not view.has_exact:
        return None
    mask = (np.ones(view.n, bool) if mask is None else mask) & view.inside
    idx = np.flatnonzero(mask)
    m = model.q(view.cv[idx])
    zx = common.logit(view.q_exact[idx].astype(float))
    w = view.w[idx].astype(float)
    sp, n = ev.iso_spread(m, zx, w, levels, half_logit)
    rmse = ev.binned_logit_rmse(view.cv[idx][:, :min(view.dim, 2)], common.logit(m), zx, w, np.ones(len(idx), bool))
    return dict(levels=list(map(float, levels)), spread=sp.tolist(), spread_n=n.tolist(),
                spread_mean=float(np.nanmean(sp)) if np.isfinite(sp).any() else np.nan, logit_rmse=rmse)
