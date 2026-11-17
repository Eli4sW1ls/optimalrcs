"""Student: the parametric lambda (M3, one fixed architecture, warm-started) and its export.

    params, info = train(view, cfg, seed, init_params)        # models.MLP on the view's training frames
    export = build_export(view, params, cfg, system, ...)       # lambda_runtime format 2

Export (lambda_runtime.LambdaModel, format 2):
- calibration m = g(z) on the training frames (evaluate.calibrate);
- the smooth-minimum ramps: kappa_A = h_lo * span / (n_A sigma_s), so the isosurface of the lowest planned level
  h_lo lies at least n_A RMS steps of s from A (kappa_B likewise with 1 - h_hi);
- the provisional front (front_s) when the teacher used one;
- the trust-region blend: h_seed(s) = isotonic regression of h(s, m) on s over the in-distribution training
  frames, t = S((d2 - thr) / width) with thr the Mahalanobis threshold of the export (quantile of the training
  frames) and width = thr;
- the OOD statistics (cv_min, cv_max, Mahalanobis mean / inverse covariance / threshold) as before.
"""
import time

import numpy as np

import common
import evaluate as ev
import models
from lambda_runtime import LambdaModel, ScoreModel


def _cfg_for_models(cfg_loop):
    """The config tables models.TrainSet / MLP / evaluate read, from the loop config."""
    st = cfg_loop.get('student', {})
    return {
        'train': {'lags': st.get('lags', [1, 2, 4, 8]), 'anchor_stride': int(st.get('anchor_stride', 4)),
                  'mid_weight': float(st.get('mid_weight', 5.0)), 'noise_q': float(st.get('noise_q', 0.0))},
        'm3': {'epochs': int(st.get('epochs', 24)), 'checkpoints': int(st.get('checkpoints', 8)),
               'batch': int(st.get('batch', 4096)), 'lr': float(st.get('lr', 3e-3)),
               'weight_decay': float(st.get('weight_decay', 1e-4)), 'max_anchors': int(st.get('max_anchors', 3_000_000)),
               'huber': float(st.get('huber', 0.0))},
        'eval': {'q_lo': float(st.get('q_lo', 1e-3)), 'q_hi': float(st.get('q_hi', 0.95)),
                 'n_interfaces': int(st.get('n_levels', 10)), 'slab_half_logit': float(st.get('slab_half_logit', 0.2)),
                 'zq_lags': st.get('zq_lags', [1, 4, 16, 64, 256]), 'select_lags': st.get('select_lags', [1, 4, 16, 64]),
                 'max_pieces_select': int(st.get('max_pieces_select', 4000))},
    }


def train(view, cfg_loop, seed=0, init_params=None, log=None, specs=None):
    """Fit M3 on the view (training split; checkpoint selected by the validation SD(Z_q)). Returns the score
    parameters and an info dict (selected epoch, metrics per checkpoint, seconds). `specs`: the input CVs
    (default all of the view's; the screening step passes subsets). A warm start is used only when the previous
    network had the same inputs."""
    st = cfg_loop.get('student', {})
    cfg = _cfg_for_models(cfg_loop)
    specs = list(view.cv_names) if specs is None else list(specs)
    if init_params is not None and 'feat_specs' in init_params and list(map(str, init_params['feat_specs'])) != specs:
        init_params = None
    ts = models.TrainSet(view, cfg, specs)
    # calibration frames: evenly spaced training frames; selection frames: validation pieces
    tr = np.flatnonzero(view.mask('train') & view.valid)
    cal_idx = tr[np.unique(np.linspace(0, len(tr) - 1, min(len(tr), 500_000)).astype(int))]
    sel_mask = view.take_pieces('val', cfg['eval']['max_pieces_select'], seed=seed)
    if not sel_mask.any():
        sel_mask = view.take_pieces('train', cfg['eval']['max_pieces_select'], seed=seed)
    score_sets = [((ts.feat(view.cv[i]) - ts.mu) / ts.sd).astype(np.float32) for i in (cal_idx, np.flatnonzero(sel_mask))]
    ckpts = []

    def on_ckpt(epoch, params, scores):
        z_cal, z = scores
        cal = ev.calibrate(z_cal, view.q[cal_idx], view.w[cal_idx].astype(float))
        out = ev.evaluate(view, sel_mask, z, cal, cfg, full=False)
        ckpts.append(dict(epoch=epoch, sdzqs=out['sdzqs'], spread_mean=out['spread_mean']))
        if log:
            common.say(f'   student epoch {epoch:3d}: sdzqs {out["sdzqs"]:8.3f} spread {out["spread_mean"]:.4f}', log=log)

    t0 = time.time()
    arch, beta = list(st.get('arch', [64, 64])), float(st.get('beta', 10.0))
    if init_params is not None and (list(init_params.get('arch', arch)) != arch):
        init_params = None
    m = models.MLP(arch, beta, cfg, seed=seed, init_params=init_params)
    m.fit(ts, on_checkpoint=on_ckpt, score_sets=score_sets)
    best = int(np.argmin([c['sdzqs'] for c in ckpts]))
    m.use_checkpoint(best)
    params = m.export()
    params['arch'] = np.array(arch)
    info = dict(selected_epoch=int(m.epoch), checkpoints=ckpts, seconds=time.time() - t0, arch=arch, beta=beta,
                n_train_frames=int(len(ts.z)), warm_start=init_params is not None, specs=specs,
                sdzqs=float(ckpts[best]['sdzqs']))
    return params, info


def rms_step(view, column):
    """RMS lag-1 step of a CV column over the training pieces (path-weighted)."""
    i0, i1 = view.pairs(view.mask('train'), 1)
    d = view.cv[i1, column] - view.cv[i0, column]
    return float(np.sqrt(np.average(d * d, weights=view.w[i0])))


def build_export(view, score_params, cfg_loop, front_s=None, blend=True, log=None):
    """LambdaModel parameters (format 2) from the score model and the view."""
    st = cfg_loop.get('student', {})
    ic = cfg_loop.get('interfaces', {})
    p = dict(score_params)
    tr = np.flatnonzero(view.mask('train') & view.valid)
    cal_idx = tr[np.unique(np.linspace(0, len(tr) - 1, min(len(tr), 500_000)).astype(int))]
    sm = ScoreModel(score_params)
    z_cal = sm.raw(view.cv[cal_idx])
    zk, qk, dk = ev.calibrate(z_cal, view.q[cal_idx], view.w[cal_idx].astype(float))
    F = sm.features(view.cv[tr[::max(1, len(tr) // 400_000)]])
    mean = F.mean(0)
    icov = np.linalg.pinv(np.atleast_2d(np.cov(F, rowvar=False)))
    d2 = np.einsum('ni,ij,nj->n', F - mean, icov, F - mean)
    thr = float(np.quantile(d2, float(st.get('ood_quantile', 0.999))))
    span = view.lambda_b - view.lambda_a
    sigma_s = rms_step(view, view.state_cv)
    h_lo, h_hi = float(st.get('q_lo', 1e-3)), float(st.get('q_hi', 0.95))
    n_a = float(ic.get('n_a', 3))
    p.update(cal_z=zk, cal_q=qk, cal_d=dk, lambda_a=view.lambda_a, lambda_b=view.lambda_b, state_cv=view.state_cv,
             ramp=float(st.get('ramp', 0.05)), scale_eps=float(st.get('scale_eps', 1e-3)),
             cv_min=view.cv[tr].min(0), cv_max=view.cv[tr].max(0), maha_mean=mean, maha_icov=icov, maha_thr=thr,
             ramp_kind='smin', smin_k=float(st.get('smin_k', np.log(2.0))), sigma_s=sigma_s, export_format=2)
    if front_s is not None:
        p['front_s'] = float(front_s)
        p['front_h'] = float(st.get('front_h', 0.5))
    # kappa (lambda per unit of s): the lowest planned level lies at least n_A RMS steps of s from A, the highest
    # at least n_A steps from B (LambdaModel floors kappa so that each cap is inactive at the opposite state)
    m0 = LambdaModel(dict(p, kappa_a=1.0, kappa_b=1.0))
    lam_lo, lam_hi = float(m0.scale(np.array([h_lo]))[0]), float(m0.scale(np.array([h_hi]))[0])
    p['kappa_a'] = (lam_lo - view.lambda_a) / (n_a * sigma_s)
    p['kappa_b'] = (view.lambda_b - lam_hi) / (n_a * sigma_s)
    if blend:
        # lambda_seed(s): monotone table of lambda against s on the in-distribution training frames
        m0 = LambdaModel(dict(p))
        idx = tr[::max(1, len(tr) // 400_000)]
        s = view.cv[idx, view.state_cv]
        lam = m0(view.cv[idx])
        Fi = sm.features(view.cv[idx]) - mean
        ok = np.einsum('ni,ij,nj->n', Fi, icov, Fi) <= thr
        sk, lk, dkk = _monotone_table(s[ok], lam[ok], view.w[idx][ok].astype(float), view.lambda_a, view.lambda_b)
        p.update(blend_s_knots=sk, blend_h_knots=lk, blend_d_knots=dkk, blend_thr=thr,
                 blend_width=float(st.get('blend_width_factor', 1.0)) * thr)
    if log:
        common.say(f'export: sigma_s {sigma_s:.4g}, kappa_A {p["kappa_a"]:.3g}, kappa_B {p["kappa_b"]:.3g}, '
                   f'maha thr {thr:.3g}' + (f', front s >= {front_s}' if front_s is not None else ''), log=log)
    return p


def _monotone_table(s, lam, w, lambda_a, lambda_b, n_bins=200, min_count=20):
    """Monotone (isotonic + PCHIP) table of lambda against s, strictly inside (lambda_a, lambda_b)."""
    from scipy.interpolate import PchipInterpolator
    from sklearn.isotonic import IsotonicRegression
    lo, hi = np.quantile(s, [0.0005, 0.9995])
    edges = np.r_[-np.inf, np.linspace(lo, hi, n_bins - 1), np.inf]
    bi = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, n_bins - 1)
    cnt, sw = np.bincount(bi, minlength=n_bins), np.bincount(bi, w, minlength=n_bins)
    ok = (cnt >= min_count) & (sw > 0)
    sb = np.bincount(bi, w * s, minlength=n_bins)[ok] / sw[ok]
    lb = np.bincount(bi, w * lam, minlength=n_bins)[ok] / sw[ok]
    eps = 1e-9 * (lambda_b - lambda_a)
    iso = IsotonicRegression(increasing=True, y_min=lambda_a + eps, y_max=lambda_b - eps).fit(sb, lb, sample_weight=sw[ok])
    li = iso.predict(sb)
    starts = np.r_[0, np.flatnonzero(np.diff(li) > 0) + 1]
    ends = np.r_[starts[1:], len(li)]
    sk = np.array([np.average(sb[a:b], weights=sw[ok][a:b]) for a, b in zip(starts, ends)])
    lk = li[starts]
    if len(sk) < 2:
        sk, lk = np.array([lambda_a, lambda_b]), np.array([lambda_a + eps, lambda_b - eps])
    keep = np.r_[True, np.diff(sk) > 0]
    sk, lk = sk[keep], lk[keep]
    # end knots at the states, so the seed lambda keeps rising beyond the data instead of holding its last value:
    # with a provisional front the out-of-distribution frames past it (loop_stag: the lower channel beyond
    # x = -0.68) otherwise all blended to one lambda (0.281), and no interface above it could be seeded
    if sk[0] > lambda_a and lk[0] > lambda_a + eps:
        sk, lk = np.r_[lambda_a, sk], np.r_[lambda_a + eps, lk]
    if sk[-1] < lambda_b and lk[-1] < lambda_b - eps:
        sk, lk = np.r_[sk, lambda_b], np.r_[lk, lambda_b - eps]
    return sk, lk, PchipInterpolator(sk, lk).derivative()(sk)
