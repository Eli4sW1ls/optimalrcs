"""Step 3: candidate models of the raw score z(x) ~ logit q_np. Common interface:

    m = Model(...); m.fit(ts); z = m.predict(cv); params = m.export()

`ts` is a TrainSet (features of the training frames, targets, weights, pairs). `export` returns the parameters
of lambda_runtime.ScoreModel; `predict` evaluates exactly that numpy forward pass.

- Seed:        z = the state CV itself (the current order parameter), for comparison.
- SparseLogit: M1, L1-penalized logistic regression of q_np (soft labels) on the standardized features.
- SplineGAM:   M1, additive uniform cubic B-splines per feature, weighted ridge regression on logit q_np.
- screen_gbt:  M2, gradient-boosted trees on logit q_np, permutation importances (feature screening only).
- MLP:         M3, SiLU network; loss = weighted MSE on logit q_np (mid-q frames up-weighted)
               + beta * sum over lags of the weighted MSE of increments (dz vs d logit q_np) along the pieces.
"""
import time

import numpy as np

import common
import data
from lambda_runtime import Features, ScoreModel, bspline_basis


class TrainSet:
    """Training arrays for one feature set: standardized features of the valid training frames, targets
    z = logit q_np, weights (path weight x mid_weight for 0.05 < q_np < 0.95), anchor subset, lag pairs."""

    def __init__(self, frames, cfg, specs, split='train', stats=None):
        t = cfg['train']
        self.feat = Features(frames.cv_names.tolist(), specs)
        sel = np.flatnonzero(frames.mask(split) & frames.valid)
        self.frame_idx = sel
        F = self.feat(frames.cv[sel])
        # standardization from the training frames only (pass the training TrainSet's stats for other splits)
        self.mu, self.sd = stats if stats is not None else data.standardization(F, frames.w[sel].astype(float))
        self.X = ((F - self.mu) / self.sd).astype(np.float32)
        q = frames.q[sel]
        self.z = frames.z[sel]
        self.q = q
        self.w_path = frames.w[sel].astype(float)
        self.w = self.w_path * np.where((q > 0.05) & (q < 0.95), t['mid_weight'], 1.0)
        # optional label-noise weight: logit q_np is noise where min(q, 1 - q) is small; weight min(1, p / noise_q)
        # (the variance of a logit estimate grows ~ 1/p). Off by default (noise_q = 0).
        if t.get('noise_q', 0) > 0:
            self.w = self.w * np.minimum(1.0, np.minimum(q, 1 - q) / t['noise_q'])
        anchors = np.arange(len(sel))[::t['anchor_stride']]
        max_a = cfg.get('m3', {}).get('max_anchors')
        if max_a and len(anchors) > max_a:
            anchors = anchors[np.unique(np.linspace(0, len(anchors) - 1, max_a).astype(int))]
        self.anchors = anchors
        # pairs (i, i + lag) inside one piece, both valid training frames; positions in `sel`
        pos = np.full(frames.n, -1)
        pos[sel] = np.arange(len(sel))
        self.pairs = {}
        for lag in t['lags']:
            i0, i1 = frames.pairs(frames.mask(split), lag, stride=t['anchor_stride'])
            self.pairs[lag] = (pos[i0], pos[i1])

    def subsample(self, n, seed=0):
        """Evenly spaced subset of at most n training frames (indices into the training arrays)."""
        if len(self.z) <= n:
            return np.arange(len(self.z))
        return np.unique(np.linspace(0, len(self.z) - 1, n).astype(int))

    def base_params(self, kind):
        return dict(kind=kind, mu=self.mu, sd=self.sd, **self.feat.to_dict())


class _Model:
    name = 'model'

    def predict(self, cv):
        return ScoreModel(self.params).raw(cv)

    def export(self):
        return dict(self.params)


class Seed(_Model):
    """z = the state CV (lambda = y today); calibrated like the others."""

    def __init__(self, frames, cfg):
        name = frames.cv_names[cfg['data']['state_cv']]
        self.name = f'seed ({name})'
        feat = Features(frames.cv_names.tolist(), [str(name)])
        self.params = dict(kind='feature', mu=np.zeros(1), sd=np.ones(1), **feat.to_dict())

    def fit(self, ts=None):
        return self


class SparseLogit(_Model):
    def __init__(self, C, max_frames, seed=0):
        self.C, self.max_frames, self.seed = C, max_frames, seed
        self.name = f'M1 L1-logistic C={C:g}'

    def fit(self, ts):
        from sklearn.linear_model import LogisticRegression
        s = ts.subsample(self.max_frames)
        X, q, w = ts.X[s].astype(float), ts.q[s], ts.w[s]
        Xd = np.vstack([X, X])
        yd = np.r_[np.ones(len(s)), np.zeros(len(s))]
        wd = np.r_[w * q, w * (1 - q)]
        clf = LogisticRegression(penalty='l1', C=self.C, solver='saga', max_iter=2000, tol=1e-5,
                                 random_state=self.seed)
        clf.fit(Xd, yd, sample_weight=wd / wd.mean())
        self.params = ts.base_params('linear')
        self.params.update(w=clf.coef_[0].astype(float), b=float(clf.intercept_[0]))
        self.n_nonzero = int(np.sum(clf.coef_[0] != 0))
        return self


class SplineGAM(_Model):
    def __init__(self, n_intervals, ridge, max_frames):
        self.n_int, self.ridge, self.max_frames = n_intervals, ridge, max_frames
        self.name = f'M1 spline GAM (P-spline) penalty={ridge:g}'

    def fit(self, ts):
        s = ts.subsample(self.max_frames)
        X, z, w = ts.X[s].astype(float), ts.z[s], ts.w[s]
        lo, hi = np.quantile(X, 0.001, axis=0), np.quantile(X, 0.999, axis=0)
        Bs = [bspline_basis(X[:, j], lo[j], hi[j], self.n_int) for j in range(X.shape[1])]
        B = np.hstack([np.ones((len(z), 1))] + Bs)
        # P-spline: minimize sum w (z - B c)^2 / sum w + ridge * sum over terms |D2 c_j|^2 (second differences:
        # shrinks each term towards a straight line), plus 1e-8 |c|^2 for the constant offsets of the terms
        nb = self.n_int + 3
        D = np.diff(np.eye(nb), n=2, axis=0)
        P = np.zeros((B.shape[1], B.shape[1]))
        for j in range(X.shape[1]):
            s0 = 1 + j * nb
            P[s0:s0 + nb, s0:s0 + nb] = self.ridge * D.T @ D + 1e-8 * np.eye(nb)
        BtW = B.T * (w / w.sum())
        c = np.linalg.solve(BtW @ B + P, BtW @ z)
        self.params = ts.base_params('gam')
        self.params.update(coef=c[1:].reshape(X.shape[1], nb), b=float(c[0]), spl_lo=lo, spl_hi=hi,
                           spl_n=self.n_int)
        return self


def screen_gbt(ts_train, ts_val, cfg, seed=0):
    """M2: gradient-boosted trees on logit q_np; permutation importance on validation frames. Not a lambda."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    c = cfg['m2']
    s = ts_train.subsample(c['max_frames'])
    gbt = HistGradientBoostingRegressor(max_iter=c['max_iter'], learning_rate=0.1, early_stopping=False,
                                        random_state=seed)
    t0 = time.time()
    gbt.fit(ts_train.X[s], ts_train.z[s], sample_weight=ts_train.w[s])
    v = ts_val.subsample(c['max_frames'] // 2)                       # ts_val uses the training standardization
    pi = permutation_importance(gbt, ts_val.X[v], ts_val.z[v], sample_weight=ts_val.w[v], n_repeats=5,
                                random_state=seed, scoring='neg_mean_squared_error')
    order = np.argsort(-pi.importances_mean)
    return dict(specs=[ts_train.feat.specs[i] for i in order], importance=pi.importances_mean[order],
                importance_sd=pi.importances_std[order], fit_seconds=time.time() - t0)


# ---------------------------------------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------------------------------------


def _net(n_in, arch):
    import torch.nn as nn
    layers, last = [], n_in
    for h in arch:
        layers += [nn.Linear(last, h), nn.SiLU()]
        last = h
    layers.append(nn.Linear(last, 1))
    return nn.Sequential(*layers)


class MLP(_Model):
    def __init__(self, arch, beta, cfg, seed=0, device=None):
        import torch
        self.arch, self.beta, self.seed = list(arch), float(beta), seed
        self.c = cfg['m3']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = f'M3 MLP {"x".join(map(str, arch))} beta={beta:g}'

    def fit(self, ts, on_checkpoint=None, score_sets=()):
        """Train for the fixed epoch budget; at each of `checkpoints` evenly spaced epochs export the network and
        call on_checkpoint(epoch, model_params, scores) (the caller evaluates the dynamical metrics there).
        `score_sets`: standardized feature arrays that the network scores at each checkpoint on the training device
        (float32); `scores` lists the results. That replaces the numpy forward pass, which dominated the sweep."""
        import torch
        torch.manual_seed(self.seed)
        rng = torch.Generator(device=self.device).manual_seed(self.seed)
        dev = self.device
        X = torch.as_tensor(ts.X, device=dev)
        z = torch.as_tensor(ts.z, dtype=torch.float32, device=dev)
        w = torch.as_tensor(ts.w, dtype=torch.float32, device=dev)
        anchors = torch.as_tensor(ts.anchors, device=dev)
        pairs = {lag: (torch.as_tensor(i0, device=dev), torch.as_tensor(i1, device=dev))
                 for lag, (i0, i1) in ts.pairs.items()}
        score_sets = [torch.as_tensor(np.asarray(S, np.float32), device=dev) for S in score_sets]
        net = _net(X.shape[1], self.arch).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.c['lr'], weight_decay=self.c['weight_decay'])
        batch, epochs = self.c['batch'], self.c['epochs']
        steps = int(np.ceil(len(anchors) / batch))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
        ckpt_epochs = sorted(set(np.round(np.linspace(epochs / self.c['checkpoints'], epochs,
                                                      self.c['checkpoints'])).astype(int).tolist()))
        self.history, self.checkpoints = [], []
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            perm = anchors[torch.randperm(len(anchors), generator=rng, device=dev)]
            tot = torch.zeros((), device=dev)
            tot_inc = torch.zeros((), device=dev)
            for s in range(steps):
                a = perm[s * batch:(s + 1) * batch]
                # one forward pass over the anchors and the frames of all lag pairs (fewer kernel launches)
                idx = [a]
                if self.beta > 0:
                    for lag, (i0, i1) in pairs.items():
                        j = torch.randint(len(i0), (batch,), generator=rng, device=dev)
                        idx += [i0[j], i1[j]]
                cat = torch.cat(idx)
                out = net(X[cat])[:, 0] - z[cat]
                na = len(a)
                wa = w[a]
                loss_pt = torch.sum(wa * self._point_loss(out[:na])) / wa.sum()
                loss = loss_pt
                if self.beta > 0:
                    inc = 0.0
                    for k in range(len(pairs)):
                        lo = na + 2 * k * batch
                        p0 = idx[1 + 2 * k]
                        wp = w[p0]
                        inc = inc + torch.sum(wp * (out[lo + batch:lo + 2 * batch] - out[lo:lo + batch]) ** 2) / wp.sum()
                    loss = loss + self.beta * inc
                    tot_inc += inc.detach()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss_pt.detach()
            self.history.append(dict(epoch=epoch, loss_point=float(tot) / steps, loss_inc=float(tot_inc) / steps,
                                     seconds=time.time() - t0))
            if epoch in ckpt_epochs:
                params = self._export_net(net, ts)
                self.checkpoints.append(dict(epoch=epoch, params=params))
                if on_checkpoint is not None:
                    on_checkpoint(epoch, params, [self._score(net, S) for S in score_sets])
        self.train_seconds = time.time() - t0
        self.params = self.checkpoints[-1]['params']
        self._net = net
        return self

    def _point_loss(self, r):
        """Squared residual of logit q; with m3.huber = delta > 0, the Huber loss (quadratic up to |r| = delta,
        linear beyond; scaled to equal r^2 in the quadratic part)."""
        delta = self.c.get('huber', 0)
        if delta > 0:
            import torch
            return 2 * torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction='none', delta=delta)
        return r ** 2

    @staticmethod
    def _score(net, S, chunk=1 << 20):
        import torch
        with torch.no_grad():
            return np.concatenate([net(S[i:i + chunk])[:, 0].double().cpu().numpy()
                                   for i in range(0, len(S), chunk)]) if len(S) else np.zeros(0)

    def _export_net(self, net, ts):
        params = ts.base_params('mlp')
        lin = [m for m in net if hasattr(m, 'weight')]
        params['n_layers'] = len(lin)
        for i, m in enumerate(lin):
            params[f'W{i}'] = m.weight.detach().cpu().double().numpy()
            params[f'b{i}'] = m.bias.detach().cpu().double().numpy()
        return params

    def use_checkpoint(self, k):
        self.params = self.checkpoints[k]['params']
        self.epoch = self.checkpoints[k]['epoch']
        return self
