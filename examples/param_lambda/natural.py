"""Natural coordinate of a learned committor, and staple path lengths predicted from the training run alone.

For overdamped dynamics the committor q is a martingale: along a trajectory it has no mean drift, only diffusion
with a q-dependent coefficient D(q) = <dq^2>/2 per frame. Here q is h, the exported model's calibrated committor
after the A/B ramps (lambda = scale(h); the interfaces are level sets of h). From the training run's own frames:

* D(h): lag-1 increments of h along the forward and mirrored pieces (h = 0 in A, 1 in B), path weights, in bins
  equally spaced in the log1p scale of the export (fine at small h, where increments scale with h).
* natural coordinate u(h) = int_0^h dh' / sqrt(2 D(h')), scaled to [0, 1]: the coordinate in which diffusion is
  uniform, so equal steps in u take about equal times. Below the data, D is continued as D ~ (h + eps)^2 (the
  log1p scale), above it as constant. Exported as a monotone table (LambdaModel scale_h / scale_u / scale_d).
* predicted staple lengths: a 1D Markov chain in h that resamples the observed increments of the current bin.
  For an ensemble [k*] (middle interface lambda_k), two independent walks start at h just past lambda_k (the
  observed first frames of upward crossings) and stop at a turn (the sampler's rule, pgmoves.turn_detected, as in
  evaluate.turn_replay), in A or in B; path length = sum of the two. A model of the path length, not the path
  ensemble: validate it against runs (validate()).

    python run.py natural z_br                    # D(h), exports <sys>_<fam>-nat.npz, figures, validation
"""
import json
import os
import re

import numpy as np

import common
from lambda_runtime import LambdaModel

N_BINS = 120


def frame_h(frames, model, idx):
    """h at the frames idx: blend(s, q) between the states, 0 in A, 1 in B."""
    cv = frames.cv[idx]
    s = cv[:, model.state_cv]
    h = np.where(frames.B[idx], 1.0, 0.0)
    inside = ~frames.A[idx] & ~frames.B[idx]
    if inside.any():
        h[inside] = model.blend(s[inside], model.q(cv[inside]))
    return h


class Increments:
    """Lag-1 pairs (h0, h1) of the training split, binned by h0 on the log1p scale of the export."""

    def __init__(self, frames, model, split='train', max_pairs=4_000_000, seed=0):
        sel = np.flatnonzero(frames.mask(split))
        i0 = sel[:-1][(frames.piece[sel[1:]] == frames.piece[sel[:-1]]) & (sel[1:] == sel[:-1] + 1)]
        i0 = i0[frames.valid[i0]]                                   # start between the states
        if len(i0) > max_pairs:
            i0 = np.sort(np.random.default_rng(seed).choice(i0, max_pairs, replace=False))
        both = np.unique(np.r_[i0, i0 + 1])
        hb = frame_h(frames, model, both)
        h0, h1 = hb[np.searchsorted(both, i0)], hb[np.searchsorted(both, i0 + 1)]
        self.eps, self.l1e = model.eps, np.log1p(1 / model.eps)
        self.edges = self.from_unit(np.linspace(0, 1, N_BINS + 1))
        b = np.clip(np.searchsorted(self.edges, h0, side='right') - 1, 0, N_BINS - 1)
        o = np.argsort(b, kind='stable')
        self.h0, self.dh, self.w, self.b = h0[o], (h1 - h0)[o], frames.w[i0][o].astype(float), b[o]
        self.start = np.searchsorted(self.b, np.arange(N_BINS + 1))
        n = np.diff(self.start)
        self.n = n
        # per-bin normalized cumulative weights, offset by the bin index (one global searchsorted samples any bin)
        cw = np.zeros(len(self.w))
        for k in range(N_BINS):
            s, e = self.start[k], self.start[k + 1]
            if e > s:
                c = np.cumsum(self.w[s:e]); cw[s:e] = k + c / c[-1]
        self.cw = cw
        D = np.full(N_BINS, np.nan)
        for k in np.flatnonzero(n >= 50):
            s, e = self.start[k], self.start[k + 1]
            D[k] = np.sum(self.w[s:e] * self.dh[s:e] ** 2) / (2 * np.sum(self.w[s:e]))
        self.D_raw = D
        self.D = self._fill(D)

    def to_unit(self, h):
        return np.log1p(np.asarray(h, float) / self.eps) / self.l1e

    def from_unit(self, u):
        return self.eps * np.expm1(np.asarray(u, float) * self.l1e)

    def _fill(self, D):
        """Bins without data: D ~ (h + eps)^2 below the data (matched at the first bin with data), constant above;
        gaps inside by log-linear interpolation; then a light smoothing of log D."""
        hc = 0.5 * (self.edges[:-1] + self.edges[1:])
        ok = np.isfinite(D) & (D > 0)
        if not ok.any():
            raise ValueError('no increments')
        k0, k1 = np.flatnonzero(ok)[[0, -1]]
        logD = np.interp(np.arange(N_BINS), np.flatnonzero(ok), np.log(D[ok]))
        logD[:k0] = np.log(D[k0]) + 2 * np.log((hc[:k0] + self.eps) / (hc[k0] + self.eps))
        logD[k1 + 1:] = np.log(D[k1])
        kern = np.array([1, 2, 3, 2, 1], float); kern /= kern.sum()
        sm = np.convolve(np.pad(logD, 2, mode='edge'), kern, mode='valid')
        return np.exp(sm)

    def natural_table(self, n_knots=241):
        """(h knots, u knots, slopes) of u(h) = int dh / sqrt(2 D) on [0, 1], monotone PCHIP data."""
        from scipy.interpolate import PchipInterpolator
        # integrate the piecewise-constant density per bin exactly, then interpolate at knots spaced in the bin scale
        rho = 1 / np.sqrt(2 * self.D)
        U = np.r_[0, np.cumsum(rho * np.diff(self.edges))]
        U /= U[-1]
        hk = self.from_unit(np.linspace(0, 1, n_knots))
        uk = np.interp(hk, self.edges, U)
        p = PchipInterpolator(hk, uk)
        return hk, uk, p.derivative()(hk)

    def sample(self, h, rng):
        """One increment for each walker at h (resampled from its bin; empty bins borrow the nearest bin with data)."""
        b = np.clip(np.searchsorted(self.edges, h, side='right') - 1, 0, N_BINS - 1)
        have = np.flatnonzero(self.n > 0)
        b = have[np.clip(np.searchsorted(have, b), 0, len(have) - 1)]
        j = np.searchsorted(self.cw, b + rng.random(len(h)) * (1 - 1e-12), side='right')
        return self.dh[np.clip(j, 0, len(self.dh) - 1)]

    def crossing_starts(self, level):
        """First frames past `level` of observed upward crossings (h1 >= level > h0), weighted."""
        m = (self.h0 < level) & (self.h0 + self.dh >= level)
        return self.h0[m] + self.dh[m], self.w[m]


def predict_lengths(inc, interfaces_h, n_walk=2000, max_steps=200_000, seed=0):
    """Predicted staple path lengths per ensemble [k*] (k = 1 .. n-2, middle interface interfaces_h[k]):
    dict k -> array of lengths (sum of two independent excursions until a turn or a state)."""
    rng = np.random.default_rng(seed)
    lv = np.sort(np.asarray(interfaces_h, float))
    out = {}
    for k in range(1, len(lv) - 1):
        hs, ws = inc.crossing_starts(lv[k])
        if len(hs) == 0:
            out[k] = np.full(n_walk, np.nan)
            continue
        start = hs[rng.choice(len(hs), 2 * n_walk, p=ws / ws.sum())]
        out[k] = _excursions(inc, lv, start, max_steps, rng).reshape(2, n_walk).sum(0)
    return out


def _excursions(inc, lv, h, max_steps, rng):
    """Steps until the turn rule fires (or A / B) for walkers starting at h."""
    h = h.copy()
    mx, mn = h.copy(), h.copy()
    t = np.zeros(len(h), int)
    alive = np.ones(len(h), bool)
    n_lv = len(lv)
    for step in range(1, max_steps + 1):
        a = np.flatnonzero(alive)
        if not len(a):
            break
        x = h[a] + inc.sample(h[a], rng)
        h[a] = x
        mx[a] = np.maximum(mx[a], x); mn[a] = np.minimum(mn[a], x)
        k_up = np.searchsorted(lv, mx[a], side='right')
        j_dn = np.searchsorted(lv, mn[a], side='left')
        turn = ((k_up >= 2) & (x < lv[np.clip(k_up - 2, 0, n_lv - 1)])) | \
               ((j_dn + 1 < n_lv) & (x > lv[np.clip(j_dn + 1, 0, n_lv - 1)]))
        done = turn | (x <= 0) | (x >= 1)
        t[a[done]] = step
        alive[a[done]] = False
    t[alive] = max_steps
    return t


# ---------------------------------------------------------------------------------------------------
# validation against staple runs
# ---------------------------------------------------------------------------------------------------
_HEAD = re.compile(r'# Cycle: (\d+), status: (\w+).*path length: (\d+)')


def observed_lengths(run, skip=0.2):
    """Median and 90th percentile of the accepted path lengths per ensemble directory (k = dir index - 1 for
    the [k*] ensembles), after the first `skip` fraction of the cycles; and the interfaces from logging.log."""
    with open(os.path.join(run, 'logging.log')) as f:
        m = re.search(r'interfaces = \[([^\]]*)\]', f.read())
    intfs = [float(v) for v in m.group(1).split(',')]
    out = {}
    for d in sorted(x for x in os.listdir(run) if x.isdigit() and int(x) >= 2):
        rows = []
        for line in open(os.path.join(run, d, 'order.txt')):
            if line.startswith('# Cycle'):
                g = _HEAD.match(line)
                if g:
                    rows.append((int(g.group(1)), g.group(2) == 'ACC', int(g.group(3))))
        c = np.array(rows)
        if not len(c):
            continue
        c = c[(c[:, 0] >= skip * c[:, 0].max()) & (c[:, 1] == 1) & (c[:, 2] > 1)]
        if len(c):
            out[int(d) - 1] = (float(np.median(c[:, 2])), float(np.percentile(c[:, 2], 90)), len(c))
    return intfs, out


def lambda_to_h(model, lam):
    """Interface values in lambda units -> h (inverse of the export's scale)."""
    return model.h_of_lambda(lam)


# ---------------------------------------------------------------------------------------------------
# run.py natural <system>
# ---------------------------------------------------------------------------------------------------
FAMILY = {'z_br': 'm3-fast', 'cc_br': 'm3'}              # the models the staple test runs used
MAX_RATIO = 10.0


def _placements(system, path, m, m_nat):
    """name -> (model, interfaces in that model's lambda) for the maps and the predictions."""
    import staple_run as sr
    out = {'seed (y)': (None, sr.SEED_INTERFACES[system]),
           'lam: equal in log1p scale': (m, sr.make_interfaces(system, path, m, 11)),
           'y: equal median y': (m, sr.make_interfaces(system, path, m, 11, placement='y')),
           'max(y, lam)': (m, sr.make_interfaces(system, path, m, 11, placement='max'))}
    nat = sr.make_interfaces(system, None, m_nat, 11)
    out[f'natural, ratio <= {MAX_RATIO:g}'] = (m_nat, sr.cap_ratio(m_nat, nat, MAX_RATIO))
    return out


def _runs_of(system, fam):
    """toytis runs of this exported model (logging.log names the file), with their placement tag."""
    root = os.path.join(common.TOYTIS, 'simulations')
    out = []
    for d in sorted(os.listdir(root)):
        log = os.path.join(root, d, 'logging.log')
        if 'lam-' not in d or not os.path.isfile(log):
            continue
        head = open(log).read(20000)
        if f'{system}_{fam}.npz' in head or f'{system}_{fam}-nat.npz' in head:
            out.append(os.path.join(root, d))
    return out


def cmd_natural(cfg, args):
    import data
    import steps
    system = cfg['name']
    fam = FAMILY.get(system, 'm3')
    out = steps._dir(cfg, 'natural')
    f = data.Frames(cfg)
    path = os.path.join(common.HERE, 'exported', f'{system}_{fam}.npz')
    m = LambdaModel(path)
    inc = Increments(f, m)
    hk, uk, dk = inc.natural_table()
    params = dict(m.params, scale_h=hk, scale_u=uk, scale_d=dk)
    nat_path = os.path.join(common.HERE, 'exported', f'{system}_{fam}-nat.npz')
    np.savez(nat_path, **params)
    m_nat = LambdaModel(nat_path)
    hc = 0.5 * (inc.edges[1:] + inc.edges[:-1])
    res = dict(family=fam, h=hc, n=inc.n, D=inc.D, D_raw=inc.D_raw, export=os.path.relpath(nat_path, common.REPO))
    common.say(f'{system}: natural-coordinate export {nat_path}')

    # predicted path lengths per placement, and the observed ones of the runs made so far
    pl = _placements(system, path, m, m_nat)
    res['placements'] = {}
    for name, (mm, intfs) in pl.items():
        if mm is None:
            continue
        pred = predict_lengths(inc, mm.h_of_lambda(intfs))
        res['placements'][name] = dict(interfaces=intfs, q=list(mm.h_of_lambda(intfs[1:-1])),
                                       median=[float(np.median(pred[k])) for k in sorted(pred)],
                                       p90=[float(np.percentile(pred[k], 90)) for k in sorted(pred)])
        common.say(f'  {name:32s} q {np.round(mm.h_of_lambda(intfs[1:-1]), 4)}\n'
                   f'  {"":32s} predicted median {np.round(res["placements"][name]["median"]).astype(int)}')
    import staple_run as sr
    res['n_scan'] = {}
    for n in (11, 13, 15, 19):                     # natural placement + ratio cap: the longest predicted median
        intfs = sr.cap_ratio(m_nat, sr.make_interfaces(system, None, m_nat, n), MAX_RATIO)
        pred = predict_lengths(inc, m_nat.h_of_lambda(intfs), n_walk=1000)
        med = [float(np.median(pred[k])) for k in sorted(pred)]
        res['n_scan'][n] = dict(n_interfaces=len(intfs), longest_median=max(med), sum_median=sum(med))
        common.say(f'  natural, --n-interfaces {n} -> {len(intfs)} interfaces: longest predicted median {max(med):.0f} '
                   f'frames (observed tends to be ~1.3x), sum over ensembles {sum(med):.0f}')
    res['runs'] = {}
    for run in _runs_of(system, fam):
        intfs, obs = observed_lengths(run)
        mm = m_nat if f'{system}_{fam}-nat.npz' in open(os.path.join(run, 'logging.log')).read(20000) else m
        pred = predict_lengths(inc, mm.h_of_lambda(intfs))
        ks = sorted(set(pred) & set(obs))
        res['runs'][os.path.basename(run)] = dict(
            interfaces=intfs, k=ks, obs_median=[obs[k][0] for k in ks], obs_p90=[obs[k][1] for k in ks],
            n_paths=[obs[k][2] for k in ks], pred_median=[float(np.median(pred[k])) for k in ks],
            pred_p90=[float(np.percentile(pred[k], 90)) for k in ks])
        common.say(f'  run {os.path.basename(run)}: observed / predicted median per ensemble\n'
                   + '\n'.join(f'     k {k:2d}  h {mm.h_of_lambda(intfs[k]):.2e}  {obs[k][0]:7.0f} / {np.median(pred[k]):7.0f}'
                               for k in ks))
    with open(os.path.join(out, 'natural.json'), 'w') as fh:
        json.dump(steps._jsonable(res), fh, indent=1)
    fig_natural(cfg, inc, m, m_nat, res, out)
    fig_interfaces(cfg, f, pl, out)


def _plt():
    import report
    return report._plt()


def fig_natural(cfg, inc, m, m_nat, res, out):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    hc = 0.5 * (inc.edges[1:] + inc.edges[:-1])
    ok = np.isfinite(inc.D_raw)
    ax[0].loglog(hc[ok], inc.D_raw[ok], 'o', ms=3, color='#52514e', label='from the training run (bins with >= 50 pairs)')
    ax[0].loglog(hc, inc.D, '-', color='#2a78d6', label='used (filled, smoothed)')
    ax[0].set(xlabel='h (committor of the export)', ylabel='D(h) = <dh^2>/2 per frame', title=f'{cfg["name"]}: diffusion along q')
    ax[0].legend(fontsize=8)
    hh = np.r_[0, np.logspace(-6, 0, 400)]
    span = m.lambda_b - m.lambda_a
    ax[1].semilogx(hh[1:], (m.scale(hh[1:]) - m.lambda_a) / span, color='#eda100', label='log1p scale (eps 1e-3)')
    ax[1].semilogx(hh[1:], (m_nat.scale(hh[1:]) - m.lambda_a) / span, color='#2a78d6', label='natural coordinate')
    ax[1].set(xlabel='h', ylabel='(lambda - lambda_A) / (lambda_B - lambda_A)', title='output scales')
    ax[1].legend(fontsize=8)
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#52514e']
    for c, (name, r) in zip(colors, res['runs'].items()):
        ax[2].loglog(r['obs_median'], r['pred_median'], 'o', color=c, label=f'{name} (median)')
        ax[2].loglog(r['obs_p90'], r['pred_p90'], '^', color=c, mfc='none', label='  90th percentile')
    lim = [10, 1e5]
    ax[2].plot(lim, lim, 'k--', lw=1)
    ax[2].set(xlabel='observed path length (frames)', ylabel='predicted from the training run',
              title='staple path length per ensemble', xlim=lim, ylim=lim)
    ax[2].legend(fontsize=6, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'natural.png'), dpi=130)
    plt.close(fig)


def fig_interfaces(cfg, f, pl, out):
    """Potential (contours in kT) with the interfaces of each placement (lambda isolines; y = const for the seed)."""
    import toy2d
    plt = _plt()
    d = cfg['data']
    settings = toy2d.read_full_settings(d['tis_dir'])
    pot = toy2d.potential_from_settings(settings, d.get('potential'), d.get('potential_kwargs'))
    la, lb = float(f.lambda_a), float(f.lambda_b)
    x = f.cv[np.flatnonzero(f.valid)[::50], 1]
    g1 = np.linspace(la - 0.06, lb + 0.06, 240)
    g2 = np.linspace(*np.quantile(x, [0.001, 0.999]), 120)
    V = toy2d.potential_grid(pot, g1, g2) / settings['temperature']
    V = V - np.nanmin(V)
    Y, X = np.meshgrid(g1, g2, indexing='ij')
    cv = np.c_[Y.ravel(), X.ravel()]
    fig, axes = plt.subplots(len(pl), 1, figsize=(9, 2.6 * len(pl)), sharex=True)
    for ax, (name, (mm, intfs)) in zip(axes, pl.items()):
        ax.grid(False)
        ax.contourf(g1, g2, np.clip(V, 0, 14).T, levels=np.arange(0, 14.5, 1), cmap='Greys', alpha=0.55)
        ax.contour(g1, g2, V.T, levels=np.arange(1, 14, 1), colors='0.55', linewidths=0.4)
        for v in (la, lb):
            ax.axvline(v, color='k', lw=1.4)
        if mm is None:
            for v in intfs[1:-1]:
                ax.axvline(v, color='#2a78d6', lw=1.2)
            label = 'y = ' + ', '.join(f'{v:g}' for v in intfs[1:-1])
        else:
            L = mm(cv).reshape(Y.shape)
            ax.contour(g1, g2, L.T, levels=intfs[1:-1], colors='#2a78d6', linewidths=1.2)
            label = 'q = ' + ', '.join(f'{q:.2g}' for q in mm.h_of_lambda(intfs[1:-1]))
        ax.set(ylabel='x', title=f'{name}:  {label}')
        ax.title.set_fontsize(8)
    axes[-1].set_xlabel('y (state CV; A left of the first black line, B right of the second)')
    fig.suptitle(f'{cfg["name"]}: potential (grey, 1 kT contours) and interfaces (blue)', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'interfaces.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
