"""A loop directory from a converged staple data set with its NPNEq fit: student, interfaces, gated freeze, production.

    python -m loop.staple_fit loop_stag_staple fit          pool from the fit npz ([staple_fit] npz), student, plan,
                                                            placement; state.json; figures (interfaces.png, maps.png)
    python -m loop.staple_fit loop_stag_staple plot         figures/interfaces_plan.png (before the runs)
    python -m loop.staple_fit loop_stag_staple production   seed candidates from the data's whole paths, the gated
                                                            freeze (verification runs) and the production run

The npz (written by the staple notebooks' NPNEq fit) holds per frame: X_all (the order parameter, the transverse
coordinate, ...), i_traj (piece), group_traj (path), t_traj, path_weights, boundary0/1, q_fit (the NPNEq committor:
the teacher's labels here, no refit) and q_exact. It replaces pass 0 + the teacher of the loop:

- pool pass 0: the pieces as they are (forward pieces from the first frame past M, mirrored pieces = the start of the
  path read backwards), the frame's recorded order parameter = the state CV (the run's lambda), weights =
  path_weights, labels = q_fit. Ensembles and MC cycles are not in the npz: every piece gets ensemble 0 and its path
  a random row (seeded), so the validation / test blocks are random paths;
- whole paths for the gate: reversed mirrored piece + forward piece of each path, in random order (path_arrays takes
  the first 5000);
- seed candidates: `seed_paths` whole paths, half with their channel section in each channel (`channel_of`), written
  as infretis load paths through artificial_paths.write (positions only; they are real paths of the data, the
  "artificial" store is the loop's slot for externally made seed paths);
- the interfaces of the staple run (pass0.json, for the figure): estimated from the first frames of the forward pieces.

Then the loop's own steps: student (passes.learn), plan_natural, place on the whole paths, and passes.freeze with a
forced verification (the gate on a run's own paths), then controller.run_production.
"""
import argparse
import json
import os
import sys

import numpy as np

import common
from lambda_runtime import LambdaModel

from loop import controller, manifest, pool as pool_mod


def _cfg(name):
    cfg = controller.load_loop_config(name)
    sf = cfg['staple_fit']
    if not os.path.isabs(sf['npz']):
        sf['npz'] = os.path.join(common.REPO, sf['npz'])
    return cfg


def channel_of(xy, lo=-1.0, hi=1.0):
    """+1 / -1: sign of the mean transverse coordinate over the frames in the channel section (lo < x < hi); 0 if
    the path never enters it."""
    m = (xy[:, 0] > lo) & (xy[:, 0] < hi)
    return int(np.sign(xy[m, 1].mean())) if m.any() else 0


def estimate_interfaces(x_first, la, lb, bin_width=0.005, min_ratio=10.0):
    """Interfaces of the run from the first frames of its forward pieces: each lies just past its ensemble's M, so
    their density has a sharp peak right above every interface. Peaks = local maxima above min_ratio x the median
    nonzero bin count; the interface is the left edge of the peak bin (accurate to bin_width)."""
    h, e = np.histogram(x_first, bins=np.arange(la, lb + bin_width, bin_width))
    med = np.median(h[h > 0]) if (h > 0).any() else 0
    pk = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] > h[i + 1] and h[i] > min_ratio * med]
    return [float(la)] + [round(float(e[i]), 3) for i in pk] + [float(lb)]


def fig_plan(cfg, path=None):
    """One large panel: the potential, the placed interfaces (lambda isosurfaces, labelled with their committor
    level) and the exact isocommittors at the same levels; the staple run's interfaces (x = const) for comparison."""
    from loop import plots
    plt = plots.plt
    pool = pool_mod.Pool(cfg['loop_dir'])
    st = controller.State(cfg).data
    rec = st['passes'][-1]
    m = LambdaModel(rec['model'])
    grid = plots.Grid(cfg, pool)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    grid.background(ax)
    inner = rec['interfaces'][1:-1]
    L = grid.field(m(grid.cv))                                # masked where V > vmax_kT (never visited)
    cs = ax.contour(grid.ys, grid.xs, L.T, levels=inner, colors=plots.BLUE, linewidths=1.6)
    h = m.h_of_lambda(np.array(inner))
    ax.clabel(cs, fmt={lv: f'{q:.2g}' for lv, q in zip(inner, h)}, fontsize=8, inline=True)
    ax.contour(grid.ys, grid.xs, grid.field(grid.q_exact).T, levels=np.sort(np.unique(np.clip(h, 1e-6, 1 - 1e-6))),
               colors=plots.RED, linewidths=1.0, linestyles='--')
    with open(os.path.join(cfg['loop_dir'], 'pass_0', 'pass0.json')) as f:
        old = json.load(f)['final_interfaces']
    for v in old[1:-1]:
        ax.axvline(v, color='#52514e', lw=0.8, ls=':')
    plots._mark_unvisited(ax, grid)
    s = cfg['system_spec']
    ax.set_xlabel(f'{s["cv_names"][0]} (state CV; A left of the first black line, B right of the second)')
    ax.set_ylabel(s['cv_names'][1])
    ax.set_title(f'{cfg["name"]}: {len(rec["interfaces"])} interfaces placed on the staple data\n'
                 f'blue: lambda isosurfaces, labelled with their committor level; red dashed: exact isocommittors at the '
                 f'same levels; dotted: the staple run\'s own interfaces (x); hatched: V > 12 kT, never visited',
                 loc='left', fontsize=9.5)
    path = path or os.path.join(cfg['loop_dir'], 'figures', 'interfaces_plan.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def build_pool(cfg, log=None):
    sf, system = cfg['staple_fit'], cfg['system_spec']
    pool = pool_mod.Pool(cfg['loop_dir'])
    if pool.has(0):
        common.say('pool pass 0 exists', log=log)
        return pool
    z = np.load(sf['npz'])
    X = z['X_all'][:, :2].astype(float)                      # the system's CVs, in cv_names order (checked: q_exact)
    piece, path, t, w = z['i_traj'], z['group_traj'], z['t_traj'], z['path_weights'].astype(float)
    starts = np.r_[0, np.flatnonzero(np.diff(piece)) + 1]
    ends = np.r_[starts[1:], len(piece)]
    n = ends - starts
    p_path = path[starts]
    mirrored = t[starts] == 0                                 # a forward piece starts at t* >= 1
    rng = np.random.default_rng(int(cfg.get('seed', 0)))
    upath, inv = np.unique(p_path, return_inverse=True)
    row_of_path = rng.permutation(len(upath))
    lam = X[:, system['state_cv']]
    pieces = dict(cv=np.column_stack([lam, X]), t=t.astype(np.int64), piece=piece, piece_n=n,
                  piece_w=w[starts], piece_pnr=p_path, piece_ens=np.zeros(len(n), np.int32),
                  piece_row=row_of_path[inv].astype(np.int32), piece_mirrored=mirrored,
                  piece_core=np.zeros(len(n), bool), piece_tainted=np.zeros(len(n), bool),
                  weighting=str(z['weights_used']), reversed_pieces=True, q_exact=z['q_exact'])
    # whole paths: reversed mirrored piece + forward piece, in random order
    ops, pnr, W = [], [], []
    by_path = {}
    for i in range(len(n)):
        by_path.setdefault(int(p_path[i]), []).append(i)
    for g in rng.permutation(upath):
        idx = by_path[int(g)]
        fw = [i for i in idx if not mirrored[i]]
        mi = [i for i in idx if mirrored[i]]
        seg = [X[starts[i]:ends[i]][::-1] for i in mi] + [X[starts[i]:ends[i]] for i in fw]
        P = np.concatenate(seg)
        ops.append(np.column_stack([P[:, system['state_cv']], P]).astype(np.float32))
        pnr.append(int(g))
        W.append([w[starts[idx[0]]]])
    raw = dict(ops=ops, pnr=pnr, rows=list(row_of_path[np.searchsorted(upath, pnr)]), W=W, ptype=[''] * len(ops),
               gen_ens=[0] * len(ops), tainted=[False] * len(ops), move=[''] * len(ops))
    intf = estimate_interfaces(X[starts[~mirrored], system['state_cv']], system['lambda_a'], system['lambda_b'])
    meta = dict(source=sf['npz'], lambda_version='seed', lambda_a=system['lambda_a'], lambda_b=system['lambda_b'],
                cv_names=system['cv_names'], interfaces=intf, mode='staple', md_steps=0, n_rows=len(upath),
                workdir=os.path.dirname(sf['npz']), note='converged staple data with its NPNEq fit (staple_fit.py)')
    pool.store(0, pieces, raw, meta)
    pool.set_labels(0, z['q_fit'].astype(float))
    p0dir = os.path.join(cfg['loop_dir'], 'pass_0')
    os.makedirs(p0dir, exist_ok=True)
    manifest.write(os.path.join(p0dir, 'pass0.json'), iterations=[dict(interfaces=intf)], reached_b=True,
                   saved_frames=int(len(piece)), final_interfaces=intf, note='staple data (staple_fit.py)')
    common.say(f'pool pass 0: {len(n)} pieces, {len(ops)} whole paths, {len(piece)} frames; run interfaces '
               f'(estimated) {intf}', log=log)
    return pool


def cmd_fit(cfg, log):
    from loop import passes, diagnostics, interfaces as itf
    pool = build_pool(cfg, log)
    state = controller.State(cfg)
    if state.data['passes']:
        common.say('state.json already has a pass: delete it to refit', log=log)
        return state
    model_path, params, sinfo, view = passes.learn(cfg, pool, 1, None, log)
    model = LambdaModel(model_path)
    levels = np.exp(np.linspace(np.log(1e-3), np.log(0.95), 10))
    ex = diagnostics.exact_metrics(model, view, levels)
    plan = itf.plan_natural(model, view, cfg['interfaces'], log=log)
    paths = itf.path_arrays(pool, model, state_cv=cfg['system_spec']['state_cv'], lineage='include')
    placed = itf.place(model, paths, plan['lam'], cfg['interfaces'], cfg['run'].get('mode', 'staple'), log=log)
    rec = dict(k=1, model=model_path, model_md5=manifest.md5(model_path), interfaces=placed['interfaces'],
               plan=plan, removed=placed['removed'], refilled=placed.get('refilled', []),
               placement_check=passes._slim(placed['check']),
               recheck=dict(ok=False, fails=[['no sampler run with this lambda yet']], unverified=[]),
               student={k: v for k, v in sinfo.items() if k != 'checkpoints'},
               diagnostics=dict(exact=ex, new=dict(zq=diagnostics.zq_profile(model, view, view.valid),
                                                    ood=diagnostics.ood_by_ensemble(model, view, view.valid))),
               cost=dict(md_steps=0), source='staple_fit', pool_pass=0)
    state.data['passes'] = [rec]
    state.data['pass0'] = dict(workdir=os.path.join(cfg['loop_dir'], 'pass_0'), pool_passes=[0], reached_b=True,
                               md_steps=0)
    state.save()
    common.say(f'fit: {len(placed["interfaces"])} interfaces {np.round(placed["interfaces"], 4).tolist()}; removed '
               f'{placed["removed"]}; refilled {placed.get("refilled", [])}; placement check ok: {placed["check"]["ok"]}'
               + (f'; exact: spread {ex["spread_mean"]:.4f}, logit RMSE {ex["logit_rmse"]:.3f}' if ex else ''), log=log)
    return state


def zero_minus_path(system, seed=0, max_len=200_000, y0=0.0):
    """A [0-] seed: overdamped dynamics of the system (its kT, gamma, dt) from just outside A (s = lambda_A + 0.01
    sigma_s), until a trajectory whose first step enters A leaves A again; frames (x, y). Staple data hold no [0-]
    paths (the loader skips ensemble 0)."""
    from loop import artificial_paths
    force = artificial_paths._force(system)
    rng = np.random.default_rng(seed)
    kT, gamma, dt, la = system['kT'], system['gamma'], system['dt'], system['lambda_a']
    sig = np.sqrt(2 * kT * dt / gamma)
    for _ in range(10_000):
        P = [np.array([la + 0.01 * sig, y0])]
        while len(P) < max_len:
            x, y = P[-1]
            fx, fy = force(np.array([x]), np.array([y]))
            P.append(np.array([x + fx[0] * dt / gamma + sig * rng.standard_normal(),
                               y + fy[0] * dt / gamma + sig * rng.standard_normal()]))
            if len(P) == 2 and P[1][0] >= la:
                break                                         # must enter A on the first step
            if len(P) > 2 and P[-1][0] > la:
                return np.array(P)
    raise RuntimeError('no [0-] path')


def find_seed_paths(cfg, pool, model, interfaces, mode, k_per=3, log=None):
    """Data paths that yield a valid seed under the new lambda and interfaces, by the seeder's own rules
    (seed.staple_candidates + validate_staple, i.e. infretis' StaplePath checks): up to k_per per ensemble and
    channel. Random staple paths of the old run end at the old turns and often hold no complete turn of the new
    interface set, so a random subset can miss an ensemble ([6*] did with 400 paths)."""
    from loop import seed as seed_mod
    z, _ = pool.load(0)
    starts = np.r_[0, np.cumsum(z['raw_n'])[:-1]]
    found = {(e, c): [] for e in range(1, len(interfaces)) for c in (1, -1)}
    for i in range(len(z['raw_n'])):                          # random order (build_pool shuffled the paths)
        cv = z['raw_cv'][starts[i]:starts[i] + z['raw_n'][i], 1:3].astype(float)
        c = channel_of(cv)
        if c == 0:
            continue
        need = [e for e in range(1, len(interfaces)) if len(found[(e, c)]) < k_per]
        if not need:
            if all(len(v) >= k_per for v in found.values()):
                break
            continue
        lam = model(cv)
        for e in need:
            for s_, t in seed_mod.staple_candidates(lam, interfaces, e):
                op = np.column_stack([lam[s_:t + 1], cv[s_:t + 1]])
                if seed_mod.validate_staple(op, interfaces, e)[0]:
                    found[(e, c)].append(i)
                    break
    rows = {e: (len(found[(e, 1)]), len(found[(e, -1)])) for e in range(1, len(interfaces))}
    common.say(f'seed search: valid data paths per ensemble (upper, lower channel, up to {k_per} each): {rows}', log=log)
    return found, z, starts


def write_seed_candidates(cfg, pool, log):
    from loop import artificial_paths
    root = os.path.join(cfg['loop_dir'], 'artificial')
    if os.path.isdir(root):
        return root
    st = controller.State(cfg).data
    rec = st['passes'][-1]
    model = LambdaModel(rec['model'])
    found, z, starts = find_seed_paths(cfg, pool, model, rec['interfaces'], cfg['run'].get('mode', 'staple'),
                                       k_per=int(cfg['staple_fit'].get('seeds_per_ensemble', 3)), log=log)
    order = []                                                # per ensemble, alternating the channel it starts with
    for e in range(1, len(rec['interfaces'])):
        a, b = (found[(e, 1)], found[(e, -1)]) if e % 2 else (found[(e, -1)], found[(e, 1)])
        for i in [x for pair in zip(a, b) for x in pair] + a[len(b):] + b[len(a):]:
            if i not in order:
                order.append(i)
    paths = [zero_minus_path(cfg['system_spec'], seed=int(cfg.get('seed', 0)))]      # ensemble [0-]
    paths += [z['raw_cv'][starts[i]:starts[i] + z['raw_n'][i], 1:3].astype(float) for i in order]
    missing = [e for e in range(1, len(rec['interfaces'])) if not found[(e, 1)] and not found[(e, -1)]]
    note = (f'one [0-] path (overdamped dynamics inside A) and {len(paths) - 1} whole paths of the converged staple '
            f'data ({cfg["staple_fit"]["npz"]}) that are valid seeds under the new lambda, both channels per ensemble '
            f'where the data have one; real dynamics, stored in the artificial-seed slot'
            + (f'; NO valid data path for ensembles {missing}' if missing else ''))
    artificial_paths.write(cfg['loop_dir'], cfg['system_spec'], paths, note)
    common.say(f'seed candidates: {note}', log=log)
    return root


def cmd_production(cfg, log):
    from loop import passes
    pool = pool_mod.Pool(cfg['loop_dir'])
    state = controller.State(cfg)
    if not state.data['passes']:
        raise SystemExit('run `fit` first')
    write_seed_candidates(cfg, pool, log)
    if not os.path.isdir(os.path.join(cfg['loop_dir'], 'production')):
        out = passes.freeze(cfg, state, pool, log)
        if out is None:
            return
    controller.run_production(cfg, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config')
    ap.add_argument('stage', choices=['fit', 'plot', 'production'])
    a = ap.parse_args()
    cfg = _cfg(a.config)
    log = os.path.join(cfg['loop_dir'], 'loop.log')
    common.say(f'== staple_fit {cfg["name"]} {a.stage} in {cfg["loop_dir"]}', log=log)
    if a.stage == 'plot':
        print(fig_plan(cfg))
        return
    {'fit': cmd_fit, 'production': cmd_production}[a.stage](cfg, log)


if __name__ == '__main__':
    main()
