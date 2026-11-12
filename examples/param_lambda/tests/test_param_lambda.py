"""Tests of the parametric order parameter: export = trained model, continuity, A/B handling, path reversal,
the toytis adapter, and the loader (frames identical to tis_data.prepare_data).

    cd examples/param_lambda && python -m pytest tests -q            (add -m "not slow" to skip the toytis run)

Tests that need the frame caches or exported models of run.py skip when those are missing.
"""
import glob
import os
import pickle
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
import common  # noqa: E402
from lambda_runtime import Features, LambdaModel, ScoreModel, bspline_basis  # noqa: E402

EXPORTED = sorted(glob.glob(os.path.join(PKG, 'exported', '*.npz')))
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------------------------------
# synthetic models of every kind (no data needed)
# ---------------------------------------------------------------------------------------------------
def _synthetic(kind, lam_a=0.05, lam_b=0.85):
    feat = Features(['y', 'x'], ['y', 'x', 'u', 'v^2'])
    p = dict(kind=kind, mu=rng.normal(size=4) * 0.1, sd=rng.uniform(0.5, 2, 4), **feat.to_dict())
    if kind == 'linear':
        p.update(w=rng.normal(size=4), b=0.3)
    elif kind == 'gam':
        p.update(coef=rng.normal(size=(4, 15)), b=-0.2, spl_lo=-np.ones(4) * 2, spl_hi=np.ones(4) * 2, spl_n=12)
    elif kind == 'mlp':
        sizes = [4, 16, 8, 1]
        p['n_layers'] = 3
        for i in range(3):
            p[f'W{i}'] = rng.normal(size=(sizes[i + 1], sizes[i])) / np.sqrt(sizes[i])
            p[f'b{i}'] = rng.normal(size=sizes[i + 1]) * 0.1
    zk = np.linspace(-6, 6, 30)
    qk = 1 / (1 + np.exp(-zk))
    from scipy.interpolate import PchipInterpolator
    p.update(cal_z=zk, cal_q=qk, cal_d=PchipInterpolator(zk, qk).derivative()(zk), lambda_a=lam_a, lambda_b=lam_b,
             state_cv=0, ramp=0.05, scale_eps=1e-3, cv_min=np.array([-0.1, -0.1]), cv_max=np.array([1.0, 0.4]),
             maha_mean=np.zeros(4), maha_icov=np.eye(4), maha_thr=50.0)
    return p


KINDS = ['linear', 'gam', 'mlp']


def _points(n=5000, lam_a=0.05, lam_b=0.85):
    y = np.concatenate([rng.uniform(-0.05, 0.95, n), [lam_a, lam_b, lam_a + 1e-15, lam_b - 1e-15,
                                                       np.nextafter(lam_a, 1), np.nextafter(lam_b, 0),
                                                       np.nextafter(lam_a, 0), np.nextafter(lam_b, 1)]])
    return np.c_[y, rng.uniform(-0.05, 0.35, len(y))]


# ---------------------------------------------------------------------------------------------------
# export matches the training model
# ---------------------------------------------------------------------------------------------------
def test_mlp_export_matches_torch():
    torch = pytest.importorskip('torch')
    import models
    net = models._net(4, [16, 8]).double()
    feat = Features(['y', 'x'], ['y', 'x', 'u', 'v^2'])

    class TS:
        mu, sd = np.zeros(4), np.ones(4)

        def base_params(self, kind):
            return dict(kind=kind, mu=self.mu, sd=self.sd, **feat.to_dict())

    m = models.MLP([16, 8], 0.0, {'m3': {}}, device='cpu')
    params = m._export_net(net, TS())
    cv = _points(2000)
    with torch.no_grad():
        zt = net(torch.as_tensor(feat(cv))).numpy()[:, 0]
    assert np.max(np.abs(ScoreModel(params).raw(cv) - zt)) < 1e-12


def test_linear_and_gam_forward():
    for kind in ('linear', 'gam'):
        p = _synthetic(kind)
        cv = _points(2000)
        f = (Features.from_arrays(p['feat_L'], p['feat_E'], p['feat_specs'], p['cv_names'])(cv) - p['mu']) / p['sd']
        if kind == 'linear':
            ref = f @ p['w'] + p['b']
        else:
            ref = p['b'] + sum(bspline_basis(f[:, j], p['spl_lo'][j], p['spl_hi'][j], 12) @ p['coef'][j]
                               for j in range(4))
        assert np.max(np.abs(ScoreModel(p).raw(cv) - ref)) < 1e-12


@pytest.mark.parametrize('kind', KINDS)
def test_one_frame_equals_batch(kind):
    m = LambdaModel(_synthetic(kind))
    cv = _points()
    assert np.max(np.abs(m(cv) - np.array([m.one(c) for c in cv.tolist()]))) < 1e-12


@pytest.mark.parametrize('path', EXPORTED)
def test_exported_one_equals_batch_and_training_run(path):
    m = LambdaModel(path)
    name = os.path.basename(path)[:-4]
    system = name.split('_m')[0].split('_seed')[0]
    frames_file = os.path.join(common.OUT_ROOT, system, 'frames.npz')
    if not os.path.isfile(frames_file):
        pytest.skip('no frame cache')
    z = np.load(frames_file, mmap_mode='r')
    cv = np.asarray(z['cv'][::max(1, len(z['cv']) // 5000)], float)
    assert np.max(np.abs(m(cv) - np.array([m.one(c) for c in cv.tolist()]))) < 1e-12
    # the exported score is the training run's score
    import json
    summary = os.path.join(common.OUT_ROOT, system, 'eval', 'export.json')
    if os.path.isfile(summary):
        run_id = next((info['run'] for info in json.load(open(summary)).values()
                       if info['file'].endswith(os.path.basename(path))), None)
        if run_id:
            p = dict(np.load(os.path.join(common.OUT_ROOT, system, 'fit', 'runs', run_id + '.npz')))
            assert np.max(np.abs(ScoreModel(p).raw(cv) - m.raw(cv))) == 0


# ---------------------------------------------------------------------------------------------------
# A/B handling, continuity, reversal
# ---------------------------------------------------------------------------------------------------
def _check_position(op, L, R):            # toytis funcs.check_position
    return 'M' if L <= op[0] <= R else 'L' if op[0] < L else 'R'


@pytest.mark.parametrize('kind', KINDS)
def test_states_from_state_cv(kind):
    m = LambdaModel(_synthetic(kind))
    cv = _points()
    lam, s = m(cv), cv[:, 0]
    la, lb = m.lambda_a, m.lambda_b
    assert np.array_equal(lam < la, s < la) and np.array_equal(lam > lb, s > lb)
    assert np.array_equal(lam <= la, s <= la) and np.array_equal(lam >= lb, s >= lb)
    lam1 = [m.one(c) for c in cv.tolist()]
    assert [_check_position([v], la, lb) for v in lam1] == [_check_position([v], la, lb) for v in s]
    assert np.all(lam[(s <= la) | (s >= lb)] == s[(s <= la) | (s >= lb)])


def test_states_with_extreme_model():
    # a model that says q -> 0 (or 1) everywhere still cannot put a frame into A (or B)
    for zshift in (-1e6, 1e6):
        p = _synthetic('linear')
        p['b'] = zshift
        m = LambdaModel(p)
        cv = _points()
        lam, s = m(cv), cv[:, 0]
        assert np.array_equal(lam < m.lambda_a, s < m.lambda_a) and np.array_equal(lam > m.lambda_b, s > m.lambda_b)


@pytest.mark.parametrize('kind', KINDS)
def test_continuity(kind):
    m = LambdaModel(_synthetic(kind))
    # across the state boundaries
    for b in (m.lambda_a, m.lambda_b):
        for x in rng.uniform(0, 0.3, 20):
            lo, hi = m.one([b - 1e-10, x]), m.one([b + 1e-10, x])
            assert abs(hi - lo) < 1e-6
    # everywhere: a small move gives a small change (finite gradient)
    cv = _points(3000)
    d = rng.normal(size=cv.shape) * 1e-7
    assert np.max(np.abs(m(cv + d) - m(cv))) < 1e-3


@pytest.mark.parametrize('path', EXPORTED)
def test_exported_continuous_along_paths(path):
    """No jumps of lambda along held-out paths: on each frame-to-frame step, refined into 32 and 64 sub-steps, the
    largest sub-step change shrinks about by half (a continuous function); a discontinuity would keep its size."""
    m = LambdaModel(path)
    system = os.path.basename(path)[:-4].split('_m')[0].split('_seed')[0]
    frames_file = os.path.join(common.OUT_ROOT, system, 'frames.npz')
    if not os.path.isfile(frames_file):
        pytest.skip('no frame cache')
    z = np.load(frames_file, mmap_mode='r')
    n = min(len(z['piece']), 400_000)
    cv, piece = np.asarray(z['cv'][:n], float), np.asarray(z['piece'][:n])
    i = np.flatnonzero(piece[1:] == piece[:-1])
    i = i[np.random.default_rng(0).choice(len(i), min(len(i), 5000), replace=False)]
    x0, dx = cv[i], cv[i + 1] - cv[i]
    worst = []
    for k in (32, 64):
        s = np.linspace(0, 1, k + 1)
        pts = x0[:, None, :] + s[None, :, None] * dx[:, None, :]
        lam = m(pts.reshape(-1, cv.shape[1])).reshape(len(i), k + 1)
        worst.append(np.max(np.abs(np.diff(lam, axis=1)), axis=1))
    assert np.mean(worst[1] <= 0.75 * worst[0] + 1e-12) > 0.999
    assert np.max(worst[1]) < 0.05 * (m.lambda_b - m.lambda_a)


@pytest.mark.parametrize('kind', KINDS)
def test_reversal(kind):
    """lambda depends on the frame only: a reversed path gives the reversed sequence; velocities do not enter."""
    import toytis_adapter
    p = _synthetic(kind)
    m = LambdaModel(p)
    path = np.cumsum(rng.normal(size=(500, 2)) * 0.005, axis=0) + [0.4, 0.15]
    lam = m(path)
    assert np.array_equal(m(path[::-1]), lam[::-1])
    toytis_adapter._DEFAULT_MODEL = p
    op = toytis_adapter.LearnedOrderX({'v_ord': False})
    for x in path[:50]:
        v = rng.normal(size=2)
        assert op.calculate((x.copy(), v)) == op.calculate((x.copy(), -v))


# ---------------------------------------------------------------------------------------------------
# toytis adapter
# ---------------------------------------------------------------------------------------------------
def test_adapter_output_cache_pickle():
    import toytis_adapter
    toytis_adapter._DEFAULT_MODEL = _synthetic('mlp')
    op = toytis_adapter.LearnedOrderX({'v_ord': False})
    ph = (np.array([0.4, 0.1]), np.zeros(2))
    out = op.calculate(ph)
    assert isinstance(out, list) and len(out) == 3 and all(type(v) is np.float64 for v in out)
    assert out[1:] == [0.4, 0.1]
    assert op.calculate(ph) == out                                 # cached
    ph[0][0] = 0.5                                                 # same object, moved: no stale cache
    assert op.calculate(ph)[1] == 0.5
    op2 = pickle.loads(pickle.dumps(op))                           # restart.pkl pickles the whole simulation
    assert op2.calculate((np.array([0.3, 0.2]), np.zeros(2))) == op.calculate((np.array([0.3, 0.2]), np.zeros(2)))
    with pytest.raises(ValueError):
        toytis_adapter.LearnedOrderX({'v_ord': True})
    # OOD: counted only between the states (A/B frames do not use the model)
    n0 = op.n_ood
    op.calculate((np.array([0.01, 0.39]), np.zeros(2)))          # in A, x beyond the range: not counted
    assert op.n_ood == n0
    op.calculate((np.array([0.4, 0.45]), np.zeros(2)))           # inside, x beyond the training range
    assert op.n_ood == n0 + 1
    m = LambdaModel(_synthetic('mlp'))
    assert not m.ood(np.array([[0.01, 0.45], [0.9, -0.5]])).any() and m.ood(np.array([[0.4, 0.45]]))[0]


@pytest.mark.slow
def test_toytis_staple_run(tmp_path):
    """A short fresh staple run of the Z potential with the learned lambda: initial paths are made (the kick
    wrapper), cycles run, order.txt holds [lambda, y, x], and the A/B frames agree with y."""
    if not os.path.isdir(common.TOYTIS):
        pytest.skip('no toytis')
    import toytis_adapter
    path = next((p for p in EXPORTED if os.path.basename(p).startswith('z_br_m3')), None)
    params = path or _synthetic('mlp')
    toytis_adapter.install(params)
    from simulation import Simulation
    from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
    m = LambdaModel(params)
    q_levels = np.exp(np.linspace(np.log(1e-3), np.log(0.9), 6))
    intfs = toytis_adapter.interfaces_for_q(params, q_levels)
    settings = {'interfaces': intfs, 'simtype': 'i*', 'method': 'load', 'max_len': 200000, 'dt': 0.002,
                'temperature': 0.05, 'friction': 10.0, 'high_friction': True, 'max_cycles': 4, 'p_shoot': 0.9,
                'include_stateB': False, 'prime_both_starts': True, 'snake_Lmax': 0, 'max_paths': 5, 'dim': 2,
                'mass': 1, 'permeability': False, 'zero_left': 0.1, 'v_ord': False, 'checkpoint_freq': 0,
                'potential': RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)}
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        np.random.seed(0)
        sim = Simulation(settings)
        sim.run()
    finally:
        os.chdir(cwd)
    rows = []
    for f in sorted(glob.glob(str(tmp_path / '0*' / 'order.txt'))):
        rows += [list(map(float, line.split())) for line in open(f) if not line.startswith('#')]
    rows = np.array([r for r in rows if len(r) == 4 and any(r[1:])])
    assert len(rows) > 100
    lam, y, x = rows[:, 1], rows[:, 2], rows[:, 3]
    assert np.array_equal(lam < intfs[0], y < intfs[0]) and np.array_equal(lam > intfs[-1], y > intfs[-1])
    # order.txt rounds every column to 6 decimals: allow |grad lambda| * (rounding of y, x) + rounding of lambda
    yx = np.c_[y, x]
    grad = np.sqrt(sum(((m(yx + e) - m(yx - e)) / 2e-7) ** 2 for e in (np.array([1e-7, 0]), np.array([0, 1e-7]))))
    assert np.all(np.abs(lam - m(yx)) <= 5e-7 * (1 + np.sqrt(2) * grad) + 1e-12)
    assert sum(getattr(e.orderparameter, '_kick', None) is None for e in sim.ensembles) == len(sim.ensembles)


# ---------------------------------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('system', ['flat1d', 'cc_br'])
def test_loader_matches_prepare_data(system):
    """read_run gives the frames of tis_data.prepare_data, and every path's cycle points at its own order.txt block."""
    import argparse
    import contextlib
    import io
    import data
    import tis_data
    cfg = common.load_config(system)
    if not os.path.isdir(cfg['data']['tis_dir']):
        pytest.skip('no run')
    args_list = [a for a in cfg['data']['loader_args']]
    if '--paths-per-ensemble' in args_list:
        k = args_list.index('--paths-per-ensemble')
        args_list[k + 1] = '40'
    else:
        args_list += ['--paths-per-ensemble', '40']
    with contextlib.redirect_stdout(io.StringIO()):
        mine = data.read_run(cfg, args_list)
        p = argparse.ArgumentParser()
        tis_data.add_data_arguments(p)
        ref = tis_data.prepare_data(p.parse_args([cfg['data']['tis_dir']] + args_list))
    assert np.array_equal(mine['cv'], ref.X_all) and np.array_equal(mine['piece'], ref.i_traj)
    assert np.array_equal(mine['path'], ref.group_traj) and np.array_equal(mine['t'], ref.t_traj)
    assert np.allclose(mine['w'], ref.path_weights) and np.array_equal(mine['A'], ref.boundary0)
    # the recorded cycle of each path: its order.txt block holds the forward piece from t* on
    starts = np.r_[0, np.flatnonzero(np.diff(mine['piece'])) + 1]
    checked = 0
    for s, g, mir in list(zip(starts, mine['piece_path'], mine['piece_mirrored']))[:60]:
        if mir:
            continue
        j = np.searchsorted(mine['path_id'], g)
        ens, cycle, tstar = mine['path_ens'][j], mine['path_cycle'][j], mine['path_tstar'][j]
        block = _order_block(os.path.join(cfg['data']['tis_dir'], f'{ens:03d}', 'order.txt'), cycle)
        n = mine['path_len'][j] - tstar
        assert np.allclose(block[tstar:, 1:], mine['cv'][s:s + n])
        checked += 1
    assert checked > 10


def _order_block(path, cycle):
    rows, on = [], False
    with open(path) as f:
        for line in f:
            if line.startswith('# Cycle:'):
                if on:
                    break
                on = line.startswith(f'# Cycle: {cycle},')
                continue
            if on and not line.startswith('#'):
                rows.append(list(map(float, line.split())))
    return np.array(rows)


def test_calibration_monotone_and_resolves_low_q():
    """g is strictly increasing, continuous, inside (0, 1), and recovers a known monotone map down to q ~ 1e-4."""
    import evaluate as ev
    z = rng.normal(size=400_000) * 3
    q_true = 1 / (1 + np.exp(-(z - 1)))
    q = np.clip(q_true + rng.normal(size=z.size) * 0.002, 1e-6, 1 - 1e-6)
    zk, qk, dk = cal = ev.calibrate(z, q, np.ones_like(z))
    assert np.all(np.diff(zk) > 0) and np.all(np.diff(qk) > 0) and np.all(dk >= 0)
    assert qk[0] >= common.Q_CLIP and qk[-1] <= 1 - common.Q_CLIP
    grid = np.linspace(zk[0], zk[-1], 20001)
    g = ev.apply_cal(cal, grid)
    assert np.all(np.diff(g) >= 0) and np.max(np.abs(np.diff(g))) < 1e-2
    mid = (grid > -2) & (grid < 4)
    assert np.max(np.abs(g[mid] - 1 / (1 + np.exp(-(grid[mid] - 1))))) < 0.01
    low = (1 / (1 + np.exp(-(grid - 1))) > 1e-3) & (1 / (1 + np.exp(-(grid - 1))) < 1e-2)
    assert np.max(np.abs(np.log(g[low]) - np.log(1 / (1 + np.exp(-(grid[low] - 1)))))) < 1.0


# ---------------------------------------------------------------------------------------------------
# embedding test: the extra coordinates
# ---------------------------------------------------------------------------------------------------


def test_embed_ou_stationary_and_independent():
    """embed._ou: stationary variance sd^2 and lag-1 correlation exp(-1/tau) within a path, from the first frame
    on; no correlation between the end of one path and the start of the next."""
    import embed
    rng = np.random.default_rng(3)
    path_len = np.full(4000, 50)
    offset = np.concatenate([[0], np.cumsum(path_len)[:-1]])
    for tau in (2.0, 1e6):
        x = embed._ou(rng, path_len, offset, tau, 0.1).reshape(len(path_len), -1)
        assert abs(x[:, 0].std() - 0.1) < 0.01 and abs(x[:, -1].std() - 0.1) < 0.01
        r = np.corrcoef(x[:, 10], x[:, 11])[0, 1]
        assert abs(r - np.exp(-1 / tau)) < 0.05
        assert abs(np.corrcoef(x[:-1, -1], x[1:, 0])[0, 1]) < 0.05
