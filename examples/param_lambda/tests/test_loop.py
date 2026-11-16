"""Tests of the learning loop (examples/param_lambda/loop): the format-2 export (smooth-minimum ramps, front,
trust blend), the loader guard, the seed cutting rules and the interface checks on synthetic data."""
import os
import sys

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, 'loop'))

from lambda_runtime import LambdaModel, Features  # noqa: E402


def _mlp_export(front_s=None, blend=True, seed=0, dim=2, la=0.05, lb=0.85):
    """A small random MLP export in format 2 with the smin ramps (and a front / blend)."""
    rng = np.random.default_rng(seed)
    names = ['y', 'x'][:dim]
    feat = Features(names, names)
    p = dict(kind='mlp', mu=np.zeros(dim), sd=np.ones(dim), **feat.to_dict(), n_layers=2,
             W0=rng.normal(size=(8, dim)) * 2, b0=rng.normal(size=8), W1=rng.normal(size=(1, 8)), b1=np.zeros(1),
             cal_z=np.array([-6.0, 0.0, 6.0]), cal_q=np.array([1e-4, 0.5, 1 - 1e-4]), cal_d=np.array([0.08, 0.08, 0.08]),
             lambda_a=la, lambda_b=lb, state_cv=0, ramp=0.05, scale_eps=1e-3, cv_min=np.full(dim, -1.0), cv_max=np.full(dim, 2.0),
             maha_mean=np.full(dim, 0.4), maha_icov=np.eye(dim) * 4, maha_thr=4.0,
             ramp_kind='smin', smin_k=float(np.log(2.0)), kappa_a=3.0, kappa_b=2.0, export_format=2)
    if front_s is not None:
        p['front_s'], p['front_h'] = front_s, 0.5
    if blend:
        sk = np.array([la, 0.3, 0.6, lb])
        lk = np.array([la + 1e-6, 0.35, 0.55, lb - 1e-6])
        p.update(blend_s_knots=sk, blend_h_knots=lk, blend_d_knots=np.gradient(lk, sk), blend_thr=4.0, blend_width=4.0)
    return p


@pytest.mark.parametrize('front_s,blend', [(None, False), (None, True), (0.6, False), (0.6, True)])
def test_format2_states_exact_and_one_equals_batch(front_s, blend):
    m = LambdaModel(_mlp_export(front_s, blend))
    rng = np.random.default_rng(1)
    cv = np.c_[rng.uniform(-0.2, 1.1, 20000), rng.uniform(-1.5, 2.5, 20000)]
    cv[:50, 0] = m.lambda_a                                      # exactly on the boundaries
    cv[50:100, 0] = m.lambda_b
    cv[100:150, 0] = np.nextafter(m.lambda_a, 1)
    cv[150:200, 0] = np.nextafter(m.lambda_b, 0)
    lam = m(cv)
    s = cv[:, 0]
    assert np.array_equal(lam <= m.lambda_a, s <= m.lambda_a)
    assert np.array_equal(lam >= m.lambda_b, s >= m.lambda_b)
    assert np.array_equal(lam < m.lambda_a, s < m.lambda_a)
    assert np.array_equal(lam > m.lambda_b, s > m.lambda_b)
    assert np.all(np.isfinite(lam))
    one = np.array([m.one(r) for r in cv[:3000]])
    assert np.allclose(one, lam[:3000], atol=1e-12, rtol=0)


@pytest.mark.parametrize('front_s', [None, 0.6])
def test_format2_continuous_and_capped_near_states(front_s):
    """No jumps along lines of constant x (across A, the front and B); the smooth-minimum caps hold:
    lambda - lambda_A <= kappa_A (s - lambda_A) everywhere inside, and lambda_B - lambda <= kappa_B (lambda_B - s)
    without a front (the blend is off here: lambda_seed is a separate fallback, not capped)."""
    m = LambdaModel(_mlp_export(front_s, False))
    for x in (-0.5, 0.3, 1.7):
        s = np.linspace(-0.1, 1.0, 200001)
        lam = m(np.c_[s, np.full_like(s, x)])
        assert np.max(np.abs(np.diff(lam))) < 2e-3                  # no jumps (steps of 5.5e-6 in s)
        inside = (s > m.lambda_a) & (s < m.lambda_b)
        assert np.all(lam[inside] - m.lambda_a <= m.kappa_a * (s[inside] - m.lambda_a) * 1.0001 + 1e-12)
        if front_s is None:
            assert np.all(m.lambda_b - lam[inside] <= m.kappa_b * (m.lambda_b - s[inside]) * 1.0001 + 1e-12)
    if front_s is not None:                                         # beyond the front lambda rises linearly in s
        m = LambdaModel(dict(_mlp_export(front_s, False), kappa_a=50.0))
        s = np.linspace(0.61, 0.84, 50)
        lam = m(np.c_[s, np.zeros_like(s)])
        assert np.allclose(np.diff(lam), np.diff(lam)[0], rtol=1e-9)


def test_trust_blend_falls_back_to_seed():
    m = LambdaModel(_mlp_export(None, True))
    far = np.c_[np.full(5, 0.45), np.linspace(50, 60, 5)]            # far outside the training data
    assert np.allclose(m.trust(far), 1.0)
    from lambda_runtime import pchip_eval
    assert np.allclose(m(far), pchip_eval(m._bs, m._bh, m._bd, far[:, 0]), atol=1e-9)
    near = np.array([[0.45, 0.4]])
    assert m.trust(near)[0] == 0.0


def test_loader_refuses_mirrored_pieces_for_retis(tmp_path):
    sys.path.insert(0, os.path.dirname(PKG))
    import infretis_data
    (tmp_path / 'infretis.toml').write_text('[simulation]\ninterfaces = [-1.0, 0.0, 1.0]\nsteps = 1\n[engine]\ntimestep = 1\nsubcycles = 1\n')
    (tmp_path / 'infretis_data.txt').write_text('# header\n')
    with pytest.raises(ValueError, match='RETIS'):
        infretis_data.load_infretis_data(str(tmp_path), reversed_pieces=True)
    import tis_data
    (tmp_path / 'logging.log').write_text("Full settings:\n{'interfaces': [-1.0, 0.0, 1.0], 'simtype': 'retis', 'dim': 1}\n")
    with pytest.raises(ValueError, match='RETIS'):
        tis_data.load_tis_data(str(tmp_path), [-1.0, 0.0, 1.0], reversed_pieces=True)


def test_seed_cutting_rules():
    from loop import seed
    la, lb = 0.0, 1.0
    intf = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # a RETIS path: A, excursion to 0.5, back to A, excursion to B
    lam = np.array([-0.05, 0.1, 0.3, 0.5, 0.3, 0.1, -0.02, 0.15, 0.45, 0.7, 0.9, 1.05])
    segs = seed.retis_segments(lam, la, lb)
    assert segs == [(0, 6), (6, 11)]
    assert not seed.is_zero_minus(lam, la)
    assert seed.is_zero_minus(np.array([0.01, -0.1, -0.2, -0.1, 0.02]), la)
    # a staple candidate for [2*] = window (0.2, 0.4, 0.6) cut from the second excursion
    cands = seed.staple_candidates(lam, intf, 3)
    assert cands, 'no staple candidate'
    s, t = cands[0]
    op = lam[s:t + 1][:, None]
    ok, pptype, region = seed.validate_staple(op, intf, 3)
    assert ok and pptype in ('LMR', 'RML', 'LML', 'RMR') and region[0] >= 1


def test_interface_checks_reject_one_step_apart():
    from loop import interfaces as itf
    rng = np.random.default_rng(0)
    paths = []
    for _ in range(200):
        x = np.cumsum(rng.normal(0, 0.01, 400)) + 0.4
        paths.append((x, x, 1.0))
    good = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    bad = [0.0, 0.2, 0.3, 0.4, 0.405, 0.5, 0.6, 0.8]                  # two interfaces half a step apart
    cfg = dict(eps=0.01, c=1.0, n_a=3, n_sh=3)
    assert itf.check(paths, good, cfg, 'staple')['ok']
    res = itf.check(paths, bad, cfg, 'staple')
    assert not res['ok'] and any(f[0] in ('gap', 'skip_up', 'skip_dn') for f in res['fails'])
    placed = itf.place(None, paths, bad, cfg, 'staple')
    assert placed['check']['ok'] and 0.405 not in placed['interfaces'] or 0.4 not in placed['interfaces']
    assert placed['interfaces'][0] == 0.0 and placed['interfaces'][-1] == 0.8
    assert np.all(np.diff(placed['interfaces']) > 0)


@pytest.mark.slow
def test_smoke_loop_flat1d_infretis(tmp_path):
    """The whole loop, 2 passes, on the 1D flat system with infretis + ASE (configs/loop_flat1d_smoke.toml): pass 0
    reaches B, every pass runs with its learned lambda, re-checks pass on the pass's own paths, the production
    directory is prepared, and every pass manifest names the model, interfaces and data by hash."""
    import json
    import subprocess
    env = dict(os.environ, OP_LOOP_ROOT=str(tmp_path), OMP_NUM_THREADS='1')
    subprocess.run([sys.executable, 'run.py', 'loop', 'loop_flat1d_smoke'], cwd=PKG, env=env, check=True,
                   capture_output=True, timeout=1800)
    d = tmp_path / 'loop_flat1d_smoke'
    st = json.loads((d / 'state.json').read_text())
    assert st['pass0']['reached_b'] and len(st['passes']) == 2
    for rec in st['passes']:
        m = json.loads((d / f'pass_{rec["k"]}' / 'manifest.json').read_text())
        assert m['model_md5'] and m['interfaces_md5'] and m['data_md5']
        assert rec['cost']['md_steps'] > 0
        assert all(v['empty'] == 0 for v in rec['recheck']['empty_shooting_regions'].values())
        lam = LambdaModel(rec['model'])
        assert np.isclose(rec['interfaces'][0], lam.lambda_a) and np.isclose(rec['interfaces'][-1], lam.lambda_b)
        assert np.all(np.diff(rec['interfaces']) > 0)
    assert (d / 'production' / 'infretis.toml').is_file()


def test_vectorized_turn_extension_matches_infretis():
    """seed.extend_turn (vectorized) stops where infretis' turn_detected, applied after every new frame, stops."""
    pytest.importorskip('infretis')
    from loop import env  # noqa: F401  (infretis checkout on the path)
    from loop import seed
    from infretis.classes.staple_path import turn_detected

    def slow(lam, intf, m_idx, j, d):
        L, R = intf[max(m_idx - 1, 0)], intf[min(m_idx + 1, len(intf) - 1)]
        if m_idx == 0:
            L = intf[0]
        k = j + d
        if lam[k] <= intf[0] or lam[k] >= intf[-1]:
            return k
        lr = -1 if lam[k] < L else (1 if lam[k] > R else None)
        if lr is None:
            return None
        orders = [lam[j], lam[k]]
        while True:
            k += d
            if not 0 <= k < len(lam):
                return None
            orders.append(lam[k])
            if turn_detected(np.asarray(orders), list(intf), m_idx, lr):
                return k

    rng = np.random.default_rng(3)
    for _ in range(500):
        intf = np.sort(rng.uniform(0, 1, 9)); intf[0], intf[-1] = 0.0, 1.0
        lam = np.cumsum(rng.normal(0, rng.choice([0.01, 0.05, 0.2]), 800)) + rng.uniform(0.1, 0.9)
        m_idx, j, d = int(rng.integers(0, 8)), int(rng.integers(5, 795)), int(rng.choice([-1, 1]))
        assert slow(lam, list(intf), m_idx, j, d) == seed.extend_turn(lam, list(intf), m_idx, j, d)
