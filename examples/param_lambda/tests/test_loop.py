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


class _IdentityScale:
    """A stand-in model for refill: h = lambda on [0, 1]."""
    lambda_a, lambda_b = 0.0, 1.0

    def h_of_lambda(self, lam):
        return np.asarray(lam, float)

    def scale(self, h):
        return np.asarray(h, float)


def test_refill_splits_wide_pairs_only_where_the_gate_allows():
    """refill re-inserts an interface into an interior pair more than max_ratio apart when a candidate passes the
    gate, and leaves the set unchanged when every candidate fails."""
    from loop import interfaces as itf
    rng = np.random.default_rng(0)
    paths = []
    for _ in range(300):
        x = np.clip(np.cumsum(rng.normal(0, 0.004, 1500)) + rng.uniform(0.02, 0.5), 0.001, 0.999)
        paths.append((x, x, 1.0))
    gate = dict(eps=0.01, c=1.0, n_a=3, n_sh=3)
    lv = [0.0, 0.03, 0.05, 0.4, 0.6, 1.0]                               # 0.05 -> 0.4: ratio 8; 0.4 -> 0.6: 1.5
    assert itf.check(paths, lv, gate, 'staple')['ok']
    out, added = itf.refill(_IdentityScale(), paths, lv, gate, 'staple', R=5.0)
    assert any(lv[2] < a['lam'] < lv[3] for a in added)                     # the interior pair over R is split
    assert itf.check(paths, out, gate, 'staple')['ok'] and np.all(np.diff(out) > 0)
    h = np.asarray(out)
    assert max(itf.q_ratio(h[i], h[i + 1]) for i in range(1, len(h) - 2)) <= 5.0 + 1e-9
    assert out[0] == 0.0 and out[-1] == 1.0 and set(lv) <= set(out)          # states kept, nothing removed
    # a gate no candidate can meet (n_sh beyond any window's occupancy): the set comes back unchanged
    out2, added2 = itf.refill(_IdentityScale(), paths, lv, dict(gate, n_sh=1e6), 'staple', R=5.0)
    assert added2 == [] and out2 == lv


def test_uncoupled_w_well_force_and_no_coupling():
    """zu_br calculator: the w force is -dV_w/dw of the double well and does not depend on (y, x); the (y, x)
    forces are those of the 2D potential."""
    from loop import env  # noqa: F401  (puts the ase checkout first)
    from loop import ase_calcs, ase_systems
    from ase import Atoms
    c = dict(ase_systems.get('zu_br')['calculator'])
    cls = getattr(ase_calcs, c.pop('class'))
    c2 = dict(c, dim=2, base_dim=2)
    w0, dV = c['w_well']['w0'], c['w_well']['dV']
    for p in ([0.3, 0.1, -0.07], [0.6, 0.25, 0.12], [0.45, 0.15, 0.0]):
        a = Atoms('H', positions=[p]); a.calc = cls(**c)
        b = Atoms('H', positions=[p[:2] + [0.0]]); b.calc = cls(**c2)
        f, f2 = a.get_forces()[0], b.get_forces()[0]
        r = p[2] / w0
        assert np.allclose(f[:2], f2[:2])
        assert np.isclose(f[2], -4 * dV * (r * r - 1) * r / w0)


def test_screen_keeps_only_cvs_whose_removal_hurts(monkeypatch):
    """screen.run (K-fold): a CV is kept only when leaving it out raises the pooled held-out Dirichlet energy by
    more than tol AND in a majority of folds; the state CV is never a candidate. Also the energy on a chain."""
    from loop import screen

    class V:
        cv_names = ['y', 'x', 'w']
        state_cv = 0
        n = 400
        has_exact = False
        rng = np.random.default_rng(0)
        cv = rng.normal(size=(400, 3))
        path = np.repeat(np.arange(40), 10)
        piece = np.repeat(np.arange(40), 10)
        piece_block = np.arange(40) % 10
        w = np.ones(400)
        valid = np.ones(400, bool)
        frame_split = np.full(400, 'train', dtype=object)

        def mask(self, split):
            return self.frame_split == split

        def pairs(self, mask, lag):
            i0 = np.array([i for i in range(self.n - lag) if mask[i] and mask[i + lag]
                           and self.path[i] == self.path[i + lag]], int)
            return i0, i0 + lag

    # per fold (5 folds) the energy sums (num, den) of each CV set: x helps in every fold; w helps a lot in one
    # fold only (pooled +26%, but up in 1 of 5 folds): dropped
    E = {('y', 'x', 'w'): [1.0] * 5, ('y', 'w'): [1.5] * 5, ('y', 'x'): [0.95, 0.95, 0.95, 0.95, 2.5]}
    monkeypatch.setattr(screen, '_cv_energy', lambda view, cfg, specs, seed, folds, k, lags:
                        ([{L: (e, 1.0) for L in lags} for e in E[tuple(specs)]], [1.0] * 5))
    res = screen.run(V(), {'screen': {'tol': 0.1}}, ['y', 'x', 'w'])
    assert res['kept'] == ['y', 'x']
    rx, rw = res['rows'][1], res['rows'][2]
    assert rx['left_out'] == 'x' and rx['keep'] and rx['folds_up'] == 5
    assert rw['left_out'] == 'w' and not rw['keep'] and rw['rel_change'] > 0.1 and rw['folds_up'] == 1
    assert set(screen.fold_of_frame(V(), 5)) == set(range(5))
    v = V()
    q = np.tile(np.linspace(0, 0.9, 10), 40)                    # increments of 0.1 within each path
    allm = np.ones(400, bool)
    assert np.isclose(screen.dirichlet(v, q, allm, lags=(1, 2))[1], 0.01)
    assert np.isclose(screen.dirichlet(v, q, allm, lags=(1, 2))[2], 0.04)


def test_frame_budget_keeps_b_paths_and_reweights_the_rest():
    """The teacher's frame cap (and the pool view's, same rule) keeps every piece of a path that reaches B and
    weights the randomly thinned rest by the inverse keep fraction, so weighted totals stay unbiased."""
    from types import SimpleNamespace
    from loop import teacher
    from loop.pool import reaches_b_rows
    rng = np.random.default_rng(0)
    n_pieces, length = 2000, 50
    piece = np.repeat(np.arange(n_pieces), length)
    path = piece // 2                                        # two pieces per path
    b_paths = rng.choice(n_pieces // 2, 10, replace=False)
    B = np.zeros(len(piece), bool)
    B[np.isin(path, b_paths) & (np.arange(len(piece)) % length == length - 1)] = True
    view = SimpleNamespace(n=len(piece), piece=piece, path=path, B=B)
    totals = []
    for seed in range(40):
        mask, scale = teacher.select_frames(view, 20_000, seed=seed)
        assert mask[np.isin(path, b_paths)].all()
        assert abs(mask.sum() - 20_000) < 3000
        totals.append(float(scale[mask].sum()))
    assert abs(np.mean(totals) / len(piece) - 1) < 0.02     # inverse-inclusion weights: unbiased frame total
    # pool rows: a piece whose state CV (column 1) reaches lambda_b marks its row
    z = dict(piece_n=np.array([3, 2, 4]), piece_row=np.array([0, 0, 1]),
             cv=np.c_[np.zeros(9), [0.1, 0.5, 0.2, 0.3, 0.9, 0.1, 0.2, 0.4, 0.3]])
    assert reaches_b_rows(z, 0.85).tolist() == [0]
    # when the B paths are most of the frames (pass-0 RETIS data), they get half the budget and the front counts
    # them with their inclusion weights
    from loop.pool import thin_strata
    f_b, f_o = thin_strata(14_000_000, 1_000_000, 8_000_000)
    assert np.isclose(f_b * 14e6, 4e6) and f_o == 1.0
    keep_paths = np.arange(15)
    v2 = SimpleNamespace(cv=np.c_[np.r_[np.full(15, 0.9), np.full(5, 0.3)]], state_cv=0, path=np.arange(20),
                         path_id=np.arange(20), lambda_a=0.05, incl=np.r_[np.full(15, 1 / f_b), np.ones(5)])
    _, arrivals = teacher.front(v2, [0.85], n_front=20)
    assert arrivals[0] == round(15 / f_b)


def _exact_tools():
    from loop import env
    sys.path.insert(0, os.path.join(env.EXAMPLES, 'figures_by_potential', '_shared', 'exact_committors'))
    import exact_tools
    return exact_tools


def test_vec_step_matches_ase():
    """exact_tools.vec_step (the exact solver's and md_well_lgv's step) is ase's Langevin step on the cos well."""
    from loop import ase_systems  # noqa: F401  (loop.env: the ASE checkout first)
    from loop.ase_calcs import ToytisPotential
    from ase import Atoms, units
    from ase.constraints import FixCartesian
    from ase.md.langevin import Langevin
    E = _exact_tools()
    s = ase_systems.get('well_lgv')
    at = Atoms('H', positions=[[-0.2, 0, 0]], masses=[1.0])
    at.set_velocities([[0.3, 0, 0]])
    at.set_constraint(FixCartesian(0, [False, True, True]))
    at.calc = ToytisPotential(**{k: v for k, v in s['calculator'].items() if k != 'class'})
    dyn = Langevin(at, timestep=s['dt'], temperature_K=s['kT'] / units.kB, friction=s['gamma'], fixcm=False,
                   rng=np.random.RandomState(3))
    rs = np.random.RandomState(3)
    c = E.vec_coefficients(s['dt'], s['gamma'], s['kT'], 1.0)
    x, v = np.array(-0.2), np.array(0.3)
    for _ in range(300):
        dyn.run(1)
        xi, eta = rs.standard_normal((1, 3)), rs.standard_normal((1, 3))
        x, v = E.vec_step_xieta(x, v, xi[0, 0], eta[0, 0], E.coswell_force, c, np)
        assert abs(at.positions[0, 0] - x) < 1e-9 and abs(at.get_velocities()[0, 0] - v) < 1e-9
    # x' depends on z1 only (what the exact solver assumes)
    z = np.linspace(-2, 2, 5)
    assert np.allclose(E.vec_step(0.1, 0.2, 0.5, z, E.coswell_force, c, np)[0], E.vec_step(0.1, 0.2, 0.5, 0.0, E.coswell_force, c, np)[0])


def test_velocity_cv_reads_the_frame_with_vel_rev(tmp_path):
    """SeedLambda with a velocity CV: flagged velocity dependent (infretis then recomputes reversed paths), and a stored
    phase point (config only, no velocities in memory) gives v from its trajectory frame, negated when vel_rev."""
    from loop import ase_systems
    import infretis_orderp
    from infretis.classes.system import System
    from ase import Atoms
    from ase.io import write
    s = ase_systems.get('well_lgv')
    op = infretis_orderp.SeedLambda(ase_systems.cv_tables(s), state_cv=0)
    assert op.velocity_dependent
    at = Atoms('H', positions=[[-0.1, 0, 0]], masses=[1.0])
    at.set_velocities([[0.7, 0, 0]])
    f = str(tmp_path / 'p.traj')
    write(f, [at])
    for rev, sign in ((False, 1), (True, -1)):
        ph = System()
        ph.config, ph.vel_rev, ph.vel = (f, 0), rev, np.zeros(0)
        lam, x, v = op.calculate(ph)
        assert np.isclose(lam, -0.1) and np.isclose(x, -0.1) and np.isclose(v, sign * 0.7)
    # positions only: not velocity dependent (the overdamped systems are unchanged)
    assert not infretis_orderp.SeedLambda(ase_systems.cv_tables(ase_systems.get('z_br'))).velocity_dependent


def test_stopped_targets():
    import teacher
    piece = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    boundary = np.array([0, 0, 0, 1, 0, 0, 0, 0], bool)
    u, starts = teacher.stopped_targets(piece, boundary, 2)
    assert u.tolist() == [2, 3, 3, 3, 4, 7, 7, 7]
    assert starts.tolist() == [0, 1, 2, 5, 6]


def test_lag_one_update_equals_library_npneq():
    """The stopped-window update at lag 1 is the library's NPNEq step (same basis, weights and gamma = 0)."""
    cp = pytest.importorskip('cupy')
    import teacher
    orc = teacher._optimalrcs_cupy()
    rng = np.random.default_rng(0)
    n = 4000
    piece = np.repeat(np.arange(40), 100)
    r = cp.asarray(np.clip(rng.uniform(size=n), 0, 1))
    b = np.zeros(n, bool)
    b[99::100] = True
    r[cp.asarray(b)] = 1.0
    y = cp.asarray(rng.normal(size=n))
    w = cp.asarray(rng.uniform(0.5, 2, n - 1))
    fk = orc.lazy_basis_poly_ry(r, y, 4, cp.asarray(1.0 - b))
    ref = orc.nonparametrics.npneq(r, fk, cp.asarray(piece), 0.0, w_traj=w)
    u, starts = teacher.stopped_targets(piece, b, 1)
    new = teacher._npneq_lag(u, starts, chunk=700)(r, fk, cp.asarray(piece), 0.0, w_traj=w)
    assert float(cp.abs(new - ref).max()) < 1e-8
