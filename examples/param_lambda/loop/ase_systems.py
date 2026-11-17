"""The toy systems as infretis + ASE runs: engine settings in ASE units from the reduced settings of the toytis
runs, the CV sources (the particle's positions), the initial configuration, and the exact committor where one
exists.

Reduced units -> ASE: sigma = 1 A, epsilon = 1 eV, m = 1 amu, so lengths, energies and masses are unchanged; the
time unit is TIME_FS = 10.18 fs (A sqrt(amu / eV)); T [K] = kT / kB; the engine takes the time step in fs and the
friction in 1/fs. The infretis ASE engine's overdamped integrator (ase checkout, OverdampedLangevin) makes the same
step as toytis' Brownian engine, x += dt F / (m gamma) + N(0, 2 kT dt / (m gamma)), so the exact committors of the
discrete toytis chains (figures_by_potential/<potential>/exact/) apply unchanged.
"""
import os

import numpy as np

from loop import env

TIME_FS = 10.180505671156725          # fs per ASE time unit (1 / ase.units.fs)
KB = 8.617330337217213e-05            # eV / K (ase.units.kB)
CALCS = os.path.join(env.HERE, 'ase_calcs.py')

SYSTEMS = {
    # the infretis examples/ase flat-walls staple run (reduced kT = 1, gamma = 5, dt = 2e-4); A x <= -0.1, B x >= 0.3
    'flat1d': dict(dim=1, kT=1.0, gamma=5.0, dt=2e-4, lambda_a=-0.1, lambda_b=0.3, cv_names=['x'], state_cv=0,
                   calculator=dict(**{'class': 'FlatWalls1D'}, xleft=-0.2, xright=0.4, k=100.0),
                   start=[-0.101, 0.0, 0.0], seed_interfaces=[-0.1, 0.0, 0.1, 0.2, 0.3], maxlength=100000,
                   exact=dict(kind='flat1d', walls=(-0.2, 0.4), k=100.0)),
    # toytis sim_istarz_br_g10: Z potential, overdamped, gamma 10, kT 0.05, dt 0.002; A y <= 0.05, B y >= 0.85
    'z_br': dict(dim=2, kT=0.05, gamma=10.0, dt=0.002, lambda_a=0.05, lambda_b=0.85, cv_names=['y', 'x'], state_cv=0,
                 calculator=dict(**{'class': 'ToytisPotential'}, spec='sjoelbak_angle:RectangularGridWithAngledBarrierPotential',
                                 params={'Lx': 0.3, 'angle': 45}, dim=2),
                 start=[0.02, 0.15, 0.0], seed_interfaces=[0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85],
                 maxlength=200000, exact=dict(kind='zpot', file='examples/figures_by_potential/2d_zpot/exact/q_exact_staple_h0.001.npz'),
                 potential='sjoelbak_angle:RectangularGridWithAngledBarrierPotential', potential_kwargs={'Lx': 0.3, 'angle': 45}),
    # toytis test_cc_br: curved channel, overdamped, gamma 10, kT 0.05, dt 0.005; A y <= 0.15, B y >= 0.85
    'cc_br': dict(dim=2, kT=0.05, gamma=10.0, dt=0.005, lambda_a=0.15, lambda_b=0.85, cv_names=['y', 'x'], state_cv=0,
                  calculator=dict(**{'class': 'ToytisPotential'}, spec='curved_channel:CurvedChannelDoubleWell', params={}, dim=2),
                  start=[0.1, -0.15, 0.0], seed_interfaces=[0.15, 0.22, 0.29, 0.36, 0.43, 0.5, 0.57, 0.64, 0.71, 0.78, 0.85],
                  maxlength=200000, exact=dict(kind='cc', file='examples/figures_by_potential/2d_curved_channel/exact/q_exact_h0.0015.npz'),
                  potential='curved_channel:CurvedChannelDoubleWell', potential_kwargs={}),
    # M3 native embedding: z_br plus a simulated slow coordinate w that does not touch (y, x): an uncoupled double
    # well dV ((w / w0)^2 - 1)^2 with the w0 and dV (3 kT) of zw_br, so the w dynamics match the gate system and only
    # the coupling differs. The committor and rate are those of z_br (`exact` reads columns y, x; `rate_ref`).
    'zu_br': dict(dim=3, kT=0.05, gamma=10.0, dt=0.002, lambda_a=0.05, lambda_b=0.85, cv_names=['y', 'x', 'w'], state_cv=0,
                  calculator=dict(**{'class': 'ToytisPotential'}, spec='sjoelbak_angle:RectangularGridWithAngledBarrierPotential',
                                  params={'Lx': 0.3, 'angle': 45}, dim=3, base_dim=2, w_well={'w0': 0.1, 'dV': 0.15}),
                  start=[0.02, 0.15, -0.1], seed_interfaces=[0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85],
                  maxlength=200000, exact=dict(kind='zpot', file='examples/figures_by_potential/2d_zpot/exact/q_exact_staple_h0.001.npz'),
                  potential='sjoelbak_angle:RectangularGridWithAngledBarrierPotential', potential_kwargs={'Lx': 0.3, 'angle': 45},
                  rate_ref='z_br'),
    # M3 gate: toytis zpot_w.ZChannelsW (figures_by_potential/3d_zpot_w), w0 0.1, dV_w 3 kT, delta 0.02: w selects
    # which of two mirrored ridges is active, so the committor depends on w. Exact committor and rate: the 3D grid
    # solve of loop/exact_zw.py (about a minute on one core at h 0.004). Check of the rate from the GPU Brownian MD scan (3d_zpot_w/explore/scan_w00.1_d0.02.json,
    # dV_w = 0.15): nAB / walker steps = 4.21e-8 per step (1683 transitions; the walkers start in A and spend most of
    # the time on the A side, so this is k_AB up to the small time after reaching B). The same estimate with w frozen
    # (8 kT) gives 2.40e-8 per step against the exact Z 2.26e-8 (+6%, 961 transitions). The 3D grid solve
    # (loop/exact_zw.py) gives 2.12e-5 per time unit at h 0.012.
    'zw_br': dict(dim=3, kT=0.05, gamma=10.0, dt=0.002, lambda_a=0.05, lambda_b=0.85, cv_names=['y', 'x', 'w'], state_cv=0,
                  calculator=dict(**{'class': 'ToytisPotential'}, spec='zpot_w:ZChannelsW',
                                  params={'w0': 0.1, 'dV_w': 0.15, 'delta': 0.02, 'Lx': 0.3, 'angle': 45}, dim=3),
                  start=[0.02, 0.15, -0.1], seed_interfaces=[0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85],
                  maxlength=200000,
                  # exact: 3D grid committor of loop/exact_zw.py (h 0.004: k_AB 2.1153e-5; h 0.008: 2.1227e-5; matches
                  # the shooting points of 3d_zpot_w/README.md within their errors)
                  exact=dict(kind='tpt_grid3', file='examples/figures_by_potential/3d_zpot_w/exact/q_tpt_h0.004.npz'),
                  md_rate=dict(rate=1683 / 4e10 / 0.002, rel_err=1 / np.sqrt(1683),
                               source='GPU Brownian MD scan, 3d_zpot_w/explore/scan_w00.1_d0.02.json (dV_w 0.15): nAB / walker steps')),
    # M3 two channels: toytis potential_staggered_channels (the benchmark landscape of 2d_staggered_channels:
    # V_peak 7.1 kT, V_dip 3 kT at kT 0.25), overdamped, gamma 1, dt 0.005 (RMS step 0.05 against the wall width
    # 0.15). Positions (x, y): x is the state CV (A x <= -1.6, B x >= 1.6), y the transverse coordinate (upper
    # channel y > 0.6, lower y < -0.6, obstacle between). Exact committor and rate: loop/exact_stag.py.
    'stag_br': dict(dim=2, kT=0.25, gamma=1.0, dt=0.005, lambda_a=-1.6, lambda_b=1.6, cv_names=['x', 'y'], state_cv=0,
                    calculator=dict(**{'class': 'ToytisPotential'}, spec='potential_staggered_channels:PotentialStaggeredChannels',
                                    params={'V_peak': 0.5 * 0.25 / 0.07, 'V_dip': 0.75}, dim=2),
                    start=[-2.0, 0.0, 0.0], seed_interfaces=[-1.6, -1.2, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6],
                    maxlength=400000,
                    exact=dict(kind='tpt_grid', ranges=[[-1.75, 1.75], [-2.05, 2.05]],
                               file='examples/figures_by_potential/2d_staggered_channels/exact/brownian_g1_dt0.005/q_tpt_h0.01.npz'),
                    potential='potential_staggered_channels:PotentialStaggeredChannels',
                    potential_kwargs={'V_peak': 0.5 * 0.25 / 0.07, 'V_dip': 0.75}),
}


def get(name, overrides=None):
    s = dict(SYSTEMS[name], name=name)
    s.update(overrides or {})
    return s


def engine_table(s, subcycles=1):
    """The [engine] table of infretis.toml (overdamped Langevin through ASE)."""
    calc = dict(s['calculator'], module=CALCS)
    return {
        'class': 'ase', 'engine': 'ase', 'temperature': s['kT'] / KB, 'input_path': '.',
        'timestep': s['dt'] * TIME_FS, 'subcycles': int(subcycles), 'integrator': 'overdamped_lgv',
        'langevin_fixcm': False, 'langevin_friction': s['gamma'] / TIME_FS,
        'calculator_settings': calc,
        'particles': {'mass': [1.0], 'name': ['H'], 'pos': [list(s['start'][:s['dim']])]},
        'box': {'periodic': [False] * s['dim']},
    }


def cv_tables(s):
    """CV sources: the particle's positions, in the toytis order (y, x, ...)."""
    return [{'class': 'Position', 'index': [0, j], 'periodic': False} for j in range(s['dim'])]


def write_initial(s, path, pos=None):
    """initial.traj: one particle of mass 1 at `pos` (default the system's start, inside A), the unused
    Cartesian components fixed."""
    from ase import Atoms
    from ase.constraints import FixCartesian
    from ase.io import write
    p = list(s['start'] if pos is None else pos) + [0.0] * 3
    atoms = Atoms('H', positions=[p[:3]], masses=[1.0])
    atoms.set_constraint(FixCartesian(0, [False] * s['dim'] + [True] * (3 - s['dim'])))
    write(path, atoms)
    return path


def exact_committor(s):
    """Callable q(cv) of the exact committor (0 in A, 1 in B), or None."""
    ex = s.get('exact')
    if not ex:
        return None
    if ex['kind'] == 'flat1d':
        import sys
        sys.path.insert(0, os.path.join(env.EXAMPLES, 'figures_by_potential', '_shared', 'exact_committors'))
        import exact_tools
        lo, hi = ex['walls']
        k = ex['k']

        def force(x, xp=np):
            return xp.where(x < lo, -k * (x - lo), xp.where(x > hi, -k * (x - hi), 0.0))
        sol = exact_tools.solve_brownian_1d(force, s['dt'], s['gamma'], s['kT'], s['lambda_a'], s['lambda_b'], 4001)
        xs, qs = sol[0], sol[1]

        def q(cv):
            cv = np.atleast_2d(cv)
            out = np.clip(np.interp(cv[:, 0], xs, qs), 0, 1)
            out[cv[:, 0] <= s['lambda_a']] = 0
            out[cv[:, 0] >= s['lambda_b']] = 1
            return out
        return q
    if ex['kind'] == 'tpt_grid3':                     # loop/exact_zw.py: q[iy, ix, iw] on the grid (y, x, w)
        from scipy.interpolate import RegularGridInterpolator
        z = np.load(os.path.join(env.REPO, ex['file']))
        f = RegularGridInterpolator((z['y'], z['x'], z['w']), z['q'].astype(float), bounds_error=False, fill_value=None)

        def q(cv):
            cv = np.atleast_2d(cv)
            out = np.empty(len(cv))
            for i in range(0, len(cv), 2_000_000):
                out[i:i + 2_000_000] = f(cv[i:i + 2_000_000, :3])
            out = np.clip(out, 0, 1)
            out[cv[:, s['state_cv']] <= s['lambda_a']] = 0
            out[cv[:, s['state_cv']] >= s['lambda_b']] = 1
            return out
        return q
    if ex['kind'] == 'tpt_grid':                      # toy2d.exact_committor_brownian grid (loop/exact_stag.py)
        import toy2d
        ref = dict(np.load(os.path.join(env.REPO, ex['file'])))
        sc, tc = s['state_cv'], 1 - s['state_cv']

        def q(cv):
            cv = np.atleast_2d(cv)
            out = np.empty(len(cv))
            for i in range(0, len(cv), 2_000_000):
                c = cv[i:i + 2_000_000]
                out[i:i + 2_000_000] = toy2d.interp_grid(ref, c[:, sc], c[:, tc])
            out[cv[:, sc] <= s['lambda_a']] = 0
            out[cv[:, sc] >= s['lambda_b']] = 1
            return out
        return q
    import sys
    sys.path.insert(0, os.path.join(env.EXAMPLES, 'figures_by_potential', '_shared', 'exact_committors'))
    import exact_tools
    sol = dict(np.load(os.path.join(env.REPO, ex['file'])))
    if ex['kind'] == 'zpot':
        force = exact_tools.zpot_force
    else:
        from loop.ase_calcs import load_toytis_potential
        pot = load_toytis_potential(s['potential'], s['potential_kwargs'])

        def force(y, x, xp=np):
            return pot.potential_and_force_xy(y, x, xp)[1:]

    def q(cv):
        cv = np.atleast_2d(cv)
        out = np.empty(len(cv))
        for i in range(0, len(cv), 2_000_000):
            out[i:i + 2_000_000] = exact_tools.eval_q_brownian_2d(force, sol, cv[i:i + 2_000_000, 0], cv[i:i + 2_000_000, 1])
        out = np.clip(out, 0, 1)
        out[cv[:, s['state_cv']] <= s['lambda_a']] = 0
        out[cv[:, s['state_cv']] >= s['lambda_b']] = 1
        return out
    return q
