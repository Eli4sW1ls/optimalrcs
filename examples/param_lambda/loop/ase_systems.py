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
