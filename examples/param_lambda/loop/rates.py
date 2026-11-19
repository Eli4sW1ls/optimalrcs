"""Crossing probability and rate of an infretis StapleTIS run through inftools' iSTAR analysis, and the exact
values of the toy systems.

    r = staple_rate(run_dir, burn_in=0.1)       # P_cross per interface, global P_cross, flux, rate per frame and
                                                 # per reduced time unit, relative block errors
    k = exact_rate(system)                       # TPT rate of the overdamped committor on a grid

The inftools chain: istar.reading.calculate_infretis_weights (path_f / path_w per ensemble, ptype),
compute_path_taus (path lengths from load/<pnr>/order.txt), istar.weights.PathTable, istar.msm.compute_plocs
(the turn-to-turn MSM) and compute_rates. Errors: inftools' recursive block analysis of the running estimates
(istar.running + istar.block_errors), the relative error at the plateau.

Rates and crossing probabilities for results come only from runs with a frozen lambda (constraint 5).
"""
import contextlib
import io
import json
import os

import numpy as np


def staple_rate(run_dir, burn_in=0.1, n_snap=40, data_file=None, toml_file=None, verbose=False):
    from inftools.istar.reading import calculate_infretis_weights, compute_path_taus
    from inftools.istar.weights import PathTable
    from inftools.istar.msm import compute_plocs, compute_rates, RATE_LABELS
    from inftools.istar.running import running_estimates
    from inftools.istar.block_errors import compute_rel_errors_2d, plateau_error
    from loop import ase_systems
    from loop.backends import infretis_backend as be
    data_file = data_file or os.path.join(run_dir, 'infretis_data.txt')
    toml_file = toml_file or os.path.join(run_dir, 'infretis.toml')
    cfg = be.read_toml(toml_file)
    dt_red = float(cfg['engine'].get('timestep', 1.0)) / ase_systems.TIME_FS      # fs -> reduced time units
    sub = int(cfg['engine'].get('subcycles', 1))
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        n_rows = sum(1 for line in open(data_file) if line.strip() and not line.startswith('#'))
        wr = calculate_infretis_weights(data_file, toml_file, nskip=int(burn_in * n_rows), verbose=verbose)
        compute_path_taus(wr, load_dir=os.path.join(run_dir, 'load'),     # the archive of a pruned run (archive_runs)
                          h5_file=be.order_h5(run_dir))
        table = PathTable(wr)
        plocs, (p, q, M) = compute_plocs(table, return_msm=True)
        rates = compute_rates(M, table.tau_matrices(), plocs[-1], time_fac=1.0)
        rows = np.unique(np.linspace(max(10, table.n_paths // n_snap), table.n_paths, n_snap).astype(int))
        run = running_estimates(table, rows, pathlengths=True)
    out = dict(pcross=list(map(float, plocs)), pcross_global=float(plocs[-1]), n_paths=int(table.n_paths),
               interfaces=list(map(float, wr['interfaces'])), time_per_frame=dt_red * sub)
    for k in RATE_LABELS:
        out[k] = float(rates[k])
    out['rate_per_time'] = out['Rate'] / (dt_red * sub)                 # per reduced time unit
    try:
        sizes = np.arange(1, max(2, len(rows) // 4) + 1)
        rel_p = compute_rel_errors_2d(run['pcross'][:, -1], sizes)
        rel_r = compute_rel_errors_2d(run['rates'][:, RATE_LABELS.index('Rate')], sizes)
        out['rel_err_pcross'] = float(plateau_error(rel_p, sizes)[0])
        out['rel_err_rate'] = float(plateau_error(rel_r, sizes)[0])
    except Exception as exc:                                   # too few snapshots
        out['rel_err_pcross'] = out['rel_err_rate'] = float('nan')
        out['error_note'] = str(exc)
    return out


def reference_rate(system):
    """The reference rate of a system, or None: exact for 2D systems; `rate_ref` = another system's name (same
    committor, e.g. an uncoupled embedding) or a dict(rate, rel_err, source) from MD."""
    ref = system.get('rate_ref')
    if isinstance(ref, dict) and 'file' in ref:              # the rate.json of a long MD run (loop/md_z_lgv.py)
        from loop import env
        f = os.path.join(env.REPO, ref['file'])
        if not os.path.isfile(f):
            return None
        with open(f) as fh:
            md = json.load(fh)
        return dict(rate=md['k_ab'], rel_err=md['k_ab_se'] / md['k_ab'], steps_per_ab=md['steps_per_ab'],
                    source=f'long MD ({md["walkers"]} walkers x {md["steps_per_walker"]} steps, {md["n_ab"]} A -> B): '
                           f'{ref["file"]}')
    if isinstance(ref, dict):
        return dict(ref)
    if isinstance(ref, str):
        from loop import ase_systems
        out = exact_rate(ase_systems.get(ref))
        out['source'] = f'{ref}: ' + out['source']
        return out
    ex = system.get('exact') or {}
    if ex.get('kind') in ('tpt_grid', 'tpt_grid3'):
        from loop import env
        z = np.load(os.path.join(env.REPO, ex['file']))
        return dict(rate=float(z['k_ab']), steps_per_ab=float(z['steps_per_ab']),
                    source=('loop/exact_zw.py (3D' if ex['kind'] == 'tpt_grid3' else 'toy2d.exact_committor_brownian (')
                    + f'TPT on a grid, h = {float(z["h"]):g}): {ex["file"]}')
    if system.get('dim') == 2 and system.get('potential'):
        return exact_rate(system)
    return None


def exact_rate(system, h=0.004):
    """A -> B rate per reduced time unit of the overdamped dynamics: the TPT rate of the grid committor
    (toy2d.exact_committor_brownian, continuous dynamics with the boundary shift of a discrete check every step)."""
    import toy2d
    from loop.ase_calcs import load_toytis_potential
    if system['dim'] != 2:
        raise NotImplementedError('exact rate: 2D systems only')
    pot = load_toytis_potential(system['potential'], system['potential_kwargs'])
    # potentials without a grid() method (the Z potential): the box of exact_tools.solve_brownian_2d
    rng = {} if hasattr(pot, 'grid') else dict(y_range=(-0.03, 0.93), x_range=(-0.03, 0.33))
    ref = toy2d.exact_committor_brownian(pot, system['kT'], system['gamma'], system['dt'], system['lambda_a'],
                                         system['lambda_b'], h=h, **rng)
    return dict(rate=float(ref['k_ab']), steps_per_ab=float(ref['steps_per_ab']),
                source=f'toy2d.exact_committor_brownian (TPT on a grid, h = {h})')
