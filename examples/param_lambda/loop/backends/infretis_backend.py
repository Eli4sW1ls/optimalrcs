"""infretis backend (ASE engine): toml generation, runs, loaders, MD-step accounting, lineage.

A pass is one infretis run in its own directory:

    <workdir>/infretis.toml      written by `write_run` (no [current] table: a fresh start from load/0..n-1)
    <workdir>/load/<n>/          the initial paths (one per ensemble, hard-linked copies) and every accepted path
    <workdir>/infretis_data.txt  the path table (path_f / path_w per ensemble)
    <workdir>/restart.toml       written by infretis after every cycle (tsubcycles = saved frames)
    <workdir>/lineage.json       the inodes of trajectory files that carry non-physical frames (see `taint`)

The order parameter is `infretis_orderp.SeedLambda` (lambda = the state CV, the CV pool recorded) or
`infretis_orderp.LearnedLambda` (an exported lambda model). Both write [lambda, cv_0, cv_1, ...] per frame, so
order.txt holds the CV pool of every sampled frame and no trajectory file has to be re-read for the analysis.

Lineage (constraint 7): infretis keeps every frame's trajectory file in traj.txt and hard-links the files of a
path into load/<n>/accepted/. A shooting move re-emits its shooting point into new files, so frames are inherited
only through swaps, copies and the initial paths. The files of the initial paths of pass 0 (made from a
hand-placed configuration) are tainted; a path is tainted while any of its files shares an inode with a tainted
file; seed paths cut from tainted paths inherit the taint. Tainted paths are kept in the pool but excluded from
training by default.
"""
import glob
import json
import os
import shutil
import subprocess
import time

import numpy as np
import tomli_w

try:
    import tomllib
except ImportError:                     # pragma: no cover
    import tomli as tomllib

from loop import env, ase_systems

import infretis_data                    # examples/

ORDERP = os.path.join(env.PARAM_LAMBDA, 'infretis_orderp.py')
PASS0 = os.path.join(env.HERE, 'backends', 'pass0_infinit.py')


# ---------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------
def seed_order_table(system):
    return {'class': 'SeedLambda', 'module': ORDERP, 'cvs': ase_systems.cv_tables(system),
            'state_cv': int(system['state_cv'])}


def learned_order_table(system, model_path):
    return {'class': 'LearnedLambda', 'module': ORDERP, 'model': os.path.abspath(model_path),
            'cvs': ase_systems.cv_tables(system)}


def base_config(system, run, interfaces, order_table, steps, mode, shooting='sh', seed=0):
    """A complete infretis.toml dict for a fresh run (no [current])."""
    n = len(interfaces)
    staple = mode == 'staple'
    sim = {
        'interfaces': [float(v) for v in interfaces], 'steps': int(steps), 'seed': int(seed), 'load_dir': 'load',
        'shooting_moves': [shooting] * max(n, 2) if shooting != 'wf' else ['sh', 'sh'] + ['wf'] * (n - 2),
        'tis_set': {'maxlength': int(system['maxlength']), 'allowmaxlength': False, 'zero_momentum': False,
                    'n_jumps': int(run.get('n_jumps', 4)), 'staple': staple},
    }
    if staple:
        sim['mode'] = 'staple'
        sim['staple_infinite_swap'] = bool(run.get('staple_infinite_swap', True))
        if not sim['staple_infinite_swap']:
            sim['staple_swap_frac'] = float(run.get('staple_swap_frac', 0.5))
    elif mode == 'pp':
        sim['mode'] = 'pp'
    workers = int(run.get('workers', 1))
    return {
        'runner': {'workers': workers, 'files': [], 'wmdrun': ['not used'] * max(8, workers)},
        'simulation': sim,
        'engine': ase_systems.engine_table(system, run.get('subcycles', 1)),
        'orderparameter': order_table,
        'output': {'data_dir': './', 'screen': int(run.get('screen', 10)), 'pattern': False, 'delete_old': False,
                   'delete_old_all': False, 'visual_validation': False},
    }


def write_toml(cfg, path):
    with open(path, 'wb') as f:
        tomli_w.dump(cfg, f)


def read_toml(path):
    with open(path, 'rb') as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------------------------------
def path_inodes(path_dir):
    """Inodes of the trajectory files of one load path (follows symlinked path directories)."""
    acc = os.path.join(os.path.realpath(path_dir), 'accepted')
    if not os.path.isdir(acc):
        return set()
    return {os.stat(os.path.join(acc, f)).st_ino for f in os.listdir(acc)}


def taint_file(workdir):
    return os.path.join(workdir, 'lineage.json')


def write_taint(workdir, inodes, note=''):
    with open(taint_file(workdir), 'w') as f:
        json.dump({'tainted_inodes': sorted(int(i) for i in inodes), 'note': note}, f)


def read_taint(workdir):
    p = taint_file(workdir)
    if not os.path.isfile(p):
        return set()
    with open(p) as f:
        return set(json.load(f)['tainted_inodes'])


def path_tainted(workdir, pnr, tainted=None):
    tainted = read_taint(workdir) if tainted is None else tainted
    return bool(path_inodes(os.path.join(workdir, 'load', str(pnr))) & tainted)


# ---------------------------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------------------------
def write_run(workdir, system, run, interfaces, order_table, steps, mode, load_paths, tainted_inodes=(),
              shooting='sh', seed=0, note=''):
    """Create a fresh run directory: infretis.toml, load/<i> = hard-linked copies of `load_paths[i]` (one per
    ensemble, each a path directory with order.txt, traj.txt, accepted/), lineage.json."""
    if os.path.exists(workdir):
        raise FileExistsError(workdir)
    os.makedirs(os.path.join(workdir, 'load'))
    if len(load_paths) != len(interfaces):
        raise ValueError(f'{len(load_paths)} load paths for {len(interfaces)} ensembles')
    for i, src in enumerate(load_paths):
        dst = os.path.join(workdir, 'load', str(i))
        shutil.copytree(os.path.realpath(src), dst, copy_function=os.link)
    cfg = base_config(system, run, interfaces, order_table, steps, mode, shooting, seed)
    write_toml(cfg, os.path.join(workdir, 'infretis.toml'))
    write_taint(workdir, set(tainted_inodes), note)
    return cfg


def run(workdir, steps=None, log='infretis.out', timeout=None):
    """Run infretisrun in `workdir` (a subprocess with the checkouts on PYTHONPATH). Returns the cost record."""
    toml = os.path.join(workdir, 'infretis.toml')
    if steps is not None:
        cfg = read_toml(toml)
        cfg['simulation']['steps'] = int(steps)
        write_toml(cfg, toml)
    if os.path.isfile(os.path.join(workdir, 'restart.toml')):
        raise FileExistsError(f'{workdir}/restart.toml exists: this run was started before (use `resume`)')
    t0 = time.time()
    with open(os.path.join(workdir, log), 'a') as out:
        proc = subprocess.run([env.INFRETISRUN, '-i', 'infretis.toml'], cwd=workdir, env=env.subprocess_env(),
                              stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f'infretisrun failed in {workdir} (see {log})')
    rec = cost(workdir)
    rec['wall_seconds'] = wall
    if steps is not None and rec['cycles'] != int(steps):
        raise RuntimeError(f'{workdir}: {rec["cycles"]} of {steps} cycles completed')
    return rec


def resume(workdir, steps, log='infretis.out', timeout=None):
    """Continue a run from its restart.toml up to `steps` cycles in total."""
    rt = os.path.join(workdir, 'restart.toml')
    cfg = read_toml(rt)
    cfg['simulation']['steps'] = int(steps)
    write_toml(cfg, rt)
    t0 = time.time()
    with open(os.path.join(workdir, log), 'a') as out:
        proc = subprocess.run([env.INFRETISRUN, '-i', 'restart.toml'], cwd=workdir, env=env.subprocess_env(),
                              stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f'infretisrun (restart) failed in {workdir}')
    rec = cost(workdir)
    rec['wall_seconds'] = time.time() - t0
    return rec


def cost(workdir):
    """Cycles done, saved frames and MD steps of a run (restart.toml: tsubcycles x engine subcycles)."""
    rt = os.path.join(workdir, 'restart.toml')
    if not os.path.isfile(rt):
        return dict(cycles=0, saved_frames=0, md_steps=0)
    cfg = read_toml(rt)
    frames = int(cfg['current'].get('tsubcycles', 0))
    sub = int(cfg.get('engine', {}).get('subcycles', 1))
    return dict(cycles=int(cfg['current']['cstep']), saved_frames=frames, md_steps=frames * sub)


def md_steps(workdir):
    return cost(workdir)['md_steps']


def ensemble_counts(workdir, burn_in=0.0, weighting=None):
    """Accepted paths of a run per ensemble, from infretis_data.txt (rows = paths in the order they ended).
    Returns dict(total = all rows, kept = rows after the first `burn_in` fraction, count = per ensemble the kept
    paths with positive weight there). Unweighted: the weights only set the distribution of the same data; with
    infinite swap a path counts in every ensemble it occupies."""
    settings = infretis_data.read_settings(workdir)
    n_ens = len(settings['interfaces'])
    f = os.path.join(workdir, 'infretis_data.txt')
    if not os.path.isfile(f):
        return dict(total=0, kept=0, count=[0] * n_ens)
    data = infretis_data.read_infretis_data(f, n_ens)
    if weighting is None:
        weighting, _ = infretis_data.default_weighting(settings)
    W = np.asarray(infretis_data._pair_weights(data, weighting, n_ens), float)
    n = len(W)
    W = W[int(burn_in * n):]
    return dict(total=n, kept=len(W), count=[int(c) for c in (W > 0).sum(axis=0)] if len(W) else [0] * n_ens)


# ---------------------------------------------------------------------------------------------------
# pass 0: infinit with the seed order parameter until B is reached
# ---------------------------------------------------------------------------------------------------
def write_pass0(workdir, system, run, p0, seed=0):
    """A directory for the infinit pass: initial.traj, infretis.toml with 2 interfaces and [infinit]."""
    if os.path.exists(workdir):
        raise FileExistsError(workdir)
    os.makedirs(workdir)
    ase_systems.write_initial(system, os.path.join(workdir, 'initial.traj'), p0.get('start'))
    mode = p0.get('mode', 'retis')
    cfg = base_config(system, run, [system['lambda_a'], system['lambda_b']], seed_order_table(system),
                      p0.get('steps_per_iter', 100), mode, p0.get('shooting', 'wf' if mode == 'retis' else 'sh'), seed)
    n_iter = int(p0.get('max_iter', 10))
    cfg['infinit'] = {'pL': float(p0.get('pL', 0.3)), 'lamres': float(p0.get('lamres', 0.001)),
                      'skip': float(p0.get('skip', 0.1)), 'initial_conf': 'initial.traj', 'cstep': -1,
                      'steps_per_iter': [int(p0.get('steps_per_iter', 100))] * n_iter}
    write_toml(cfg, os.path.join(workdir, 'infretis.toml'))
    return cfg


def run_pass0(workdir, n_b=20, log='pass0.out', timeout=None):
    """Run the infinit loop (pass0_infinit.py in a subprocess) until `n_b` paths have reached B or the
    iterations are exhausted. Returns the record it writes (pass0.json)."""
    t0 = time.time()
    with open(os.path.join(workdir, log), 'a') as out:
        proc = subprocess.run([env.PYTHON, PASS0, '--n-b', str(int(n_b))], cwd=workdir, env=env.subprocess_env(),
                              stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f'pass 0 failed in {workdir} (see {log})')
    with open(os.path.join(workdir, 'pass0.json')) as f:
        rec = json.load(f)
    rec['wall_seconds'] = time.time() - t0
    return rec


# ---------------------------------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------------------------------
def data_files(workdir):
    """The path tables of a run directory, in order: infretis_data.txt, infretis_data_1.txt, ... (infinit
    writes one per iteration)."""
    files = [os.path.join(workdir, 'infretis_data.txt')]
    files += sorted(glob.glob(os.path.join(workdir, 'infretis_data_[0-9]*.txt')),
                    key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
    return [f for f in files if os.path.isfile(f)]


def order_h5(workdir):
    """The run's archive (<run>/archive.h5, loop/archive_runs.py) when it holds every path: its check passed, or
    load/ is gone. None otherwise (read load/<pnr>/order.txt). A pruned load/ keeps only the active paths, so its
    existence does not mean it is complete."""
    h5 = os.path.join(workdir, 'archive.h5')
    if not os.path.isfile(h5):
        return None
    if not os.path.isdir(os.path.join(workdir, 'load')):
        return h5
    rep = os.path.join(workdir, 'archive_report.json')
    try:
        with open(rep) as f:
            return h5 if json.load(f).get('check', {}).get('ok') else None
    except (OSError, ValueError):
        return None


def raw_paths(workdir, data_file=None, toml_file=None, weighting=None, max_paths=None):
    """Whole paths of a run: per path the order array [lambda, cv_0, ...] (n_frames, 1 + dim), the row of the
    path table, the W row (path_f / path_w by the run's default weighting), ptype, generating ensemble, and the
    lineage flag. The path table of a run is read in file order (the order in which paths ended)."""
    from inftools.istar.reading import OrderSource
    settings = infretis_data.read_settings(workdir, toml_file)
    n_ens = len(settings['interfaces'])
    data = infretis_data.read_infretis_data(data_file or os.path.join(workdir, 'infretis_data.txt'), n_ens)
    if weighting is None:
        weighting, _ = infretis_data.default_weighting(settings)
    W = infretis_data._pair_weights(data, weighting, n_ens)
    rows = np.arange(len(data['pnr']))
    if max_paths is not None and len(rows) > max_paths:
        rows = rows[np.unique(np.round(np.linspace(0, len(rows) - 1, max_paths)).astype(int))]
    tainted = read_taint(workdir)
    load_dir = os.path.join(workdir, 'load')
    out = dict(settings=settings, weighting=weighting, rows=[], pnr=[], ops=[], W=[], ptype=[], gen_ens=[],
               move=[], tainted=[], length=[])
    with OrderSource(load_dir=load_dir, h5_file=order_h5(workdir)) as src:   # the archive of a pruned run
        for r in rows:
            pnr = int(data['pnr'][r])
            op = src.get(pnr)
            if op is None:
                continue
            out['rows'].append(int(r))
            out['pnr'].append(pnr)
            out['ops'].append(np.asarray(op, float))
            out['W'].append(W[r])
            out['ptype'].append(str(data['ptype'][r]))
            out['gen_ens'].append(int(data['gen_ens'][r]))
            out['move'].append(str(data['move'][r]))
            out['tainted'].append(path_tainted(workdir, pnr, tainted))
            out['length'].append(int(data['length'][r]))
    out['W'] = np.array(out['W']) if out['W'] else np.zeros((0, n_ens))
    return out


def has_velocity_cvs(workdir, toml_file=None):
    """True if the run's order parameter records a velocity CV (an infretis 'velocity' CV source)."""
    cfg = read_toml(toml_file or os.path.join(workdir, 'infretis.toml'))
    return any(str(t.get('class', '')).lower() == 'velocity' for t in cfg.get('orderparameter', {}).get('cvs', []))


def load_pieces(workdir, reversed_pieces=None, burn_in_fraction=0.0, thin=1, paths_per_ensemble=None,
                weighting=None, include_zero_minus=False, data_file=None, toml_file=None):
    """Training pieces of a run (cut at M as sampled; mirrored pieces for staple / pp runs only). Returns
    arrays in the layout of param_lambda's frame cache (cv = [lambda, cv_0, ...] per frame) plus per-piece
    ensemble, path-table row, mirrored flag, path number and lineage flag."""
    settings = infretis_data.read_settings(workdir, toml_file)
    if reversed_pieces is None:
        # no mirrored pieces when the CVs hold velocities: a time-reversed piece would need v -> -v, and the ASE
        # Langevin step is not exactly time-reversible (M5)
        reversed_pieces = settings['mode'] in ('staple', 'pp') and not has_velocity_cvs(workdir, toml_file)
    cvs, order, t, w, pnrs, settings, info = infretis_data.load_infretis_data(
        workdir, include_zero_minus=include_zero_minus, weighting=weighting, cut_at_m=True,
        burn_in_fraction=burn_in_fraction, thin=thin, paths_per_ensemble=paths_per_ensemble,
        reversed_pieces=reversed_pieces, data_file=data_file, toml_file=toml_file, with_info=True)
    tainted = read_taint(workdir)
    taint_cache = {}
    n = np.array([len(p) for p in cvs], int)
    pieces = dict(
        cv=np.concatenate(cvs) if cvs else np.zeros((0, 1)),
        t=np.concatenate(t).astype(np.int32) if t else np.zeros(0, np.int32),
        piece=np.repeat(np.arange(len(cvs), dtype=np.int32), n),
        piece_n=n, piece_w=np.array(w, float), piece_pnr=np.array(pnrs, np.int64),
        piece_ens=np.array([i['ens'] for i in info], np.int32), piece_row=np.array([i['row'] for i in info], np.int32),
        piece_mirrored=np.array([i['mirrored'] for i in info], bool),
        piece_core=np.array([i['core'] for i in info], np.int32).reshape(-1, 2),
        piece_tainted=np.array([taint_cache.setdefault(p, path_tainted(workdir, p, tainted)) for p in pnrs], bool),
        settings=settings, weighting=settings['weighting'], reversed_pieces=reversed_pieces)
    return pieces
