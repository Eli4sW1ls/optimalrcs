"""The loop controller: passes, pool, teacher, student, export, placement, seeding, diagnostics, stop, freeze.

    python run.py loop <config>                 run (resumes from the loop directory's state)
    python run.py loop <config> --pass0-only    pass 0 only (infinit with the seed order parameter)
    python run.py loop <config> --max-passes N

Loop directory: <examples/figures_by_potential/_shared/op_loop/<name>/> (common.out_dir of the config name):
    pass_0/            the infinit run (pass0.json, one infretis_data_<k>.txt + infretis_<k>.toml per iteration)
    pass_<k>/          the infretis run of pass k (k >= 1), its manifest.json
    pool/              frame pool (pool.py)
    models/            lambda exports per pass (lambda_<k>.npz) and their training info
    state.json         what has been done (resumable)
"""
import json
import os
import time

import numpy as np

import common

from loop import env, ase_systems, manifest, pool as pool_mod
from loop.backends import infretis_backend as be


def loop_dir(cfg):
    root = os.environ.get('OP_LOOP_ROOT', os.path.join(common.OUT_ROOT, 'op_loop'))     # tests use a temporary root
    d = os.path.join(root, cfg['name'])
    os.makedirs(d, exist_ok=True)
    return d


def load_loop_config(name_or_path):
    cfg = common._read_toml(name_or_path)
    cfg['loop_dir'] = loop_dir(cfg)
    cfg['system_spec'] = ase_systems.get(cfg['system'], cfg.get('system_overrides'))
    return cfg


class State:
    def __init__(self, cfg):
        self.path = os.path.join(cfg['loop_dir'], 'state.json')
        self.data = manifest.read(self.path) if os.path.isfile(self.path) else dict(passes=[], md_steps=0, done=False)

    def save(self):
        manifest.write(self.path, **self.data)


# ---------------------------------------------------------------------------------------------------
# pass 0
# ---------------------------------------------------------------------------------------------------
def run_pass0(cfg, log=None):
    """infinit with the seed order parameter until B is reached; every iteration is stored as a pool pass."""
    system, run, p0 = cfg['system_spec'], cfg['run'], cfg['pass0']
    workdir = os.path.join(cfg['loop_dir'], 'pass_0')
    if not os.path.isfile(os.path.join(workdir, 'pass0.json')):
        if os.path.isdir(workdir):
            raise FileExistsError(f'{workdir} exists without pass0.json: remove it to redo pass 0')
        be.write_pass0(workdir, system, run, p0, seed=int(cfg.get('seed', 0)))
        common.say(f'pass 0: infinit in {workdir}', log=log)
        rec = be.run_pass0(workdir, n_b=int(p0.get('n_b', 20)))
    else:
        with open(os.path.join(workdir, 'pass0.json')) as f:
            rec = json.load(f)
    common.say(f'pass 0: {len(rec["iterations"])} iterations, reached B: {rec["reached_b"]}, '
               f'{rec["saved_frames"]} saved frames, final interfaces {np.round(rec["final_interfaces"], 4).tolist()}', log=log)
    return workdir, rec


def ingest_pass0(cfg, workdir, rec, log=None):
    """Store every infinit iteration as a pool pass (pieces cut with its own interfaces, seed lambda)."""
    system, pc = cfg['system_spec'], cfg['pool']
    pool = pool_mod.Pool(cfg['loop_dir'])
    sub = int(cfg['run'].get('subcycles', 1))
    stored = []
    for i, it in enumerate(rec['iterations']):
        k = i
        if pool.has(k):
            stored.append(k)
            continue
        data_file = os.path.join(workdir, os.path.basename(it['data_file']))
        toml_file = os.path.join(workdir, it.get('toml', f'infretis_{it["cstep"] + 1}.toml'))
        pieces = be.load_pieces(workdir, burn_in_fraction=0.0, paths_per_ensemble=pc.get('cap_paths_per_ensemble'),
                                data_file=data_file, toml_file=toml_file,
                                # pass 0 runs RETIS with wire fencing: path_f / path_w undoes the WF acceptance
                                # weight (the loader's default); the path_f override is for staple runs
                                weighting=pc.get('weighting') if cfg['pass0'].get('mode', 'retis') == 'staple' else None)
        raw = be.raw_paths(workdir, data_file=data_file, toml_file=toml_file)
        meta = dict(kind='infinit', lambda_version='seed', lambda_model=None, interfaces=it['interfaces'],
                    mode=cfg['pass0'].get('mode', 'retis'), lambda_a=system['lambda_a'], lambda_b=system['lambda_b'],
                    cv_names=system['cv_names'], state_cv=system['state_cv'], n_rows=len(raw['rows']), workdir=workdir,
                    data_file=data_file, toml=toml_file, md_steps=int(it['saved_frames']) * sub,
                    wall_seconds=it.get('wall_seconds'), cycles=it['steps'])
        if system.get('exact'):
            q = ase_systems.exact_committor(system)
            pieces['q_exact'] = q(pieces['cv'][:, 1:]).astype(np.float32)
        m = pool.store(k, pieces, raw, meta)
        common.say(f'pool: pass {k} (infinit iteration {it["cstep"]}): {m["n_pieces"]} pieces, {m["n_frames"]} frames, '
                   f'{m["n_paths"]} paths', log=log)
        stored.append(k)
    return pool, stored


# ---------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------
def cmd_loop(cfg, args):
    log = os.path.join(cfg['loop_dir'], 'loop.log')
    common.say(f'== loop {cfg["name"]} ({cfg["system"]}, {cfg["backend"]}) in {cfg["loop_dir"]}', log=log)
    state = State(cfg)
    workdir, rec = run_pass0(cfg, log=log)
    pool, stored = ingest_pass0(cfg, workdir, rec, log=log)
    md0 = int(sum(pool.meta(k)['md_steps'] for k in stored))
    state.data['pass0'] = dict(workdir=workdir, pool_passes=stored, reached_b=rec['reached_b'], md_steps=md0)
    state.data['md_steps'] = md0 + int(sum(p['cost']['md_steps'] for p in state.data['passes']))
    state.save()
    if getattr(args, 'pass0_only', False):
        return
    from loop import passes
    if state.data.get('done') and not os.path.isdir(os.path.join(cfg['loop_dir'], 'production')):
        state.data['done'] = False                       # the freeze was refused: continue
    if not state.data.get('done') and not os.path.isdir(os.path.join(cfg['loop_dir'], 'production')):
        passes.run_passes(cfg, state, pool, log=log, max_passes=getattr(args, 'max_passes', None))
    if getattr(args, 'production', False):
        run_production(cfg, log)


def run_production(cfg, log=None):
    """Run the frozen production directory (lambda frozen) and its rate analysis (inftools iSTAR)."""
    from loop import rates
    out = os.path.join(cfg['loop_dir'], 'production')
    if not os.path.isfile(os.path.join(out, 'infretis.toml')):
        raise FileNotFoundError(f'{out}: no frozen production run (the loop has not finished)')
    if not os.path.isfile(os.path.join(out, 'restart.toml')):
        common.say(f'production: running {out}', log=log)
        cost = be.run(out)
    else:
        cost = be.cost(out)
    r = rates.staple_rate(out, burn_in=float(cfg.get('production', {}).get('burn_in', 0.1)))
    r['cost'] = cost
    if cfg['system_spec'].get('dim') == 2:
        r['exact'] = rates.exact_rate(cfg['system_spec'])
    manifest.write(os.path.join(out, 'rates.json'), **r)
    common.say(f'production: P_cross {r["pcross_global"]:.4g} (rel. error {r["rel_err_pcross"]:.2f}), rate '
               f'{r["rate_per_time"]:.4g} per time unit (rel. error {r["rel_err_rate"]:.2f})'
               + (f'; exact rate {r["exact"]["rate"]:.4g}' if 'exact' in r else ''), log=log)
    return r
