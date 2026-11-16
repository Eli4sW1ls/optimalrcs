#!/usr/bin/env python3
"""Pass 0 of the loop: inftools' infinit with the seed order parameter, stopped when B is reached.

Run in the pass directory (infretis.toml with 2 interfaces, [infinit] with cstep = -1, initial.traj):

    python pass0_infinit.py --n-b 20

The loop is infinit's (inftools/tistools/initial_paths.py), step by step through its helper functions, with
one addition: after every iteration the finished paths of all iterations are counted, and the loop stops once
`n_b` paths have reached B (max lambda >= lambda_B). infinit itself has no such criterion: it runs every entry
of steps_per_iter and only brings the front close to the fixed lambda_B. The pass directory is left in the
state infinit leaves it (infretis.toml with the latest interfaces and active paths, restart_<k>.toml,
infretis_data_<k>.txt per iteration, load/), and pass0.json records the iterations.

infretis is run in-process by inftools (spawned workers), so this file must be run as a script.
"""
import argparse
import json
import sys
sys.path.pop(0)                 # the script's own directory (loop/backends) must not shadow the infretis package
import os
import pathlib as pl
import shutil
import time

import numpy as np


def count_b(data_files, lambda_b):
    """Finished paths whose max order parameter reaches lambda_B, over the given path tables."""
    n = 0
    for f in data_files:
        if not os.path.isfile(f):
            continue
        with open(f) as fh:
            for line in fh:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if float(parts[2]) >= lambda_b:
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-b', type=int, default=20, help='stop once this many finished paths reached B')
    ap.add_argument('--toml', default='infretis.toml')
    args = ap.parse_args()

    from inftools.misc.infinit_helper import (set_default_infinit, read_toml, write_toml, run_infretis_ext,
                                              update_toml_interfaces, update_toml, update_actives_toml, rename_file)
    from inftools.tistools.initial_paths import generate_zero_paths
    from inftools.exercises.puckering import initial_path_from_iretis

    config = read_toml(args.toml)
    iset = set_default_infinit(config)
    lambda_b = float(config['simulation']['interfaces'][-1])
    record = dict(iterations=[], n_b=args.n_b, reached_b=False)
    t_start = time.time()

    if iset['cstep'] == -1:
        init_conf = pl.Path(iset['initial_conf']).resolve()
        max_op = generate_zero_paths(str(init_conf), toml=args.toml)
        iset['cstep'] = 0
        intf = config['simulation']['interfaces']
        nworkers = config['runner']['workers']
        lamres0 = (max_op - intf[0]) / nworkers
        intf = [intf[0]] + [intf[0] + lamres0 * (i + 1) for i in range(nworkers - 1)] + [intf[1]]
        mode = config['simulation'].get('mode', 'retis')
        sh_moves = ['sh', 'sh'] + [('sh' if mode == 'staple' else 'wf') for _ in range(len(intf) - 2)]
        load_dir = pl.Path(config['simulation'].get('load_dir', 'load'))
        for i in range(1, nworkers):
            shutil.copytree(load_dir / '1', load_dir / str(i + 1))
        c0 = read_toml(args.toml)
        c0['infinit'] = iset
        c0['simulation']['interfaces'] = intf
        c0['simulation']['shooting_moves'] = sh_moves
        write_toml(c0, 'infretis.toml')
        record['zero_paths'] = dict(max_op=float(max_op), interfaces=list(map(float, intf)))
        # lineage: the zero paths come from a hand-placed configuration; their trajectory files are tainted
        inodes = set()
        for i in range(nworkers + 1):
            acc = load_dir / str(i) / 'accepted'
            if acc.is_dir():
                inodes |= {os.stat(acc / f).st_ino for f in os.listdir(acc)}
        with open('lineage.json', 'w') as f:
            json.dump({'tainted_inodes': sorted(inodes), 'note': 'zero paths of infinit (load/0..%d)' % nworkers}, f)

    data_files = []
    for k, steps in enumerate(iset['steps_per_iter'][iset['cstep']:]):
        t0 = time.time()
        c_before = read_toml('infretis.toml')
        success = run_infretis_ext(steps)
        if not success:
            record['error'] = f'infretis did not complete iteration {iset["cstep"]}'
            break
        rt = read_toml('restart.toml')
        data_file = rt.get('output', {}).get('data_file', 'infretis_data.txt')
        data_files.append(data_file)
        n_b = count_b(data_files, lambda_b)
        # infretis carries current.tsubcycles over infinit iterations (update_actives_toml copies [current]):
        # the frames of this iteration are the difference
        total = int(rt['current'].get('tsubcycles', 0))
        frames = total - int(c_before.get('current', {}).get('tsubcycles', 0))
        it = dict(cstep=int(iset['cstep']), steps=int(steps), interfaces=list(map(float, c_before['simulation']['interfaces'])),
                  data_file=data_file, saved_frames=frames, saved_frames_total=total, n_reached_b=int(n_b),
                  wall_seconds=time.time() - t0)
        record['iterations'].append(it)
        print(f'[pass0] iteration {it["cstep"]}: {steps} steps, {n_b} paths reached B, '
              f'interfaces {np.round(it["interfaces"], 4).tolist()}', flush=True)
        # infinit's tail: new interfaces from the crossing probability, relabel the active paths, restart layout
        update_toml_interfaces(config)
        iset['cstep'] += 1
        update_toml(config)                      # backs up this iteration's toml as infretis_<cstep>.toml
        it['toml'] = f'infretis_{iset["cstep"]}.toml'
        out = initial_path_from_iretis(config['simulation'].get('load_dir', 'load'), 'infretis.toml',
                                       restart='restart.toml', return_pathnr=True)
        update_actives_toml(out)
        rename_file('restart.toml', f'restart_{iset["cstep"]}.toml')
        if n_b >= args.n_b:
            record['reached_b'] = True
            break
    record['final_interfaces'] = list(map(float, read_toml('infretis.toml')['simulation']['interfaces']))
    record['data_files'] = data_files
    record['saved_frames'] = int(sum(it['saved_frames'] for it in record['iterations']))
    record['wall_seconds'] = time.time() - t_start
    with open('pass0.json', 'w') as f:
        json.dump(record, f, indent=1)
    print(f'[pass0] done: reached B = {record["reached_b"]}, {len(record["iterations"])} iterations, '
          f'{record["saved_frames"]} saved frames', flush=True)


if __name__ == '__main__':
    main()
