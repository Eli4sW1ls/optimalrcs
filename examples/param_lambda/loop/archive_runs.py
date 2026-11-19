#!/usr/bin/env python3
"""Compact finished infretis runs: the text files of every path in load/ (order.txt, energy.txt, traj.txt) into
<run>/archive.h5 with inftools' `store_txt`, a check of the archive against load/, and on request the deletion of
every path directory in load/ except the active ones (restart.toml [current] active: the paths the ensembles hold
at the end, which a restart or a new seed needs). The trajectory frame files (load/<pnr>/accepted/*.traj) are not
archived: only the kept active paths can still seed new runs (loop/seed.py). The rates, path lengths, raw paths and
error analyses read the archive (rates.py, infretis_backend.raw_paths, inft error_analysis_staple) once load/ no
longer holds every path.

    python -m loop.archive_runs <run dir> ... [--delete]

A run is archived only when it is finished (restart.toml cstep >= steps) and idle (no file in it written for 10
minutes). paths are deleted only with --delete and only when the check passes: every path directory of load/ with
an order.txt is in the archive, every path of infretis_data.txt is there, and a sample of 50 paths reads back equal.
Report per run: <run>/archive_report.json. A run whose report records a pruned or deleted load/ is not archived
again (its load/ no longer holds every path); one with a passed check and nothing deleted goes to the pruning.
"""
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from loop import env  # noqa: E402,F401
from loop.backends import infretis_backend as be  # noqa: E402


def finished(run):
    rt = os.path.join(run, 'restart.toml')
    if not os.path.isfile(rt):
        return False, 'no restart.toml'
    cfg = be.read_toml(rt)
    done, steps = int(cfg['current']['cstep']), int(cfg['simulation']['steps'])
    if done < steps:
        return False, f'cycle {done} of {steps}'
    newest = max(os.path.getmtime(os.path.join(run, f)) for f in ('restart.toml', 'infretis_data.txt', 'sim.log')
                 if os.path.isfile(os.path.join(run, f)))
    if time.time() - newest < 600:
        return False, 'written in the last 10 minutes'
    return True, f'{done} cycles'


def prune(run):
    """Delete every path directory of load/ except the active paths (and the directories they link to)."""
    load = os.path.join(run, 'load')
    active = {str(int(p)) for p in be.read_toml(os.path.join(run, 'restart.toml'))['current'].get('active', [])}
    keep = set(active)
    for a in active:
        d = os.path.join(load, a)
        if os.path.islink(d) and os.path.dirname(os.path.realpath(d)) == os.path.realpath(load):
            keep.add(os.path.basename(os.path.realpath(d)))
    n = 0
    for d in os.listdir(load):
        if d.isdigit() and d not in keep:
            p = os.path.join(load, d)
            if os.path.islink(p):
                os.remove(p)
            else:
                shutil.rmtree(p)
            n += 1
    return dict(kept=sorted(keep, key=int), removed=n, missing_active=sorted(a for a in active if not os.path.isdir(os.path.join(load, a))))


def check(run, n_sample=50, seed=0):
    import h5py
    from inftools.istar.reading import load_order_from_h5, load_order_from_path
    load, h5f = os.path.join(run, 'load'), os.path.join(run, 'archive.h5')
    dirs = [d for d in os.listdir(load) if d.isdigit() and os.path.isfile(os.path.join(load, d, 'order.txt'))]
    data = [ln.split() for ln in open(os.path.join(run, 'infretis_data.txt')) if ln.strip() and not ln.startswith('#')]
    pnrs = sorted({int(r[0]) for r in data})
    with h5py.File(h5f, 'r') as h5:
        missing_dirs = [d for d in dirs if d not in h5 or 'order.txt' not in h5[d]]
        missing_data = [p for p in pnrs if str(p) in dirs and str(p) not in h5]
        rng = np.random.default_rng(seed)
        sample = rng.choice(dirs, size=min(n_sample, len(dirs)), replace=False) if dirs else []
        bad = []
        for d in sample:
            a, b = load_order_from_path(os.path.join(load, d)), load_order_from_h5(h5, int(d))
            if a is None or b is None or a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
                bad.append(str(d))
    no_dir = [p for p in pnrs if str(p) not in dirs]          # paths of the data file without a load directory
    ok = not missing_dirs and not missing_data and not bad
    return dict(ok=ok, n_dirs=len(dirs), n_data_paths=len(pnrs), missing_dirs=missing_dirs[:20],
                n_missing_dirs=len(missing_dirs), data_paths_without_dir=len(no_dir), sample=len(sample),
                sample_mismatch=bad)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    delete = '--delete' in sys.argv
    from inftools.tistools.store_txt import store_txt
    for run in args:
        run = os.path.abspath(run)
        load = os.path.join(run, 'load')
        rep = dict(run=run, time=time.strftime('%Y-%m-%d %H:%M:%S'))
        ok, why = finished(run)
        rep['finished'] = why
        if not ok:
            print(f'{run}: skipped ({why})', flush=True)
            continue
        if not os.path.isdir(load):
            print(f'{run}: no load/ (archived before: {os.path.isfile(os.path.join(run, "archive.h5"))})', flush=True)
            continue
        rpath, h5f = os.path.join(run, 'archive_report.json'), os.path.join(run, 'archive.h5')
        old = json.load(open(rpath)) if os.path.isfile(rpath) else {}
        if old.get('load_deleted') or old.get('load_pruned'):
            print(f'{run}: archived and pruned before, left as is', flush=True)
            continue
        if old.get('check', {}).get('ok') and os.path.isfile(h5f):
            rep = old                           # archived and checked by an earlier call: prune only
        else:
            t0 = time.time()
            if os.path.isfile(h5f):
                os.remove(h5f)                  # a partial archive of an interrupted earlier call
            store_txt(name=os.path.join(run, 'archive'), load=load, order=True, energy=True, traj=True)
            rep['archive_seconds'] = time.time() - t0
            rep['archive_bytes'] = os.path.getsize(h5f)
            rep['check'] = check(run)
            print(f'{run}: archive {rep["archive_bytes"] / 1e6:.0f} MB in {rep["archive_seconds"]:.0f} s, check '
                  f'{"OK" if rep["check"]["ok"] else "FAILED"} ({rep["check"]["n_dirs"]} paths)', flush=True)
            with open(rpath, 'w') as f:
                json.dump(rep, f, indent=1)
        if delete and rep['check']['ok']:
            rep['prune'] = prune(run)
            rep['load_pruned'] = True
            print(f'{run}: load/ pruned to the {len(rep["prune"]["kept"])} active paths ({rep["prune"]["removed"]} '
                  f'removed' + (f'; active paths missing: {rep["prune"]["missing_active"]}' if rep['prune']['missing_active'] else '')
                  + ')', flush=True)
            with open(rpath, 'w') as f:
                json.dump(rep, f, indent=1)


if __name__ == '__main__':
    main()
