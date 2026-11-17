"""Artificial seed paths: full A -> B paths made outside the sampler, used only as seed candidates of last resort.

The loop seeds every ensemble from stored paths (seed.build_seeds). When the pass-0 data never reach B (loop_stag:
the initial data cover one channel, and no lower-channel path of pass 0 got past x ~ -0.74), the ensembles above
the data front cannot be seeded. infinit should provide such paths in the future; until then this module makes
them: overdamped dynamics of the system's own potential at a raised temperature (`kT_factor`), integrated with
`substeps` steps per saved frame (the walls are stiff: at 3 x kT and one step per frame, walkers were kicked by up to
1.9 per frame and one left the box through the outer wall), many walkers from A, keeping each walker's segment from
its last frame in A to its first frame in B. A path is kept only if `accept` holds: loop_stag requires every frame
inside the channel section (-1 < x < 1) to lie in the lower channel (y < -0.6) and no frame step above 0.3.

The paths are stored as infretis load paths under <loop_dir>/artificial/load/<n>/ (order.txt, traj.txt,
accepted/art_<n>.traj with the positions) and every one of their trajectory files is tainted (lineage.json), so
seeds cut from them, and every path that still shares frames with them, are excluded from training (constraint 7).
seed.build_seeds appends them after all stored pool paths. Their MD steps are counted in state.json
(`artificial_md_steps`, added to `md_steps`).

    python -m loop.artificial_paths loop_stag [--n 8] [--kt-factor 2]
"""
import argparse
import json
import os
import sys

import numpy as np

from loop import env, controller, manifest
from loop.backends import infretis_backend as be

CV_FROM_POS = {'stag_br': lambda X, Y: np.c_[X, Y]}


def _force(system):
    if system['name'] != 'stag_br':
        raise NotImplementedError('artificial paths: stag_br only so far')
    sys.path.insert(0, os.path.join(env.EXAMPLES, 'figures_by_potential', '_shared', 'exact_committors'))
    import lgv2d_tools
    return lambda X, Y: lgv2d_tools.stag_force(X, Y)


def _in_box(box, names, X, Y):
    cols = {names[0]: X, names[1]: Y}
    m = np.ones_like(X, bool)
    for c, (lo, hi) in box.items():
        m &= (cols[c] >= lo) & (cols[c] <= hi)
    return m


ACCEPT = {'stag_br': lambda P: bool(np.all(P[(P[:, 0] > -1) & (P[:, 0] < 1), 1] < -0.6)
                                    and np.abs(np.diff(P, axis=0)).max() < 0.3)}


def generate(system, n_paths, kt_factor=2.0, avoid_box=None, walkers=200, max_len=60000, seed=0, max_steps=50_000_000,
             substeps=4):
    """Return (list of (n_frames, 2) position arrays A -> B, total walker steps (substeps counted))."""
    rng = np.random.default_rng(seed)
    force = _force(system)
    kT, gamma, dt = system['kT'] * kt_factor, system['gamma'], system['dt']
    la, lb = system['lambda_a'], system['lambda_b']
    names = system['cv_names']
    accept = ACCEPT[system['name']]
    h = dt / substeps
    sig = np.sqrt(2 * kT * h / gamma)
    X = rng.uniform(-2.3, -1.7, walkers)
    Y = rng.uniform(-1.4, 1.4, walkers)
    buf = np.zeros((max_len, walkers, 2))
    n = np.zeros(walkers, int)                         # frames since the last frame in A (that frame is buf[0])
    alive = np.zeros(walkers, bool)                    # has left A without touching the avoided box
    out, steps = [], 0
    while len(out) < n_paths and steps < max_steps:
        for _ in range(substeps):
            fx, fy = force(X, Y)
            X = X + h * fx / gamma + sig * rng.standard_normal(walkers)
            Y = Y + h * fy / gamma + sig * rng.standard_normal(walkers)
        steps += walkers * substeps
        inA = X <= la
        bad = _in_box(avoid_box, names, X, Y) if avoid_box else np.zeros(walkers, bool)
        # restart the record at every frame in A; kill walkers that enter the box or run too long
        buf[0, inA, 0], buf[0, inA, 1] = X[inA], Y[inA]
        n[inA] = 1
        alive[inA] = True
        go = alive & ~inA
        idx = np.flatnonzero(go)
        buf[n[idx], idx, 0], buf[n[idx], idx, 1] = X[idx], Y[idx]
        n[idx] += 1
        alive &= ~bad & (n < max_len - 1)
        done = np.flatnonzero(go & alive & (X >= lb))
        for w in done:
            P = buf[:n[w], w].copy()
            if accept(P):
                out.append(P)
            alive[w] = False                           # the walker continues; a new record starts in A
        if len(out) >= n_paths:
            break
    return out[:n_paths], steps


def write(loop_dir, system, paths, note):
    from ase import Atoms
    from ase.constraints import FixCartesian
    from ase.io import write as ase_write
    root = os.path.join(loop_dir, 'artificial')
    os.makedirs(os.path.join(root, 'load'), exist_ok=True)
    inodes = set()
    for i, P in enumerate(paths):
        d = os.path.join(root, 'load', str(i))
        os.makedirs(os.path.join(d, 'accepted'))
        frames = []
        for x, y in P:
            a = Atoms('H', positions=[[x, y, 0.0]], masses=[1.0])
            a.set_momenta(np.zeros((1, 3)))
            a.set_constraint(FixCartesian(0, [False, False, True]))
            frames.append(a)
        fn = f'art_{i}.traj'
        ase_write(os.path.join(d, 'accepted', fn), frames)
        cv = CV_FROM_POS[system['name']](P[:, 0], P[:, 1])
        lam = cv[:, system['state_cv']]                  # the seed lambda; seeds re-evaluate lambda anyway
        with open(os.path.join(d, 'traj.txt'), 'w') as f:
            f.write(f'# artificial path: {note}\n#     Step              Filename       index    vel\n')
            for k in range(len(P)):
                f.write(f'{k:>10}  {fn:>20s}  {k:>10}  {1:>5}\n')
        with open(os.path.join(d, 'order.txt'), 'w') as f:
            f.write(f'# artificial path: {note}\n#     Time       Orderp\n')
            for k in range(len(P)):
                f.write(f'{k:>10d}' + ''.join(f'{v:>12.6f}' for v in [lam[k]] + list(cv[k])) + '\n')
        inodes |= be.path_inodes(d)
    be.write_taint(root, inodes, note='artificial seed paths (all tainted): ' + note)
    return root


def load(loop_dir):
    """Seed candidates in the layout of seed._stored_paths (empty if there are none)."""
    root = os.path.join(loop_dir, 'artificial')
    d = os.path.join(root, 'load')
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d), key=int):
        o = np.loadtxt(os.path.join(d, name, 'order.txt'), comments='#')
        out.append(dict(pass_index='artificial', workdir=root, pnr=int(name), cv=o[:, 1:], tainted=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--kt-factor', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    system = cfg['system_spec']
    root = os.path.join(cfg['loop_dir'], 'artificial')
    if os.path.isdir(root):
        raise FileExistsError(f'{root} exists')
    box = cfg.get('pass0', {}).get('exclude_box')
    paths, steps = generate(system, a.n, a.kt_factor, box, seed=a.seed)
    note = (f'overdamped {system["name"]} at {a.kt_factor} x kT, 4 substeps per frame, avoiding {box}, lower channel '
            f'only, {len(paths)} paths, '
            f'{steps} walker steps')
    write(cfg['loop_dir'], system, paths, note)
    lens = [len(p) for p in paths]
    manifest.write(os.path.join(root, 'manifest.json'), note=note, kt_factor=a.kt_factor, avoid_box=box,
                   n_paths=len(paths), lengths=lens, md_steps=int(steps), seed=a.seed)
    st = controller.State(cfg)
    st.data['artificial_md_steps'] = int(steps)
    st.data['md_steps'] = int(st.data.get('md_steps', 0)) + int(steps)
    st.save()
    print(f'{len(paths)} artificial paths (lengths {min(lens)}-{max(lens)} frames), {steps} walker steps -> {root}')


if __name__ == '__main__':
    main()
