#!/usr/bin/env python3
"""Continue a finished loop past its max_passes in a copy, leaving the original untouched:

    python -m loop.experiments.extend_loop <source loop name> <new loop name> [--pass0-only]
    python run.py loop <new config>                 # the new config: inherit the source, name = <new>, max_passes

The copy holds state.json (done = False), the pool without the verification passes (copied: the teacher writes
its labels into the pool files), models/ (copied) and links to the source's pass_<k> directories (the stored paths
the seeds are cut from). Passes k + 1, ... then run exactly as a longer loop would have, from the pass-k labels.

--pass0-only: a fresh loop on the source's pass 0 (a variant of the later passes at no pass-0 cost): pass_0 linked,
the infinit pool passes copied without the teacher labels (pass 1 then starts cold, as in the source), no state.
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
import common  # noqa: E402

def pass0_only(src, dst):
    import numpy as np
    if os.path.exists(os.path.join(dst, 'pass_0')):
        sys.exit(f'{dst}/pass_0 exists')
    with open(os.path.join(src, 'state.json')) as f:
        stored = json.load(f)['pass0']['pool_passes']
    os.makedirs(os.path.join(dst, 'pool'), exist_ok=True)
    for k in stored:
        with open(os.path.join(src, 'pool', f'pass_{k}.json')) as f:
            assert json.load(f)['kind'] == 'infinit', f'pool pass {k} of {src} is not an infinit iteration'
        shutil.copy2(os.path.join(src, 'pool', f'pass_{k}.json'), os.path.join(dst, 'pool', f'pass_{k}.json'))
        z = dict(np.load(os.path.join(src, 'pool', f'pass_{k}.npz'), allow_pickle=False))
        z.pop('q_np', None)                          # the source's teacher labels: the new loop starts cold
        np.savez(os.path.join(dst, 'pool', f'pass_{k}.npz'), **z)
    os.symlink(os.path.realpath(os.path.join(src, 'pass_0')), os.path.join(dst, 'pass_0'))
    print(f'{dst}: pass 0 of {src} (pool passes {stored}, labels removed)')


if __name__ == '__main__':
    src_name, dst_name = sys.argv[1:3]
    root = os.environ.get('OP_LOOP_ROOT', os.path.join(common.OUT_ROOT, 'op_loop'))
    src, dst = os.path.join(root, src_name), os.path.join(root, dst_name)
    if os.path.exists(os.path.join(dst, 'state.json')):
        sys.exit(f'{dst} already holds a loop')
    if '--pass0-only' in sys.argv[3:]:
        pass0_only(src, dst)
        sys.exit(0)
    os.makedirs(os.path.join(dst, 'pool'), exist_ok=True)
    with open(os.path.join(src, 'state.json')) as f:
        st = json.load(f)
    st['done'] = False
    st['extended_from'] = src
    with open(os.path.join(dst, 'state.json'), 'w') as f:
        json.dump(st, f, indent=1)
    n = 0
    for fn in sorted(os.listdir(os.path.join(src, 'pool'))):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(src, 'pool', fn)) as f:
            kind = json.load(f)['kind']
        if kind == 'verify':
            continue
        for ext in ('.json', '.npz'):
            p = os.path.join(src, 'pool', fn[:-5] + ext)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(dst, 'pool', fn[:-5] + ext))
        n += 1
    shutil.copytree(os.path.join(src, 'models'), os.path.join(dst, 'models'), dirs_exist_ok=True)
    for d in sorted(os.listdir(src)):
        if d.startswith('pass_') and not os.path.exists(os.path.join(dst, d)):
            os.symlink(os.path.realpath(os.path.join(src, d)), os.path.join(dst, d))
    print(f'{dst}: {len(st["passes"])} passes, {n} pool passes (verification passes left out)')
