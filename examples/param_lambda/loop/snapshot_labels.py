#!/usr/bin/env python3
"""Save the pool's current teacher labels as the labels of pass k (pass_<k>/teacher_q.npz: per pool store, the
q of every frame). The loop overwrites the pool labels at every teacher fit; passes.teach calls `save` itself,
this command snapshots a running loop started before that.

    python -m loop.snapshot_labels <loop_dir> <k>
"""
import os
import sys

import numpy as np


def save(loop_dir, k):
    pdir = os.path.join(loop_dir, 'pool')
    out = {}
    for f in sorted(os.listdir(pdir)):
        if f.startswith('pass_') and f.endswith('.npz'):
            z = np.load(os.path.join(pdir, f))
            if 'q_np' in z.files:
                out[f[:-4]] = z['q_np']
    os.makedirs(os.path.join(loop_dir, f'pass_{k}'), exist_ok=True)
    path = os.path.join(loop_dir, f'pass_{k}', 'teacher_q.npz')
    np.savez_compressed(path, **out)
    return path


if __name__ == '__main__':
    print(save(sys.argv[1], int(sys.argv[2])))
