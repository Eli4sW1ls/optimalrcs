#!/usr/bin/env python3
"""Are the student's epochs and architecture enough? Train the student on a finished loop's pool labels (the last
pass's teacher) with several epoch budgets and architectures, and score every checkpoint against the exact
committor and against the teacher it was fitted to. Read-only on the loop.

    python -m loop.experiments.student_variants <loop config> [--variants base,e96,w128_e96,d3_e96,w256_e96,warm]

The two errors separate the causes: student vs teacher large -> capacity or training budget; student ~ teacher
but far from exact -> the labels limit it. Output: <loop_dir>/experiments/student_variants.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

import common  # noqa: E402
import evaluate as ev  # noqa: E402
import models  # noqa: E402
from lambda_runtime import LambdaModel  # noqa: E402
from loop import controller, diagnostics, gpu, passes, pool as pool_mod, student  # noqa: E402

VARIANTS = {
    'base': dict(arch=[64, 64], epochs=24),
    'e96': dict(arch=[64, 64], epochs=96),
    'w128_e96': dict(arch=[128, 128], epochs=96),
    'd3_e96': dict(arch=[64, 64, 64], epochs=96),
    'w256_e96': dict(arch=[256, 256], epochs=96),
    'warm': dict(arch=None, epochs=24, warm=True),          # what pass k + 1 would do: warm start, the loop's budget
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    with open(os.path.join(cfg['loop_dir'], 'state.json')) as f:
        st = json.load(f)
    last = st['passes'][-1]
    pool = pool_mod.Pool(cfg['loop_dir'])
    view = passes._view(pool, cfg, need_labels=True)
    specs = list(last.get('cvs_kept') or view.cv_names)
    levels = np.exp(np.linspace(np.log(1e-3), np.log(0.95), 10))
    idx = np.flatnonzero(view.inside & view.valid)
    cv2 = view.cv[idx][:, :min(view.dim, 2)]
    zt = common.logit(np.clip(view.q[idx].astype(float), 1e-6, 1 - 1e-6))
    w = view.w[idx].astype(float)
    zx = common.logit(view.q_exact[idx].astype(float))
    teacher_vs_exact = ev.binned_logit_rmse(cv2, zt, zx, w, np.ones(len(idx), bool))
    prev = dict(np.load(last['model'].replace('.npz', '_score.npz')))
    out = dict(loop=cfg['name'], specs=specs, n_frames=int(view.n), teacher_vs_exact=teacher_vs_exact,
               loop_pass=dict(k=last['k'], exact=last['diagnostics'].get('exact', {}).get('logit_rmse'),
                              arch=last['student'].get('arch'), selected_epoch=last['student'].get('selected_epoch')),
               variants={})
    print(f'{cfg["name"]}: specs {specs}, teacher vs exact (binned logit RMSE) {teacher_vs_exact:.3f}', flush=True)
    od = os.path.join(cfg['loop_dir'], 'experiments')
    os.makedirs(od, exist_ok=True)
    path = os.path.join(od, 'student_variants.json')
    if os.path.isfile(path):                    # keep the variants of earlier runs (one process per variant is fine)
        with open(path) as f:
            out['variants'] = json.load(f).get('variants', {})
    for name in a.variants.split(','):
        if name in out['variants']:
            print(name, 'done before, skipped', flush=True)
            continue
        v = VARIANTS[name]
        stc = dict(cfg.get('student', {}), epochs=v['epochs'], checkpoints=8)
        if v.get('arch'):
            stc['arch'] = v['arch']
        c = dict(cfg, student=stc)
        mcfg = student._cfg_for_models(c)
        t0 = time.time()
        with gpu.exclusive(what=f'student_variants {cfg["name"]} {name}'):
            ts = models.TrainSet(view, mcfg, specs)
            arch = list(stc.get('arch', [64, 64]))
            init = prev if v.get('warm') and list(prev.get('arch', arch)) == arch else None
            m = models.MLP(arch, float(stc.get('beta', 10.0)), mcfg, seed=a.seed, init_params=init)
            m.fit(ts)
        train_s = time.time() - t0
        rows = []
        for i, ck in enumerate(m.checkpoints):
            m.use_checkpoint(i)
            p = m.export()
            p['arch'] = np.array(arch)
            model = LambdaModel(student.build_export(view, p, c, front_s=last['teacher'].get('provisional_b'),
                                                     blend=bool(stc.get('blend', True))))
            ex = diagnostics.exact_metrics(model, view, levels)
            zm = common.logit(np.clip(model.q(view.cv[idx]), 1e-6, 1 - 1e-6))
            vs_t = ev.binned_logit_rmse(cv2, zm, zt, w, np.ones(len(idx), bool))
            rows.append(dict(epoch=int(ck['epoch']), exact_logit_rmse=ex['logit_rmse'], spread=ex['spread_mean'],
                             vs_teacher=vs_t))
        out['variants'][name] = dict(arch=arch, epochs=v['epochs'], warm=init is not None, train_seconds=train_s,
                                     checkpoints=rows)
        print(name, json.dumps(rows[-1]), f'best exact {min(r["exact_logit_rmse"] for r in rows):.3f}',
              f'({train_s:.0f} s)', flush=True)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
    print(path)


if __name__ == '__main__':
    main()
