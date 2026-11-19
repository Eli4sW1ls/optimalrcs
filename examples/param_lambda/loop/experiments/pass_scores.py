#!/usr/bin/env python3
"""Which internal per-pass score tracks the exact error of lambda_k? (offline, state.json only)

    python -m loop.experiments.pass_scores [loop ...] [--json out.json]

Per pass k (lambda_k): the target is diagnostics.exact.logit_rmse (never used by the loop). Candidate scores, all
computed by the loop without the exact committor:
  recheck_ok     the spacing checks of lambda_k's interfaces on lambda_k's own run (pass k record)
  n_fails        number of failing checks there
  stab           max stability spread of lambda_k against lambda_{k-1} (pass k record)
  sdzq_new       SD(Z_q) of lambda_k on all pooled data, in-sample (pass k record)
  sdzq_prosp     SD(Z_q) of lambda_k on its own run's paths, before refitting on them (pass k+1 record: out of
                 sample; missing for the last pass)
  sdzq_student   the student's validation SD(Z_q)
  ood            OOD fraction of the pooled frames
Output: Spearman rank correlation within each loop, and the exact error of the pass each rule would freeze
(last, best by score, last pass whose recheck passes) against the best pass.
"""
import json
import os
import sys

import numpy as np

OP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                  '..', 'figures_by_potential', '_shared', 'param_lambda', 'op_loop')
DEFAULT = ['loop_z', 'loop_zw_x12', 'loop_zw_s2', 'loop_zw_paths', 'loop_zu', 'loop_zu_xy', 'loop_zu_b',
           'loop_zu_b_s2', 'loop_zu_xy_b', 'loop_zu_xy_b_s2_x12', 'loop_stag', 'loop_cc', 'loop_well_lgv']


def rows_of(loop):
    st = json.load(open(os.path.join(OP, loop, 'state.json')))
    P = st['passes']
    out = []
    for i, p in enumerate(P):
        d = p.get('diagnostics') or {}
        ex = (d.get('exact') or {}).get('logit_rmse')
        if ex is None:
            continue
        stab = d.get('stability')
        sv = [v for v in (stab['new_on_old'] + stab['old_on_new'])] if stab else []
        sv = [v for v in sv if v is not None and np.isfinite(v)]
        nxt = P[i + 1]['diagnostics'] if i + 1 < len(P) else {}
        rc = p.get('recheck') or {}
        out.append(dict(loop=loop, k=p['k'], exact=ex, n_intf=len(p['interfaces']),
                        recheck_ok=bool(rc.get('ok')) and not rc.get('unverified'), n_fails=len(rc.get('fails', [])),
                        stab=max(sv) if sv else np.nan,
                        sdzq_new=d.get('new', {}).get('zq', {}).get('sdzqs', np.nan),
                        sdzq_prosp=(nxt.get('prospective') or {}).get('zq', {}).get('sdzqs', np.nan),
                        sdzq_student=(p.get('student') or {}).get('sdzqs', np.nan),
                        ood=d.get('new', {}).get('ood', {}).get('all', np.nan),
                        teacher_rms=(p.get('teacher') or {}).get('rms_vs_exact', np.nan)))
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


SCORES = ['stab', 'sdzq_new', 'sdzq_prosp', 'sdzq_student', 'ood', 'n_fails']


def main():
    argv = sys.argv[1:]
    out_json = None
    if '--json' in argv:
        i = argv.index('--json')
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    loops = [a for a in argv if not a.startswith('--')] or DEFAULT
    allrows, summary = [], []
    for L in loops:
        if not os.path.isfile(os.path.join(OP, L, 'state.json')):
            continue
        R = rows_of(L)
        if len(R) < 3:
            continue
        allrows += R
        ex = np.array([r['exact'] for r in R])
        print(f'\n== {L}: {len(R)} passes')
        print('  k  exact  intf ok fails  stab    sdzq_new sdzq_prosp sdzq_stud  ood')
        for r in R:
            print(f"  {r['k']:<2d} {r['exact']:6.3f} {r['n_intf']:4d} {'Y' if r['recheck_ok'] else '-'}  {r['n_fails']:3d}  "
                  f"{r['stab']:7.4f} {r['sdzq_new']:8.3f} {r['sdzq_prosp']:9.3f} {r['sdzq_student']:9.3f}  {r['ood']:.4f}")
        s = dict(loop=L, n=len(R), best=float(ex.min()), best_k=R[int(ex.argmin())]['k'], last=float(ex[-1]))
        for sc in SCORES:
            v = np.array([r[sc] for r in R], float)
            s['rho_' + sc] = spearman(v, ex)
            okv = np.isfinite(v)
            s['pick_' + sc] = float(ex[okv][np.argmin(v[okv])]) if okv.any() else np.nan
        okr = [r for r in R if r['recheck_ok']]
        s['pick_last_ok'] = okr[-1]['exact'] if okr else np.nan
        # last pass whose checks pass, among those with the lowest prospective score in the second half
        summary.append(s)
    print('\n== Spearman rho (score vs exact logit RMSE), per loop')
    print('loop                    n   ' + '  '.join(f'{c:>12s}' for c in SCORES))
    for s in summary:
        print(f"{s['loop']:22s} {s['n']:3d}   " + '  '.join(f"{s['rho_' + c]:12.2f}" for c in SCORES))
    med = {c: np.nanmedian([s['rho_' + c] for s in summary]) for c in SCORES}
    print(f"{'median':22s}       " + '  '.join(f'{med[c]:12.2f}' for c in SCORES))
    print('\n== exact logit RMSE of the frozen pass under each rule (best = oracle)')
    rules = ['last', 'pick_last_ok'] + ['pick_' + c for c in SCORES]
    print('loop                   best(k)      ' + '  '.join(f'{r.replace("pick_", ""):>10s}' for r in rules))
    for s in summary:
        print(f"{s['loop']:22s} {s['best']:6.3f}({s['best_k']:2d})  " + '  '.join(f"{s[r]:10.3f}" for r in rules))
    for r in rules:
        reg = [s[r] / s['best'] for s in summary if np.isfinite(s[r])]
        if not reg:
            continue
        print(f'  {r:16s} mean ratio to best {np.mean(reg):.2f}, worst {np.max(reg):.2f}, n {len(reg)}')
    if out_json:
        json.dump(dict(rows=allrows, summary=summary), open(out_json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
