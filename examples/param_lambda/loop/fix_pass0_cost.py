#!/usr/bin/env python3
"""One-off repair of loops run before 2026-11-16 16:00: pass0.json recorded infretis' cumulative frame counter per
infinit iteration. Rewrites the per-iteration frames as differences, the pool meta of the pass-0 stores, and the
MD-step totals in state.json.  python -m loop.fix_pass0_cost <loop_dir>"""
import json
import os
import sys

d = sys.argv[1]
p0 = os.path.join(d, 'pass_0', 'pass0.json')
rec = json.load(open(p0))
if not rec.get('fixed_cumulative'):
    prev = 0
    for it in rec['iterations']:
        tot = it['saved_frames']
        it['saved_frames'], it['saved_frames_total'] = tot - prev, tot
        prev = tot
    rec['saved_frames'] = prev
    rec['fixed_cumulative'] = True
    json.dump(rec, open(p0, 'w'), indent=1)
md0 = 0
for k, it in enumerate(rec['iterations']):
    mp = os.path.join(d, 'pool', f'pass_{k}.json')
    if os.path.isfile(mp):
        m = json.load(open(mp))
        m['md_steps'] = it['saved_frames']
        json.dump(m, open(mp, 'w'), indent=1)
        md0 += it['saved_frames']
sp = os.path.join(d, 'state.json')
if os.path.isfile(sp):
    st = json.load(open(sp))
    st['pass0']['md_steps'] = md0
    st['md_steps'] = md0 + sum(p['cost']['md_steps'] for p in st['passes'])
    json.dump(st, open(sp, 'w'), indent=1)
print(d, 'pass 0 frames', md0, 'total', st['md_steps'] if os.path.isfile(sp) else None)
