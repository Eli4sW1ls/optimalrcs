#!/usr/bin/env python3
"""inftools' staple error analysis (`inft error_analysis_staple`: running estimates + recursive block errors of
P_cross, the MSM matrices and the rate) on the production runs of a loop:

    python -m loop.pcross_errors <loop config> [--dirs production production_seed] [--interval 50] [--burn-in 0.1]

Output per run in <run>/error_analysis/ (pcross_runav.txt, pcross_block_errors_<i>.txt/.png, rate_runav.txt, ...),
read by `loop.plots.pcross_convergence`."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, controller  # noqa: E402,F401

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--dirs', nargs='+', default=['production', 'production_seed'])
    ap.add_argument('--interval', type=int, default=50)
    ap.add_argument('--burn-in', type=float, default=0.1)
    ap.add_argument('--out-name', default='error_analysis', help='output folder inside each run (the convergence '
                    'plot reads error_analysis/)')
    a = ap.parse_args()
    cfg = controller.load_loop_config(a.config)
    from inftools.analysis.rec_error_staple import error_analysis_staple
    from loop.backends import infretis_backend as be
    for sub in a.dirs:
        d = os.path.join(cfg['loop_dir'], sub)
        df = os.path.join(d, 'infretis_data.txt')
        n = sum(1 for line in open(df) if line.strip() and not line.startswith('#'))
        out = os.path.join(d, a.out_name)
        error_analysis_staple(d, nskip=int(a.burn_in * n), interval=a.interval, pathlengths=True, h5=be.order_h5(d),
                              outdir=out, output=os.path.join(out, 'report.txt'), quiet_progress=True)
        print('done', sub)
