#!/usr/bin/env python3
"""Memory analysis of the production runs: inftools' staple error analysis (q-matrix block errors) followed by the
tistools memory-analysis plots (tistools/scripts/infretis_memory_analysis.py), with the q errors drawn as error bars.

    python -m loop.memory_plots [loop ...] [--min-cycles 20000] [--interval 100] [--nskip-min 100] [--burn-in 0.1]

Runs: every <loop>/production* directory with an infretis_data.txt and at least --min-cycles cycles (all loops if
none are named). Skipped paths: max(--nskip-min, --burn-in x the number of paths), the burn-in of the rates.
Output per run in <run>/memory_analysis/: qmat_block_errors_<interval>.txt (+ pcross, pmat, rate block errors),
crossing_probability_profile.png, memory_matrices.png, memory_probabilities.png, free_energy_momentum.png,
q_matrix.png, memory_landscape.png, and nskip.txt.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, controller  # noqa: E402,F401

SCRIPT = os.path.join(os.path.dirname(env.EXAMPLES), '..', '..', 'APPTIS', 'tistools', 'scripts',
                      'infretis_memory_analysis.py')
SCRIPT = os.path.normpath(SCRIPT)


def cycles(run):
    try:
        m = re.search(r'^cstep\s*=\s*(\d+)', open(os.path.join(run, 'restart.toml')).read(), re.M)
        return int(m.group(1)) if m else 0
    except OSError:
        return 0


def runs(names, min_cycles):
    root = os.path.dirname(controller.loop_dir({'name': 'x'}))
    loops = names or sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, 'loop_*')) if os.path.isdir(p))
    out = []
    for name in loops:
        for d in sorted(glob.glob(os.path.join(root, name, 'production*'))):
            if os.path.isfile(os.path.join(d, 'infretis_data.txt')) and cycles(d) >= min_cycles:
                out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('loops', nargs='*')
    ap.add_argument('--min-cycles', type=int, default=20000)
    ap.add_argument('--interval', type=int, default=100)
    ap.add_argument('--nskip-min', type=int, default=100)
    ap.add_argument('--burn-in', type=float, default=0.1)
    ap.add_argument('--redo', action='store_true', help='recompute the block errors even if the file exists')
    a = ap.parse_args()
    from inftools.analysis.rec_error_staple import error_analysis_staple
    for d in runs(a.loops, a.min_cycles):
        out = os.path.join(d, 'memory_analysis')
        os.makedirs(out, exist_ok=True)
        n = sum(1 for line in open(os.path.join(d, 'infretis_data.txt')) if line.strip() and not line.startswith('#'))
        nskip = max(a.nskip_min, int(a.burn_in * n))
        with open(os.path.join(out, 'nskip.txt'), 'w') as f:
            f.write(f'{nskip} of {n} paths skipped; q errors: interval {a.interval}\n')
        qerr = os.path.join(out, f'qmat_block_errors_{a.interval}.txt')
        print(f'== {d}: {n} paths, nskip {nskip}', flush=True)
        if a.redo or not os.path.isfile(qerr):
            error_analysis_staple(d, data_file=None, toml_file=None, nskip=nskip, interval=a.interval,
                                  pathlengths=False, load_dir=None, h5=None, time_unit=1.0, recompute_taus=False,
                                  outdir=out, output=os.path.join(out, 'report.txt'), verbose=False,
                                  quiet_progress=True)
        r = subprocess.run([sys.executable, SCRIPT, d, '--nskip', str(nskip), '--q-errors', qerr, '--outdir', out],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
        else:
            print('   plots in', out, flush=True)


if __name__ == '__main__':
    main()
