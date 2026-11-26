#!/usr/bin/env python3
"""Path lengths and shooting-region lengths per ensemble of finished runs of one loop, side by side:

    python -m loop.plot_lengths <loop config> <run sub-dir>[=label] ...
    e.g. python -m loop.plot_lengths loop_z_dp production="9 interfaces" production_refill="12 interfaces"

Per run (accepted paths, unweighted counts, at most 6000 paths spread over the run):
  - path length (frames) of the paths in each ensemble (a path counts in every ensemble whose W column is > 0);
  - shooting-region frames: per path of ensemble e >= 2, the frames strictly inside (lambda_{e-2}, lambda_e) in the
    runs whose excursion crosses lambda_{e-1} (seed.window_cores, the region the shooting points are drawn from);
  - the share of all stored path frames each ensemble holds (where the MD work goes).
Writes <loop dir>/figures/path_lengths.png and prints a table per run."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from loop import controller, seed as seed_mod  # noqa: E402
from loop.backends import infretis_backend as be  # noqa: E402

COLS = ['#2a6fdb', '#e8590c', '#1baf7a', '#c2255c']


def per_ensemble(run_dir, max_paths=6000):
    raw = be.raw_paths(run_dir, max_paths=max_paths)
    lv = [float(v) for v in raw['settings']['interfaces']]
    n_ens = len(lv)
    L = np.array([len(o) for o in raw['ops']])
    W = np.asarray(raw['W'])
    out = []
    for e in range(1, n_ens):
        sel = np.flatnonzero(W[:, e] > 0)
        reg = []
        if e >= 2:
            lo, mid, hi = lv[e - 2], lv[e - 1], lv[e]
            for i in sel:
                lam = raw['ops'][i][:, 0]
                reg.append(sum(b - a + 1 for a, b in seed_mod.window_cores(lam, lo, mid, hi)))
        out.append(dict(e=e, lam_m=lv[e - 1] if e >= 2 else lv[0], window=(lv[max(e - 2, 0)], lv[e]),
                        lengths=L[sel], region=np.array(reg), frames=int(L[sel].sum())))
    tot = sum(r['frames'] for r in out)
    for r in out:
        r['share'] = r['frames'] / tot if tot else np.nan
    return lv, out


def main():
    cfg = controller.load_loop_config(sys.argv[1])
    runs = []
    for arg in sys.argv[2:]:
        sub, _, label = arg.partition('=')
        d = os.path.join(cfg['loop_dir'], sub)
        if not os.path.isfile(os.path.join(d, 'infretis_data.txt')):
            d = os.path.join(d, 'run')
        lv, rows = per_ensemble(d)
        runs.append((label or sub, lv, rows))
        print(f'\n{label or sub}: {len(lv)} interfaces {[round(v, 3) for v in lv]}')
        print('  ens  window            paths  length median / p90   region median / p5 / p95   share of frames')
        for r in rows:
            rg = r['region']
            print(f'  {r["e"]:3d}  {r["window"][0]:.3f}-{r["window"][1]:.3f}  {len(r["lengths"]):6d}  '
                  f'{np.median(r["lengths"]):8.0f} / {np.percentile(r["lengths"], 90):6.0f}   '
                  + (f'{np.median(rg):8.0f} / {np.percentile(rg, 5):4.0f} / {np.percentile(rg, 95):5.0f}' if len(rg) else ' ' * 26)
                  + f'   {r["share"]:.3f}')
    fig, axs = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
    width = 0.012
    for j, (label, lv, rows) in enumerate(runs):
        c = COLS[j % len(COLS)]
        off = (j - (len(runs) - 1) / 2) * width * 1.1
        for ax, key in ((axs[0], 'lengths'), (axs[1], 'region')):
            data = [(r['lam_m'] + off, r[key]) for r in rows if len(r[key])]
            bp = ax.boxplot([np.maximum(d, 0.5) for _, d in data], positions=[p for p, _ in data], widths=width,
                            whis=(5, 95), showfliers=False, patch_artist=True, manage_ticks=False)
            for b in bp['boxes']:
                b.set(facecolor=c, alpha=0.35, edgecolor=c)
            for k in ('whiskers', 'caps', 'medians'):
                for ln in bp[k]:
                    ln.set(color=c)
        axs[2].bar([r['lam_m'] + off for r in rows], [r['share'] for r in rows], width=width, color=c, alpha=0.7,
                   label=label)
        for ax in axs:
            for v in lv:
                ax.axvline(v, color=c, lw=0.6, ls=':', alpha=0.7)
        axs[0].plot([], [], color=c, lw=6, alpha=0.5, label=label)
    axs[0].set(yscale='log', ylabel='path length [frames]',
               title='accepted path length per ensemble (box: quartiles, whiskers: 5-95%), at the ensemble\'s middle interface')
    axs[1].set(yscale='log', ylabel='shooting-region frames',
               title='frames inside the window (lambda_{i-1}, lambda_{i+1}) of the runs crossing lambda_i, per path (ensembles >= 2)')
    axs[2].set(ylabel='share of path frames', xlabel='lambda (middle interface of the ensemble; [0+] at lambda_A)',
               title='where the stored path frames are: share per ensemble')
    for ax in axs:
        ax.grid(color='#e6e5e1', lw=0.5)
    axs[0].legend(fontsize=8)
    fig.suptitle(f'{cfg["name"]}: path and shooting-region lengths per ensemble (dotted: interfaces)', x=0.01,
                 ha='left', fontsize=10)
    fig.tight_layout()
    out = os.path.join(cfg['loop_dir'], 'figures')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'path_lengths.png')
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == '__main__':
    main()
