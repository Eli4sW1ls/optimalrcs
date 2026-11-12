"""Crossing-probability (iSTAR MSM) check on the 2D Brownian Z staple run sim_istarz_br_g10.

Committor data: forward + mirrored pieces, rejection weights, 1500 paths per ensemble, seed 0 (the
data of istarz_br_g10_reversed). q: the exact committor, and the fit continued to 25000 iterations
(r_continued.npy, plateau +0.031). MSM: all cycles of the run (tistools), 'staple' and 'rejections'
weighting; its error from 4 blocks of MC cycles.
"""
import argparse, contextlib, io, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/examples/')
import tis_data, crossing_check as cc
OUT = os.path.dirname(os.path.abspath(__file__))
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10'
FIG = REPO + '/examples/figures_by_potential/'

p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args([SIM, '--reversed', '--paths-per-ensemble', '1500', '--seed', '0']))
    interfaces = tis_data.read_interfaces(SIM)
y, x = data.X_all[:, 0], data.X_all[:, 1]
ref = np.load(FIG + '2d_zpot/brownian_g10/reference_and_analyses/committor_reference_h0.001.npz')
q_exact = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)
q_fit = np.load(FIG + '2d_zpot/brownian_g10/staple/reversed/r_continued.npy')
assert len(q_fit) == len(y)
print(f'{len(y)} frames; interfaces {interfaces}', flush=True)

cr = cc.crossings(data.lam, data.i_traj, data.group_traj, data.path_weights, interfaces)
print(f'{len(cr["start"])} crossings with a known turn', flush=True)
del data

results = {}
for weighting in ('staple', 'rejections'):
    msm = cc.istar_msm(SIM, interfaces, weighting)
    res = cc.compare(cr, msm, {'exact': q_exact, 'fit': q_fit})
    # MSM error from 4 blocks of MC cycles (same crossings, MSM re-estimated per block)
    from tistools.reading import read_pathensemble
    last = int(read_pathensemble(os.path.join(SIM, '002', 'pathensemble.txt')).cyclenumbers[-1]) + 1
    blocks = []
    for b in range(4):
        mb = cc.istar_msm(SIM, interfaces, weighting, cycles=(b * last // 4, (b + 1) * last // 4))
        blocks.append(cc.compare(cr, mb, {}, n_boot=0)['est']['MSM'])
    res['msm_block_err'] = np.std(blocks, axis=0) / np.sqrt(len(blocks) - 1)
    res['pcross'] = msm['pcross']
    results[weighting] = res
    print(f'\n=== MSM weighting: {weighting}; P_cross(A->B) {msm["pcross"]:.3e} ===')
    print(' m  dir  lambda   n     MSM (+-block)        exact (+-boot)   fit (+-boot)    exact-MSM   fit-MSM')
    for c, (m, d) in enumerate(res['cells']):
        e, er = res['est'], res['err']
        print(f'{m:2d} {"up" if d > 0 else "dn":>4} {interfaces[m]:6.2f} {res["count"][c]:6d}  '
              f'{e["MSM"][c]:.3f} +- {res["msm_block_err"][c]:.3f}   {e["exact"][c]:.3f} +- {er["exact"][c]:.3f}   '
              f'{e["fit"][c]:.3f} +- {er["fit"][c]:.3f}   {e["exact"][c] - e["MSM"][c]:+.3f}     {e["fit"][c] - e["MSM"][c]:+.3f}', flush=True)

np.savez(f'{OUT}/z_check.npz', **{f'{w}_{k}': np.asarray(v) for w, r in results.items()
                                  for k, v in (('MSM', r['est']['MSM']), ('exact', r['est']['exact']), ('fit', r['est']['fit']),
                                               ('err_exact', r['err']['exact']), ('err_fit', r['err']['fit']),
                                               ('msm_block_err', r['msm_block_err']), ('count', r['count']))},
         cells=np.array(results['staple']['cells']), interfaces=interfaces)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
for col, d in enumerate((1, -1)):
    for w, r in results.items():
        sel = [c for c, (m, dd) in enumerate(r['cells']) if dd == d]
        lam = np.array([interfaces[r['cells'][c][0]] for c in sel])
        e, er = r['est'], r['err']
        ls = '-' if w == 'staple' else ':'
        axes[0, col].errorbar(lam, e['MSM'][sel], yerr=r['msm_block_err'][sel], fmt='s' + ls, color='k', ms=4,
                              label=f'MSM ({w} weights)')
        if w == 'staple':
            axes[0, col].errorbar(lam, e['exact'][sel], yerr=er['exact'][sel], fmt='o-', color='C0', ms=4, label='exact q')
            axes[0, col].errorbar(lam, e['fit'][sel], yerr=er['fit'][sel], fmt='^-', color='C3', ms=4, label='fit (plateau +0.03)')
        tot_ex = np.sqrt(r['diff_err']['exact'][sel] ** 2 + r['msm_block_err'][sel] ** 2)
        tot_fit = np.sqrt(r['diff_err']['fit'][sel] ** 2 + r['msm_block_err'][sel] ** 2)
        axes[1, col].errorbar(lam, e['exact'][sel] - e['MSM'][sel], yerr=tot_ex, fmt='o' + ls, color='C0', ms=4,
                              label=f'exact $-$ MSM ({w})')
        axes[1, col].errorbar(lam + 0.006, e['fit'][sel] - e['MSM'][sel], yerr=tot_fit, fmt='^' + ls, color='C3', ms=4,
                              label=f'fit $-$ MSM ({w})')
    axes[0, col].set(title=f'{"upward" if d > 0 else "downward"} crossings: mean q at the crossing points', ylabel='$q$')
    axes[1, col].axhline(0, color='k', lw=0.8)
    axes[1, col].set(xlabel='interface $\\lambda_m$', ylabel='difference')
    for ax in axes[:, col]:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/z_check.png', dpi=110)
print('saved', OUT)
