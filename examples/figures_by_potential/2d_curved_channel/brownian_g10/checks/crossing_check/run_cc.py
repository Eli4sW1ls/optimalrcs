"""MSM crossing check on the curved-channel Brownian staple test run (2000 cycles), using the fit saved by
toy2d_staple_analysis.ipynb, the exact committor, and deliberately shifted copies of the exact committor
(+0.03 and +0.06 where 0.5 < q < 0.95) to show what the check detects."""
import glob, sys
import numpy as np
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/examples/')
import crossing_check as cc, toy2d
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/test_cc_br'
f = sorted(glob.glob(REPO + '/examples/figures_by_potential/2d_curved_channel/brownian_g10/staple_test/committor_*.npz'))[-1]
d = np.load(f)
run = toy2d.read_full_settings(SIM); intf = run['interfaces']; pot = toy2d.potential_from_settings(run)
y, x = d['X_all'][:, 0].astype(float), d['X_all'][:, 1].astype(float)
ref = toy2d.exact_committor_brownian(pot, run['temperature'], run['friction'], run['dt'], intf[0], intf[-1])
q_exact = toy2d.interp_grid(ref, y, x)
plateau = (q_exact > 0.5) & (q_exact < 0.95)
q_shift3 = np.clip(q_exact + 0.03 * plateau, 0, 1)
q_shift6 = np.clip(q_exact + 0.06 * plateau, 0, 1)
cr = cc.crossings(y, d['i_traj'], d['group_traj'], d['path_weights'], intf)
print(f'{f.split("/")[-1]}: {len(y)} frames, {len(cr["start"])} crossings; interfaces {intf}')
for weighting in ('staple',):
    msm = cc.istar_msm(SIM, intf, weighting)
    res = cc.compare(cr, msm, {'exact': q_exact, 'fit': d['q_fit'], 'exact+0.03': q_shift3, 'exact+0.06': q_shift6}, n_boot=200)
    last = 2000; blocks = []
    for b in range(4):
        blocks.append(cc.compare(cr, cc.istar_msm(SIM, intf, weighting, cycles=(b * last // 4, (b + 1) * last // 4)), {}, n_boot=0)['est']['MSM'])
    berr = np.std(blocks, axis=0) / np.sqrt(3)
    print(f'=== {weighting} weights, P_cross {msm["pcross"]:.3e}')
    print(' m  dir lambda    n   MSM +- err    exact-MSM   fit-MSM   (exact+0.03)-MSM  (exact+0.06)-MSM   [+- combined error]')
    for c, (m, dd) in enumerate(res['cells']):
        e = res['est']; tot = np.sqrt(res['diff_err']['exact'][c] ** 2 + berr[c] ** 2)
        print(f'{m:2d} {"up" if dd > 0 else "dn":>3} {intf[m]:5.2f} {res["count"][c]:5d}  {e["MSM"][c]:.3f} +- {berr[c]:.3f}   '
              f'{e["exact"][c] - e["MSM"][c]:+.3f}     {e["fit"][c] - e["MSM"][c]:+.3f}     {e["exact+0.03"][c] - e["MSM"][c]:+.3f}            '
              f'{e["exact+0.06"][c] - e["MSM"][c]:+.3f}     [+-{tot:.3f}]')
    for k in ('exact', 'fit', 'exact+0.03', 'exact+0.06'):
        diff = res['est'][k] - res['est']['MSM']
        err = np.sqrt(res['diff_err'][k] ** 2 + berr ** 2)
        chi2 = np.sum((diff / err) ** 2)
        print(f'  {k:11s}: RMS(q - MSM) over cells {np.sqrt(np.mean(diff ** 2)):.4f}, chi2 {chi2:.1f} for {len(diff)} cells')
