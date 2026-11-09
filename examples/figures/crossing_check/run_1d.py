"""Validation of the MSM crossing check on the 1D flat staple run (memory loss exact in 1D)."""
import argparse, contextlib, io, sys
import numpy as np
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/examples/')
import tis_data, crossing_check as cc
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istar0br10intf'
LA, LB = -0.1, 0.8
p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args([SIM, '--reversed', '--paths-per-ensemble', '1000', '--seed', '0',
                                               '--lambda-a', str(LA), '--lambda-b', str(LB)]))
    intf = tis_data.read_interfaces(SIM)
shift = 0.5826 * np.sqrt(2 * 2e-4)
q_exact = np.clip((data.lam - LA + shift) / (LB - LA + 2 * shift), 0, 1)
print('interfaces', intf)
cr = cc.crossings(data.lam, data.i_traj, data.group_traj, data.path_weights, intf)
for weighting in ('staple', 'rejections'):
    msm = cc.istar_msm(SIM, intf, weighting)
    print(f'\n=== {weighting}: P_cross {msm["pcross"]:.4e} ===')
    print('q_t matrix:\n', np.array2string(np.asarray(msm['q_t']), precision=3, suppress_small=True, max_line_width=200))
    print('z:', np.round(msm['z'], 3))
    res = cc.compare(cr, msm, {'exact': q_exact}, n_boot=50)
    for c, (m, d) in enumerate(res['cells']):
        print(f'  m {m} {"up" if d > 0 else "dn"} lam {intf[m]:5.2f} n {res["count"][c]:5d}  MSM {res["est"]["MSM"][c]:.3f}  exact {res["est"]["exact"][c]:.3f}')
