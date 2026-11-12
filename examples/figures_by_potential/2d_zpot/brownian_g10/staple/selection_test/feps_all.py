"""plots_feps (path weights passed) of the five data-selection fits, rebuilt from their data and saved q.
    python feps_all.py <base|flat_qn|select_paths|blind_flat_y|blind_uniform>"""
import sys
import numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures_by_potential/2d_zpot/brownian_g10/staple/selection_test/'
name = sys.argv[1]
if name.startswith('blind_'):
    mode = name[len('blind_'):]
    src = open(HERE + 'blind_starts.py').read()
    sys.argv = ['blind_starts.py', mode, '3000', '1e7', '0']
    exec(src[:src.index('if MAX_ITER == 0:')])
    import optimalrcs
    d = np.load(HERE + f'{TAG}/fit.npz')
    assert len(d['r']) == len(y) and np.allclose(d['y'], y.astype(np.float32))
    q = optimalrcs.CommittorNE(boundary0=A, boundary1=B, i_traj=seg, t_traj=tt.astype(float), path_weights=w, group_traj=grp)
    out = HERE + f'{TAG}/feps.png'
else:
    src = open(HERE + 'selection_test.py').read()
    sys.argv = ['selection_test.py', 'fit', name]
    exec(src[:src.index("if sys.argv[1] == 'diag':")])
    w, keep = version(name)
    ix = np.flatnonzero(keep)
    d = {'r': np.load(HERE + f'{name}/r.npy')}
    assert len(d['r']) == len(ix)
    q = optimalrcs.CommittorNE(boundary0=A[ix], boundary1=B[ix], i_traj=it[ix], t_traj=t_traj[ix],
                               path_weights=w[ix], group_traj=grp[ix])
    q.set_fixed_traj_length_trap(b[ix], 2 ** 16)
    out = HERE + f'{name}/feps.png'
q.r_traj = cp.asarray(d['r'].astype(np.float64))
plt.show = lambda *a, **k: None
q.plots_feps(delta_t_sim=0.002)
fig = plt.gcf()
fig.suptitle(f'{name}: feps after 11000 iterations (path weights passed)', y=1.02)
fig.savefig(out, dpi=100, bbox_inches='tight')
print('saved', out, flush=True)
