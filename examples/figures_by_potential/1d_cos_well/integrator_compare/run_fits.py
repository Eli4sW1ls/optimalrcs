"""EM and BBK counterparts of the ASE fits in infretis_md_1d_langevin.ipynb (same settings), for the integrator comparison.

Data (first 2M frames, gamma 1, kT 1, dt 0.005, A: x <= -0.35, B: x >= 0.35):
    EM   langevin_md/fits/euler_md_wel_lgv_3M.npy                  exact: exact/q_exact_EM_on_md_frames.npy
    BBK  toytis simulations/md_istarwell2_100M/md_path.npy         exact: exact/q_exact_BBK_on_md_frames.npy
Fits (as the notebook: warm-up 1000 iterations at envelope 1, then envelope 0.01, gamma 0.1, ny 6, lazy basis):
    x_v        CVs x, v (recorded), no history, seeds 1 2 3, 10000 main iterations
    history_x  CV x, history [0, 1] ('y(t-d),y(t)'), seeds 1 2 3, 10000 main iterations
    sub20_hist_fine  CV x, history [0, 1], fit lag and stride 20 (delta_t=20, delta_t_stride=20), seed 1, 15000 main
    history_x5 CV x, history [0, 1, 2, 4], seed 1, 10000 main iterations
    history_ry CV x, history [0, 1], history_type drawn from ['y(t-d),y(t)', 'y(t-d),r(t-d)'], seed 2611, 25000 main
Output: <integrator>/<variant>/seed<s>/ (q_traj.npy, metrics.json, it<N>_{metrics,feps,obs_pred}.png), results.json.
Usage: python run_fits.py [--integrators EM BBK]
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.abspath(os.path.join(HERE, '../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(EX), 'optimalrcs-cupy'))
sys.path.insert(0, EX)
import cupy as cp
import optimalrcs
import seed_runs

ap = argparse.ArgumentParser()
ap.add_argument('--integrators', nargs='+', default=['EM', 'BBK'])
a = ap.parse_args()

WELL = os.path.dirname(HERE)
DATA = {'EM': (f'{WELL}/langevin_md/fits/euler_md_wel_lgv_3M.npy', f'{WELL}/exact/q_exact_EM_on_md_frames.npy'),
        'BBK': ('/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/md_istarwell2_100M/md_path.npy',
                f'{WELL}/exact/q_exact_BBK_on_md_frames.npy')}
N, LA, LB, DT = 2_000_000, -0.35, 0.35, 0.005
WARMUP, ENV_W, ENV, GAMMA, NY, CHUNK = 1000, 1.0, 0.01, 0.1, 6, 5000
FITS = [('x_v', [0, 1], None, 1, [1, 2, 3], 10000),
        ('history_x', [0], [0, 1], 1, [1, 2, 3], 10000),
        ('sub20_hist_fine', [0], [0, 1], 20, [1], 15000),
        ('history_x5', [0], [0, 1, 2, 4], 1, [1], 10000),
        ('history_ry', [0], [0, 1], 1, [2611], 25000)]   # new type, as the ASE notebook run (seed 2611, 26000 it in all)
HIST_TYPE = {'history_ry': ['y(t-d),y(t)', 'y(t-d),r(t-d)']}
res_file = f'{HERE}/results.json'
rows = json.load(open(res_file)) if os.path.exists(res_file) else []

for integ in a.integrators:
    md, ex = DATA[integ]
    d = np.load(md, mmap_mode='r')[:N]
    lam, v = np.array(d[:, 0], float), np.array(d[:, 1], float)
    q_exact = np.array(np.load(ex, mmap_mode='r')[:N], float)
    b0, b1 = lam <= LA, lam >= LB
    free = ~b0 & ~b1
    edges_l, edges_v = np.linspace(LA, LB, 71), np.linspace(*np.quantile(v[free], [0.002, 0.998]), 61)
    bm = seed_runs.BinnedMean((lam, v), (edges_l, edges_v), mask=free)
    ex_bin = bm(q_exact)
    for variant, cols, hist, sub, seeds, n_main in FITS:
        Y = cp.asarray(np.c_[lam, v][:, cols])
        kw = dict(ny=NY, gamma=GAMMA, print_step=1000, save_min_delta_zq=False, lazy_basis=True)
        if hist:
            kw.update(history_delta_t=list(hist), history_type=HIST_TYPE.get(variant, 'y(t-d),y(t)'), history_shift_type='r(t)')
        if sub > 1:
            kw.update(delta_t=sub, delta_t_stride=sub)
        for seed in seeds:
            if any(r['integrator'] == integ and r['variant'] == variant and r['seed'] == seed for r in rows):
                continue
            folder = f'{HERE}/{integ}/{variant}/seed{seed}'
            os.makedirs(folder, exist_ok=True)
            np.random.seed(seed); cp.random.seed(seed)
            q = optimalrcs.CommittorNE(boundary0=b0, boundary1=b1, i_traj=np.ones(N), t_traj=np.arange(N, dtype=float))
            comp_y = lambda: Y[:, np.random.randint(Y.shape[1])]
            t, done = time.time(), 0
            progress = []
            for n_it, env in [(WARMUP, ENV_W)] + [(CHUNK, ENV)] * (n_main // CHUNK):
                optimalrcs.envelope_scale = env
                q.fit_transform(comp_y, max_iter=n_it - 1, **kw)
                done += n_it
                r = cp.asnumpy(q.r_traj)
                rms_f = float(np.sqrt(np.mean((r[free] - q_exact[free]) ** 2)))
                rms_b = float(np.sqrt(np.nanmean((bm(r) - ex_bin) ** 2)))
                progress.append(dict(iter=done, rms_frame=rms_f, rms_bin=rms_b))
                seed_runs.save_and_show(q, folder, f'it{done}', ('metrics', 'feps', 'obs_pred'),
                                        dict(feps_kw=dict(delta_t_sim=DT)))
                plt.close('all')
                print(f'{integ} {variant} seed {seed}: {done} it, {(time.time() - t) / 60:.1f} min, '
                      f'RMS frame {rms_f:.4f} bin {rms_b:.4f}', flush=True)
            np.save(f'{folder}/q_traj.npy', r.astype(np.float32))
            mh = {k: [float(x) for x in vals] for k, vals in q.metrics_history.items()}
            json.dump(dict(metrics=mh, progress=progress), open(f'{folder}/metrics.json', 'w'))
            rows.append(dict(integrator=integ, variant=variant, seed=seed, iters=done, minutes=(time.time() - t) / 60,
                             rms_frame=rms_f, rms_bin=rms_b,
                             **{k: mh[k][-1] for k in ('cross_entropy', 'auc', 'max_sd_zq', 'max_grad_zq') if k in mh}))
            json.dump(rows, open(res_file, 'w'), indent=1)
            del q; cp.get_default_memory_pool().free_all_blocks()
print('DONE', flush=True)
