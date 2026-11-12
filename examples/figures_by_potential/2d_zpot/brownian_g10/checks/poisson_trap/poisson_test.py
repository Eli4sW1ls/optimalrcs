import sys, numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
R='/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0,R+'/optimalrcs-cupy/'); import optimalrcs, cut_profiles, metrics_tis as mt
OUT=R+'/examples/figures_by_potential/2d_zpot/brownian_g10/checks/poisson_trap/'
d=np.load(R+'/examples/figures_by_potential/2d_zpot/brownian_g10/staple/reversed/committor.npz')
ref=np.load(R+'/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/committor_reference_h0.001.npz')
b0,b1=d['boundary0'],d['boundary1']
q=np.clip(RegularGridInterpolator((ref['y'],ref['x']),ref['q'],bounds_error=False,fill_value=None)(d['X_all']),0,1)
q[b0]=0; q[b1]=1
c=optimalrcs.CommittorNE(boundary0=b0,boundary1=b1,i_traj=d['i_traj'],t_traj=d['t_traj'],path_weights=d['path_weights'])
c.r_traj=cp.asarray(q); w=c.path_weights; trap=b0|b1
ends=np.r_[np.flatnonzero(np.diff(d['i_traj'])!=0),len(q)-1]; starts=np.r_[0,ends[:-1]+1]
L=ends-starts+1; tr=trap[ends]
print(f'{len(L)} pieces; end in A/B: {tr.mean():.3f}; mean length A/B-ending {L[tr].mean():.0f}, other {L[~tr].mean():.0f}',flush=True)
lags=[1,4,16,64,256,1024,4096]
def zq_comp(dt):
    lx,z=cut_profiles.comp_zq(c.r_traj,c.b_traj,c.i_traj,c.future_boundary,c.past_boundary,w_traj=w,dt=dt,nbins=200)
    return lx.get()[:-2], z.get()[:-1]
res={}
c.set_fixed_traj_length_trap(trap,2**16); res['fixed trap 2^16 (current)']=[zq_comp(dt) for dt in lags]
for s in (0,1):
    np.random.seed(s); c.set_poisson_traj_length_trap(trap)
    res[f'Poisson trap (draw {s})']=[zq_comp(dt) for dt in lags]
tau=mt.stop_index(c.i_traj,c.b_traj)
st=[]
for dt in lags:
    lx,z=mt.comp_zq_stopped(c.r_traj,c.b_traj,c.i_traj,w_traj=w,dt=dt,nbins=200,tau=tau); st.append((lx.get(),z.get()))
res['stopped (metrics_tis)']=st
zs=[float(cp.max(cp.abs(mt.comp_zq_stopped_zscore(c.r_traj,c.b_traj,c.i_traj,w_traj=w,dt=dt,nbins=200,tau=tau)[1][5:]))) for dt in lags]
print('stopped max|z| (Z/sigma) per lag:',' '.join(f'{z:.2f}' for z in zs))
print('\nmethod: max_q |Z(dt)| per lag', lags)
for k,v in res.items(): print(f'{k:28s}',' '.join(f'{np.abs(z).max():7.1f}' for _,z in v))
print('\nmethod: max_q |Z(dt) - Z(1)| per lag', lags[1:])
for k,v in res.items():
    z1=np.interp(v[0][0],v[0][0],v[0][1]); 
    print(f'{k:28s}',' '.join(f'{np.abs(np.interp(v[0][0],x,z)-z1).max():7.1f}' for x,z in v[1:]))
fig,ax=plt.subplots(1,len(res),figsize=(5*len(res),4.2),sharey=True)
for a,(k,v) in zip(ax,res.items()):
    for dt,(x,z) in zip(lags,v): a.plot(x,z,lw=1,label=f'{dt}')
    a.set_title(k,fontsize=10); a.grid(alpha=.3); a.set_xlabel('q (exact)')
ax[0].set_ylabel('$Z_q$ (path weights)'); ax[-1].legend(title='lag',fontsize=7)
fig.suptitle('2D Brownian Z, mirrored pieces (istarz_br_g10_reversed), exact committor: should be the same at every lag',fontsize=10)
fig.tight_layout(); fig.savefig(OUT+'zpot_exact_q.png',dpi=100); print('saved',OUT)
