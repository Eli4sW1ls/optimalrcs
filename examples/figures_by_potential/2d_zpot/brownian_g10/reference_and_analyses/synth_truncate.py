"""Cut the synthetic walker trajectories like path-sampling pieces: each walker stops at the first saved frame
outside a window of three interface intervals around its start ([l_{i-1}, l_{i+2}] for a start in
[l_i, l_{i+1}]), or at A/B. A stopping rule, so the data stay unbiased; only 'pieces end at interfaces' is new.
Writes synth/synth_trunc.npz with the same fields as synth_shoot.py."""
import numpy as np
HERE = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/'
d = np.load(HERE + 'synth/synth_30k_s50.npz')
it, t, y, x = d['i_traj'], d['t_traj'], d['y'], d['x']
lam = np.array([0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85])
n = len(y)
starts = np.r_[0, np.flatnonzero(it[1:] != it[:-1]) + 1]; ends = np.r_[starts[1:] - 1, n - 1]
i0 = np.clip(np.searchsorted(lam, y[starts]) - 1, 0, len(lam) - 2)
lo = lam[np.clip(i0 - 1, 0, None)]; hi = lam[np.clip(i0 + 2, None, len(lam) - 1)]
LO = np.repeat(lo, ends - starts + 1); HI = np.repeat(hi, ends - starts + 1)
out = (y <= LO) | (y >= HI) | (y <= 0.05) | (y >= 0.85)
out[starts] = False                                   # the start frame itself never stops the piece
first_out = np.minimum.reduceat(np.where(out, np.arange(n), n), starts)
stop = np.where(first_out < n, np.minimum(first_out, ends), ends)
keep = np.arange(n) <= np.repeat(stop, ends - starts + 1)
print(f'kept {keep.sum()} of {n} frames; pieces reaching A or B: {np.sum((y[stop] <= 0.05) | (y[stop] >= 0.85))} of {len(starts)}')
np.savez(HERE + 'synth/synth_trunc.npz', i_traj=it[keep], t_traj=t[keep], y=y[keep], x=x[keep], lambda_a=0.05, lambda_b=0.85, stride=50)
