"""Synthetic path data with a known committor for the Brownian Z potential: N walkers started uniformly in
the region between A and B, run with the toytis overdamped update (x += dt/gamma F + N(0, sigma), force as
in rate_walkers.py, checked against the potential class) until they reach A (y <= 0.05) or B (y >= 0.85),
checked every step. Positions saved every STRIDE steps, plus the absorbing frame. Each walker is one
segment: unbiased dynamics from an arbitrary starting distribution, the ideal input for NPNEq.
    python synth_shoot.py N_WALKERS STRIDE MAX_STEPS OUT.npz
"""
import sys, time, numpy as np, cupy as cp
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses')
import importlib.util
spec = importlib.util.spec_from_file_location('rw', '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/rate_walkers.py')
src = open(spec.origin).read()
# reuse the checked force function without running the rate test
exec(src[:src.index('cp.random.seed(0)')].replace("NW = int(sys.argv[1]) if len(sys.argv) > 1 else 32768", "NW = 1")
     .replace("NS = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000", "NS = 1"))
LA, LB = 0.05, 0.85
N = int(sys.argv[1]); STRIDE = int(sys.argv[2]); MAX = int(sys.argv[3]); OUT = sys.argv[4]
cp.random.seed(12345)
Y = cp.random.uniform(0.07, 0.84, N); X = cp.random.uniform(0.0, 0.3, N)
alive = cp.arange(N)
rec_id, rec_t, rec_y, rec_x = [cp.arange(N)], [cp.zeros(N, dtype=cp.int64)], [Y.copy()], [X.copy()]
t0 = time.time()
Ya, Xa = Y.copy(), X.copy()
for step in range(1, MAX + 1):
    fy, fx = force(Ya, Xa)
    Ya = Ya + DT / GAMMA * fy + SIGMA * cp.random.standard_normal(Ya.size)
    Xa = Xa + DT / GAMMA * fx + SIGMA * cp.random.standard_normal(Xa.size)
    hit = (Ya <= LA) | (Ya >= LB)
    if step % STRIDE == 0 or bool(hit.any()):
        rec = hit | (step % STRIDE == 0)
        if bool(rec.any()):
            rec_id.append(alive[rec]); rec_t.append(cp.full(int(rec.sum()), step, dtype=cp.int64))
            rec_y.append(Ya[rec]); rec_x.append(Xa[rec])
        if bool(hit.any()):
            keep = ~hit
            alive, Ya, Xa = alive[keep], Ya[keep], Xa[keep]
            if alive.size == 0:
                break
    if step % 20000 == 0:
        print(f'step {step}: {alive.size} walkers left ({time.time() - t0:.0f} s)', flush=True)
ids = cp.asnumpy(cp.concatenate(rec_id)); ts = cp.asnumpy(cp.concatenate(rec_t))
ys = cp.asnumpy(cp.concatenate(rec_y)); xs = cp.asnumpy(cp.concatenate(rec_x))
order = np.lexsort((ts, ids))
ids, ts, ys, xs = ids[order], ts[order], ys[order], xs[order]
print(f'{len(ids)} frames from {N} walkers; {int(alive.size)} not absorbed after {step} steps; '
      f'{np.sum(ys >= LB)} reached B, {np.sum(ys <= LA)} reached A ({time.time() - t0:.0f} s)', flush=True)
np.savez(OUT, i_traj=ids, t_traj=ts, y=ys, x=xs, lambda_a=LA, lambda_b=LB, stride=STRIDE)
