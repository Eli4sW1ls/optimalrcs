import cupy as cp
import numpy as np


def npq(r_traj, fk, i_traj=None, w_traj=None):
    """ implements NPq (non-parametric committor optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = (i_traj[1:] == i_traj[:-1]).astype(r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    # A basis with transition_moments/apply (polybasis.PolyBasisRY, from
    # lazy_basis_poly_ry) is never materialized: akj, b and the update are
    # computed frame by frame in fused kernels, with the same sums as below.
    if hasattr(fk, "transition_moments"):
        akj, b = fk.transition_moments(itw, -(r_traj[1:] - r_traj[:-1]) * itw, equilibrium=True)
        al_j = cp.linalg.lstsq(akj, cp.reshape(b, [b.shape[0], 1]), rcond=None)[0]
        return fk.apply(r_traj, cp.reshape(al_j, [al_j.shape[0]]), 0, 1)

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = cp.tensordot(dfk * itw, dfk, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1]
    b = cp.tensordot(dfk, -delta_r * itw, 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + cp.tensordot(al_j, fk, 1)
    rn_traj = cp.clip(rn_traj, 0, 1)
    return rn_traj



def npqsoft(r_traj, fk, ia_traj, ib_traj, lmbd_a, lmbd_b, i_traj=None, w_traj=None):
    """ implements NPlmbd (non-parametric committor optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = (i_traj[1:] == i_traj[:-1]).astype(r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]

    akj = cp.tensordot(dfk * itw, dfk, axes=[1, 1])
    akj = akj + cp.tensordot(fk * (lmbd_a * ia_traj + lmbd_b * ib_traj), fk, axes=[1, 1])

    delta_r = -(r_traj[1:] - r_traj[:-1])

    b = cp.tensordot(dfk, delta_r * itw, 1)
    b = b + cp.tensordot(fk, (lmbd_b * ib_traj * (1 - r_traj) - r_traj * lmbd_a * ia_traj), 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + cp.tensordot(al_j, fk, 1)
    rn_traj = cp.clip(rn_traj, 0, 1)
    return rn_traj



def npt(r_traj, fk, i_traj=None, w_traj=None):
    """ implements NPt (non-parametric mfpt optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a re-weighting factor,
        to use with multiple short trajectories. Default value is 1.
    """

    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = (i_traj[1:] == i_traj[:-1]).astype(r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = cp.tensordot(dfk * itw, dfk, axes=[1, 1])

    delta_r = -(r_traj[1:] - r_traj[:-1])
    b = cp.tensordot(dfk, delta_r * itw, 1) + 2 * cp.sum(fk, 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + cp.tensordot(al_j, fk, 1)
    return rn_traj


# Transitions per block when accumulating akj/b (the original cupy default).
npneq_chunk = 1024 * 64 + 1


# Opt-in conventions for the npneq family. The defaults reproduce the original
# cupy implementation exactly; both alternatives are what the TensorFlow version
# (optimalrcs/nonparametrics.py) uses.
#
# gamma_all_frames=False (original): akj = sum fa (fb - fa (1 + gamma)) itw, i.e.
#     the gamma regularization is only applied to transitions with itw > 0.
#     POSSIBLE_BUG: transitions across trajectory ends, masked out by train_mask,
#     or with zero path weight then carry no regularization, unlike npnet / TF.
# gamma_all_frames=True: akj = sum fa (fb itw - fa (itw + gamma)), as in npnet and
#     TF. Use it to match the TensorFlow results, or when training on many short
#     trajectories (TIS) with gamma > 0. Mathematically identical to the default
#     when gamma == 0, or when every itw is 1.
#
# stable_negative=False (original): with stable=True, akj = +sum fa fa itw.
#     POSSIBLE_BUG: the TF version, and an earlier (shadowed) definition of npneq in
#     this file, use -sum fa fa itw, so this flips the sign of the update.
# stable_negative=True: akj = -sum fa fa itw, as in TF. Use it with stable=True to
#     match the TensorFlow results. Irrelevant when stable=False.


def _npneq_chunks(n_trans, step):
    """[s, e) blocks of transitions, as cut by the original cupy npneq: the block
    size is made odd and an even-sized last block has its last transition split
    off. Only the summation order depends on this, kept for exact reproducibility."""
    if step == 0: step = n_trans
    if step % 2 == 0: step -= 1  # to keep even the chunks
    s = 0
    while s < n_trans:
        e = min(s + step, n_trans)
        if e == n_trans and e - s > 2 and (e - s) % 2 == 0: e = e - 1
        yield s, e
        s = e


def _npneq_moments(fa, fb, itw, delta_r, gamma, stable, gamma_all_frames=False, stable_negative=False):
    """akj/b contributions of one block of consecutive transitions.

    fa/fb are the basis functions at the start/end frames of each transition.
    Kept separate from `npneq` so that the (n_basis, block) temporaries stay
    bounded instead of scaling with the full trajectory length.
    """
    if stable:
        akj = cp.tensordot(fa, fa * itw, axes=[1, 1])  # POSSIBLE_BUG: sign, see stable_negative
        if stable_negative:
            akj = -akj
    elif gamma_all_frames:
        akj = cp.tensordot(fa, fb * itw - fa * (itw + gamma), axes=[1, 1])
    else:
        akj = cp.tensordot(fa, (fb - fa * (1 + gamma)) * itw, axes=[1, 1])  # POSSIBLE_BUG: see gamma_all_frames
    return akj, cp.tensordot(fa, -delta_r * itw, 1)


def _npneq_accumulate(r_traj, basis, itw, gamma, stable, step, akj=None, b=None,
                      gamma_all_frames=False, stable_negative=False):
    """Sums the akj/b moments over all transitions, block by block."""
    n_trans = r_traj.shape[0] - 1
    n_basis = basis(0, 1).shape[0]
    if akj is None:
        akj = cp.zeros((n_basis, n_basis), dtype=r_traj.dtype)
        b = cp.zeros(n_basis, dtype=r_traj.dtype)
    for s, e in _npneq_chunks(n_trans, step):
        f = basis(s, e + 1)
        akj_i, b_i = _npneq_moments(f[:, :-1], f[:, 1:], itw[s:e], r_traj[s + 1:e + 1] - r_traj[s:e],
                                    gamma, stable, gamma_all_frames, stable_negative)
        akj += akj_i
        b += b_i
    return akj, b


def _npneq_solve(akj, b):
    al_j = cp.linalg.lstsq(akj, cp.reshape(b, [b.shape[0], 1]), rcond=None)[0]
    return cp.reshape(al_j, [al_j.shape[0]])


def _npneq_update(r_traj, fk, al_j, step):
    """r + sum_j al_j f_j, clipped to [0, 1]. A basis array is contracted in one go,
    as in the original; a callable basis is evaluated block by block."""
    if not callable(fk):
        return cp.clip(r_traj + cp.tensordot(al_j, fk, 1), 0, 1)
    n_frames = r_traj.shape[0]
    rn_traj = r_traj.copy()
    for s in range(0, n_frames, step):
        e = min(s + step, n_frames)
        rn_traj[s:e] += cp.tensordot(al_j, fk(s, e), 1)
    return cp.clip(rn_traj, 0, 1, out=rn_traj)


def _npneq_itw(r_traj, i_traj, train_mask, w_traj=None, weight=None):
    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = cp.asarray(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)
    if weight is not None:
        itw = itw * weight
    if train_mask is not None:
        itw = itw * train_mask
    if w_traj is not None:
        itw = itw * cp.asarray(w_traj, dtype=r_traj.dtype)
    return itw


def _as_basis(fk):
    return fk if callable(fk) else (lambda s, e: fk[:, s:e])


def npneq(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None,
          chunk=None, w_traj=None, gamma_all_frames=False, stable_negative=False):
    """ implements NPNEq (non-parametric non-equilibrium committor
    optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    It is the trajectory indicator function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    w_traj is an optional weight of each transition X(i) -> X(i+1), e.g. the
        Monte Carlo weight of the path it belongs to. Default value is 1.
    gamma_all_frames, stable_negative: opt-in conventions, see the comment above
        `_npneq_chunks`. The defaults reproduce the original cupy implementation.

    akj and b are sums over transitions, so they are accumulated block by
    block of `chunk` transitions (default `npneq_chunk`).

    fk is either the (n_basis, N) basis array or a callable fk(s, e) returning
    its columns s..e-1; the callable form never materializes the full basis.
    A basis with `transition_moments`/`apply` (polybasis.PolyBasisRY) is never
    materialized at all: akj, b and the update are computed frame by frame.
    """
    itw = _npneq_itw(r_traj, i_traj, train_mask, w_traj)
    if hasattr(fk, "transition_moments"):
        akj, b = fk.transition_moments(itw, -(r_traj[1:] - r_traj[:-1]) * itw, gamma, stable,
                                       gamma_all_frames=gamma_all_frames, stable_negative=stable_negative)
        return fk.apply(r_traj, _npneq_solve(akj, b), 0, 1)

    step = chunk or npneq_chunk
    akj, b = _npneq_accumulate(r_traj, _as_basis(fk), itw, gamma, stable, step,
                               gamma_all_frames=gamma_all_frames, stable_negative=stable_negative)
    return _npneq_update(r_traj, fk, _npneq_solve(akj, b), step)

def lag_transitions(boundary, i_traj=None, delta_t=1, stride=1):
    """Start and end frames (t0, t1) of the lag-delta_t transitions, for `npneq_lag`.

    Starts: frames outside the boundary states whose offset from the first frame of
    their trajectory is a multiple of `stride` (stride = delta_t: the transitions of
    the trajectory subsampled every delta_t frames). End: t1 = min(t0 + delta_t, first
    boundary frame after t0), so a transition stops at the first entry into A or B, also
    between subsampled frames (stopped increment; the committor is a martingale at every
    lag). Transitions that run past the end of their trajectory without reaching a
    boundary are dropped. Returns two sorted int64 cupy arrays.
    """
    b = np.asarray(cp.asnumpy(boundary), dtype=bool)
    n = len(b)
    it = np.zeros(n) if i_traj is None else np.asarray(cp.asnumpy(i_traj))
    idx = np.arange(n)
    first = np.r_[True, it[1:] != it[:-1]]
    last = np.r_[it[1:] != it[:-1], True]
    start = np.maximum.accumulate(np.where(first, idx, 0))                # first frame of each frame's trajectory
    end = np.minimum.accumulate(np.where(last, idx, n)[::-1])[::-1]        # last frame of each frame's trajectory
    nxt = np.r_[np.minimum.accumulate(np.where(b, idx, n)[::-1])[::-1], n]  # first boundary frame >= j
    t0 = idx[((idx - start) % stride == 0) & ~b]
    t1 = np.minimum(t0 + delta_t, nxt[t0 + 1])
    keep = t1 <= end[t0]
    return cp.asarray(t0[keep]), cp.asarray(t1[keep])


def npneq_lag(r_traj, fk, t0, t1, gamma=0, w_traj=None, chunk=None):
    """NPNEq iteration on the transitions t0 -> t1 of `lag_transitions` (lag delta_t):

        akj = sum_t w_t fk(t0) (fj(t1) - (1 + gamma) fj(t0)),   b_k = -sum_t w_t fk(t0) (r(t1) - r(t0)),

    the lag-1 `npneq` (default conventions) with t + 1 replaced by t1. w_traj: optional
    weight per FRAME (e.g. the path weight), each transition gets the weight of t0.
    fk: basis array, callable fk(s, e) or polybasis.PolyBasisRY (used as a callable); it is
    evaluated in blocks of about `chunk` frames (default `npneq_chunk`).
    """
    basis = _as_basis(fk)
    step = chunk or npneq_chunk
    n_basis = basis(0, 1).shape[0]
    akj = cp.zeros((n_basis, n_basis), dtype=r_traj.dtype)
    b = cp.zeros(n_basis, dtype=r_traj.dtype)
    t0h = cp.asnumpy(t0)
    for lo_frame in range(0, r_traj.shape[0], step):
        s, e = np.searchsorted(t0h, [lo_frame, lo_frame + step])
        if e == s:
            continue
        i0, i1 = t0[s:e], t1[s:e]
        lo, hi = int(t0h[s]), int(i1.max()) + 1
        f = basis(lo, hi)
        fa, fb = f[:, i0 - lo], f[:, i1 - lo]
        w = 1.0 if w_traj is None else w_traj[i0]
        akj += cp.tensordot(fa, (fb - fa * (1 + gamma)) * w, axes=[1, 1])
        b += cp.tensordot(fa, -(r_traj[i1] - r_traj[i0]) * w, 1)
    return _npneq_update(r_traj, fk, _npneq_solve(akj, b), step)


def npneq_(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk_size=None,
           gamma_all_frames=False, stable_negative=False):
    """ as `npneq`, but also returns the expansion coefficients al_j (on the host). """
    step = chunk_size or npneq_chunk
    itw = _npneq_itw(r_traj, i_traj, train_mask)
    akj, b = _npneq_accumulate(r_traj, _as_basis(fk), itw, gamma, stable, step,
                               gamma_all_frames=gamma_all_frames, stable_negative=stable_negative)

    al_j = _npneq_solve(akj, b)
    return _npneq_update(r_traj, fk, al_j, step), al_j.get()

def npneq_2(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk_size=None, _=None, weight=1,
            gamma_all_frames=False, stable_negative=False):
    """ as `npneq`, but the transitions are weighted by `weight` and the akj/b
    moments are added to previously accumulated ones `_` = (akj, b, ...).

    returns the new RC and (akj, b, al_j) on the host.
    """
    step = chunk_size or npneq_chunk
    itw = _npneq_itw(r_traj, i_traj, train_mask, weight=weight)
    if _ is None:
        akj, b = None, None
    else:
        akj, b = cp.asarray(_[0]), cp.asarray(_[1])
    akj, b = _npneq_accumulate(r_traj, _as_basis(fk), itw, gamma, stable, step, akj, b,
                               gamma_all_frames=gamma_all_frames, stable_negative=stable_negative)

    al_j = _npneq_solve(akj, b)
    return _npneq_update(r_traj, fk, al_j, step), (akj.get(), b.get(), al_j.get())


def npneq_dt(r_traj, fk, i_traj, future_boundary, gamma=0, dt=1, gamma_all_frames=False):
    """ implements NPNEq (non-parametric non-equilibrium committor
    optimization) iteration.
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    it = (i_traj[dt:] == i_traj[:-dt]).astype(r_traj.dtype)
    not_crossed = (cp.logical_or(future_boundary.index[:-dt] == -1,
                                  future_boundary.delta_t[:-dt] > dt)).astype(r_traj.dtype)

    if gamma_all_frames:  # see the comment above `_npneq_chunks`
        delta_fj = fk[:, dt:] * not_crossed * it - fk[:, :-dt] * (it + gamma)
    else:  # POSSIBLE_BUG: gamma only on transitions within a trajectory, see gamma_all_frames
        delta_fj = (fk[:, dt:] * not_crossed - fk[:, :-dt] * (1 + gamma)) * it
    akj = cp.tensordot(fk[:, :-dt], delta_fj, axes=[1, 1])

    r_plus = cp.where(cp.logical_and(future_boundary.index[:-dt] > -1, future_boundary.delta_t[:-dt] <= dt),
                      future_boundary.r[:-dt], r_traj[dt:])
    delta_r = r_plus - r_traj[:-dt]
    b = cp.tensordot(fk[:, :-dt], -delta_r * it, 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + cp.tensordot(al_j, fk, 1)
    rn_traj = cp.clip(rn_traj, 0, 1)
    return rn_traj



def npnew(r, fk, it):
    """ implements NPNEw (non-parametric non-equilibrium re-weighting factors
    optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indicator function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """

    dfk = fk[:, 1:] - fk[:, :-1]

    b = -cp.tensordot(dfk * it, r[:-1], 1)
    b = cp.reshape(b, [b.shape[0], 1])
    scale = cp.sum(1 - r[:-1] * it)
    scale = cp.reshape(scale, [1, 1])
    b = cp.concatenate((b, scale), 0)

    ones = cp.reshape(it, [1, it.shape[0]])
    dfk = cp.concatenate((dfk * it, ones), 0)
    akj = cp.tensordot(dfk, fk[:, :-1], axes=[1, 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn = r + cp.tensordot(al_j, fk, 1)
    rn = abs(rn)

    return rn



def npnet(r_traj, fk, t_traj, i_traj, gamma=0, t_max=1e10, subsample=None, train_mask=None):
    """ implements NPNEt (non-parametric non-equilibrium mfpt
    optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = (i_traj[1:] == i_traj[:-1]).astype(r_traj.dtype)

    if train_mask is not None:
        itw = itw * train_mask

    if hasattr(fk, "transition_moments"):  # never materialized, see npneq
        delta_r = r_traj[1:] - r_traj[:-1] + t_traj[1:] - t_traj[:-1]
        akj, b = fk.transition_moments(itw, -delta_r * itw, gamma, gamma_all_frames=True)
        al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
        if subsample is None:
            return fk.apply(r_traj, al_j, 0, t_max)
        rn = fk.apply(r_traj, al_j, -cp.inf, cp.inf)
        k = int(len(rn) / subsample)
        t_max = min(t_max, float(cp.min(cp.max(cp.reshape(rn[:k * subsample], [subsample, k]), 1))))
        return cp.clip(rn, 0, t_max)

    dfj = fk[:, 1:] * itw - fk[:, :-1] * (itw + gamma)
    

    akj = cp.tensordot(fk[:, :-1], dfj, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1] + t_traj[1:] - t_traj[:-1]

    b = cp.tensordot(fk[:, :-1], -delta_r * itw, 1)
    b = cp.reshape(b, [b.shape[0], 1])

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    al_j = cp.reshape(al_j, [al_j.shape[0]])

    rn = r_traj + cp.tensordot(al_j, fk, 1)
    if subsample is not None:
        k = int(len(rn) / subsample)
        t_max = min(t_max, float(cp.min(cp.max(cp.reshape(rn[:k * subsample], [subsample, k]), 1))))
    rn = cp.clip(rn, 0, t_max)
    return rn
