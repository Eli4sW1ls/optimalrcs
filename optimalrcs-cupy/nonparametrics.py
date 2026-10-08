import cupy as cp


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


npneq_chunk = 1_000_000


def _npneq_moments(fa, fb, itw, delta_r, gamma, stable):
    """akj/b contributions of one block of consecutive transitions.

    fa/fb are the basis functions at the start/end frames of each transition.
    Kept separate from `npneq` so that the (n_basis, block) temporaries stay
    bounded instead of scaling with the full trajectory length.
    """
    if stable:
        akj = -cp.matmul(fa, (fa * itw).T)
    else:
        akj = cp.matmul(fa, (fb * itw - fa * (itw + gamma)).T)
    return akj, cp.matmul(fa, -delta_r * itw)


def _npneq_accumulate(r_traj, basis, itw, gamma, stable, step, akj=0, b=0):
    """Sums the akj/b moments over all transitions, block by block."""
    n_frames = r_traj.shape[0]
    delta_r = r_traj[1:] - r_traj[:-1]
    for s in range(0, n_frames - 1, step):
        e = min(s + step, n_frames - 1)
        f = basis(s, e + 1)
        akj_i, b_i = _npneq_moments(f[:, :-1], f[:, 1:], itw[s:e],
                                    delta_r[s:e], gamma, stable)
        akj = akj + akj_i
        b = b + b_i
    return akj, b


def _npneq_update(r_traj, basis, al_j, step):
    """r + sum_j al_j f_j, clipped to [0, 1], built block by block."""
    n_frames = r_traj.shape[0]
    rn_traj = r_traj.copy()
    for s in range(0, n_frames, step):
        e = min(s + step, n_frames)
        rn_traj[s:e] += cp.matmul(al_j, basis(s, e))
    return cp.clip(rn_traj, 0, 1, out=rn_traj)


def _npneq_itw(r_traj, i_traj, train_mask):
    if i_traj is None:
        itw = cp.ones_like(r_traj[:-1])
    else:
        itw = cp.asarray(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)
    if train_mask is not None:
        itw = itw * train_mask
    return itw


def _as_basis(fk):
    return fk if callable(fk) else (lambda s, e: fk[:, s:e])


def npneq(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None,
          chunk=None):
    """ implements NPNEq (non-parametric non-equilibrium committor
    optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Ib is the boundary indicator function:
        Ib(i)=1 when X(i) belongs to the boundary states and 0 otherwise
    It is the trajectory indicator function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory

    akj and b are sums over transitions, so they are accumulated block by
    block: the full-length intermediates this would otherwise allocate
    dominate peak memory for multi-million-frame trajectories.

    fk is either the (n_basis, N) basis array or a callable fk(s, e) returning
    its columns s..e-1; the callable form never materializes the full basis.
    A basis with `transition_moments`/`apply` (polybasis.PolyBasisRY) is never
    materialized at all: akj, b and the update are computed frame by frame.
    """
    itw = _npneq_itw(r_traj, i_traj, train_mask)
    if hasattr(fk, "transition_moments"):
        akj, b = fk.transition_moments(itw, -(r_traj[1:] - r_traj[:-1]) * itw, gamma, stable)
        al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
        return fk.apply(r_traj, al_j, 0, 1)

    basis = _as_basis(fk)
    step = chunk or npneq_chunk
    akj, b = _npneq_accumulate(r_traj, basis, itw, gamma, stable, step)

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    return _npneq_update(r_traj, basis, al_j, step)

def npneq_(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk_size=None):
    """ as `npneq`, but also returns the expansion coefficients al_j (on the host). """
    basis = _as_basis(fk)
    step = chunk_size or npneq_chunk
    itw = _npneq_itw(r_traj, i_traj, train_mask)
    akj, b = _npneq_accumulate(r_traj, basis, itw, gamma, stable, step)

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    return _npneq_update(r_traj, basis, al_j, step), al_j.get()

def npneq_2(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk_size=None, _=None, weight=1):
    """ as `npneq`, but the transitions are weighted by `weight` and the akj/b
    moments are added to previously accumulated ones `_` = (akj, b, ...).

    returns the new RC and (akj, b, al_j) on the host.
    """
    basis = _as_basis(fk)
    step = chunk_size or npneq_chunk
    itw = _npneq_itw(r_traj, i_traj, train_mask) * weight
    if _ is None:
        akj, b = 0, 0
    else:
        akj, b = cp.asarray(_[0]), cp.asarray(_[1])
    akj, b = _npneq_accumulate(r_traj, basis, itw, gamma, stable, step, akj, b)

    al_j = cp.linalg.lstsq(akj, b, rcond=None)[0]
    return _npneq_update(r_traj, basis, al_j, step), (akj.get(), b.get(), al_j.get())


def npneq_dt(r_traj, fk, i_traj, future_boundary, gamma=0, dt=1):
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

    delta_fj = fk[:, dt:] * not_crossed * it - fk[:, :-dt] * (it + gamma)
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
        akj, b = fk.transition_moments(itw, -delta_r * itw, gamma)
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
