# Copyright (c) 2025 Sergei Krivov
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

import tensorflow as tf


@tf.function
def npq(r_traj, fk, i_traj=None, w_traj=None):
    """ implements NPq (non-parametric committor optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a rewighting factor,
        to use with multiple short trajectores. Default value is 1.
    """

    if i_traj is None:
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = tf.tensordot(dfk * itw, dfk, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1]
    b = tf.tensordot(dfk, -delta_r * itw, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    rn_traj = tf.clip_by_value(rn_traj, 0, 1)
    return rn_traj


@tf.function
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
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]

    akj = tf.tensordot(dfk * itw, dfk, axes=[1, 1])
    akj = akj + tf.tensordot(fk * (lmbd_a * ia_traj + lmbd_b * ib_traj), fk, axes=[1, 1])

    delta_r = -(r_traj[1:] - r_traj[:-1])

    b = tf.tensordot(dfk, delta_r * itw, 1)
    b = b + tf.tensordot(fk, (lmbd_b * ib_traj * (1 - r_traj) - r_traj * lmbd_a * ia_traj), 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    rn_traj = tf.clip_by_value(rn_traj, 0, 1)
    return rn_traj


@tf.function
def npt(r_traj, fk, i_traj=None, w_traj=None):
    """ implements NPt (non-parametric mfpt optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    Itw trajectories indicator function multiplied by a re-weighting factor,
        to use with multiple short trajectories. Default value is 1.
    """

    if i_traj is None:
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)
    if w_traj is not None:
        itw = itw * w_traj

    dfk = fk[:, 1:] - fk[:, :-1]
    akj = tf.tensordot(dfk * itw, dfk, axes=[1, 1])

    delta_r = -(r_traj[1:] - r_traj[:-1])
    b = tf.tensordot(dfk, delta_r * itw, 1) + 2 * tf.math.reduce_sum(fk, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    return rn_traj

npneq_chunk = 1_000_000


@tf.function(reduce_retracing=True)
def _npneq_moments(fa, fb, itw, delta_r, gamma, stable):
    """akj/b contributions of one block of consecutive transitions.

    fa/fb are the basis functions at the start/end frames of each transition.
    Kept separate from `npneq` so that the (n_basis, block) temporaries stay
    bounded instead of scaling with the full trajectory length.
    """
    if stable:
        akj = -tf.matmul(fa, fa * itw, transpose_b=True)
    else:
        akj = tf.matmul(fa, fb * itw - fa * (itw + gamma), transpose_b=True)
    return akj, tf.linalg.matvec(fa, -delta_r * itw)


def npneq(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None,
          chunk=None, w_traj=None):
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

    akj and b are sums over transitions, so they are accumulated block by
    block: the full-length intermediates this would otherwise allocate
    dominate peak memory for multi-million-frame trajectories.

    fk is either the (n_basis, N) basis tensor or a callable fk(s, e) returning
    its columns s..e-1; the callable form never materializes the full basis.
    """
    basis = fk if callable(fk) else (lambda s, e: fk[:, s:e])

    if i_traj is None:
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)

    if train_mask is not None:
        itw = itw * train_mask
    if w_traj is not None:
        itw = itw * tf.cast(w_traj, r_traj.dtype)

    delta_r = r_traj[1:] - r_traj[:-1]

    n_frames = int(r_traj.shape[0])
    gamma = tf.cast(gamma, r_traj.dtype)
    akj, b = 0, 0
    step = chunk or npneq_chunk
    for s in range(0, n_frames - 1, step):
        e = min(s + step, n_frames - 1)
        f = basis(s, e + 1)
        akj_i, b_i = _npneq_moments(f[:, :-1], f[:, 1:], itw[s:e],
                                    delta_r[s:e], gamma, stable)
        akj += akj_i
        b += b_i

    b = tf.reshape(b, [-1, 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    delta = tf.concat([tf.tensordot(al_j, basis(s, min(s + step, n_frames)), 1)
                       for s in range(0, n_frames, step)], axis=0)
    rn_traj = tf.clip_by_value(r_traj + delta, 0, 1)
    return rn_traj

@tf.function
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
    it = tf.cast(i_traj[dt:] == i_traj[:-dt], dtype=r_traj.dtype)
    delta_t_prec = tf.cast(dt, dtype=r_traj.dtype)
    not_crossed = tf.cast(tf.logical_or(future_boundary.index[:-dt] == - 1,
                                        future_boundary.delta_t[:-dt] > delta_t_prec), dtype=r_traj.dtype)

    delta_fj = fk[:, dt:] * not_crossed * it - fk[:, :-dt] * (it + gamma)
    akj = tf.tensordot(fk[:, :-dt], delta_fj , axes=[1, 1])

    r_plus = tf.where(tf.logical_and(future_boundary.index[:-dt] > - 1, future_boundary.delta_t[:-dt] <= delta_t_prec),
                      future_boundary.r[:-dt], r_traj[dt:])
    delta_r = r_plus - r_traj[:-dt]
    b = tf.tensordot(fk[:, :-dt], -delta_r * it, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn_traj = r_traj + tf.tensordot(al_j, fk, 1)
    rn_traj = tf.clip_by_value(rn_traj, 0, 1)
    return rn_traj


@tf.function
def npnew(r, fk, it):
    """ implements NPNEw (non-parametric non-equilibrium re-weighting factors
    optimization) iteration.

    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indicator function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """

    dfk = fk[:, 1:] - fk[:, :-1]

    b = -tf.tensordot(dfk * it, r[:-1], 1)
    b = tf.reshape(b, [b.shape[0], 1])
    scale = tf.math.reduce_sum(1 - r[:-1] * it)
    scale = tf.reshape(scale, [1, 1])
    b = tf.concat((b, scale), 0)

    ones = tf.reshape(it, [1, it.shape[0]])
    dfk = tf.concat((dfk * it, ones), 0)
    akj = tf.tensordot(dfk, fk[:, :-1], axes=[1, 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r + tf.tensordot(al_j, fk, 1)
    rn = abs(rn)

    return rn

@tf.function
def npnet(r_traj, fk, t_traj, i_traj, gamma=0, tmax=1e10, train_mask=None):
    """ implements NPNEt (non-parametric non-equilibrium mfpt
    optimization) iteration.
    
    r is the putative RC time-series
    fk are the basis functions of the variation delta r
    It is the trajectory indictor function:
        It(i)=1 if X(i) and X(i+1) belong to the same short trajectory
    """
    if i_traj is None:
        itw = tf.ones_like(r_traj[:-1])
    else:
        itw = tf.cast(i_traj[1:] == i_traj[:-1], dtype=r_traj.dtype)
        
    if train_mask is not None:
        itw = itw * train_mask

    dfj = fk[:, 1:] * itw - fk[:,:-1] * (itw + gamma)
    
    akj = tf.tensordot(fk[:, :-1], dfj, axes=[1, 1])

    delta_r = r_traj[1:] - r_traj[:-1] + t_traj[1:] - t_traj[:-1]

    b = tf.tensordot(fk[:, :-1], -delta_r * itw, 1)
    b = tf.reshape(b, [b.shape[0], 1])

    al_j = tf.linalg.lstsq(akj, b, fast=False)
    al_j = tf.reshape(al_j, [al_j.shape[0]])

    rn = r_traj + tf.tensordot(al_j, fk, 1)
    rn = tf.clip_by_value(rn, 0, tmax)
    return rn

