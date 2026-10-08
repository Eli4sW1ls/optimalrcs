import numpy as np
import cupy as cp


def _segment_bounds(n, i_traj):
    """Start/end indices of each contiguous run of equal `i_traj` values."""
    if i_traj is None:
        return np.array([0]), np.array([n])
    seg_start = np.empty(n, dtype=bool)
    seg_start[0] = True
    seg_start[1:] = i_traj[1:] != i_traj[:-1]
    starts = np.flatnonzero(seg_start)
    ends = np.append(starts[1:], n)
    return starts, ends


def _delta_i(index):
    return cp.where(index > -1, index - cp.arange(index.shape[0], dtype=index.dtype), 0)


def _delta_t(index, t_traj):
    return cp.where(index > -1, t_traj[index] - t_traj, 0)


class _Boundary:
    """The delta_i/delta_t arrays are derived from the stored indices when they
    are accessed, instead of being kept as length-N arrays: only the metrics
    and plots use them, and holding them would cost 56 bytes per frame."""

    @property
    def delta_i(self):
        return _delta_i(self.index)

    @property
    def delta_i2(self):
        return _delta_i(self.index2)

    @property
    def delta_t(self):
        return self.delta_i if self._t_traj is None else _delta_t(self.index, self._t_traj)

    @property
    def delta_t2(self):
        return self.delta_i2 if self._t_traj is None else _delta_t(self.index2, self._t_traj)


def _traj_starts_ends(i_traj):
    """First/last frame of each trajectory, as host arrays."""
    starts, ends = _segment_bounds(len(i_traj), cp.asnumpy(i_traj))
    return starts, ends - 1


class FutureBoundary(_Boundary):
    """
    class to contain information about boundaries in the future, to correctly describe martingale at the boundaries
    self.index[i] - index of the next boundary in the future for the current point with index i
            if i is a boundary itself, then index[i]=i
    self.index3[i] - as index[i], but the last frame of the trajectory if no boundary is reached
    self.r[i] - r value of the next boundary in the future
    self.delta_i - difference in indices between the current point and the future boundary
    self.index2[i] - index for the next boundary in the future for the current point eith index i, however
            if i is a boundary itself, then index2[i] points to the next boundary
    self.r2[i] - r value of the next boundary in the future, however
            if i is a boundary itself, then r2[i] points to the next boundary
    self.delta_i2 - difference in indices between the current point and the future boundary, however
            if i is a boundary itself, then delta_i2[i]>0 points to the next boundary
    self.delta_i_to_end - distance to the end of trajectory
    """
    def __init__(self, r_traj: np.ndarray, b_traj: np.ndarray, t_traj: np.ndarray = None, i_traj: np.ndarray = None) -> None:
        n = len(r_traj)
        r_traj = cp.asarray(r_traj)
        _b_traj = cp.asnumpy(b_traj)
        _i_traj = None if i_traj is None else cp.asnumpy(i_traj)
        starts, ends = _segment_bounds(n, _i_traj)

        # The indices are built on the host, per trajectory (not per frame), via
        # a reversed running-min: looping over O(n) individual frames in Python
        # is prohibitively slow for multi-million-frame data.
        marker = np.where(_b_traj > 0, np.arange(n), n)  # n = "not found" sentinel
        index = np.empty(n, dtype=np.int64)
        index3 = np.empty(n, dtype=np.int64)
        delta_i_to_end = np.empty(n, dtype='int32')
        for s, e in zip(starts, ends):
            index[s:e] = np.minimum.accumulate(marker[s:e][::-1])[::-1]
            index3[s:e] = np.where(index[s:e] == n, e - 1, index[s:e])
            delta_i_to_end[s:e] = np.arange(e - s - 1, -1, -1)
        index = np.where(index == n, -1, index).astype('int32')

        index2 = np.roll(index, -1)
        index2[-1] = -1
        if _i_traj is not None:
            index2[_i_traj != np.roll(_i_traj, -1)] = -1

        self.index = cp.asarray(index)
        self.index2 = cp.asarray(index2)
        self.index3 = cp.asarray(index3.astype('int32'))
        self.delta_i_to_end = cp.asarray(delta_i_to_end)

        self.r = cp.where(self.index > -1, r_traj[self.index], 0)
        self.r2 = cp.where(self.index2 > -1, r_traj[self.index2], 0)
        self._t_traj = None if t_traj is None else cp.asarray(t_traj)

    @property
    def delta_t3(self):
        """time (or frames) to the next boundary, or to the end of the trajectory"""
        if self._t_traj is None:
            return self.index3 - cp.arange(self.index3.shape[0], dtype=self.index3.dtype)
        return self._t_traj[self.index3] - self._t_traj

    def set_distance_to_end(self, i_traj):
        traj_starts, traj_ends = _traj_starts_ends(i_traj)
        delta_i_to_end = np.zeros(len(i_traj), 'int32')
        for i_start, i_end in zip(traj_starts, traj_ends):
            delta_i_to_end[i_start:i_end+1] = range(i_end-i_start,-1,-1)
        self.delta_i_to_end = cp.asarray(delta_i_to_end)

    def set_distance_to_end_fixed_traj_length_trap(self, i_traj, trap_boundary, traj_length):
        traj_starts, traj_ends = _traj_starts_ends(i_traj)
        trap_boundary = cp.asnumpy(trap_boundary)
        delta_i_to_end = np.zeros(len(i_traj), 'int32')
        traj_length=traj_length-1
        for i_start, i_end in zip(traj_starts, traj_ends):
            delta_i_to_end[i_start:i_end+1] = range(i_end-i_start,-1,-1)
            if trap_boundary[i_end] and (traj_length > i_end-i_start):
                delta_i_to_end[i_start:i_end+1] += traj_length - (i_end - i_start)
        self.delta_i_to_end = cp.asarray(delta_i_to_end)

    def set_distance_to_end_poisson_traj_length_trap(self, i_traj, trap_boundary, average_traj_length=None):
        traj_starts, traj_ends = _traj_starts_ends(i_traj)
        trap_boundary = cp.asnumpy(trap_boundary)
        delta_i_to_end = np.zeros(len(i_traj), 'int32')
        tb=0
        nb=0
        if average_traj_length is None:
            for i_start, i_end in zip(traj_starts, traj_ends):
                if trap_boundary[i_end]:
                    tb+=i_end-i_start
                    nb+=1
            tnb=(len(i_traj)-tb)/(len(traj_ends)-nb)
            tb=tb/nb
            average_traj_length=int(tnb-tb)
            #print (tb,tnb,average_traj_length)
        for i_start, i_end in zip(traj_starts, traj_ends):
            delta_i_to_end[i_start:i_end+1] = range(i_end-i_start,-1,-1)
            if trap_boundary[i_end]:
                traj_length=int(-np.log(np.random.random())*average_traj_length -1)
                delta_i_to_end[i_start:i_end+1] += traj_length
        self.delta_i_to_end = cp.asarray(delta_i_to_end)

class PastBoundary(_Boundary):
    def __init__(self, r_traj: np.ndarray, b_traj: np.ndarray, t_traj: np.ndarray = None, i_traj: np.ndarray = None) -> None:
        n = len(r_traj)
        r_traj = cp.asarray(r_traj)
        _b_traj = cp.asnumpy(b_traj)
        _i_traj = None if i_traj is None else cp.asnumpy(i_traj)
        starts, ends = _segment_bounds(n, _i_traj)

        # self.index[i]: nearest j <= i within the same trajectory with
        # b_traj[j] > 0, else -1. Mirrors FutureBoundary but with a forward
        # running-max, computed per-trajectory for the same reason.
        marker = np.where(_b_traj > 0, np.arange(n), -1)
        index = np.empty(n, dtype=np.int64)
        delta_i_from_start = np.empty(n, dtype='int32')
        for s, e in zip(starts, ends):
            index[s:e] = np.maximum.accumulate(marker[s:e])
            delta_i_from_start[s:e] = np.arange(e - s)
        index = index.astype('int32')

        index2 = np.roll(index, 1)
        index2[0] = -1
        if _i_traj is not None:
            index2[_i_traj != np.roll(_i_traj, 1)] = -1

        self.index = cp.asarray(index)
        self.index2 = cp.asarray(index2)
        self.delta_i_from_start = cp.asarray(delta_i_from_start)

        self.r = cp.where(self.index > -1, r_traj[self.index], 0)
        self.r2 = cp.where(self.index2 > -1, r_traj[self.index2], 0)
        self._t_traj = None if t_traj is None else cp.asarray(t_traj)
