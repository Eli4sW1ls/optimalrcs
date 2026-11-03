# Copyright (c) 2025 Sergei Krivov
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

import numpy as np


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
    return np.where(index > -1, index - np.arange(len(index), dtype=index.dtype), 0)


def _delta_t(index, t_traj):
    return np.where(index > -1, t_traj[index] - t_traj, 0)


class _Boundary:
    """delta_i/delta_t are derived from the stored indices when accessed rather than
    kept as length-N arrays: only the standard Z_q and the plots read them, and
    storing them costs 24 bytes per frame for each boundary object."""

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


class FutureBoundary(_Boundary):
    """
    class to contain information about boundaries in the future, to correctly describe martingale at the boundaries
    self.index[i] - index of the next boundary in the future for the current point with index i
            if i is a boundary itself, then index[i]=i
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
        starts, ends = _segment_bounds(n, i_traj)

        # self.index[i]: nearest j >= i within the same trajectory with
        # b_traj[j] > 0, else -1. Computed per-trajectory (not per-frame) via
        # a reversed running-min, since looping over O(n) individual frames
        # in Python is prohibitively slow for multi-million-frame data.
        marker = np.where(b_traj > 0, np.arange(n), n)  # n = "not found" sentinel
        index = np.empty(n, dtype=np.int64)
        for s, e in zip(starts, ends):
            index[s:e] = np.minimum.accumulate(marker[s:e][::-1])[::-1]
        self.index = np.where(index == n, -1, index).astype('int32')

        # self.delta_i_to_end[i]: distance from i to the last frame of its trajectory.
        self.delta_i_to_end = np.empty(n, dtype='int32')
        for s, e in zip(starts, ends):
            self.delta_i_to_end[s:e] = np.arange(e - s - 1, -1, -1)

        self.r = np.where(self.index > -1, r_traj[self.index], 0)
        self.index2=np.roll(self.index, -1)
        self.index2[-1]=-1
        if i_traj is not None:
            self.index2[i_traj!=np.roll(i_traj,-1)]=-1
        self.r2 = np.where(self.index2 > -1, r_traj[self.index2], 0)
        self._t_traj = t_traj

    def set_distance_to_end(self, i_traj):
        traj_ends=np.where(np.roll(i_traj,-1)!=i_traj)[0]
        traj_starts=np.concatenate(([0],traj_ends[:-1]+1))
        self.delta_i_to_end=np.zeros(len(i_traj), 'int32')
        for i_start, i_end in zip(traj_starts, traj_ends):
            self.delta_i_to_end[i_start:i_end+1] = range(i_end-i_start,-1,-1)
            
    def set_distance_to_end_fixed_traj_length_trap(self, i_traj, trap_boundary, traj_length):
        traj_ends=np.where(np.roll(i_traj,-1)!=i_traj)[0]
        traj_starts=np.concatenate(([0],traj_ends[:-1]+1))
        self.delta_i_to_end=np.zeros(len(i_traj), 'int32')
        traj_length=traj_length-1
        for i_start, i_end in zip(traj_starts, traj_ends):
            self.delta_i_to_end[i_start:i_end+1] = range(i_end-i_start,-1,-1)
            if trap_boundary[i_end] and (traj_length > i_end-i_start):
                self.delta_i_to_end[i_start:i_end+1] += traj_length - (i_end - i_start)

    def set_distance_to_end_poisson_traj_length_trap(self, i_traj, trap_boundary, average_traj_length=None):
            traj_ends=np.where(np.roll(i_traj,-1)!=i_traj)[0]
            traj_starts=np.concatenate(([0],traj_ends[:-1]+1))
            self.delta_i_to_end=np.zeros(len(i_traj), 'int32')
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
                self.delta_i_to_end[i_start:i_end+1] = range(i_end-i_start,-1,-1)
                if trap_boundary[i_end]:
                    traj_length=int(-np.log(np.random.random())*average_traj_length -1)
                    self.delta_i_to_end[i_start:i_end+1] += traj_length

    def set_distance_to_end_trap(self, i_traj, trap_boundary, extra=10 ** 9):
        """Segments whose last frame lies in trap_boundary continue there indefinitely: the segment
        end is not treated as a cut (as in a continuous run, where frames dwelling inside A or B
        contribute nothing). Used for data cut into excursions between the boundaries."""
        _trap = np.asarray(trap_boundary) > 0
        _i_traj = None if i_traj is None else np.asarray(i_traj)
        starts, ends = _segment_bounds(len(_trap), _i_traj)
        delta_i_to_end = np.empty(len(_trap), dtype='int32')
        for s, e in zip(starts, ends):
            delta_i_to_end[s:e] = np.arange(e - s - 1, -1, -1) + (extra if _trap[e - 1] else 0)
        self.delta_i_to_end = delta_i_to_end


class PastBoundary(_Boundary):
    def __init__(self, r_traj: np.ndarray, b_traj: np.ndarray, t_traj: np.ndarray = None, i_traj: np.ndarray = None) -> None:
        n = len(r_traj)
        starts, ends = _segment_bounds(n, i_traj)

        # self.index[i]: nearest j <= i within the same trajectory with
        # b_traj[j] > 0, else -1. Mirrors FutureBoundary but with a forward
        # running-max, computed per-trajectory for the same reason.
        marker = np.where(b_traj > 0, np.arange(n), -1)
        index = np.empty(n, dtype=np.int64)
        for s, e in zip(starts, ends):
            index[s:e] = np.maximum.accumulate(marker[s:e])
        self.index = index.astype('int32')

        # self.delta_i_from_start[i]: distance from i to the first frame of its trajectory.
        self.delta_i_from_start = np.empty(n, dtype='int32')
        for s, e in zip(starts, ends):
            self.delta_i_from_start[s:e] = np.arange(e - s)

        self.r = np.where(self.index > -1, r_traj[self.index], 0)
        self.index2=np.roll(self.index, 1)
        self.index2[0]=-1
        if i_traj is not None:
            self.index2[i_traj!=np.roll(i_traj,1)]=-1
        self.r2 = np.where(self.index2 > -1, r_traj[self.index2], 0)
        self._t_traj = t_traj

    def set_distance_from_start_trap(self, i_traj, trap_boundary, extra=10 ** 9):
        """Segments whose first frame lies in trap_boundary continue there indefinitely: the segment
        start is not treated as a cut (as in a continuous run, where frames dwelling inside A or B
        contribute nothing). Used for data cut into excursions between the boundaries."""
        _trap = np.asarray(trap_boundary) > 0
        _i_traj = None if i_traj is None else np.asarray(i_traj)
        starts, ends = _segment_bounds(len(_trap), _i_traj)
        delta_i_from_start = np.empty(len(_trap), dtype='int32')
        for s, e in zip(starts, ends):
            delta_i_from_start[s:e] = np.arange(e - s) + (extra if _trap[s] else 0)
        self.delta_i_from_start = delta_i_from_start
