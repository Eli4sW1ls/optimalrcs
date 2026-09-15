import numpy as np
import optimalrcs

# For each TIS path p:
#   cvs[p]:       (n_frames, n_features), all candidate CVs at each frame
#   order[p]:     (n_frames,), order parameter used to define A/B
#   time[p]:      (n_frames,), physical time, starting at 0 for each path

tis_dir = "path/to/tis/data"


X = np.concatenate(cvs, axis=0)
lam = np.concatenate(order, axis=0)
t_traj = np.concatenate(time, axis=0)
i_traj = np.concatenate([
    np.full(len(path), path_id, dtype=int)
    for path_id, path in enumerate(cvs)
])

# Define basin membership from the same operational-state definitions used in TIS.
boundary0 = lam <= lambda_A  # state A: q = 0
boundary1 = lam >= lambda_B  # state B: q = 1

# Validate the prepared data before fitting.
assert X.shape[0] == lam.size == t_traj.size == i_traj.size
assert not np.any(boundary0 & boundary1)
assert np.all(np.diff(i_traj) >= 0)
assert all(np.all(np.diff(t_traj[i_traj == path_id]) > 0)
           for path_id in np.unique(i_traj))

q = optimalrcs.CommittorNE(
    boundary0=boundary0,
    boundary1=boundary1,
    i_traj=i_traj,
    t_traj=t_traj,
)

def comp_y():
    return X[:, np.random.randint(X.shape[1])]

q.fit_transform(
    comp_y,
    history_delta_t=[0, 1, 2, 4, 8, 16],
    gamma=0.05,
    max_iter=10_000,
)