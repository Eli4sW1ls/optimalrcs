import glob
import importlib.util
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optimalrcs

# For each TIS path p:
#   cvs[p]:       (n_frames, n_features), all candidate CVs at each frame
#   order[p]:     (n_frames,), order parameter used to define A/B
#   time[p]:      (n_frames,), physical time, starting at 0 for each path

tis_dir = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_2603/"
lambda_A = 0.05
lambda_B = 0.85

# Path to the potential module, relative to tis_dir.
potential_file = os.path.join(tis_dir, "../../potentials/sjoelbak.py")

_CYCLE_RE = re.compile(r"# Cycle:\s*\d+,\s*status:\s*(\S+?),")


def load_order_file(order_file, acc_only=True):
    """Load per-path time/order/CV arrays from a single TIS `order.txt` file.

    An `order.txt` file holds one or more paths, each introduced by a
    `# Cycle: <n>, status: <FLAG>, ...` header line. Within a path, every
    frame line holds the time in column 0, the order parameter in column 1,
    and any additional collective variables (CVs) in the remaining columns.
    """
    # First pass: locate path boundaries and acceptance flags by scanning
    # lines as plain text (cheap - no per-line list/float allocation).
    starts, accepted_flags = [], []
    n_data_lines = 0
    with open(order_file) as f:
        for line in f:
            match = _CYCLE_RE.match(line)
            if match:
                starts.append(n_data_lines)
                accepted_flags.append(match.group(1) == "ACC")
                continue
            if line.startswith("#"):
                continue
            n_data_lines += 1
    starts.append(n_data_lines)

    # Second pass: parse all numeric data in one bulk call with pandas' C
    # parser. This avoids building millions of Python list/string objects
    # (as a line-by-line float() loop would), which is what previously blew
    # up memory and runtime on multi-million-line order.txt files.
    data = pd.read_csv(
        order_file, comment="#", header=None, sep=r"\s+",
    ).to_numpy(dtype=float)

    cvs, order, time = [], [], []
    for i in range(len(starts) - 1):
        lo, hi = starts[i], starts[i + 1]
        if hi <= lo or not (accepted_flags[i] or not acc_only):
            continue
        block = data[lo:hi]
        time.append(block[:, 0])
        order.append(block[:, 1])
        cvs.append(block[:, 1:])
    return cvs, order, time


def load_tis_data(tis_dir, ensemble_glob="0[0-9][0-9]", acc_only=True):
    """Load per-path time/order/CV arrays from all TIS ensemble folders in `tis_dir`.

    Ensemble folders are expected at `tis_dir/<ensemble_glob>/order.txt`,
    following the standard (RE)PPTIS output layout.
    """
    cvs, order, time = [], [], []
    folders = sorted(glob.glob(os.path.join(tis_dir, ensemble_glob)))[1:]
    print(f"Found {len(folders)} ensemble folders in {tis_dir}")
    for i, folder in enumerate(folders):
        order_file = os.path.join(folder, "order.txt")
        if not os.path.isfile(order_file):
            print(f"  [{i + 1}/{len(folders)}] {folder}: no order.txt, skipping")
            continue
        c, o, t = load_order_file(order_file, acc_only=acc_only)
        cvs += c
        order += o
        time += t
        n_frames = sum(len(p) for p in o)
        print(f"  [{i + 1}/{len(folders)}] {folder}: "
              f"loaded {len(c)} paths, {n_frames} frames")
    print(f"Loaded {len(cvs)} paths, {sum(len(p) for p in order)} frames total")
    return cvs, order, time


cvs, order, time = load_tis_data(tis_dir)
cvs, order, time = cvs[int(1.*len(cvs)//3):int(1.7*len(cvs)//3)], order[int(1.*len(order)//3):int(1.7*len(order)//3)], time[int(1.*len(time)//3):int(1.7*len(time)//3)]

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

max_iter = 600
print(f"Starting CommittorNE training for {max_iter} iterations...")
q.fit_transform(
    comp_y,
    # history_delta_t=[0, 1, 2, 4, 8, 16],
    # gamma=0.02,
    max_iter=max_iter,
    print_step=1000,
    min_delta_x=1e-5,
)
print("CommittorNE training complete.")

q.plots_feps()
q.plots_obs_pred()
# Plot the potential with data points colored by their fitted committor value.
# Assumes a 2D potential where the order parameter (x-axis) and the first CV
# (y-axis) are the same coordinates the potential is defined on.
spec = importlib.util.spec_from_file_location("potential_module", potential_file)
potential_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(potential_module)
potential = potential_module.RectangularGridWithBarrierPotential()

fig, ax = plt.subplots()
potential.plot_potential(ax)
sc = ax.scatter(lam, X[:, 1], c=q.r_traj, cmap="coolwarm", s=2, edgecolors="none")
fig.colorbar(sc, ax=ax, label="committor")
ax.set_xlabel("order parameter")
ax.set_ylabel("CV[1]")
plt.show()

