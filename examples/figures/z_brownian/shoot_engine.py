"""Committor at a few plateau points by shooting with the toytis ndLangevinEngine itself
(overdamped, gamma = 10, kT = 0.05, dt = 0.002), absorption checked every step at
A: y <= 0.05, B: y >= 0.85 (the staple run's definitions). Compared with the PDE reference
and with the staple NPNEq and MD NPq fits near each point."""
import os, sys, numpy as np
from multiprocessing import Pool
TOY = "/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis"
HERE = os.path.dirname(os.path.abspath(__file__))
PTS = [(0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]
N, LA, LB = 1200, 0.05, 0.85

def shoot(args):
    seed, y0, x0, n = args
    sys.path.insert(0, TOY); os.chdir(TOY)
    from engine import ndLangevinEngine
    from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
    np.random.seed(seed)
    eng = ndLangevinEngine({"dt": 0.002, "temperature": 0.05, "friction": 10.0, "mass": 1.0, "dim": 2,
                            "high_friction": True,
                            "potential": RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)})
    hits = 0
    for _ in range(n):
        ph = (np.array([y0, x0]), np.zeros(2))
        while True:
            ph = eng.step(ph)
            y = ph[0][0]
            if y <= LA: break
            if y >= LB: hits += 1; break
    return hits

if __name__ == "__main__":
    jobs = [(1000 * i + j, y0, x0, N // 16) for i, (y0, x0) in enumerate(PTS) for j in range(16)]
    with Pool(16) as pool:
        res = np.array(pool.map(shoot, jobs)).reshape(len(PTS), 16).sum(1)
    from scipy.interpolate import RegularGridInterpolator
    ref = np.load(f"{HERE}/committor_reference_h0.001.npz")
    f = RegularGridInterpolator((ref["y"], ref["x"]), ref["q"])
    st = np.load(f"{os.path.dirname(HERE)}/istarz_br_g10/committor.npz")
    md = np.load(f"{HERE}/npq_md/npq_md.npz")
    ys, xs = st["X_all"][:, 0], st["X_all"][:, 1]
    for (y0, x0), h in zip(PTS, res):
        n = N // 16 * 16; p = h / n
        near_st = (ys - y0) ** 2 + (xs - x0) ** 2 < 0.01 ** 2
        near_md = (md["y"] - y0) ** 2 + (md["x"] - x0) ** 2 < 0.01 ** 2
        print(f"(y={y0:.2f}, x={x0:.2f}): engine shooting q = {p:.3f} +- {np.sqrt(p * (1 - p) / n):.3f} | "
              f"PDE {float(f([[y0, x0]])[0]):.3f} | staple NPNEq {st['r'][near_st].mean():.3f} ({near_st.sum()} frames) | "
              f"MD NPq {md['q_npq'][near_md].mean():.3f} ({near_md.sum()} rows)", flush=True)
