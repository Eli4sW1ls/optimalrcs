"""Write a plain MD trajectory (position and velocity per frame) of an infretis ASE system, for infretis_md_1d_langevin.ipynb.

    python make_md_path.py <run folder with infretis.toml> <out.npy> [--frames 5000000] [--stride 1] [--seed 0]

Uses the [engine] settings of the toml: its calculator (module, class, keys), mass, start position, temperature,
timestep and Langevin friction, with ASE's own Langevin integrator (the integrator of the infretis ASE engine for
integrator = "langevin"). Saves float32 (frames, 2): x and v of particle 0, one row every `stride` steps.
The folder is put on sys.path and entered, so a calculator module given relative to it is found.
"""
import argparse, importlib.util, os, sys, tomllib
import numpy as np
from ase import Atoms, units
from ase.constraints import FixCartesian
from ase.md.langevin import Langevin

p = argparse.ArgumentParser()
p.add_argument('run'); p.add_argument('out')
p.add_argument('--frames', type=int, default=5_000_000); p.add_argument('--stride', type=int, default=1)
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
a.out = os.path.abspath(a.out)
os.chdir(a.run); sys.path.insert(0, os.getcwd())
eng = tomllib.load(open('infretis.toml', 'rb'))['engine']
cal = dict(eng['calculator_settings'])
spec = importlib.util.spec_from_file_location('calc_module', os.path.abspath(cal.pop('module')))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
calc = getattr(mod, cal.pop('class'))(**cal)
pos = np.zeros((1, 3)); pos[0, 0] = eng['particles']['pos'][0][0]
atoms = Atoms('H', positions=pos, masses=eng['particles']['mass'], pbc=False)
atoms.set_constraint(FixCartesian(0, mask=[False, True, True]))
atoms.calc = calc
dyn = Langevin(atoms, eng['timestep'] * units.fs, temperature_K=eng['temperature'],
               friction=eng['langevin_friction'] / units.fs, fixcm=False, rng=np.random.default_rng(a.seed))
out = np.empty((a.frames, 2), np.float32)
dyn.run(1000)                                               # equilibrate
for i in range(a.frames):
    dyn.run(a.stride)
    out[i, 0] = atoms.positions[0, 0]; out[i, 1] = atoms.get_velocities()[0, 0]
np.save(a.out, out)
print('saved', a.out, out.shape, 'x range', out[:, 0].min(), out[:, 0].max(), 'v std', out[:, 1].std())
