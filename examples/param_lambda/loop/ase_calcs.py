"""ASE calculators of the toy systems, loaded by the infretis ASE engine from this file's absolute path
([engine.calculator_settings] module = ".../ase_calcs.py", class = ...; the other keys are __init__ arguments).

- FlatWalls1D: V = 0 on [xleft, xright], harmonic walls outside (as infretis' examples/ase flat_walls_ase.py).
- ToytisPotential: any toytis potential class (`spec` = "module:Class" in toytis/potentials, `params` its
  keyword arguments) on the particle's positions (y, x[, w, ...]) = atoms.positions[0, :dim]. The force on the
  remaining Cartesian components is zero; fix them with an ASE constraint in the initial configuration.

Units: reduced units of the toy systems map one to one onto ASE units with sigma = 1 A, epsilon = 1 eV, m = 1 amu
(ase_systems.py converts the engine settings: time unit 10.18 fs, kT = kB T).
"""
import importlib
import os
import sys

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

TOYTIS = os.environ.get('TOYTIS', '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis')


class FlatWalls1D(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, xleft=-0.2, xright=0.4, k=100.0, **kwargs):
        super().__init__(**kwargs)
        self.xleft, self.xright, self.k = float(xleft), float(xright), float(k)

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.get_positions()[:, 0]
        d = np.where(x < self.xleft, x - self.xleft, np.where(x > self.xright, x - self.xright, 0.0))
        forces = np.zeros_like(atoms.get_positions())
        forces[:, 0] = -self.k * d
        self.results['energy'] = float(0.5 * self.k * np.sum(d * d))
        self.results['forces'] = forces


def load_toytis_potential(spec, params=None, toytis=TOYTIS):
    """Instance of a toytis potential: spec 'module:Class' (looked up in toytis/potentials, then as a module)."""
    if toytis not in sys.path:
        sys.path.insert(0, toytis)
    module_name, _, class_name = spec.partition(':')
    try:
        module = importlib.import_module('potentials.' + module_name)
    except ModuleNotFoundError:
        module = importlib.import_module(module_name)
    return getattr(module, class_name)(**(params or {}))


class ToytisPotential(Calculator):
    """`base_dim` < `dim`: the toytis potential acts on the first base_dim coordinates only, and each further
    coordinate w (up to dim) gets its own uncoupled double well dV ((w / w0)^2 - 1)^2 (`w_well` = {w0, dV}):
    native embedding, the same w dynamics as toytis zpot_w.ZChannelsW but without the coupling."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, spec, params=None, dim=2, toytis=TOYTIS, base_dim=None, w_well=None, **kwargs):
        super().__init__(**kwargs)
        self.spec, self.params, self.dim = spec, dict(params or {}), int(dim)
        self.base_dim = self.dim if base_dim is None else int(base_dim)
        ww = dict(w_well or {})
        self.w0, self.dV = float(ww.get('w0', 0.1)), float(ww.get('dV', 0.0))
        self.potential = load_toytis_potential(spec, self.params, toytis)

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions()
        forces = np.zeros_like(pos)
        energy = 0.0
        bd = self.base_dim
        for i in range(len(pos)):
            v, f = self.potential.potential_and_force((np.asarray(pos[i, :bd], float), None))
            energy += float(v)
            forces[i, :bd] = np.asarray(f, float)
            for j in range(bd, self.dim):
                r = pos[i, j] / self.w0
                energy += self.dV * (r * r - 1.0) ** 2
                forces[i, j] = -4.0 * self.dV * (r * r - 1.0) * r / self.w0
        self.results['energy'] = energy
        self.results['forces'] = forces
