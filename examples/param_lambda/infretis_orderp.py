"""An exported lambda (lambda_runtime.LambdaModel) as the order parameter of infretis.

infretis.toml:

    [orderparameter]
    class = "LearnedLambda"
    module = "/abs/path/to/examples/param_lambda/infretis_orderp.py"
    model = "/abs/path/to/examples/param_lambda/exported/<system>_m3-fast.npz"
    # the CV source: the order parameter(s) of the run lambda was trained on, in the same order (their returned
    # values are the CV columns of that run's order.txt, which the infretis loader reads)
    cvs = [{class = "PositionX", module = "orderp.py", index = [0, 0], periodic = false}]
    # cv_columns = [0, 1]      # optional: which of the CV values are the model's CVs (default: all, in order)

    [simulation]
    interfaces = [...]           # python infretis_orderp.py interfaces <model> --q 1e-3 0.01 0.1 0.5

calculate(system) returns [lambda, v_0, v_1, ...]: lambda first (infretis uses element 0 for interfaces, crossings and
the states), then every value of the CV source, so order.txt keeps the CVs for the next analysis (the infretis
loader reads columns 1.. as CVs: lambda and the old values).

- States: lambda = s (the state CV) outside (lambda_A, lambda_B), strictly inside it otherwise, so infretis'
  tests (order <= interfaces[0] in A, >= interfaces[-1] in B) give the states of the state CV. The first and last
  interfaces must be the model's lambda_A and lambda_B (`check` verifies it).
- The CV sources are built with infretis' own create_orderparameter: built-in classes (position, distance, ...)
  or external ones (class + module, the module path relative to the launch directory, as in infretis).
  Velocity sources (class "velocity", underdamped systems, M5) flag the order parameter as velocity dependent:
  infretis then recomputes the orders of a time-reversed path (Path.reverse), and `_ensure_phase_point` reads the
  frame's velocities with the sign of vel_rev. The state CV must be configurational (the states are x-regions).
- Load paths are not recomputed by infretis (it reads frame orders from order.txt): convert them with
  `python infretis_orderp.py relabel <model> <load_dir> <new_load_dir>`.
- Each infretis worker process (the APPTIS checkout spawns them) builds its own object; BLAS is limited to one
  thread (a one-frame forward pass is slower with threads). `SeedLambda` is the seed order parameter of the
  learning loop: lambda = the state CV, the CV pool recorded after it.

Command line:
    python infretis_orderp.py interfaces <model.npz> --q 1e-3 1e-2 0.1 0.5     # interfaces = [...] line
    python infretis_orderp.py interfaces <model.npz> --n 10 --q-lo 1e-3 --q-hi 0.95
    python infretis_orderp.py relabel <model.npz> <load_dir> <out_dir> [--cv-columns 0 1]
    python infretis_orderp.py check <infretis.toml>
"""
import argparse
import logging
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lambda_runtime import LambdaModel  # noqa: E402

logger = logging.getLogger(__name__)

try:                                            # a subclass when infretis is there; the module also works without
    from infretis.classes.orderparameter import OrderParameter as _Base
except ImportError:                             # pragma: no cover
    class _Base:
        def __init__(self, description='', velocity=False):
            self.description, self.velocity_dependent = description, velocity


def _limit_blas_threads(n=1):
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:                         # pragma: no cover
        pass


def build_cv_sources(cvs):
    """infretis order parameter objects from a list of [orderparameter]-style tables."""
    from infretis.classes.orderparameter import create_orderparameter
    return [create_orderparameter({'orderparameter': dict(table)}) for table in cvs]


def _velocity_dependent(sources):
    return any(getattr(op, 'velocity_dependent', False) for op in sources)


def _ensure_phase_point(system):
    """Velocity CV sources read system.vel. The engine's calculate_order fills it (with the sign of vel_rev applied);
    Path.reverse recomputes the orders of a velocity-dependent order parameter on stored phase points, which hold
    only (trajectory file, frame) and vel_rev: read that frame here, so the reversed path gets v -> -v."""
    vel = getattr(system, 'vel', None)
    if vel is not None and np.size(vel) > 0:
        return
    from ase.io import read
    fname, idx = system.config
    atoms = read(fname, index=int(idx) if idx is not None else 0)
    system.pos = atoms.positions
    v = atoms.get_velocities()
    system.vel = -v if getattr(system, 'vel_rev', False) else v


class SeedLambda(_Base):
    """infretis order parameter [s, v_0, v_1, ...]: the state CV s = v_state_cv itself (the seed order parameter of
    the learning loop), with every CV source value recorded after it, so order.txt holds the CV pool."""

    def __init__(self, cvs, state_cv=0):
        self.cv_sources = build_cv_sources(cvs)
        self.state_cv = int(state_cv)
        if getattr(self.cv_sources[self.state_cv], 'velocity_dependent', False):
            raise ValueError('the state CV must not depend on velocities')
        # velocity CVs: flagged, so infretis recomputes the orders of reversed paths (the recorded v changes sign)
        super().__init__(description=f'seed lambda: CV {self.state_cv} of the CV pool',
                         velocity=_velocity_dependent(self.cv_sources))

    def values(self, system):
        if self.velocity_dependent:
            _ensure_phase_point(system)
        out = []
        for op in self.cv_sources:
            out.extend(float(v) for v in np.ravel(op.calculate(system)))
        return out

    def calculate(self, system):
        v = self.values(system)
        return [v[self.state_cv]] + v


class LearnedLambda(_Base):
    """infretis order parameter [lambda(cv), v_0, v_1, ...], cv = the CV source values (cv_columns of them)."""

    def __init__(self, model, cvs, cv_columns=None, blas_threads=1):
        self.model_path = os.path.abspath(model)
        self.model = LambdaModel(self.model_path)
        self.cv_sources = build_cv_sources(cvs)
        self.cv_columns = None if cv_columns is None else [int(c) for c in cv_columns]
        self.n_cv = len(self.model.cv_names)
        self.n_ood = 0
        if blas_threads:
            _limit_blas_threads(int(blas_threads))
        # lambda(x, v) for underdamped systems (M5): the states stay those of the (configurational) state CV
        super().__init__(description=f'learned lambda {os.path.basename(self.model_path)} of '
                                     f'{", ".join(self.model.cv_names)}', velocity=_velocity_dependent(self.cv_sources))

    def values(self, system):
        """The CV source values of one frame (all of them, in order)."""
        if self.velocity_dependent:
            _ensure_phase_point(system)
        out = []
        for op in self.cv_sources:
            out.extend(float(v) for v in np.ravel(op.calculate(system)))
        return out

    def lam(self, values):
        cv = values if self.cv_columns is None else [values[c] for c in self.cv_columns]
        if len(cv) != self.n_cv:
            raise ValueError(f'the model takes {self.n_cv} CVs ({", ".join(self.model.cv_names)}), the CV source '
                             f'gives {len(cv)}: set cv_columns')
        cv = np.asarray(cv, float)
        lam = self.model.one(cv)
        if self.model.lambda_a < lam < self.model.lambda_b and self.model.ood_one(cv):
            self.n_ood += 1
            if self.n_ood == 1:
                logger.warning('learned lambda: frame outside the training range of the CVs (counted in n_ood)')
        return float(lam)

    def calculate(self, system):
        v = self.values(system)
        return [self.lam(v)] + v


# ---------------------------------------------------------------------------------------------------
# command line: interfaces, relabel load paths, check a toml
# ---------------------------------------------------------------------------------------------------


def interfaces_for_q(model, q_levels):
    """[lambda_A, lambda(q_1), ..., lambda(q_k), lambda_B] for committor levels 0 < q_1 < ... < q_k < 1."""
    m = model if isinstance(model, LambdaModel) else LambdaModel(model)
    q = sorted(float(x) for x in q_levels)
    return [m.lambda_a] + [float(m.interface_for_q(x)) for x in q] + [m.lambda_b]


def _fmt_line(step, values):
    # infretis OrderFormatter: step {:>10d}, then {:>12.6f} per value
    return f'{step:>10d}' + ''.join(f'{v:>12.6f}' for v in values)


def relabel(model, load_dir, out_dir, cv_columns=None):
    """Copy an infretis load directory and rewrite every <path>/order.txt with lambda in front of the old values
    (computed from the old values, which must be the CV source's output: the old order parameter). Returns per
    path (name, min lambda, max lambda)."""
    m = LambdaModel(model)
    op = LearnedLambda.__new__(LearnedLambda)          # the lambda part only, without CV sources
    op.model, op.cv_columns, op.n_cv, op.n_ood = m, cv_columns, len(m.cv_names), 0
    if os.path.exists(out_dir):
        raise SystemExit(f'{out_dir} exists')
    shutil.copytree(load_dir, out_dir)
    report = []
    for name in sorted(os.listdir(out_dir), key=lambda s: (len(s), s)):
        path = os.path.join(out_dir, name, 'order.txt')
        if not os.path.isfile(path):
            continue
        lines, lams = [], []
        with open(path) as fh:
            for line in fh:
                if line.lstrip().startswith('#') or not line.strip():
                    lines.append(line.rstrip('\n'))
                    continue
                parts = line.split()
                step, old = int(float(parts[0])), [float(x) for x in parts[1:]]
                lam = op.lam(old)
                lams.append(lam)
                lines.append(_fmt_line(step, [lam] + old))
        with open(path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        report.append((name, min(lams), max(lams)))
    return report


def check(toml_file):
    """Load the toml's order parameter as infretis does (from the toml's directory) and compare the interfaces
    with the model's lambda_A, lambda_B. Prints the committor level of every interface."""
    import tomllib
    from infretis.classes.orderparameter import create_orderparameter
    with open(toml_file, 'rb') as fh:
        cfg = tomllib.load(fh)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(toml_file)))
    try:
        op = create_orderparameter(cfg)
    finally:
        os.chdir(cwd)
    m = op.model
    intf = cfg['simulation']['interfaces']
    ok = math.isclose(intf[0], m.lambda_a, abs_tol=1e-12) and math.isclose(intf[-1], m.lambda_b, abs_tol=1e-12)
    print(f'order parameter: {op.description}')
    print(f'model lambda_A = {m.lambda_a}, lambda_B = {m.lambda_b}; interfaces[0] = {intf[0]}, [-1] = {intf[-1]}: '
          f'{"OK" if ok else "MISMATCH: the states would no longer be those of the state CV"}')
    inner = np.asarray(intf[1:-1], float)
    if len(inner):
        # invert the output scale: h = eps * expm1((lam - lambda_A) / (lambda_B - lambda_A) * log1p(1 / eps))
        h = m.eps * np.expm1((inner - m.lambda_a) / (m.lambda_b - m.lambda_a) * m._log1p_inv_eps)
        print('committor level of each interface (away from the A/B ramps): '
              + ', '.join(f'{x:.4g}' for x in h))
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('interfaces')
    a.add_argument('model')
    a.add_argument('--q', type=float, nargs='+')
    a.add_argument('--n', type=int, help='n levels equally spaced in ln q between --q-lo and --q-hi')
    a.add_argument('--q-lo', type=float, default=1e-3)
    a.add_argument('--q-hi', type=float, default=0.95)
    r = sub.add_parser('relabel')
    r.add_argument('model')
    r.add_argument('load_dir')
    r.add_argument('out_dir')
    r.add_argument('--cv-columns', type=int, nargs='+')
    c = sub.add_parser('check')
    c.add_argument('toml')
    args = p.parse_args(argv)
    if args.cmd == 'interfaces':
        q = args.q if args.q else np.exp(np.linspace(np.log(args.q_lo), np.log(args.q_hi), args.n or 10)).tolist()
        print('interfaces = [' + ', '.join(f'{x:.10g}' for x in interfaces_for_q(args.model, q)) + ']')
    elif args.cmd == 'relabel':
        for name, lo, hi in relabel(args.model, args.load_dir, args.out_dir, args.cv_columns):
            print(f'path {name}: lambda {lo:.6f} .. {hi:.6f}')
    else:
        sys.exit(0 if check(args.toml) else 1)


if __name__ == '__main__':
    main()
