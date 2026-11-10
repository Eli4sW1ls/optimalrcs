"""Use an exported lambda (lambda_runtime.LambdaModel) as the order parameter of the toytis staple sampler,
without changing the sampler's code.

    import toytis_adapter
    toytis_adapter.install('exported/z_br_m3.npz')         # before Simulation(settings) is created
    settings['interfaces'] = toytis_adapter.interfaces_for_q('exported/z_br_m3.npz', q_levels)
    settings['v_ord'] = False
    sim = Simulation(settings); sim.run()

What install() changes (module attributes only, inside the toytis process):

- ensemble.OrderX and ensemble.OrderParameter -> LearnedOrderX. Ensemble.set_order_parameter builds one per
  ensemble, so every ensemble (and every swap, which reuses stored orders) uses the same lambda.
  calculate(ph) returns [lambda, cv_0, cv_1, ...] as numpy float64 (as OrderX): element 0 is what the sampler uses for
  interfaces, turns and the states; the CVs (the positions, as OrderX wrote them) follow, so order.txt keeps
  them for the analysis. This needs v_ord = False (toytis writes velocities only for len(orders) == dim).
- The states stay those of the state CV: lambda = s outside (lambda_A, lambda_B) and strictly inside it
  otherwise (lambda_runtime, step 4), so the sampler's A/B tests (op[0] < interfaces[0], op[0] > interfaces[-1])
  give the same answer as s < lambda_A, s > lambda_B. The first and last interfaces must stay lambda_A, lambda_B.
- ensemble.kick_star (initial paths of fresh runs) -> a wrapper around the original: kick_star puts the kick
  point at s = interface value; while the wrapper is active, LearnedOrderX moves that point (in place, before
  any propagation) onto the learned interface lambda(s, x) = the same value, solving along s by bisection
  with x redrawn from potential.kick_range(s). Everything else is the original kick_star. Only for dim >= 2
  (1D kick points are floats); restarts (method 'restart') keep the pickled order parameter they were run with.

LearnedOrderX also caches the last frame: the staple extension asks for its fixed start point again at every
step (pgmoves.ext_propagate). It counts frames outside the training range in `n_ood` (warning once).
"""
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lambda_runtime import LambdaModel  # noqa: E402

TOYTIS = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis'
logger = logging.getLogger(__name__)

_DEFAULT_MODEL = None
_original_kick_star = None


class LearnedOrderX:
    """toytis order parameter: [lambda(x), cv_0, cv_1, ...] of the frame's positions."""

    def __init__(self, params=None):
        self.params = params
        path = (params or {}).get('lambda_model') or _DEFAULT_MODEL
        if path is None:
            raise ValueError('no lambda model: call toytis_adapter.install(path) or set settings["lambda_model"]')
        if (params or {}).get('v_ord'):
            raise ValueError('LearnedOrderX returns more values than dim: run with v_ord = False')
        self.model_path = path
        self.model = LambdaModel(path)
        self.dim = len(self.model.cv_names)
        self.n_ood = 0
        self._last_ph = self._last_cv = self._last_op = None
        self._kick = None

    def calculate(self, ph):
        pos = ph[0]
        cv = np.ravel(pos)[:self.dim].tolist()
        if ph is self._last_ph and cv == self._last_cv:
            return list(self._last_op)
        if self._kick is not None:
            cv = self._place_kick_point(ph, cv)
        lam = self.model.one(cv)
        m = self.model
        if m.lambda_a < cv[m.state_cv] < m.lambda_b and m.ood_one(cv):     # A/B frames never use the model
            self.n_ood += 1
            if self.n_ood == 1:
                logger.warning('lambda evaluated outside the training range at %s', cv)
        # numpy float64 like OrderX (ph[0][0]): funcs.validate_staple compares the interface list with orders
        # (`intfs > min_val`), which only works through numpy's broadcasting
        op = [np.float64(lam)] + [np.float64(c) for c in cv]
        self._last_ph, self._last_cv, self._last_op = ph, cv, op
        return list(op)

    def _place_kick_point(self, ph, cv):
        k, m = self._kick, self.model
        s = m.state_cv
        # every retry of kick_star makes a new kick point at exactly s = marker; propagated frames never are
        if not isinstance(ph[0], np.ndarray) or cv[s] != k['marker']:
            return cv
        target = k['marker']                                    # in lambda units
        if not m.lambda_a < target < m.lambda_b:                # in A or B lambda = s: the kick is already right
            return cv
        other = 1 - s
        pot = k['potential']
        sv = target
        for _ in range(2):                                      # x from the channel at the current s guess
            lo, hi = pot.kick_range(sv) if hasattr(pot, 'kick_range') else (cv[other], cv[other])
            x = (hi - lo) * np.random.rand() + lo
            a, b = m.lambda_a, m.lambda_b                       # lambda(a+) = lambda_A < target < lambda_B = lambda(b-)
            for _ in range(60):
                mid = 0.5 * (a + b)
                c = list(cv)
                c[s], c[other] = mid, x
                if m.one(c) < target:
                    a = mid
                else:
                    b = mid
            sv = 0.5 * (a + b)
        ph[0][s], ph[0][other] = sv, x
        k['moved'] = k.get('moved', 0) + 1
        return np.ravel(ph[0])[:self.dim].tolist()


def _kick_marker(ens):
    """The s value pgmoves.kick_star gives the kick point of this ensemble (dim > 1)."""
    i = ens.intfs
    if ens.id == 0:
        return i['R'] * (1 - np.sign(i['R']) * 0.02) - 0.0001
    if ens.id == 1:
        return i['L'] * (1 + np.sign(i['L']) * 0.02) + 0.0001
    return i['M'] * (1 - np.sign(i['M']) * 0.02) - 0.0001


def kick_star(ens):
    op = ens.orderparameter
    if not isinstance(op, LearnedOrderX) or ens.settings['dim'] == 1:
        return _original_kick_star(ens)
    op._kick = dict(marker=_kick_marker(ens), potential=ens.engine.potential)
    try:
        return _original_kick_star(ens)
    finally:
        op._kick = None


def install(model_path=None, toytis=TOYTIS):
    """Patch the toytis modules in this process (see the module docstring). Call before Simulation(settings)."""
    global _DEFAULT_MODEL, _original_kick_star
    if toytis not in sys.path:
        sys.path.insert(0, toytis)
    import ensemble
    _DEFAULT_MODEL = model_path
    ensemble.OrderX = LearnedOrderX
    ensemble.OrderParameter = LearnedOrderX
    if _original_kick_star is None:
        _original_kick_star = ensemble.kick_star
    ensemble.kick_star = kick_star


def interfaces_for_q(model_path, q_levels):
    """Interfaces in lambda units: lambda_A, the lambda of each committor level, lambda_B."""
    m = LambdaModel(model_path)
    inner = [m.interface_for_q(q) for q in q_levels]
    return [m.lambda_a] + inner + [m.lambda_b]
