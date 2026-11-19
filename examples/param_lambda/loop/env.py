"""Paths and environment of the order-parameter learning loop.

The samplers live in checkouts outside this repository (APPTIS): infretis with StapleTIS (the installed infretis
2025.4 has none), inftools, and an ASE checkout with the overdamped Langevin integrator the infretis ASE engine
imports. They are put first on sys.path here (for in-process use) and on PYTHONPATH for the subprocesses that
run infretis (`infretisrun`). Override with APPTIS, ASE_SRC, INFRETIS_SRC, INFTOOLS_SRC, TOYTIS.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM_LAMBDA = os.path.dirname(HERE)
EXAMPLES = os.path.dirname(PARAM_LAMBDA)
REPO = os.path.dirname(EXAMPLES)

APPTIS = os.environ.get('APPTIS', '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/APPTIS')
ASE_SRC = os.environ.get('ASE_SRC', os.path.join(APPTIS, 'ase'))
INFRETIS_SRC = os.environ.get('INFRETIS_SRC', os.path.join(APPTIS, 'infretis'))
INFTOOLS_SRC = os.environ.get('INFTOOLS_SRC', os.path.join(APPTIS, 'inftools'))
TOYTIS = os.environ.get('TOYTIS', '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis')

PYTHON = sys.executable
INFRETISRUN = os.path.join(os.path.dirname(PYTHON), 'infretisrun')

_SOURCES = [p for p in (ASE_SRC, INFRETIS_SRC, INFTOOLS_SRC) if os.path.isdir(p)]


def put_on_path():
    """The checkouts first on sys.path (ASE before the installed ase), plus param_lambda and examples."""
    for p in [INFTOOLS_SRC, INFRETIS_SRC, ASE_SRC][::-1] + [EXAMPLES, PARAM_LAMBDA, HERE]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    # the ASE checkout must win over the installed package even if ase was imported before
    if 'ase' in sys.modules and os.path.isdir(ASE_SRC) and not sys.modules['ase'].__file__.startswith(ASE_SRC):
        raise ImportError(f'ase was imported from {sys.modules["ase"].__file__} before loop.env; import loop.env first')


def subprocess_env(threads=1):
    """Environment for subprocesses: the checkouts on PYTHONPATH, single-threaded BLAS."""
    env = os.environ.copy()
    old = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = ':'.join(_SOURCES + [EXAMPLES, PARAM_LAMBDA] + ([old] if old else []))
    if os.path.isdir(TOYTIS):
        env.setdefault('TOYTIS', TOYTIS)          # ase_calcs.py (loaded by infretis) reads it
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[v] = str(threads)
    return env


put_on_path()
