#!/usr/bin/env python3
"""Run (or retry) the gated freeze of a finished loop: python -m loop.freeze_cmd <loop config>"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop import env, controller, passes, pool as pool_mod  # noqa: E402,F401

if __name__ == '__main__':
    cfg = controller.load_loop_config(sys.argv[1])
    st = controller.State(cfg)
    passes.freeze(cfg, st, pool_mod.Pool(cfg['loop_dir']), os.path.join(cfg['loop_dir'], 'loop.log'))
