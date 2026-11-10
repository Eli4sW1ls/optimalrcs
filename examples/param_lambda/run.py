"""Command line of the parametric-order-parameter pipeline (see common.py for the steps).

    python run.py extract  <system>     frame cache from the staple run and its q_np
    python run.py fit-qnp  <system>     NPNEq fit of q_np with load_tis_data_cupy.py (systems without one)
    python run.py screen   <system>     M2 feature ranking
    python run.py fit      <system> [--only m1|m3] [--quick]
    python run.py evaluate <system>
    python run.py export   <system>
    python run.py report   [<system> ...]

<system> is a config name in configs/ (z_br, cc_br, flat1d, bumps1d). Long steps go through systemd-run, e.g.
    systemd-run --user --unit=pl-z --collect -p MemoryMax=7G -p MemorySwapMax=0 \\
        /home/elias/anaconda3/envs/test/bin/python run.py extract z_br
"""
import argparse
import os
import subprocess
import sys
import time

import common


def cmd_extract(cfg, args):
    import data
    t0 = time.time()
    data.extract(cfg)
    f = data.Frames(cfg)
    common.say(f.summary())
    common.say(f'extract done in {time.time() - t0:.0f} s')


def cmd_fit_qnp(cfg, args):
    d = cfg['data']
    fig_dir = os.path.dirname(d['q_np'])
    os.makedirs(fig_dir, exist_ok=True)
    argv = [os.path.join(common.EXAMPLES, 'load_tis_data_cupy.py'), d['tis_dir']]
    argv += list(d['loader_args']) + list(d.get('qnp_fit_args', [])) + ['--figure-dir', fig_dir]
    # load_tis_data_cupy.py draws the CV of every iteration from np.random without a seed: seed it here.
    boot = (f'import sys, runpy, numpy, cupy; numpy.random.seed({cfg["seed"]}); cupy.random.seed({cfg["seed"]}); '
            f'sys.argv = {argv!r}; runpy.run_path(sys.argv[0], run_name="__main__")')
    common.say('seed', cfg['seed'], ':', ' '.join(argv))
    subprocess.run([sys.executable, '-u', '-c', boot], check=True, cwd=common.EXAMPLES)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command')
    p.add_argument('systems', nargs='*')
    p.add_argument('--only', choices=['m1', 'm3'])
    p.add_argument('--quick', action='store_true', help='small sweep for testing the pipeline')
    p.add_argument('--fresh', action='store_true', help='fit: start a new results list (default: skip runs done)')
    args = p.parse_args()
    commands = {'extract': cmd_extract, 'fit-qnp': cmd_fit_qnp}
    if args.command not in commands:
        import steps
        commands[args.command] = getattr(steps, 'cmd_' + args.command.replace('-', '_'))
    if args.command == 'report':
        commands['report'](args.systems or ['z_br', 'cc_br', 'flat1d', 'bumps1d'], args)
        return
    for name in args.systems:
        commands[args.command](common.load_config(name), args)


if __name__ == '__main__':
    main()
