"""Exclusive use of the GPU for the fits of a pass (teacher, screening, student).

    with gpu.exclusive(log=log):
        ...

1. A file lock (`OP_LOOP_GPU_LOCK`, default ~/.cache/op_loop_gpu.lock): loops running side by side take turns.
2. Then wait until no other process (any user, any program, e.g. a fit started by hand) has a compute context on
   the GPU, polled every `poll` seconds through nvidia-smi. Our own process and its children are not counted, nor
   other loop processes (`run.py loop`, `-m loop.*`), which hold an idle context between fits and take the lock.
   Set OP_LOOP_GPU_WAIT=0 to skip the wait (the lock is still taken).

The wait is only at the start: a process that starts on the GPU after we began shares it (nothing can prevent that).
"""
import contextlib
import fcntl
import os
import subprocess
import time

import common

LOCK = os.environ.get('OP_LOOP_GPU_LOCK', os.path.expanduser('~/.cache/op_loop_gpu.lock'))


def _own_pids():
    pids = {os.getpid()}
    try:
        out = subprocess.run(['pgrep', '-P', str(os.getpid())], capture_output=True, text=True).stdout
        pids |= {int(p) for p in out.split()}
    except OSError:
        pass
    return pids


def foreign_compute_pids():
    """PIDs with a compute context on the GPU that are not this process (empty if nvidia-smi is missing)."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    own = _own_pids()
    # other loop processes keep an idle CUDA context between their fits; they are coordinated by the lock
    return [int(p) for p in out.split() if p.strip().isdigit() and int(p) not in own
            and not _is_loop_process(int(p))]


def _is_loop_process(pid):
    """A process coordinated by the lock: it has the lock file open (holding or waiting for it), or it is a loop
    command. Counting a waiting lock user as foreign deadlocks: it keeps an idle CUDA context from its earlier fits
    while the lock holder waits for the GPU to be free (2026-11-17, an offline screening script and loop_stag)."""
    try:
        fd_dir = f'/proc/{pid}/fd'
        for fd in os.listdir(fd_dir):
            try:
                if os.readlink(os.path.join(fd_dir, fd)) == os.path.realpath(LOCK):
                    return True
            except OSError:
                continue
    except OSError:
        pass
    cmd = _describe(pid)
    return ('run.py loop' in cmd) or (' -m loop.' in cmd)


def _describe(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return f.read().replace(b'\0', b' ').decode(errors='replace')[:160]
    except OSError:
        return '?'


@contextlib.contextmanager
def exclusive(log=None, poll=60, what='fits'):
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    t0 = time.time()
    with open(LOCK, 'a') as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            common.say(f'gpu: waiting for the lock {LOCK} (another loop is fitting)', log=log)
            fcntl.flock(fh, fcntl.LOCK_EX)
        if os.environ.get('OP_LOOP_GPU_WAIT', '1') != '0':
            said = False
            while True:
                pids = foreign_compute_pids()
                if not pids:
                    break
                if not said:
                    common.say('gpu: waiting, in use by ' + '; '.join(f'{p} ({_describe(p)})' for p in pids), log=log)
                    said = True
                time.sleep(poll)
        waited = time.time() - t0
        if waited > 5:
            common.say(f'gpu: free after {waited / 60:.1f} min; starting the {what}', log=log)
        try:
            yield waited
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
