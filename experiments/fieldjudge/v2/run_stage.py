"""Run one existing resumable stage, reporting a machine-readable exit reason.

Use either project Python: this wrapper has no third-party dependencies. systemd
owns retries, not this process. Nothing bypasses the stage's existing checkpoints,
quota guards, evidence checks or pilot gate.
"""
from __future__ import annotations
import json
import os
import runpy
import sys
import time
from pathlib import Path
from control import RETRY, BLOCKED, TemporaryFailure, NeedsAttention, transient


def execute(argv):
    target = Path(argv[0]).resolve()
    sys.argv = [str(target), *argv[1:]]
    sys.path.insert(0, str(target.parent))
    try:
        runpy.run_path(str(target), run_name='__main__')
    except SystemExit as error:
        if error.code in (None, 0):
            return 0, 'complete', ''
        # Unknown child exit codes are not an excuse for an unbounded retry loop.
        if error.code == RETRY:
            return RETRY, 'retry_wait', 'stage requested a delayed retry'
        return BLOCKED, 'needs_attention', 'stage exited unsuccessfully'
    except Exception as error:
        retry = isinstance(error, TemporaryFailure) or (not isinstance(error, NeedsAttention) and transient(error))
        code = RETRY if retry else BLOCKED
        print(f'{target.name}: {"temporary failure; delayed retry" if retry else "paused for inspection"} ({type(error).__name__})', flush=True)
        return code, 'retry_wait' if retry else 'needs_attention', type(error).__name__
    return 0, 'complete', ''


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: run_stage.py SCRIPT [ARGS]')
    argv = sys.argv[1:]
    started = time.time()
    code, state, reason = execute(argv)
    root = Path(os.environ.get('FIELDJUDGE_V2', '/mnt/fast/scholarsreadinglist/fieldjudge/v2'))
    folder = root / 'stage-status'
    folder.mkdir(parents=True, exist_ok=True)
    name = '-'.join([Path(argv[0]).stem, *argv[1:]])
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)[:150]
    target = folder / f'{name}.json'
    tmp = target.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps({'stage': argv, 'state': state, 'reason': reason, 'exit_code': code,
                               'started': started, 'finished': time.time()}))
    os.replace(tmp, target)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
