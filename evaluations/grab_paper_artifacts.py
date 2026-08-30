"""Back-compat shim (Tom 2026-08-30 consolidation): grab is now a verb
of run_all_paper_evaluations. This forwards `grab_paper_artifacts.py
[--check] [--only S] [--intent I]` to the consolidated entry point."""
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[1:]
verb = 'check' if '--check' in args else 'grab'
args = [a for a in args if a != '--check']
intent = None
if '--intent' in args:
    i = args.index('--intent')
    intent = args[i + 1]
    del args[i:i + 2]
argv = [sys.executable,
        os.path.join(_REPO, 'evaluations', 'run_all_paper_evaluations.py'),
        verb] + ([intent] if intent else []) + args
print('[shim] -> {}'.format(' '.join(argv[1:])))
sys.exit(subprocess.call(argv))
