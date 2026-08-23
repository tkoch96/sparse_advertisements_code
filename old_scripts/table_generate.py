"""DEPRECATED shim (2026-08-23): consolidated into generate_paper_table.py."""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from evaluations.generate_paper_table import *          # noqa: F401,F403
from evaluations.generate_paper_table import main
if __name__ == '__main__':
    main()
