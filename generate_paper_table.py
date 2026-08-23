#!/usr/bin/env python
"""Root-level launcher for evaluations/generate_paper_table.py -- so
`python generate_paper_table.py ...` works from the repo root (Tom's
requested invocation). All logic lives in the evaluations module."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluations.generate_paper_table import main
if __name__ == '__main__':
    main()
