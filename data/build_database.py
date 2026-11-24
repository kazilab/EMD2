#!/usr/bin/env python3
"""Import the EMD2 source extracts into SQLite."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from emd2_simulation.processing.build_database import build, main

__all__ = ['build', 'main']

if __name__ == '__main__':
    raise SystemExit(main())
