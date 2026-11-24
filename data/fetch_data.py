#!/usr/bin/env python3
"""Retrieve or verify EMD2 source files."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from emd2_simulation.acquisition.download import checked_copy, main, rows, sha256

__all__ = ['checked_copy', 'main', 'rows', 'sha256']

if __name__ == '__main__':
    raise SystemExit(main())
