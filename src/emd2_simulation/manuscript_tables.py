"""Export supplementary model tables from a completed reference run."""
from .reporting.manuscript_tables import audit, build, main

if __name__ == "__main__":
    raise SystemExit(main())
