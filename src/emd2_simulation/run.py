"""Analyse published BCPN measurements and run the EMD2 compartmental model."""
import argparse
from .paths import OUTPUT_DIR
from .simulation.settings import SimulationSettings
from .workflow import run_workflow, reproduce


def main() -> int:
    defaults = SimulationSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empirical-only", action="store_true")
    parser.add_argument("--no-identifiability", action="store_true")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--n-ensemble", type=int, default=defaults.n_ensemble)
    parser.add_argument("--n-null-audit", type=int, default=defaults.n_null_audit)
    parser.add_argument("--t-end", type=float, default=defaults.t_end_hours)
    parser.add_argument("--n-time-points", type=int, default=defaults.n_time_points)
    parser.add_argument("--ensemble-seed", type=int, default=defaults.ensemble_seed)
    parser.add_argument("--null-audit-seed", type=int, default=defaults.null_audit_seed)
    parser.add_argument("--n-bootstrap", type=int, default=defaults.n_bootstrap)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument("--outdir", default=str(OUTPUT_DIR / "reproduced"))
    args = parser.parse_args()
    try:
        settings = SimulationSettings(
            t_end_hours=args.t_end, n_time_points=args.n_time_points,
            n_ensemble=args.n_ensemble, n_null_audit=args.n_null_audit,
            ensemble_seed=args.ensemble_seed, null_audit_seed=args.null_audit_seed,
            run_identifiability=not args.no_identifiability,
            run_ensemble=not args.no_ensemble, run_model=not args.empirical_only,
            n_bootstrap=args.n_bootstrap)
    except ValueError as error:
        parser.error(str(error))
    return run_workflow(args.outdir, settings, make_figure=not args.no_figure,
                        make_tables=not args.no_tables).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
