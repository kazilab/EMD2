"""Run the published-data analysis and the compartmental model."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .simulation.model import Params
from .simulation.settings import SimulationSettings, validated_parameters
from .simulation.engine import SimulationResult, simulate_experiment
from .analysis.empirical import analyse
from .analysis.results import analyze_simulation


@dataclass
class ExperimentResult:
    simulation: SimulationResult | None
    empirical: dict | None
    summary: dict | None
    metadata: dict
    paths: list[Path]
    exit_code: int


def run_workflow(outdir: str | Path, settings: SimulationSettings | None = None, *,
                 params: Params | None = None, make_figure: bool = True,
                 make_tables: bool = True, empirical: bool = True) -> ExperimentResult:
    """Return the empirical estimates, simulated trajectories and exported files."""
    settings = settings if settings is not None else SimulationSettings()
    params = validated_parameters(params)
    if not isinstance(empirical, bool):
        raise ValueError("empirical must be True or False")
    if not empirical and not settings.run_model:
        raise ValueError("Request published-data analysis, model simulation or both")

    observed = None
    if empirical:
        print("Estimating published-data contrasts across animals and donors", flush=True)
        observed = analyse(n_bootstrap=settings.n_bootstrap)
    simulated = None
    summary = None
    if settings.run_model:
        simulated = simulate_experiment(params, settings, progress=lambda text: print(text, flush=True))
        print("Evaluating model outputs and consistency checks", flush=True)
        summary = analyze_simulation(simulated, observed)

    from .reporting.provenance import reproduction_metadata
    from .reporting.export import export_results
    metadata = reproduction_metadata(params, settings)
    metadata["empirical_performed"] = observed is not None
    metadata["simulation_performed"] = simulated is not None
    metadata["output_options"] = {"figure": bool(make_figure), "tables": bool(make_tables)}
    if summary is not None:
        summary["reproducibility"] = metadata
    paths = export_results(simulated, observed, summary, outdir,
                           make_figure=make_figure, make_tables=make_tables,
                           metadata=metadata)

    if observed is not None:
        cecum = next(row for row in observed["consortium"] if row["Tissue"] == "CEC")
        print(f"Community caecal BCPN contrast: {cecum['mean_difference']:.6g} source-table µM; "
              f"ratio of means {cecum['mean_ratio']:.6g}; Holm P={cecum['p_holm']:.6g}")
    code = 0
    if summary is not None:
        diagnostics = summary["diagnostics"]
        print(f"Internal model checks: {diagnostics['checks_passed']}/{diagnostics['checks_evaluated']} evaluated")
        for check_id, reason in diagnostics["skipped_checks"].items():
            print(f"  NOT RUN {check_id}: {reason}")
        if simulated.ensemble is not None:
            print(f"Sensitivity ensemble: {simulated.ensemble['n_ok']}/{simulated.ensemble['n_requested']} accepted draws")
        code = 0 if all(summary["checks"].values()) else 1
    print(f"Results saved in {Path(outdir).resolve()}")
    return ExperimentResult(simulated, observed, summary, metadata, paths, code)


def reproduce(outdir: str | Path, n_ensemble: int = 400, *, params: Params | None = None,
              make_figure: bool = True, make_tables: bool = True,
              t_end_hours: float = 720.0, n_time_points: int = 1441,
              n_null_audit: int = 40, run_identifiability: bool = True,
              run_ensemble: bool = True, run_model: bool = True,
              ensemble_seed: int = 20260825, null_audit_seed: int = 20260826,
              n_bootstrap: int = 20000, empirical: bool = True) -> int:
    """Convenience API with explicit run choices and an integer exit status."""
    settings = SimulationSettings(
        t_end_hours=t_end_hours, n_time_points=n_time_points, n_ensemble=n_ensemble,
        n_null_audit=n_null_audit, run_identifiability=run_identifiability,
        run_ensemble=run_ensemble, run_model=run_model, ensemble_seed=ensemble_seed,
        null_audit_seed=null_audit_seed, n_bootstrap=n_bootstrap)
    return run_workflow(outdir, settings, params=params, make_figure=make_figure,
                        make_tables=make_tables, empirical=empirical).exit_code
