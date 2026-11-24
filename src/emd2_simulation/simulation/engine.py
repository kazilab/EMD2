"""Numerical simulation and explicitly requested synthetic diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .model import Params
from .conditions import ARMS, RATIO_SEGMENT
from .protocols import run_arms
from .ensemble import run_ensemble
from .scenarios import (dose_only_sensitivity, capacity_decomposition,
                        null_cost_subsample, two_member_gus_sensitivity)
from . import nulls
from .settings import SimulationSettings, validated_parameters
from ..analysis import identifiability as idf


@dataclass
class SimulationResult:
    params: Params
    time: np.ndarray
    nominal: dict
    endpoints: dict
    null: dict
    dose_sensitivity: dict
    capacity_decomposition: dict
    identifiability: dict | None
    scope: dict
    ensemble: dict | None
    gus_sensitivity: dict | None
    settings: SimulationSettings

    @property
    def runs(self):
        return self.nominal

    @property
    def t(self):
        return self.time

    @property
    def ctx(self):
        return {"end": self.endpoints, "null": self.null, "scope": self.scope,
                "ident": self.identifiability or {"urine_only_degenerate": None, "skipped": True}}


def simulate_experiment(params: Params, settings: SimulationSettings, *, progress=None) -> SimulationResult:
    """Run the requested simulations, including fits to synthetic targets."""
    params = validated_parameters(params)
    time = settings.time_grid()
    notify = progress if progress is not None else lambda text: None
    notify("Model: integrating six assigned-capacity arms")
    runs = run_arms(params, time)
    ends = {key: {name: (float(values[-1]) if hasattr(values, "__len__") else float(values))
                  for name, values in trace.items()} for key, trace in runs.items()}
    notify("Model: dose and microbial-capacity sensitivity scenarios")
    dose = dose_only_sensitivity(params, time, ratio_segment=RATIO_SEGMENT)
    decomp = capacity_decomposition(params, time, RATIO_SEGMENT)
    notify("Model: synthetic host-only null and biliary-route refits")
    null = nulls.evaluate_null(params, time, RATIO_SEGMENT)
    null["host_cost"] = nulls.host_parameter_cost(params, time)
    null["biliary_escape"] = nulls.biliary_escape(params, time, RATIO_SEGMENT)
    scope = idf.host_response_verdict(idf.load_design())
    ident = None
    if settings.run_identifiability:
        notify("Model: conditional synthetic recoverability profiles (five measurement sets)")
        ident = idf.run(params, time, progress=notify)
    else:
        notify("Skipping synthetic profiles and check V10")
    ensemble = None
    gus = None
    if settings.run_ensemble:
        ensemble = run_ensemble(
            ARMS, time, RATIO_SEGMENT, n=settings.n_ensemble,
            seed=settings.ensemble_seed, base=params, progress=notify)
        if settings.n_null_audit:
            ensemble["null_cost"] = null_cost_subsample(
                time, n=settings.n_null_audit, seed=settings.null_audit_seed,
                base=params, progress=notify)
        gus = two_member_gus_sensitivity(params, time)
    return SimulationResult(params, time, runs, ends, null, dose, decomp,
                            ident, scope, ensemble, gus, settings)
