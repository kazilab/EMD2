"""Validated settings for empirical analysis and the assigned-kinetics model."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from numbers import Real
import numpy as np
from .model import Params


@dataclass(frozen=True)
class SimulationSettings:
    """Time grid, random seeds and options for the model and empirical analysis."""
    t_end_hours: float = 720.0
    n_time_points: int = 1441
    n_ensemble: int = 400
    n_null_audit: int = 40
    ensemble_seed: int = 20260825
    null_audit_seed: int = 20260826
    run_identifiability: bool = True
    run_ensemble: bool = True
    run_model: bool = True
    n_bootstrap: int = 20000

    def __post_init__(self):
        for name, minimum in (("n_time_points", 2), ("n_ensemble", 1),
                              ("n_null_audit", 0), ("ensemble_seed", 0),
                              ("null_audit_seed", 0), ("n_bootstrap", 100)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if (isinstance(self.t_end_hours, bool) or not isinstance(self.t_end_hours, Real)
                or not np.isfinite(self.t_end_hours) or self.t_end_hours <= 0):
            raise ValueError("t_end_hours must be finite and positive")
        for name in ("run_identifiability", "run_ensemble", "run_model"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False")

    def time_grid(self) -> np.ndarray:
        return np.linspace(0.0, float(self.t_end_hours), self.n_time_points)


def validated_parameters(params: Params | None = None) -> Params:
    """Require finite supported parameter values without fitting any rates."""
    params = params if params is not None else Params()
    zero_allowed = {"k_back", "k_bile_BCPN"}
    for name, value in asdict(params).items():
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0 or (value == 0 and name not in zero_allowed):
            allowed = "non-negative" if name in zero_allowed else "positive"
            raise ValueError(f"{name} must be {allowed} for this diagnostic workflow")
    return params


def prepare_parameters(overrides: dict | None = None) -> Params:
    """Apply independent assigned-parameter edits for exploratory simulation."""
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(asdict(Params()))
    if unknown:
        raise ValueError(f"Unknown model parameters: {', '.join(sorted(unknown))}")
    return validated_parameters(replace(Params(), **overrides))
