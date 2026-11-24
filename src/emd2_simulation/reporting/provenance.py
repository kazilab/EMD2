"""Record input checksums, model settings and software versions."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import platform
from ..paths import DATABASE_PATH
from ..simulation.model import Params
from ..simulation.settings import SimulationSettings
from ..analysis.empirical import SEED as EMPIRICAL_SEED
from ..processing.store import source_checksum


def reproduction_metadata(params: Params, settings: SimulationSettings) -> dict:
    defaults = SimulationSettings()
    canonical_model = bool(settings.run_model and params == Params()
                           and settings.t_end_hours == defaults.t_end_hours
                           and settings.n_time_points == defaults.n_time_points)
    canonical_ensemble = bool(settings.run_model and settings.run_ensemble and settings.n_ensemble == defaults.n_ensemble
                              and settings.ensemble_seed == defaults.ensemble_seed)
    canonical_diagnostics = bool(settings.run_model and settings.run_identifiability
                                and settings.n_null_audit == defaults.n_null_audit
                                and settings.null_audit_seed == defaults.null_audit_seed)
    canonical_empirical = settings.n_bootstrap == defaults.n_bootstrap
    return {
        'input_database': 'data/emd2.sqlite',
        'input_database_sha256': hashlib.sha256(DATABASE_PATH.read_bytes()).hexdigest(),
        'published_extract_sha256': source_checksum('published_extract'),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name)
                     for name in ('numpy', 'scipy', 'matplotlib', 'pandas')},
        'settings': asdict(settings),
        'parameters': asdict(params) if settings.run_model else {},
        'empirical_seed': EMPIRICAL_SEED,
        'canonical_empirical': bool(canonical_empirical),
        'canonical_model_and_grid': canonical_model,
        'canonical_ensemble': canonical_ensemble,
        'canonical_diagnostics': canonical_diagnostics,
        'reference_compatible': bool(canonical_model and canonical_ensemble
                                     and canonical_diagnostics and canonical_empirical),
        'interpretation': ('Published-data resampling estimates experimental contrasts. '
                           'Model percentiles describe sensitivity to variation in the assigned kinetics.'),
    }
