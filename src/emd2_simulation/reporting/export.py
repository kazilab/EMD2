"""Export EMD2 summaries, figures and tables."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


FIGURE_STEMS = ('emd2_empirical', 'emd2_empirical_controls', 'emd2_mechanistic')


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def export_results(result, empirical: dict | None, summary: dict | None, outdir: str | Path, *,
                   make_figure: bool = True, make_tables: bool = True,
                   metadata: dict | None = None) -> list[Path]:
    """Validate JSON and replace the requested run outputs."""
    metadata = dict(metadata or (summary or {}).get('reproducibility', {}))
    metadata['simulation_performed'] = result is not None
    metadata['empirical_analysis_performed'] = empirical is not None
    metadata['output_options'] = {'figure': bool(make_figure), 'tables': bool(make_tables)}
    if summary is not None:
        metadata['diagnostics_completed'] = summary.get('diagnostics', {})
    documents = {'emd2_run_metadata.json': metadata}
    if empirical is not None:
        documents['emd2_empirical_summary.json'] = empirical
    if summary is not None:
        documents['emd2_simulation_summary.json'] = summary
    try:
        encoded = {name: json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False, default=_json_value) + '\n'
                   for name, value in documents.items()}
    except (ValueError, TypeError) as exc:
        raise ValueError('EMD2 analysis contains non-finite or unsupported values; no outputs were written') from exc
    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, payload in encoded.items():
        path = directory / name
        path.write_text(payload, encoding='utf-8')
        paths.append(path)
    if summary is None:
        (directory / 'emd2_simulation_summary.json').unlink(missing_ok=True)
        (directory / 'emd2_manuscript_tables.json').unlink(missing_ok=True)
    if empirical is None:
        (directory / 'emd2_empirical_summary.json').unlink(missing_ok=True)

    active = set()
    if make_figure and empirical is not None:
        from .empirical_figure import make_figure as measured_figures
        paths.extend(measured_figures(empirical, directory))
        active.update(('emd2_empirical', 'emd2_empirical_controls'))
    if make_figure and result is not None:
        from .figure import make_figure as model_figure
        paths.extend(model_figure(result.nominal, result.time, result.ctx,
                                  result.identifiability, directory, result.ensemble))
        active.add('emd2_mechanistic')
    for stem in set(FIGURE_STEMS) - active:
        for suffix in ('pdf', 'svg', 'png', 'eps', 'tif'):
            (directory / f'{stem}.{suffix}').unlink(missing_ok=True)
        (directory / f'{stem}_legend.md').unlink(missing_ok=True)
    for stem in FIGURE_STEMS:
        (directory / f'{stem}_with_key.png').unlink(missing_ok=True)

    from .tables import TABLE_DESCRIPTIONS, export_tables
    if make_tables:
        table_paths = export_tables(empirical, summary, result.nominal if result else None,
                                    result.time if result else None, directory)
        paths.extend(table_paths.values())
        active_tables = set(table_paths)
    else:
        active_tables = set()
    for name in set(TABLE_DESCRIPTIONS) | {'table_catalog'}:
        if name not in active_tables:
            (directory / 'tables' / f'{name}.csv').unlink(missing_ok=True)
    table_dir = directory / 'tables'
    if table_dir.is_dir() and not any(table_dir.iterdir()):
        table_dir.rmdir()
    return paths
