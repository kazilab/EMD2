"""Tabulate published measurements and results of the compartmental model."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


TABLE_DESCRIPTIONS = {
    'empirical_consortium': 'Treatment contrasts within each compartment, with animal-level bootstrap intervals and within-family Holm adjustment.',
    'empirical_urine': 'Urinary amounts (nmol/24 h) by collection week and analytical batch.',
    'empirical_bladder': 'Bladder concentrations (source-table nM) by collection week and analytical batch.',
    'empirical_germ_free_controls': 'Germ-free antibiotic versus control contrasts for tissues with unambiguous animal records.',
    'empirical_human_culture': 'Paired oxygen-condition contrasts using one mean per donor and condition.',
    'empirical_human_donor_means': 'Donor means, well counts and source rows. Each donor is a biological replicate.',
    'empirical_complete_donor_sensitivity': 'Oxygen-condition contrasts restricted to donors with complete well replication.',
    'empirical_eligible_observations': 'Individual observations and source rows used in the treatment comparisons.',
    'empirical_acute_descriptive': 'Acute observations and means. Treatment labels, counts and published statistics differ across sources.',
    'empirical_isolate_signals': 'Isolate assay signals in arbitrary units. Wells are technical replicates.',
    'empirical_tumour_outcomes': 'Organism-level tumour outcomes in the source study.',
    'empirical_tumour_experiments': 'BBN tumour incidence by experiment. The analysis stratifies on experiment.',
    'empirical_source_audits': 'Duplicate observations, exclusions and unresolved source discrepancies by source table.',
    'model_endpoints': 'Simulated concentrations, exposure indices and source fractions at the final time point.',
    'model_trajectories': 'Model trajectories in native units by arm and time. Terminal ratios appear in model_endpoints.',
    'model_observable_dictionary': 'Units and interpretation of each model output.',
    'model_sensitivity_endpoints': '5th, 50th and 95th percentiles across parameter draws.',
    'model_checks': 'Model consistency checks, results and supporting sources.',
    'model_check_retention': 'Fraction of accepted parameter draws meeting each evaluated criterion.',
    'model_capacity_decomposition': 'Crossed model scenarios for glucuronidase and oxidase capacities.',
    'model_dose_sensitivity': 'Effects of changing dose alone or jointly with Vmax and Km.',
    'model_biliary_escape': 'Host-only scenarios with alternative values for biliary BCPN transfer.',
    'model_gus_sensitivity': 'Sensitivity to assumed glucuronidase capacity in the two-member arm.',
    'model_identifiability': 'Conditional profile widths for synthetic observations under assigned kinetics.',
    'model_null_comparison': 'Host-only model fitted to the simulated urinary concentration.',
    'model_parameters': 'Assigned parameter values used in the run.',
    'model_acquisition_coverage': 'LC–MS acquisitions by compartment and study arm. Counts include technical records.',
}


def _cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _records(rows, *, drop=()):
    """Keep scalar columns and encode nested provenance in explicit JSON cells."""
    rows = list(rows or [])
    if not rows:
        return pd.DataFrame(columns=["status"])
    return pd.DataFrame([{key: _cell(value) for key, value in row.items() if key not in drop}
                         for row in rows])


def _contrasts(rows):
    converted = []
    for source in rows:
        row = {key: _cell(value) for key, value in source.items()
               if key not in {'observations_a', 'observations_b', 'difference_ci95', 'ratio_ci95'}}
        for name in ('difference', 'ratio'):
            interval = source.get(name + '_ci95')
            row[name + '_ci95_lower'] = interval[0] if interval is not None else None
            row[name + '_ci95_upper'] = interval[1] if interval is not None else None
        converted.append(row)
    return pd.DataFrame(converted) if converted else pd.DataFrame(columns=["status"])


def empirical_tables(empirical: dict | None) -> dict[str, pd.DataFrame]:
    """Build measured-data DataFrames without changing the empirical estimates."""
    if empirical is None:
        return {}
    families = {
        'empirical_consortium': 'consortium', 'empirical_urine': 'urine',
        'empirical_bladder': 'bladder', 'empirical_germ_free_controls': 'germ_free_controls',
        'empirical_human_culture': 'human_culture',
        'empirical_complete_donor_sensitivity': 'human_complete_sensitivity',
    }
    out = {name: _contrasts(empirical[key]) for name, key in families.items()}
    # The complete-donor sensitivity uses the same BCPN measurements and
    # concentration scale as the primary donor analysis.
    out['empirical_complete_donor_sensitivity']['unit'] = 'nM culture medium'
    donor_rows = [dict(row, unit='nM culture medium') for row in empirical['human_donors']]
    out['empirical_human_donor_means'] = _records(donor_rows)
    observations = []
    for name, key in families.items():
        for index, contrast in enumerate(empirical[key]):
            for side in ('a', 'b'):
                group = contrast.get('group_' + side, contrast.get('condition' if side == 'a' else 'reference'))
                for row in contrast.get('observations_' + side, []):
                    observations.append({
                        'analysis_table': name, 'contrast_index': index, 'group': group,
                        'tissue': contrast.get('Tissue'), 'collection_week': contrast.get('Length'),
                        'analytical_batch': contrast.get('Batch'), 'unit': contrast.get('unit'),
                        **row})
    out['empirical_eligible_observations'] = _records(observations)
    out['empirical_acute_descriptive'] = _records(empirical['acute']['descriptive'])
    out['empirical_isolate_signals'] = _records(empirical['isolate_signals'])
    outcomes, experiments = [], []
    for row in empirical['tumour_outcome']['results']:
        outcomes.append({key: value for key, value in row.items() if key != 'per_experiment'})
        experiments.extend({'agent': row['agent'], 'source_table': row['source_table'], **experiment}
                           for experiment in row.get('per_experiment', []))
    out['empirical_tumour_outcomes'] = _records(outcomes)
    out['empirical_tumour_experiments'] = _records(experiments)
    out['empirical_source_audits'] = _records([
        {'source_table': sheet, **record} for sheet, record in empirical['audits'].items()])
    return out


def observable_unit(observable: str) -> str:
    if observable.startswith('c_'):
        return 'nmol/mL (assigned model scale)'
    if observable.startswith('AUC_uro'):
        return 'nmol h/mL (assumed exposure index)'
    if observable == 'U_total':
        return 'nmol cumulative urinary excretion'
    if observable in {'micro_fraction', 'AUC_rel_conventional', 'ratio_cecum'}:
        return 'dimensionless'
    return 'see model definition'


def trajectory_table(runs: dict, t: np.ndarray) -> pd.DataFrame:
    """Return one row of native time-course values per arm and time point."""
    times = np.asarray(t, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError('Model times must be a finite one-dimensional array')
    frames = []
    for arm, trace in runs.items():
        row = {'arm': arm, 'time_h': times}
        for name, value in trace.items():
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                continue  # ratio_cecum is explicitly a terminal statistic.
            if array.shape != times.shape or not np.isfinite(array).all():
                raise ValueError(f'Invalid model trajectory: {arm}/{name}')
            row[name] = array
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def simulation_tables(summary: dict | None, runs: dict | None = None,
                      t: np.ndarray | None = None) -> dict[str, pd.DataFrame]:
    """Build model-result tables, preserving skipped diagnostic status explicitly."""
    if summary is None:
        return {}
    out = {}
    endpoints = [{'arm': arm, 'observable': name, 'value': value, 'unit': observable_unit(name)}
                 for arm, trace in summary['arms'].items() for name, value in trace.items()]
    out['model_endpoints'] = pd.DataFrame(endpoints)
    names = list(next(iter(summary['arms'].values())))
    out['model_observable_dictionary'] = pd.DataFrame([
        {'observable': name, 'unit': observable_unit(name),
         'interpretation': ('modeled source fraction or pool, indistinguishable by assay' if 'BCPNm_' in name or 'BCPNh_' in name
                            else 'proportional to cumulative urinary excretion' if name.startswith('AUC_uro')
                            else 'terminal caecum-to-plasma ratio, conditional on transport assumptions' if name == 'ratio_cecum'
                            else 'model output under the assigned kinetics')}
        for name in names])
    ensemble = summary.get('ensemble') or {}
    bands = ensemble.get('bands') or {}
    out['model_sensitivity_endpoints'] = pd.DataFrame([
        {'arm': arm, 'observable': name, **percentiles, 'unit': observable_unit(name),
         'accepted_draws': ensemble.get('n_ok'), 'requested_draws': ensemble.get('n_requested')}
        for arm, observables in bands.items() for name, percentiles in observables.items()],
        columns=['arm', 'observable', 'p5', 'p50', 'p95', 'unit', 'accepted_draws', 'requested_draws'])
    from ..analysis.checks import CHECKS
    register = {check.cid: check for check in CHECKS}
    if summary.get('model_checks'):
        out['model_checks'] = _records([
            {'check_id': cid, **record} for cid, record in summary['model_checks'].items()])
    else:
        out['model_checks'] = pd.DataFrame([
            {'check_id': cid, 'passed': passed, 'status': 'passed' if passed else 'failed',
             'statement': register[cid].statement if cid in register else '',
             'source': register[cid].source if cid in register else ''}
            for cid, passed in summary['checks'].items()])
    out['model_check_retention'] = pd.DataFrame([
        {'check_id': cid, 'retention_fraction': value, 'accepted_draws': ensemble.get('n_ok'),
         'interpretation': 'frequency across accepted parameter draws'}
        for cid, value in ensemble.get('audit', {}).items()],
        columns=['check_id', 'retention_fraction', 'accepted_draws', 'interpretation'])
    out['model_capacity_decomposition'] = _records([
        {'capacity_cell': key, **row} for key, row in summary['capacity_decomposition']['cells'].items()])
    out['model_dose_sensitivity'] = _records([
        {'rescaling': key, **row} for key in ('dose_only', 'joint_rescaling')
        for row in summary['dose_sensitivity'][key]])
    out['model_biliary_escape'] = _records(summary['null']['biliary_escape']['rows'])
    out['model_gus_sensitivity'] = _records((summary.get('two_member_gus_sensitivity') or {}).get('rows', []))
    ident = summary.get('identifiability') or {}
    sets = ident.get('sets') or {}
    out['model_identifiability'] = _records([
        {'measurement_set': name, **values} for name, values in sets.items()])
    if not len(out['model_identifiability']):
        out['model_identifiability'] = pd.DataFrame([
            {'measurement_set': None, 'status': 'not computed for this run'}])
    out['model_null_comparison'] = pd.DataFrame([
        {'quantity': key, 'value': value}
        for key, value in summary['null'].items() if not isinstance(value, (dict, list))])
    metadata = summary.get('reproducibility', {})
    out['model_parameters'] = pd.DataFrame([
        {'parameter': key, 'value': value, 'status': 'assigned'}
        for key, value in metadata.get('parameters', {}).items()],
        columns=['parameter', 'value', 'status'])
    coverage = ident.get('facts', {}).get('coverage', {})
    out['model_acquisition_coverage'] = _records([
        {'compartment': key, **values} for key, values in coverage.items()])
    if runs is not None and t is not None:
        out['model_trajectories'] = trajectory_table(runs, t)
    return out


def export_tables(empirical: dict | None, summary: dict | None, runs: dict | None,
                  t: np.ndarray | None, outdir: str | Path) -> dict[str, Path]:
    """Write the analysis tables and their catalog."""
    tables = {**empirical_tables(empirical), **simulation_tables(summary, runs, t)}
    directory = Path(outdir) / 'tables'
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        path = directory / f'{name}.csv'
        table.to_csv(path, index=False)
        paths[name] = path
    catalog = pd.DataFrame([
        {'table': name, 'filename': path.name, 'rows': len(tables[name]),
         'description': TABLE_DESCRIPTIONS[name],
         'status': ('not computed or unavailable' if len(tables[name]) == 0 else
                    'not computed' if 'status' in tables[name] and tables[name]['status'].eq('not computed for this run').all() else
                    'available')}
        for name, path in paths.items()])
    path = directory / 'table_catalog.csv';catalog.to_csv(path, index=False)
    paths['table_catalog'] = path
    return paths
