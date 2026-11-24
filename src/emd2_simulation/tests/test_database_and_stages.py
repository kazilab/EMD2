"""Portable checks for referenced inputs and reusable EMD2 workflow stages."""
from dataclasses import replace
import io
import sqlite3

import numpy as np
import pytest

from emd2_simulation.processing.store import (
    connect_readonly, load_acquisitions, load_observed, load_published,
    source_checksum,
)
from emd2_simulation.simulation.model import Params
from emd2_simulation.simulation.settings import SimulationSettings


def test_sqlite_snapshot_integrity_units_and_provenance_links():
    with connect_readonly() as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not connection.execute('PRAGMA foreign_key_check').fetchall()
        for table, expected in [('published_rows', 1820), ('acquisition_records', 2957),
                                ('observed_components', 11), ('references', 2)]:
            assert connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] == expected
        assert connection.execute(
            'SELECT count(*) FROM published_rows WHERE reference_id IS NULL OR provenance_id IS NULL').fetchone()[0] == 0
        assert not connection.execute(
            'SELECT sheet, source_row FROM published_rows GROUP BY sheet, source_row HAVING count(*) > 1').fetchall()
    data = load_published()
    assert len(data['tables']) == 15
    assert set(data['source']['selection'].split(', ')) == {key.split('_')[0] for key in data['tables']}
    assert sum(map(len, data['tables'].values())) == 1820
    assert len(data['tables']['S39']) == 512
    assert len(data['tables']['S3']) == 10 and len(data['tables']['S42']) == 2
    assert 'technical replicates nested in donor' in data['units']['S23']
    assert 'source-table' in data['units']['S39']
    assert source_checksum('published_extract') == 'a228e264c9c833c8bf5e12da93b6f1def999c96d6844135dc69106765460a76f'
    assert data['source']['sha256'] == source_checksum('source_workbook')


def test_database_is_read_only_and_missing_input_is_not_created(tmp_path):
    with connect_readonly() as connection:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            connection.execute('CREATE TABLE unexpected_write (value INTEGER)')
    missing = tmp_path / 'absent.sqlite'
    with pytest.raises(FileNotFoundError, match='missing'):
        load_published(missing)
    assert not missing.exists()


def test_acquisitions_keep_technical_records_and_original_anatomy_labels():
    rows = load_acquisitions()
    assert len(rows) == 2957
    assert len({row['source_row'] for row in rows}) == 2957
    assert min(row['source_row'] for row in rows) == 2
    assert max(row['source_row'] for row in rows) == 2958
    assert any(row['sample_type'] != 'experimental sample' for row in rows)
    conflicts = [row for row in rows if row['arm'] == '2Mem'
                 and row['compartment'] == 'colon' and '_LVR' in row['source_name'].upper()]
    assert len(conflicts) == 16
    assert 'comparability' in load_observed()


def test_acquisition_checksum_failure_preserves_existing_source(tmp_path):
    from emd2_simulation.acquisition.download import checked_copy
    target = tmp_path / 'source.dat'
    target.write_bytes(b'original')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        checked_copy(io.BytesIO(b'wrong'), target, '0' * 64)
    assert target.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [target]


@pytest.fixture(scope='module')
def completed_model():
    from emd2_simulation.simulation.engine import simulate_experiment
    settings = SimulationSettings(n_ensemble=2, n_null_audit=0,
                                  n_time_points=61, run_identifiability=False,
                                  n_bootstrap=100)
    return simulate_experiment(Params(), settings)


def test_skipped_profiles_are_not_scored_as_passes(completed_model):
    from emd2_simulation.analysis.results import analyze_simulation
    summary = analyze_simulation(completed_model)
    assert 'V10' not in summary['checks']
    assert summary['model_checks']['V10']['status'] == 'not_run'
    assert summary['model_checks']['V10']['passed'] is None
    assert summary['diagnostics']['checks_total'] == 13
    assert summary['diagnostics']['checks_evaluated'] == 12
    assert summary['diagnostics']['checks_passed'] <= 12
    assert not summary['diagnostics']['identifiability_performed']
    assert not summary['diagnostics']['null_cost_audit_performed']
    assert summary['diagnostics']['complete'] is False


def test_analysis_and_reporting_do_not_reintegrate(completed_model, monkeypatch, tmp_path):
    from emd2_simulation.simulation import model
    from emd2_simulation.analysis.results import analyze_simulation
    from emd2_simulation.reporting.export import export_results

    def no_integration(*args, **kwargs):
        pytest.fail('Analysis/reporting must use completed trajectories.')
    monkeypatch.setattr(model, 'solve_ivp', no_integration)
    summary = analyze_simulation(completed_model)
    from emd2_simulation.reporting.export import FIGURE_STEMS
    for stem in FIGURE_STEMS:
        (tmp_path / f'{stem}_with_key.png').write_bytes(b'obsolete preview')
    custom = tmp_path / 'custom_figure.png'
    custom.write_bytes(b'user export')
    paths = export_results(completed_model, None, summary, tmp_path,
                           make_figure=True, make_tables=True)
    assert (tmp_path / 'emd2_simulation_summary.json').is_file()
    assert (tmp_path / 'tables/model_trajectories.csv').is_file()
    assert (tmp_path / 'tables/model_checks.csv').is_file()
    assert (tmp_path / 'emd2_mechanistic.svg').is_file()
    svg = (tmp_path / 'emd2_mechanistic.svg').read_text()
    caption = (tmp_path / 'emd2_mechanistic_legend.md').read_text()
    assert caption.splitlines()[0].startswith('# Figure 4.')
    assert 'Conventional' in svg
    assert 'Conv. plasma' in svg
    assert 'Unity' in svg
    assert not list(tmp_path.glob('*_with_key.png'))
    assert custom.read_bytes() == b'user export'
    assert paths and all(path.is_file() for path in paths)


def test_strict_json_rejects_nan_before_overwriting_existing_results(tmp_path):
    from emd2_simulation.reporting.export import export_results
    existing = tmp_path / 'emd2_simulation_summary.json'
    existing.write_text('previous valid run')
    with pytest.raises(ValueError, match='non-finite'):
        export_results(None, None, {'invalid': float('nan')}, tmp_path,
                       make_figure=False, make_tables=False)
    assert existing.read_text() == 'previous valid run'
    assert not (tmp_path / 'emd2_run_metadata.json').exists()


def test_custom_exposure_index_gain_reaches_nominal_and_ensemble_outputs():
    from emd2_simulation.simulation.conditions import ARMS, BY_KEY
    from emd2_simulation.simulation.model import simulate, trace
    from emd2_simulation.simulation.ensemble import run_ensemble
    times = np.linspace(0.0, 720.0, 31)
    baseline = Params()
    doubled = replace(baseline, k_uro=2 * baseline.k_uro)
    nominal = trace(simulate(baseline, BY_KEY['B'], times), baseline)
    changed = trace(simulate(doubled, BY_KEY['B'], times), doubled)
    assert changed['AUC_uro'][-1] == pytest.approx(2 * nominal['AUC_uro'][-1], rel=2e-5)
    default_draws = run_ensemble(ARMS, times, n=2, seed=31, base=baseline)
    changed_draws = run_ensemble(ARMS, times, n=2, seed=31, base=doubled)
    for q in ('p5', 'p50', 'p95'):
        assert changed_draws['bands']['B']['AUC_uro'][q] == pytest.approx(
            2 * default_draws['bands']['B']['AUC_uro'][q], rel=2e-5)


def test_workflow_keeps_live_results_and_records_incomplete_diagnostics(tmp_path):
    from emd2_simulation.workflow import run_workflow
    custom = replace(Params(), k_uro=2 * Params().k_uro)
    settings = SimulationSettings(n_ensemble=2, n_null_audit=0, n_time_points=31,
                                  run_identifiability=False, n_bootstrap=100)
    result = run_workflow(tmp_path, settings, params=custom,
                          make_figure=False, make_tables=False, empirical=False)
    assert result.simulation.params == custom
    assert result.simulation.nominal and result.simulation.ensemble
    assert result.empirical is None
    assert result.summary['model_checks']['V10']['passed'] is None
    assert result.metadata['parameters']['k_uro'] == custom.k_uro
    assert result.metadata['simulation_performed'] is True
    assert result.metadata['reference_compatible'] is False
    assert not result.summary['diagnostics']['complete']
    assert all(path.is_file() for path in result.paths)
