"""Checks of published-data units, eligibility and statistical independence."""
import tempfile
from pathlib import Path
import json
import numpy as np
import pytest

from emd2_simulation.empirical import (
    DATA_PATH, analyse, contrast, holm, load_data, unique_observations,
)
from emd2_simulation.fit.observed import _same_animal_ratios


@pytest.fixture(scope="module")
def result():
    return analyse(n_bootstrap=2000)


def test_duplicates_do_not_become_animals_and_conflicts_are_not_averaged():
    rows = [{"ID": i, "value": v, "source_row": k}
            for k, (i, v) in enumerate((("a", 2), ("a", 2), ("b", 1), ("b", 3)), 2)]
    kept, audit = unique_observations(rows, ["ID"], "value")
    assert [(r["ID"], r["value"]) for r in kept] == [("a", 2)]
    assert kept[0]["source_rows"] == [2, 3]
    assert audit["exact_duplicate_rows_collapsed"] == 1
    assert audit["conflicting_keys_excluded"][0]["key"] == ["b"]


def test_donor_not_well_is_culture_inference_unit(result):
    assert len(result["human_donors"]) == 30
    assert sum(r["technical_n"] for r in result["human_donors"]) == 113
    assert all(r["n_a"] == r["n_b"] == 10 and r["paired"] for r in result["human_culture"])
    assert all(r["n_a"] == 9 for r in result["human_complete_sensitivity"])
    assert result["human_culture"][0]["p_value"] == pytest.approx(0.09375)


def test_consortium_uses_eight_animals_and_all_compartments(result):
    rows = result["consortium"]
    assert len(rows) == 9
    assert all(r["n_a"] == r["n_b"] == 8 and r["family_size"] == 9 for r in rows)
    assert result["audits"]["S39"]["exact_duplicate_rows_collapsed"] == 16
    cec = next(r for r in rows if r["Tissue"] == "CEC")
    assert cec["mean_difference"] == pytest.approx(2.394654468, abs=1e-8)
    assert cec["mean_ratio"] == pytest.approx(3.48662043, abs=1e-6)
    # Four extreme label allocations / choose(16,8); Holm family has 9 tests.
    assert cec["p_value"] == pytest.approx(4 / 12870)
    assert cec["p_holm"] == pytest.approx(36 / 12870)


def test_gf_conflicts_are_quarantined_and_legacy_ratio_removed(result):
    assert len(result["audits"]["S15"]["conflicting_keys_excluded"]) == 12
    assert {r["Tissue"] for r in result["germ_free_controls"]} == {"PL", "BLD", "LVR", "KID"}
    tables = load_data()["tables"]
    assert _same_animal_ratios(tables["S15"], "uM", dg=False) == {}
    paired = _same_animal_ratios(tables["S39"], "uM_raw", dg=True)
    assert len(paired["2Mem"]) == len(paired["3Mem"]) == 8


def test_collection_strata_are_not_pooled(result):
    for family in ("urine", "bladder"):
        rows = result[family]
        assert len(rows) == 5
        assert len({(r["Length"], r["Batch"]) for r in rows}) == 5
        assert all(r["family_size"] == 5 for r in rows)


def test_paired_bootstrap_preserves_within_donor_difference():
    r = contrast([2, 11, 101], [1, 10, 100], key="paired", paired=True, n_bootstrap=1000)
    assert r["difference_ci95"] == pytest.approx([1, 1])
    assert r["p_value"] == pytest.approx(2/8)


def test_zero_denominator_does_not_receive_a_pseudocount():
    r = contrast([1, 2, 3], [0, 0, 2], key="zeros", n_bootstrap=1000)
    assert r["mean_ratio"] == pytest.approx(3)
    assert r["ratio_ci95"] is None
    assert np.isfinite(r["difference_ci95"]).all()


def test_holm_adjustment_restores_original_order_and_is_monotone():
    assert holm([0.03, 0.001, 0.02]) == pytest.approx([0.04, 0.003, 0.04])


def test_s32_side_table_does_not_overwrite_strain_labels(result):
    rows = load_data()["tables"]["S32"]
    assert rows[0]["Strain"] is not None
    strains = {r["strain"] for r in result["isolate_signals"]}
    assert "Escherichia coli ED1a" in strains
    assert not result["audits"]["S32"]["missing_strain_rows_excluded"]


def test_acute_discrepancies_prevent_calibration_claim(result):
    acute = result["acute"]
    assert sum(r["agreement_within_1nM"] for r in acute["published_contrasts"]) == 11
    assert {r["n_recorded_ids"] for r in acute["descriptive"]} == {5, 6}
    assert all(len(r["ids"]) == len(set(r["ids"])) for r in acute["descriptive"])
    assert result["kinetic_calibration"]["performed"] is False
    assert "0.085" in " ".join(acute["limitations"])


def test_provenance_cannot_claim_bundled_hash_for_modified_data():
    data = load_data()
    data["source"]["citation"] += " modified fixture"
    r = analyse(data, n_bootstrap=100)
    assert r["input_sha256"] is None
    assert len(r["input_content_sha256"]) == 64


@pytest.mark.parametrize("a,b", [([1, np.nan], [1, 2]), ([np.inf, 1], [0, 1])])
def test_nonfinite_observations_are_rejected(a, b):
    with pytest.raises(ValueError, match="Finite"):
        contrast(a, b, key="invalid")


def test_stated_limits_are_single_sourced_and_cover_each_area():
    """The limits paragraph is emitted, not hand-typed downstream.

    Both Word generators and the README quote these strings, so a missing or
    renamed key would let the documents drift into paraphrase.
    """
    limits = analyse(n_bootstrap=100)["limits"]
    assert set(limits) == {
        "evidence_basis", "data_handling", "acute_timecourse", "unity_cutoff",
        "kcc_scope", "acquisition_counts", "model_magnitudes",
        "model_mechanism_divergence", "power", "sequence_level_outcome"}
    assert all(isinstance(v, str) and len(v) > 80 for v in limits.values())
    # The load-bearing qualifications, asserted so they cannot be softened away.
    assert "not inferred from taxonomic" in limits["evidence_basis"]
    assert "withdrawn" in limits["data_handling"]
    assert "reconciliation would not remove" in limits["acute_timecourse"]
    assert "prior to the model-side" in limits["unity_cutoff"]
    assert "genotoxicity" in limits["kcc_scope"]
    assert "not animals" in limits["acquisition_counts"]
    assert "refuted" in limits["model_magnitudes"]
    assert "not evidence of no effect" in limits["power"]


def test_acute_limitations_lead_with_the_irreducible_design_limit():
    """Reconciling labels, counts and statistics would not make the acute table
    support source attribution; being sampled first at 1 h would still. The
    unfixable reason is stated first so it is not read as a pending to-do."""
    acute = analyse(n_bootstrap=100)["acute"]
    assert acute["limitations"][0].startswith("Design limit, not reconcilable")
    assert "1 h" in acute["limitations"][0]
    assert "reconciliation cannot remove" in acute["use"]


def test_canvas_guard_ignores_ticks_outside_the_view():
    """The guard checks rendered text. A locator also emits Text objects for
    ticks beyond the axis limits, which matplotlib never draws; counting those
    made the guard reject a correct figure over a log-axis decade tick the
    panel does not show. Regression: a view-excluded tick must not trip it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from emd2_simulation.empirical_figure import _save

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    # Generous margins so the only candidate for overflow is the out-of-view
    # tick this test is about, not an edge label on a cramped canvas.
    fig.subplots_adjust(left=0.25, right=0.75, top=0.75, bottom=0.25)
    ax.set_xscale("log")
    ax.plot([0.1, 5.0], [1, 2])
    ax.set_xlim(0.05, 10.0)          # locator still makes a label at 100
    assert any(t.get_text() and t.get_position()[0] > 10.0
               for t in ax.xaxis.get_ticklabels()), "no out-of-view tick to test"
    with tempfile.TemporaryDirectory() as tmp:
        _save(fig, Path(tmp), "guard_probe")      # must not raise


def test_tumour_outcome_is_stratified_by_experiment_not_pooled():
    """S3 is five separate experiments. Every other family in this module keeps
    its strata apart, and pooling them into one 2x2 -- which is what the source
    article reports -- would treat experiment as ignorable. The stratified
    statistic is primary; the pooled one is kept only to reconcile with the
    printed value."""
    t = analyse(n_bootstrap=200)["tumour_outcome"]
    s3 = next(r for r in t["results"] if r["source_table"] == "S3")
    assert s3["n_experiments"] == 5
    assert s3["primary_test"].startswith("Cochran-Mantel-Haenszel")
    assert s3["p_value"] < s3["pooled_yates_p_for_reconciliation"]
    # the article's printed value, reproduced by the pooled Yates test only
    assert s3["pooled_yates_p_for_reconciliation"] == pytest.approx(1.6e-5, rel=0.05)
    # direction consistent in every experiment
    assert s3["experiments_favouring_control"] == 5
    for s in s3["per_experiment"]:
        assert s["treated_n"] == s["treated_normal"] + s["treated_neoplasia"]
        assert s["control_n"] == s["control_normal"] + s["control_neoplasia"]


def test_tumour_outcome_counts_match_the_published_table():
    t = analyse(n_bootstrap=200)["tumour_outcome"]
    s3 = next(r for r in t["results"] if r["source_table"] == "S3")
    assert (s3["pooled_treated"]["normal"], s3["pooled_treated"]["neoplasia"]) == (26, 6)
    assert (s3["pooled_control"]["normal"], s3["pooled_control"]["neoplasia"]) == (7, 23)
    assert s3["pooled_treated"]["n"] == 32 and s3["pooled_control"]["n"] == 30
    s42 = next(r for r in t["results"] if r["source_table"] == "S42")
    assert s42["treated"]["n"] == 8 and s42["control"]["n"] == 9
    assert s42["p_value"] == pytest.approx(0.0152, abs=5e-4)
    assert s42["n_experiments"] == 1


def test_tumour_outcome_scores_no_kcc_and_says_which():
    """Sequence-level evidence must not be read as KCC2, KCC6 or KCC10, and the
    antibiotic handle is confounded. Asserted so the disclaimer cannot be
    dropped while the result is kept."""
    t = analyse(n_bootstrap=200)["tumour_outcome"]
    assert t["scores_kcc"] is None
    joined = " ".join(t["does_not_establish"])
    for token in ("KCC2", "KCC6", "KCC10", "Isolate-level"):
        assert token in joined
    assert "everything else" in joined          # the confounding statement
    limits = analyse(n_bootstrap=200)["limits"]
    assert "stratified by experiment, not pooled" in limits["sequence_level_outcome"]
