"""Checks of published-data units, eligibility and statistical independence."""
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
