"""Regressions for the EMD2 build.

These lock down the invariants that a later edit could plausibly break without
any check failing loudly: the steady-state neutrality that justified adding the
kidney, the metric corrections in the identifiability profile, and -- most
importantly -- the fact that V8's superseded form is NOT robust, so the
restatement cannot be quietly reverted.
"""

from __future__ import annotations

from dataclasses import replace, fields

import numpy as np
import pytest

from emd2_simulation import identifiability as idf
from emd2_simulation.conditions import ARMS, BY_KEY, CHECKS
from emd2_simulation.ensemble import (FREE_SIGMA, ROBUSTNESS, run_ensemble,
                                      sample_params, two_member_gus_sensitivity)
from emd2_simulation.model import (IDX, SEGMENTS, Arm, Params, lumen_plasma_ratio,
                                   simulate, trace)

T = np.linspace(0.0, 720.0, 1441)
P = Params()


@pytest.fixture(scope="module")
def ends() -> dict:
    out = {}
    for a in ARMS:
        y = simulate(P, a, T)
        tr = trace(y, P)
        e = {k: float(v[-1]) for k, v in tr.items()}
        e["ratio_cecum"] = lumen_plasma_ratio(y, P, "cecum")
        out[a.key] = e
    return out


# --- the kidney compartment ------------------------------------------------

def test_kidney_is_steady_state_neutral() -> None:
    """The kidney was interposed on the promise that it changes no published
    quantity. At steady state k_kidney_out * K == k_renal * P, so urine and AUC
    must be invariant to k_kidney_out over a wide range."""
    ref = None
    for k_out in (2.0, 6.0, 30.0):
        q = replace(P, k_kidney_out=k_out)
        tr = trace(simulate(q, BY_KEY["B"], T), q)
        got = np.array([tr["c_BCPN_urine"][-1], tr["AUC_uro"][-1],
                        tr["c_BCPN_cecum"][-1], tr["c_BCPN_plasma"][-1]])
        if ref is None:
            ref = got
        else:
            assert np.max(np.abs(got - ref) / ref) < 5e-3


def test_kidney_flux_balances_at_steady_state(ends: dict) -> None:
    for a in ARMS:
        y = simulate(P, a, T)
        k_amt = y[IDX["BCPNh_k"], -1] + y[IDX["BCPNm_k"], -1]
        p_amt = y[IDX["BCPNh_p"], -1] + y[IDX["BCPNm_p"], -1]
        assert P.k_kidney_out * k_amt == pytest.approx(P.k_renal * p_amt, rel=1e-3)


# Parameters whose default is deliberately ZERO, so a multiplicative probe
# cannot move them. They are structural switches, not rates, and each is swept
# explicitly somewhere else in the build.
ZERO_DEFAULT_SWITCHES = {"k_bile_BCPN": 0.05}


def test_no_dead_parameters() -> None:
    """Every field of Params must change some observable. Q_bile and V_liver
    were both declared and unused before the audit; V_liver was removed and
    Q_bile was wired to the biliary concentration observable.

    Zero-default switches are probed ADDITIVELY -- multiplying zero by 1.5 tests
    nothing, and would have let a dead switch through unnoticed.
    """
    base = trace(simulate(P, BY_KEY["B"], T), P)
    keys = [k for k in base if k.startswith(("c_", "AUC", "U_", "micro"))]
    ref = np.array([float(base[k][-1]) for k in keys])
    for f in fields(Params):
        if f.name in ZERO_DEFAULT_SWITCHES:
            assert getattr(P, f.name) == 0.0, f"{f.name} no longer defaults to 0"
            probe = ZERO_DEFAULT_SWITCHES[f.name]
        else:
            probe = float(getattr(P, f.name)) * 1.5
        q = replace(P, **{f.name: probe})
        got = np.array([float(trace(simulate(q, BY_KEY["B"], T), q)[k][-1])
                        for k in keys])
        moved = np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12))
        assert moved > 1e-6, f"{f.name} changes no observable"


# --- V7b: the headline's stated scope --------------------------------------

def test_biliary_bcpn_is_off_by_default_so_no_published_number_moved() -> None:
    """The route was added to PRICE the headline, not to change it."""
    assert P.k_bile_BCPN == 0.0
    tr = trace(simulate(P, BY_KEY["B"], T), P)
    assert float(tr["c_BCPN_cecum"][-1]) == pytest.approx(9074.805, rel=1e-4)
    assert lumen_plasma_ratio(simulate(P, BY_KEY["B"], T), P, "cecum") == (
        pytest.approx(141.76, rel=1e-3))


def test_independent_exchange_coefficients_do_not_enforce_a_unity_bound() -> None:
    """Guard against restoring an unsupported physical source boundary."""
    baseline = lumen_plasma_ratio(simulate(P, BY_KEY["Germ-Free"], T), P)
    q = replace(P, k_back=0.05, k_bile_BCPN=0.0)
    altered = lumen_plasma_ratio(simulate(q, BY_KEY["Germ-Free"], T), q)
    assert baseline < 1.0
    assert altered == pytest.approx(1.00353296, rel=1e-6)
    assert altered > 1.0


def test_biliary_secretion_breaks_v7_while_the_null_still_matches_urine() -> None:
    """The finding V7b exists to record, asserted so it cannot be lost again.

    A host-only null with a few per cent of BCPN clearance through bile matches
    urinary BCPN to optimiser tolerance AND pushes the caecal lumen/plasma ratio
    above 1 -- so V7 is a statement about a missing route, not about host
    metabolism in general. If a later edit reinstates V7 as unconditional, this
    fails.
    """
    from emd2_simulation import nulls as nl
    be = nl.biliary_escape(P, T, "cecum", grid=(0.0, 0.05))
    zero, bile = be["rows"][0], be["rows"][1]
    assert zero["null_ratio"] < 1.0
    assert bile["null_ratio"] > 1.0
    assert bile["urine_rel_err"] < 1e-6          # still matches urine exactly
    assert be["min_frac_breaking_V7"] < 0.10     # a few per cent of clearance


def test_biliary_rate_that_reproduces_the_measured_germfree_ratio() -> None:
    """0.0227/h puts the host-only ratio at the MEASURED 0.663 (Suppl. S15,
    n=6), which the build without this route puts at 0.080. Recorded because it
    makes the omission the leading candidate for the Pass 5 discrepancy rather
    than an unexplained 'structural' gap."""
    q = replace(P, k_bile_BCPN=0.0227)
    r = lumen_plasma_ratio(simulate(q, BY_KEY["Germ-Free"], T), q, "cecum")
    assert r == pytest.approx(0.663, rel=0.02)


# --- deconjugation vs conversion -------------------------------------------

def test_microbial_oxidation_is_a_net_sink_for_systemic_exposure() -> None:
    """Adding the oxidase to a deconjugating community LOWERS total urothelial
    exposure, because luminal BCPN is a poorly absorbed acid. Counter-intuitive
    enough that it must be asserted, not rediscovered."""
    from emd2_simulation.ensemble import capacity_decomposition
    d = capacity_decomposition(P, T)
    assert d["oxidase_is_a_systemic_sink"] is True
    assert d["oxidase_effect_on_total_AUC"] < 1.0
    # ...while raising LUMINAL BCPN by three orders of magnitude
    gus_only = d["cells"]["gus1.00_ox0.00"]
    both = d["cells"]["gus1.00_ox1.00"]
    assert both["c_BCPN_cecum"] > 1000.0 * gus_only["c_BCPN_cecum"]


def test_v2_is_carried_by_deconjugation_not_by_conversion() -> None:
    """Depleting beta-glucuronidase alone passes V2; depleting the oxidase alone
    fails it. So V2 does not test microbial CONVERSION, and the check text now
    says so."""
    from emd2_simulation.ensemble import capacity_decomposition
    v2 = capacity_decomposition(P, T)["v2_attribution"]
    assert v2["deplete_both"]["v2_holds"] is True
    assert v2["deplete_gus_only"]["v2_holds"] is True
    assert v2["deplete_ox_only"]["v2_holds"] is False


def test_states_stay_non_negative() -> None:
    for a in ARMS:
        y = simulate(P, a, T)
        assert y.min() > -1e-8 * max(y.max(), 1.0)      # relative, not absolute


def test_no_absolute_scale_assumptions_survive_a_rescaling() -> None:
    """The model is exactly homogeneous under a joint rescaling of dose, Vmax
    and Km. Anything that changes under one is a hidden absolute-scale
    assumption -- which is how the hard-coded optimiser bounds in
    profile_capacity were caught."""
    lam = 7.0
    q = replace(P, dose_rate=P.dose_rate * lam,
                Vmax_gus=P.Vmax_gus * lam, Km_gus=P.Km_gus * lam,
                Vmax_ox=P.Vmax_ox * lam, Km_ox=P.Km_ox * lam)
    for key in ("B", "BA", "2Mem"):
        a = trace(simulate(P, BY_KEY[key], T), P)
        b = trace(simulate(q, BY_KEY[key], T), q)
        assert float(b["c_BCPN_cecum"][-1]) == pytest.approx(
            lam * float(a["c_BCPN_cecum"][-1]), rel=1e-4)
        assert float(b["micro_fraction"][-1]) == pytest.approx(
            float(a["micro_fraction"][-1]), rel=1e-4)
        assert lumen_plasma_ratio(simulate(q, BY_KEY[key], T), q, "cecum") == (
            pytest.approx(lumen_plasma_ratio(simulate(P, BY_KEY[key], T), P,
                                             "cecum"), rel=1e-4))
    # and the identifiability profile must still run at the rescaled magnitude
    grid = np.linspace(0.05, 2.5, 3) * q.Vmax_ox
    idf.profile_capacity(q, T, ["cecum"], [BY_KEY["B"]], grid=grid)


def test_dose_matches_the_drinking_water_mass_balance() -> None:
    """0.05% treated as w/v (0.5 mg/mL), ~5 mL/day, C8H18N2O2 = 174.244 g/mol.

    The paper sometimes writes v/v; near-unit density keeps the same order.
    """
    mw = 8 * 12.011 + 18 * 1.008 + 2 * 14.007 + 2 * 15.999
    expected = (0.5 * 5.0) / mw * 1e6 / 24.0                  # nmol/h
    assert P.dose_rate == pytest.approx(expected, rel=0.02)


# --- identifiability metric ------------------------------------------------

def test_arms_without_oxidase_carry_no_information_about_vmax_ox() -> None:
    """The mean-RMSE dilution bug turned on this fact; keep it asserted."""
    grid = np.linspace(0.05, 2.5, 4) * P.Vmax_ox
    for key in ("Germ-Free", "2Mem"):
        assert not idf._informative(P, T, ["cecum", "colon"], BY_KEY[key], grid)
    for key in ("B", "BA"):
        assert idf._informative(P, T, ["cecum", "colon"], BY_KEY[key], grid)


def test_profile_reports_uninformative_arms_by_name() -> None:
    grid = np.linspace(0.05, 2.5, 4) * P.Vmax_ox
    d = idf.profile_capacity(P, T, ["cecum", "colon"],
                             [BY_KEY[k] for k in ("B", "BA", "Germ-Free")],
                             grid=grid)
    assert d["uninformative_arms"] == ["Germ-Free"]
    assert d["n_informative_arms"] == 2


def test_urine_only_ridge_is_reported_as_saturated_not_as_a_width() -> None:
    """The '245%' quoted in the first release was the grid width, not the
    design's ridge. Saturation must be detected so it is never quoted again."""
    d = idf.profile_capacity(P, T, ["urine"], [BY_KEY[k] for k in ("B", "BA")],
                             grid=np.linspace(0.05, 2.5, 6) * P.Vmax_ox)
    assert d["saturated"] is True


def test_worst_residual_metric_is_never_looser_than_the_mean() -> None:
    grid = np.linspace(0.05, 2.5, 6) * P.Vmax_ox
    d = idf.profile_capacity(P, T, ["urine", "cecum", "colon"],
                             [BY_KEY[k] for k in ("B", "BA")], grid=grid)
    assert d["ridge_width"] <= d["ridge_width_mean_metric"] + 1e-12


# --- scope verdict ---------------------------------------------------------

def test_host_response_layer_is_unidentifiable_from_the_real_design() -> None:
    v = idf.host_response_verdict(idf.load_design())
    assert v["imported_endpoints"] == list(idf.IMPORTED_ENDPOINTS)
    assert v["host_response_unidentifiable"] is True
    assert "kidney" in v["compartments"] and "urine" in v["compartments"]


# --- ensemble --------------------------------------------------------------

def test_sample_params_perturbs_only_declared_free_parameters() -> None:
    held_fixed = [f.name for f in fields(Params) if f.name not in FREE_SIGMA]
    # The assertion below was vacuous for a while: every Params field was in
    # FREE_SIGMA, so the loop body never ran and the test passed on an empty
    # set. Assert that there is something to check before checking it.
    assert held_fixed, "nothing is held fixed; this test asserts nothing"
    rng = np.random.default_rng(0)
    q = sample_params(rng)
    for name in held_fixed:
        assert getattr(q, name) == getattr(P, name), name


def test_sample_params_does_not_touch_arm_capacities() -> None:
    """Arm capacities define what an arm IS. Sampling them would make the
    germ-free arm stop being germ-free."""
    before = [(a.gus, a.ox) for a in ARMS]
    sample_params(np.random.default_rng(1))
    assert [(a.gus, a.ox) for a in ARMS] == before


@pytest.mark.parametrize("n", [0, -1])
def test_run_ensemble_rejects_nonpositive_draw_counts(n: int) -> None:
    with pytest.raises(ValueError):
        run_ensemble(ARMS, T, n=n)


def test_superseded_v8_is_reported_as_not_robust() -> None:
    """The guard on the restatement.

    V8 originally asserted 2Mem >= 0.95 x conventional. That holds in ~64% of
    draws; the restated form (2Mem above the ANTIBIOTIC arm) holds in ~99.8%.
    If a later edit reverts the restatement, this fails.
    """
    ens = run_ensemble(ARMS, T, n=24, seed=7)
    assert ens["audit"]["V8"] > ens["audit"]["V8-superseded"]
    assert ens["audit"]["V8-superseded"] < 0.90
    assert ens["audit"]["V8"] >= 0.95


def test_v7_null_ratio_is_invariant_to_the_nulls_free_parameters() -> None:
    """Why V7 needs no per-draw null refit in the ensemble.

    The host-only null is the germ-free structure. Its free knobs
    (k_ox_host, k_renal, k_clear_p) rescale the host BCPN pool's amplitude;
    lumen and plasma both move with that amplitude, so the lumen/plasma ratio
    cancels. If this ever stops being true, the ensemble's V7 shortcut is wrong.
    """
    base = lumen_plasma_ratio(simulate(P, BY_KEY["Germ-Free"], T), P, "cecum")
    assert base < 1.0
    for name, mul in (("k_ox_host", 5.0), ("k_renal", 0.2), ("k_clear_p", 4.0)):
        q = replace(P, **{name: getattr(P, name) * mul})
        got = lumen_plasma_ratio(simulate(q, BY_KEY["Germ-Free"], T), q, "cecum")
        assert got == pytest.approx(base, rel=1e-9, abs=1e-12)


def test_ratio_separates_converting_from_nonconverting_in_every_draw() -> None:
    """The actual headline. If this ever stops being 100% the build's central
    claim is in trouble and should not be papered over."""
    ens = run_ensemble(ARMS, T, n=24, seed=11)
    assert ens["bands"]["B"]["ratio_cecum"]["p5"] > 10.0
    assert ens["bands"]["2Mem"]["ratio_cecum"]["p95"] < 1.0
    assert ens["bands"]["Germ-Free"]["ratio_cecum"]["p95"] < 1.0


def test_robustness_audit_covers_every_battery_check_it_claims_to() -> None:
    cids = {c.cid for c in CHECKS}
    for cid in ROBUSTNESS:
        if cid.endswith("-superseded"):
            continue
        assert cid in cids, f"{cid} audits a check that no longer exists"


def test_two_member_claim_survives_a_far_lower_glucuronidase_capacity() -> None:
    g = two_member_gus_sensitivity(P, T, grid=(0.05, 0.10, 0.50, 1.00))
    assert g["min_gus_beating_abx"] <= 0.10
    # the superseded form needed roughly half of conventional capacity
    needs_half = [r["gus"] for r in g["rows"] if r["exceeds_conventional_95"]]
    assert min(needs_half) >= 0.50


# --- figure sensitivity bands, 2026-09-20 ----------------------------------

def test_band_helper_returns_absolute_percentiles_not_offsets() -> None:
    """The whiskers used to be drawn as median-derived offsets attached to the
    NOMINAL bar, so their endpoints were not p5 and p95 whenever nominal
    differed from the median -- which it generally does. `_band` now returns
    the percentiles themselves and the panels draw the interval where it is."""
    from emd2_simulation import figure as fg
    ens = run_ensemble(ARMS, T, n=24, seed=5)
    lo, mid, hi = fg._band(ens, "B", "ratio_cecum")
    b = ens["bands"]["B"]["ratio_cecum"]
    assert (lo, mid, hi) == (b["p5"], b["p50"], b["p95"])
    assert lo <= mid <= hi
    # and the nominal is NOT the median, which is the whole point
    nominal = lumen_plasma_ratio(simulate(P, BY_KEY["B"], T), P, "cecum")
    assert abs(nominal - mid) / mid > 1e-3


# --- exposure ordering ------------------------------------------------------

def test_total_and_microbial_exposure_have_different_arm_orderings(ends) -> None:
    """V12 compares simulated exposure orderings, without tumour endpoints."""
    assert ends["2Mem"]["AUC_uro"] > ends["BA"]["AUC_uro"]          # different ordering
    assert ends["2Mem"]["AUC_uro_micro"] == pytest.approx(0.0, abs=1e-9)
    assert (ends["B"]["AUC_uro_micro"] > ends["BA"]["AUC_uro_micro"]
            > ends["2Mem"]["AUC_uro_micro"])
    assert ends["B"]["AUC_uro"] == pytest.approx(
        ends["B"]["AUC_uro_micro"] + ends["B"]["AUC_uro_host"], rel=1e-9)


def test_measurable_bcpn_is_the_sum_of_the_two_latent_pools(ends) -> None:
    for k, e in ends.items():
        for s in SEGMENTS:
            assert e[f"c_BCPN_{s}"] == pytest.approx(
                e[f"c_BCPNm_{s}"] + e[f"c_BCPNh_{s}"], rel=1e-9)
        assert e["c_BCPN_kidney"] == pytest.approx(
            e["c_BCPNh_kidney"] + e["c_BCPNm_kidney"], rel=1e-9)


def test_ratio_band_is_taken_within_draw_not_from_marginals() -> None:
    """Panel b plots each arm as a ratio to conventional. Banding the marginal
    AUCs and dividing by a fixed denominator overstates a ratio's spread,
    because the arms move together under a shared parameter draw.

    The conventional arm makes this exact and noise-free: its ratio to itself is
    1.0 in every draw, so the correct band has ZERO width -- while its marginal
    AUC band spans several-fold. Under the naive construction the bar that sits
    at exactly 100% by definition would have carried the widest whisker on the
    panel.
    """
    ens = run_ensemble(ARMS, T, n=24, seed=3)
    ref = ens["bands"]["B"]["AUC_rel_conventional"]
    assert ref["p5"] == pytest.approx(1.0)
    assert ref["p50"] == pytest.approx(1.0)
    assert ref["p95"] == pytest.approx(1.0)

    marg_b = ens["bands"]["B"]["AUC_uro"]
    assert marg_b["p95"] > 2.0 * marg_b["p5"]        # the naive whisker was huge

    # And for a genuinely varying arm the within-draw band is the narrower one.
    rel = ens["bands"]["2Mem"]["AUC_rel_conventional"]
    marg = ens["bands"]["2Mem"]["AUC_uro"]
    within = rel["p95"] - rel["p5"]
    naive = (marg["p95"] - marg["p5"]) / marg_b["p50"]
    assert within < naive


# --- audit fixes, 2026-08-26 -----------------------------------------------
# Three bookkeeping errors that an external audit caught. None of them changed a
# scientific conclusion, and none of them would have failed a check -- which is
# exactly why they need asserting.

def test_arm_count_excludes_analytical_standards() -> None:
    """`len(design)` counted 13 arms because the `Abx treatment` column also
    carries nitrosamine standards (DBN/PBN/EHBN + acid analogues) and a blank.
    There are six experimental arms."""
    design = idf.load_design()
    facts = idf.design_facts(design)
    assert facts["n_arms_total"] == 6
    assert set(facts["non_arm_labels"]).isdisjoint(idf.REAL_ARMS)
    # the standards must still be PRESENT in the raw design -- the fix is to
    # exclude them from the count, not to drop them on load
    assert len(design) > 6


def test_kidney_arm_coverage_is_three_times_urine_not_six() -> None:
    """The prose claimed 'six times urine's arm coverage'. Kidney is in all six
    arms and urine in two, which is three times by arms."""
    kid, uri = idf.coverage("kidney"), idf.coverage("urine")
    assert kid["n_arms"] == 6 and uri["n_arms"] == 2
    assert kid["n_arms"] / uri["n_arms"] == 3.0
    assert kid["records"] == 109 and uri["records"] == 90
    assert kid["records"] / uri["records"] == pytest.approx(1.21, abs=0.01)
    assert kid["biological_n_verified"] is False
    assert "biological_samples" not in kid


def test_anatomy_conflicts_are_flagged_without_changing_source_counts() -> None:
    conflicts = idf.metadata_conflicts()
    two = [r for r in conflicts if r["arm"] == "2Mem"]
    assert len(two) == 16
    assert all(r["recorded_compartment"] == "colon" and
               r["filename_compartment"] == "liver" for r in two)
    assert sum(idf.load_design()["2Mem"]["colon"].values()) == 48


def test_coverage_counts_acquisitions_without_inventing_statistical_n() -> None:
    design = idf.load_design()
    assert sum(design["2Mem"]["cecum"].values()) == 32
    for part in ("cecum", "colon", "kidney", "urine", "bile", "Plasma"):
        c = idf.coverage(part, design)
        assert c["count_unit"] == "LC-MS acquisition"
        assert c["biological_n_verified"] is False
        assert "biological_samples" not in c


# --- the exposure index is urine rescaled ----------------------------------

def test_urothelial_index_is_exactly_proportional_to_urinary_excretion() -> None:
    """There is no urothelial compartment. `AUC_uro` is cumulative urinary
    excretion times a constant, so it carries no independent information and
    supplies no host-response evidence. Asserted because the manuscript used to
    present it as a distinct endpoint."""
    from emd2_simulation.model import auc_is_urine_rescaled
    k = auc_is_urine_rescaled(P)
    assert k == pytest.approx(P.k_uro / (P.V_bladder * P.k_void))
    for a in ARMS:
        tr = trace(simulate(P, a, T), P)
        assert float(tr["AUC_uro"][-1] / tr["U_total"][-1]) == pytest.approx(
            k, rel=1e-12), a.key


# --- dose invariance is JOINT, not dose-alone ------------------------------

def test_dose_alone_moves_the_ratio_but_joint_rescaling_does_not() -> None:
    """The manuscript said +/-30% on the drinking volume moves no ratio, citing
    homogeneity. Homogeneity needs Vmax and Km to move WITH the dose; drinking
    volume does not do that. The effect is ~1%, not zero."""
    from emd2_simulation.ensemble import dose_only_sensitivity
    d = dose_only_sensitivity(P, T)
    assert d["max_rel_change_joint"] < 1e-9          # exact
    assert 1e-3 < d["max_rel_change_dose_only"] < 0.05   # small but not zero
    got = [r["ratio_cecum"] for r in d["dose_only"]]
    assert got == pytest.approx([142.768, 141.764, 140.589], abs=0.01)


# --- scope covers the downstream links too ---------------------------------

def test_scope_verdict_covers_kcc2_and_kcc10_not_only_the_host_layer() -> None:
    v = idf.host_response_verdict(idf.load_design())
    assert v["genotoxicity_unidentifiable"] is True
    assert v["kcc10_response_unidentifiable"] is True
    assert set(v["unconstrainable_edges"]) >= {"KCC2", "KCC10", "KCC6"}
    assert "assay absence" in v["basis"]
    assert "cell proliferation" in v["functional_endpoints_required"]["KCC10"]


def test_no_tumour_or_genotoxicity_data_is_imported_anywhere() -> None:
    """The imported summary has no tumour or genotoxicity endpoint keys."""
    import json
    from pathlib import Path
    obs = json.loads((Path(idf.__file__).parent / "data"
                      / "observed_roje2024.json").read_text())
    keys = set(obs)
    assert not any(t in k.lower() for k in keys
                   for t in ("tumour", "tumor", "incidence", "adduct",
                             "histolog", "surviv"))


def test_null_reports_relative_error_and_never_an_r_squared() -> None:
    """R^2 is undefined for three parameters fitted to one scalar observation:
    it evaluated to exactly 1.0 by construction and read as a perfect fit."""
    from emd2_simulation import nulls as nl
    res = nl.evaluate_null(P, T)
    assert "urine_r2" not in res
    assert res["urine_rel_err"] < 1e-6      # recovered to optimiser tolerance
    # and the fit function itself returns the residual, not a fit statistic
    _, rel = nl.fit_host_only(P, res["urine_target"], T)
    assert rel == pytest.approx(res["urine_rel_err"], rel=1e-9)


def test_cumulative_exposure_is_not_a_steady_state_quantity() -> None:
    """Concentrations converge by 720 h; the urothelial integral does not, and
    must not be presented under a 'steady-state' heading."""
    tr = trace(simulate(P, BY_KEY["B"], T), P)
    half = int(np.argmin(np.abs(T - 360.0)))
    assert tr["c_BCPN_cecum"][half] == pytest.approx(tr["c_BCPN_cecum"][-1], rel=1e-9)
    assert tr["AUC_uro"][-1] / tr["AUC_uro"][half] == pytest.approx(2.0, abs=0.15)


def test_scope_is_not_inferred_from_compartment_names() -> None:
    """An anatomy label cannot reveal what endpoints the build imports."""
    synthetic_design = {"B": {"cytokine": {"3": 1}}}
    v = idf.host_response_verdict(synthetic_design)
    assert v["host_response_unidentifiable"] is True
    assert v["imported_endpoints"] == list(idf.IMPORTED_ENDPOINTS)
