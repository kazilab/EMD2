"""Estimate experimental contrasts from Roje et al. (2024).

Animals and donors are the resampling units. Multiplicity adjustments cover
all eligible compartments or collection strata in each stated family.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import permutation_test

from ..paths import DATABASE_PATH
from ..processing.store import load_published, source_checksum

DATA_PATH = DATABASE_PATH
SEED = 20260920
N_BOOTSTRAP = 20000
N_PERMUTATIONS = 50000
TISSUES = ("SI1", "SI2", "SI3", "CEC", "COL", "REC", "PL", "LVR", "KID")
TISSUE_LABEL = {"SI1": "Small intestine 1", "SI2": "Small intestine 2",
                "SI3": "Small intestine 3", "CEC": "Caecum", "COL": "Colon",
                "REC": "Rectum", "PL": "Plasma", "LVR": "Liver",
                "KID": "Kidney", "BLD": "Bladder"}


def load_data(path: Path | None = None) -> dict:
    """Load the published measurements from SQLite or a specified JSON file."""
    if path is None or Path(path).suffix in {".sqlite", ".db"}:
        data = load_published(database=path)
    else:
        data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported published-data schema")
    return data


def _seed(key: str) -> int:
    return (SEED + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")) % 2**32


def unique_observations(rows, keys, value_key):
    """Collapse identical records within each key and exclude conflicting readings."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get(value_key) is None:
            continue
        value = float(row[value_key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid measured value at row {row['source_row']}")
        grouped[tuple(str(row[k]) for k in keys)].append(row)
    kept, conflicts = [], []
    duplicate_rows = 0
    for key, group in sorted(grouped.items()):
        values = {float(r[value_key]) for r in group}
        if len(values) != 1:
            conflicts.append({"key": list(key), "rows": [r["source_row"] for r in group],
                              "values": sorted(values)})
            continue
        record = dict(group[0])
        record["source_rows"] = [r["source_row"] for r in group]
        kept.append(record)
        duplicate_rows += len(group) - 1
    return kept, {"exact_duplicate_rows_collapsed": duplicate_rows,
                  "conflicting_keys_excluded": conflicts}


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return result.tolist()


def _difference(a, b, axis=-1):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)


def contrast(a, b, *, key, paired=False, n_bootstrap=N_BOOTSTRAP):
    """Estimate the mean difference (A minus B) with a percentile bootstrap
    interval and two-sided permutation test. Paired culture contrasts use
    donors as the unit. Reported zeros retain their recorded value.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite one-dimensional observations required")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if min(len(a), len(b)) < 2:
        raise ValueError("At least two independent units per condition required")
    if paired and len(a) != len(b):
        raise ValueError("Paired observations must have equal lengths")
    rng = np.random.default_rng(_seed(key))
    ia = rng.integers(len(a), size=(n_bootstrap, len(a)))
    ib = ia if paired else rng.integers(len(b), size=(n_bootstrap, len(b)))
    ma, mb = a[ia].mean(axis=1), b[ib].mean(axis=1)
    delta = float(a.mean() - b.mean())
    if paired:
        # Enumerate all sign assignments to the paired donor differences.
        d = a - b
        signs = 1 - 2 * ((np.arange(2**len(d))[:, None] >> np.arange(len(d))) & 1)
        distribution = (signs * d).mean(axis=1)
        pvalue = float(np.mean(np.abs(distribution) >= abs(delta) - 1e-12))
        method = "exact paired sign-flip"
    else:
        possibilities = math.comb(len(a) + len(b), len(a))
        result = permutation_test((a, b), _difference, vectorized=True,
                                  n_resamples=N_PERMUTATIONS, batch=2048,
                                  alternative="two-sided", rng=rng)
        pvalue = float(result.pvalue)
        method = "exact independent-label" if possibilities <= N_PERMUTATIONS else "Monte Carlo independent-label (50000)"
    ratio_valid = bool(np.all(mb > 0) and b.mean() > 0)
    return {
        "n_a": len(a), "n_b": len(b), "paired": paired,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "mean_difference": delta,
        "difference_ci95": np.quantile(ma - mb, [0.025, 0.975]).tolist(),
        "mean_ratio": float(a.mean() / b.mean()) if b.mean() > 0 else None,
        "ratio_ci95": np.quantile(ma / mb, [0.025, 0.975]).tolist() if ratio_valid else None,
        "ratio_interval_status": "estimable" if ratio_valid else "zero denominator in bootstrap; not estimated",
        "p_value": pvalue, "permutation_method": method,
        "n_bootstrap": n_bootstrap,
    }


def _family(rows, name):
    for row, p in zip(rows, holm([r["p_value"] for r in rows])):
        row["p_holm"] = p
        row["multiplicity_family"] = name
        row["family_size"] = len(rows)
    return rows


def _group_contrasts(rows, *, group_a, group_b, value_key, strata, family,
                     unit, group_key="Group", n_bootstrap=N_BOOTSTRAP):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[tuple(str(r[k]) for k in strata)][str(r[group_key])].append(r)
    result = []
    for stratum, groups in sorted(grouped.items()):
        aa, bb = groups.get(group_a, []), groups.get(group_b, [])
        if min(len(aa), len(bb)) < 2:
            continue
        result.append({
            **dict(zip(strata, stratum)), "group_a": group_a, "group_b": group_b,
            "unit": unit,
            **contrast([r[value_key] for r in aa], [r[value_key] for r in bb],
                       key=family + repr(stratum), n_bootstrap=n_bootstrap),
            "observations_a": [{"id": str(r["ID"]), "value": float(r[value_key]),
                                "source_rows": r.get("source_rows", [r["source_row"]])} for r in aa],
            "observations_b": [{"id": str(r["ID"]), "value": float(r[value_key]),
                                "source_rows": r.get("source_rows", [r["source_row"]])} for r in bb],
        })
    return _family(result, family)


NEOPLASIA_COLUMNS = ("CIS", "Invasion", "Sarcomatoid carcinoma")


def _counts(row):
    """(normal, any neoplasia) for one histology row; blank cells are zero."""
    def n(key):
        value = row.get(key)
        return 0 if value in (None, "") else int(value)
    return n("Normal"), sum(n(k) for k in NEOPLASIA_COLUMNS)


def _mantel_haenszel(strata):
    """Compute the uncorrected Cochran–Mantel–Haenszel statistic across experiments."""
    numerator = denominator = 0.0
    for s in strata:
        a, b = s["treated_neoplasia"], s["treated_normal"]
        c, d = s["control_neoplasia"], s["control_normal"]
        n = a + b + c + d
        if n < 2:
            continue
        numerator += a - (a + c) * (a + b) / n
        denominator += (a + b) * (c + d) * (a + c) * (b + d) / (n ** 2 * (n - 1))
    if denominator <= 0:
        return None, None
    chi2 = numerator ** 2 / denominator
    from scipy.stats import chi2 as chi2_dist
    return float(chi2), float(chi2_dist.sf(chi2, 1))


def analyse_tumour_outcome(tables):
    """Compare terminal bladder neoplasia by treatment in source tables S3 and S42."""
    from scipy.stats import chi2_contingency, fisher_exact

    strata, order = {}, []
    for row in tables["S3"]:
        experiment = str(row["experiment"])
        arm = str(row["Treatment"]).strip()
        normal, neoplasia = _counts(row)
        if experiment not in strata:
            strata[experiment] = {"experiment": experiment, "source_rows": []}
            order.append(experiment)
        side = "treated" if arm.startswith("ABX") else "control"
        strata[experiment][f"{side}_arm"] = arm
        strata[experiment][f"{side}_normal"] = normal
        strata[experiment][f"{side}_neoplasia"] = neoplasia
        strata[experiment]["source_rows"].append(row["source_row"])

    per_experiment = []
    for key in order:
        s = strata[key]
        for side in ("treated", "control"):
            total = s[f"{side}_normal"] + s[f"{side}_neoplasia"]
            s[f"{side}_n"] = total
            s[f"{side}_neoplasia_fraction"] = (s[f"{side}_neoplasia"] / total
                                               if total else None)
        per_experiment.append(s)

    chi2_cmh, p_cmh = _mantel_haenszel(per_experiment)
    same_direction = sum(
        1 for s in per_experiment
        if s["control_neoplasia_fraction"] is not None
        and s["treated_neoplasia_fraction"] is not None
        and s["control_neoplasia_fraction"] > s["treated_neoplasia_fraction"])

    pooled = {side: [sum(s[f"{side}_normal"] for s in per_experiment),
                     sum(s[f"{side}_neoplasia"] for s in per_experiment)]
              for side in ("treated", "control")}
    table = [pooled["treated"], pooled["control"]]
    p_pooled_yates = float(chi2_contingency(table, correction=True)[1])

    s3 = {
        "agent": "BBN", "source_table": "S3",
        "n_experiments": len(per_experiment),
        "per_experiment": per_experiment,
        "pooled_treated": {"arm": per_experiment[0]["treated_arm"],
                           "normal": pooled["treated"][0],
                           "neoplasia": pooled["treated"][1],
                           "n": sum(pooled["treated"]),
                           "neoplasia_fraction": pooled["treated"][1] / sum(pooled["treated"])},
        "pooled_control": {"arm": per_experiment[0]["control_arm"],
                           "normal": pooled["control"][0],
                           "neoplasia": pooled["control"][1],
                           "n": sum(pooled["control"]),
                           "neoplasia_fraction": pooled["control"][1] / sum(pooled["control"])},
        "primary_test": "Cochran-Mantel-Haenszel, stratified by experiment",
        "cmh_chi2": chi2_cmh, "p_value": p_cmh,
        "experiments_favouring_control": same_direction,
        "pooled_yates_p_for_reconciliation": p_pooled_yates,
        "pooled_note": ("Pooling the five experiments gives the published "
                        "Yates-corrected P = 1.6e-5. The primary test is stratified by experiment."),
    }

    row_a, row_b = tables["S42"][0], tables["S42"][1]
    a_normal, a_neo = _counts(row_a)
    b_normal, b_neo = _counts(row_b)
    s42 = {
        "agent": "EHBN", "source_table": "S42", "n_experiments": 1,
        "treated": {"arm": str(row_a["Treatment"]).strip(), "normal": a_normal,
                    "neoplasia": a_neo, "n": a_normal + a_neo,
                    "neoplasia_fraction": a_neo / (a_normal + a_neo)},
        "control": {"arm": str(row_b["Treatment"]).strip(), "normal": b_normal,
                    "neoplasia": b_neo, "n": b_normal + b_neo,
                    "neoplasia_fraction": b_neo / (b_normal + b_neo)},
        "primary_test": "Fisher exact, two-sided",
        "p_value": float(fisher_exact([[a_normal, a_neo], [b_normal, b_neo]])[1]),
        "source_rows": [row_a["source_row"], row_b["source_row"]],
        "note": "S42 is a single small EHBN experiment. Its direction agrees with the BBN comparison.",
    }

    rows = _family([s3, s42], "Organism-level outcome: two agents")
    return {
        "evidence_class": "sequence-level outcome under a microbial handle",
        "scores_kcc": None,
        "results": rows,
        "does_not_establish": [
            "KCC2 would require an adduct or mutation measurement.",
            "KCC6 would require an inflammatory endpoint.",
            "KCC10 would require a measure of proliferation, cell death or nutrient supply.",
            "Isolate-level attribution is unresolved. Antibiotics affect the wider microbiota, "
            "and the reconstructed-consortium experiments measured metabolites.",
        ],
    }


def analyse(data=None, *, n_bootstrap=N_BOOTSTRAP):
    bundled_input = data is None
    data = load_data() if bundled_input else data
    tables = data["tables"]
    audits = {}
    selected = [r for r in tables["S39"] if r["Drug"] == "BCPN" and r["Deglucuronidation"] == "BeforeDG"]
    s39, audits["S39"] = unique_observations(selected, ["Group", "ID", "Tissue"], "uM_raw")
    consortium = _group_contrasts(s39, group_a="3Mem", group_b="2Mem", value_key="uM_raw",
                                  strata=["Tissue"], family="S39: all nine BCPN compartments",
                                  unit="source-table µM", n_bootstrap=n_bootstrap)
    # Every technical replicate contributes to its donor-condition mean, but
    # inference gives each donor equal weight, even with incomplete replication.
    technical = defaultdict(list)
    for r in tables["S23"]:
        if r["BCPN_nM"] is not None:
            technical[(str(r["ID"]), r["O2_level"])].append(r)
    donors = []
    for (donor, oxygen), rows in sorted(technical.items()):
        donors.append({"donor": donor, "oxygen": oxygen,
                       "mean_nM": float(np.mean([r["BCPN_nM"] for r in rows])),
                       "technical_n": len(rows), "source_rows": [r["source_row"] for r in rows]})
    by_donor = defaultdict(dict)
    for r in donors:
        by_donor[r["donor"]][r["oxygen"]] = r["mean_nM"]
    culture = []
    for oxygen in ("10%O2", "21%O2"):
        ids = sorted(k for k, v in by_donor.items() if oxygen in v and "0%O2" in v)
        culture.append({"condition": oxygen, "reference": "0%O2", "donor_ids": ids,
                        "unit": "nM culture medium",
                        **contrast([by_donor[k][oxygen] for k in ids],
                                   [by_donor[k]["0%O2"] for k in ids],
                                   key="S23:"+oxygen, paired=True, n_bootstrap=n_bootstrap)})
    _family(culture, "S23: two oxygen contrasts")
    # Completeness sensitivity includes donors with four readings
    # at every oxygen condition; it is not a significance-based exclusion.
    complete = sorted(k for k in by_donor if all(len(technical[k, o]) == 4 for o in ("0%O2", "10%O2", "21%O2")))
    culture_complete = []
    for oxygen in ("10%O2", "21%O2"):
        culture_complete.append({"condition": oxygen, "donor_ids": complete,
                                 **contrast([by_donor[k][oxygen] for k in complete],
                                            [by_donor[k]["0%O2"] for k in complete],
                                            key="S23:complete:"+oxygen, paired=True,
                                            n_bootstrap=n_bootstrap)})
    _family(culture_complete, "S23: complete-replicate sensitivity")

    families = {}
    for sheet, value, unit in (("S5", "nmole", "nmol/24 h"), ("S4", "nM", "source-table nM")):
        selected = [r for r in tables[sheet] if r["Drug"] == "BCPN"]
        unique, audits[sheet] = unique_observations(selected, ["Group", "ID", "Length", "Batch"], value)
        families[sheet] = _group_contrasts(unique, group_a="BBN", group_b="ABX/BBN", value_key=value,
                                          strata=["Length", "Batch"], family=sheet+": all week/batch strata",
                                          unit=unit, n_bootstrap=n_bootstrap)
    s15, audits["S15"] = unique_observations([r for r in tables["S15"] if r["Drug"] == "BCPN"],
                                             ["Group", "ID", "Tissue"], "uM")
    gf = _group_contrasts(s15, group_a="GFA", group_b="GF", value_key="uM", strata=["Tissue"],
                         family="S15: all four eligible GF antibiotic controls", unit="source-table µM",
                         n_bootstrap=n_bootstrap)

    acute = acute_analysis(tables)
    # S32 is descriptive: well identifiers do not establish independent
    # biological replication. Missing strain labels are explicitly excluded.
    strain = defaultdict(list)
    missing_strain = []
    for r in tables["S32"]:
        if r.get("Strain") is None:
            missing_strain.append(r["source_row"])
        elif r.get("A. U.") is not None:
            strain[(r["Strain"], r["O2"], r["Time_hr"])].append(r)
    isolate = [{"strain": k[0], "oxygen": k[1], "time_h": k[2], "technical_n": len(v),
                "mean_signal": float(np.mean([r["A. U."] for r in v])),
                "signals": [float(r["A. U."]) for r in v], "source_rows": [r["source_row"] for r in v]}
               for k, v in sorted(strain.items())]
    norm_bad = [r["source_row"] for r in tables["S37"]
                if not all(isinstance(r.get(k), (int, float)) for k in ("SI1", "Norm", r["Tissue"]))
                or not r["SI1"]
                or not np.isclose(r["Norm"], r[r["Tissue"]] / r["SI1"], rtol=1e-7, atol=1e-9)]
    audits["S37"] = {"missing_or_nonreconcilable_normalisation_rows": norm_bad,
                     "decision": "Excluded from quantitative inference. Tissue signals lack units and SI1 uses inconsistent scales between animals."}
    audits["S32"] = {"missing_strain_rows_excluded": missing_strain,
                     "decision": "Well signals are descriptive because biological replication is unverified."}
    return {
        "schema_version": 1, "analysis_kind": "exploratory secondary analysis of published experiments",
        "source": data["source"],
        "input_sha256": source_checksum("published_extract") if bundled_input else None,
        "input_content_sha256": hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "seed": SEED, "n_bootstrap": n_bootstrap,
        "methods": {
            "estimand": "Difference of arithmetic means (group A minus group B) and ratio of means within each matrix.",
            "intervals": "Pointwise 95% percentile bootstrap intervals resample animals or paired donors.",
            "tests": "Two-sided label permutations for independent groups or sign flips for paired donors. Holm adjustment within each family.",
            "zeros": "Reported zero values enter the analysis as zero.",
            "strata": "Each collection week and analytical batch is analysed separately.",
            "interpretation": "Within-matrix contrasts estimate experimental effects. Interpreting a cross-matrix ratio requires comparable assay normalisation.",
        },
        "consortium": consortium, "human_culture": culture, "human_donors": donors,
        "human_complete_sensitivity": culture_complete, "urine": families["S5"], "bladder": families["S4"],
        "germ_free_controls": gf, "acute": acute, "isolate_signals": isolate,
        "published_consortium_contrasts": [r for r in tables["S40"] if r["term"] == "Group" and r[".y."] == "BCPN"],
        "tumour_outcome": analyse_tumour_outcome(tables),
        "audits": audits,
        "kinetic_calibration": {"performed": False,
            "reason": "Acute sampling, unresolved discrepancies in source table S18 and unscaled relative signals preclude fitting physiological rates."},
        "scope": {"EMD2": "Measured transformation products and community perturbation",
                  "KCC1": "Evidence for metabolic activation. Electrophilicity was not assayed.",
                  "KCC6_KCC7_KCC8_KCC10": "Require functional host-response endpoints beyond the imported measurements",
                  "KCC2": "Requires genotoxicity evidence"},
        "limits": {
            "evidence_basis":
                "Measured transformation products and reconstructed-community comparisons "
                "support microbial metabolism. No taxonomic abundance data enter these estimates.",
            "data_handling":
                "Identical plasma records in S39 are counted once. Conflicting caecal records "
                "in S15 leave no germ-free caecum/plasma ratio. Culture wells are averaged "
                "within donors. Urine and bladder comparisons retain their collection strata.",
            "acute_timecourse":
                "BCPN is present in both compartments at the earliest sample (1 h), leaving "
                "the order of appearance unresolved. S18 treatment labels, animal counts and "
                "published contrasts also limit quantitative use. S37 mixes normalisation scales.",
            "unity_cutoff":
                "Pairing by animal leaves matrix-specific differences in extraction, recovery "
                "and normalisation unresolved. Within the model, exchange and biliary delivery "
                "can each give a host-only lumen/plasma ratio above one.",
            "kcc_scope":
                "KCC6, KCC7, KCC8 and KCC10 require functional host-response measurements. "
                "KCC2 requires genotoxicity evidence. The model exposure index is exactly "
                "proportional to cumulative urinary excretion.",
            "acquisition_counts":
                "LC-MS acquisition counts include repeated analyses of aliquots across ionisation "
                "modes and deglucuronidation conditions. Biological sample sizes come from animal "
                "or donor identifiers.",
            "model_magnitudes":
                "At the reference settings, luminal and plasma concentrations exceed published "
                "values by two to three orders of magnitude, and the ordering of luminal "
                "concentrations differs. Model ratios and fractions are conditional on assigned kinetics.",
            "model_mechanism_divergence":
                "In the model, deconjugation drives the antibiotic contrast and microbial oxidation "
                "lowers total systemic exposure. The source study attributes its effect to microbial conversion.",
            "sequence_level_outcome":
                "Antibiotic-treated animals had fewer neoplasms in S3 and S42. S3 is stratified "
                "by experiment. These outcomes cannot distinguish the converting isolate from other "
                "effects of microbiota depletion or identify a functional KCC endpoint.",
            "power":
                "Most strata contain about five to eleven animals per group. After Holm adjustment, "
                "several comparisons remain inconclusive. Their intervals allow both small and large effects.",
        },
    }


def acute_analysis(tables):
    """Audit acute measurements without treating terminal cohorts as repeated animals."""
    grouped = defaultdict(list)
    for r in tables["S18_BCPN"]:
        group = "ABX/BBN" if str(r["ID"]).startswith("A") else "BBN"
        grouped[(r["Time"], r["Tissue"], group)].append(r)
    descriptive = []
    for (time, tissue, group), rows in sorted(grouped.items()):
        values = [float(r["BCPN_uM"]) for r in rows]
        descriptive.append({"time_h": time, "tissue": tissue, "group": group,
                            "n_recorded_ids": len({str(r['ID']) for r in rows}),
                            "mean_uM": float(np.mean(values)), "values_uM": values,
                            "ids": [str(r["ID"]) for r in rows],
                            "source_rows": [r["source_row"] for r in rows]})
    checks = []
    for r in tables["S18_contrasts"]:
        if r["term"] != "Group":
            continue
        key = (r["Time_or_Group"], r[".y."])
        a = grouped[(*key, "ABX/BBN")]
        b = grouped[(*key, "BBN")]
        raw = float(np.mean([v["BCPN_nM"] for v in a]) - np.mean([v["BCPN_nM"] for v in b]))
        checks.append({"time_h": key[0], "tissue": key[1],
                       "published_abx_minus_bbn_nM": r["estimate"],
                       "raw_abx_minus_bbn_nM": raw,
                       "agreement_within_1nM": abs(raw - r["estimate"]) < 1.0,
                       "published_ci95_nM": [r["conf.low"], r["conf.high"]],
                       "published_p_adjusted": r["p.adj"], "source_row": r["source_row"]})
    if sum(r["agreement_within_1nM"] for r in checks) < 11:
        raise ValueError("Acute ID-prefix treatment mapping disagrees with the labelled source contrasts")
    return {
        "descriptive": descriptive, "published_contrasts": checks,
        "treatment_mapping": "A-prefixed IDs denote inferred ABX/BBN treatment. Eleven of 12 published contrasts agree within 1 nM. The raw table has no group field.",
        "resampling_unit": "Terminal animal ID within each time and inferred treatment group.",
        "limitations": [
            "BCPN is present in both compartments at the first observation (1 h), leaving the order of appearance unresolved.",
            "The methods report five animals per time and group. S18 contains five or six IDs per cell.",
            "For the 1-h caecal contrast, the article reports P = 0.085 and S18 reports adjusted P = 0.0108913. Its estimate differs slightly from the raw mean difference.",
            "Treatment is inferred from the animal ID.",
            "Zeros may represent values below detection. S18 supplies no detection limit.",
        ],
        "use": "Descriptive temporal comparison. The sampling times cannot resolve the origin of BCPN. Discrepancies in group labels, counts and statistics preclude kinetic fitting and new significance tests.",
    }


def write_outputs(outdir: Path, *, data=None, figure=True) -> dict:
    result = analyse(data)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "emd2_empirical_summary.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if figure:
        from ..reporting.empirical_figure import make_figure
        make_figure(result, outdir)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("figures/output"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    result = write_outputs(args.outdir, figure=not args.no_figure)
    c = next(r for r in result["consortium"] if r["Tissue"] == "CEC")
    print(f"S39 caecum: 3Mem−2Mem = {c['mean_difference']:.6g}; 95% CI {c['difference_ci95']}; Holm P={c['p_holm']:.6g}")
    print(f"Wrote {args.outdir / 'emd2_empirical_summary.json'}")


if __name__ == "__main__":
    main()
