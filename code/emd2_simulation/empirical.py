"""Reproducible secondary analysis of published EMD2 experiments.

This module estimates experimental contrasts, not physiological rate constants.
Animal/donor identifiers define resampling units. No hypothesis is selected by
its p-value. Families cover all represented compartments or collection strata.
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

DATA_PATH = Path(__file__).parent / "data" / "published_roje2024.json"
SEED = 20260920
N_BOOTSTRAP = 20000
N_PERMUTATIONS = 50000
TISSUES = ("SI1", "SI2", "SI3", "CEC", "COL", "REC", "PL", "LVR", "KID")
TISSUE_LABEL = {"SI1": "Small intestine 1", "SI2": "Small intestine 2",
                "SI3": "Small intestine 3", "CEC": "Caecum", "COL": "Colon",
                "REC": "Rectum", "PL": "Plasma", "LVR": "Liver",
                "KID": "Kidney", "BLD": "Bladder"}


def load_data(path: Path = DATA_PATH) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported published-data schema")
    return data


def _seed(key: str) -> int:
    return (SEED + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")) % 2**32


def unique_observations(rows, keys, value_key):
    """Collapse identical records; exclude conflicting values for the same key.

    Repeated rows do not become new experimental units. Different readings are
    never silently averaged when no analytical replicate field is supplied.
    """
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
    """A minus B; percentile intervals and two-sided permutation tests.

    Pairing is used only for the same donor across culture conditions. Censored
    zero reports remain zero; no pseudocount or fabricated detection limit.
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
        # Every donor-level sign assignment; preserve technical replication.
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
                     "decision": "Not used for new quantitative inference: wide tissue signals lack units and SI1 scales differ between animals; some SI1 entries equal 1 while others are raw-scale signals."}
    audits["S32"] = {"missing_strain_rows_excluded": missing_strain,
                     "decision": "Descriptive technical-replicate signals only; no biological-n inference."}
    return {
        "schema_version": 1, "analysis_kind": "exploratory secondary analysis of published experiments",
        "source": data["source"],
        "input_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() if bundled_input else None,
        "input_content_sha256": hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "seed": SEED, "n_bootstrap": n_bootstrap,
        "methods": {
            "estimand": "difference of arithmetic means (group_a minus group_b); same-matrix ratio of arithmetic means",
            "intervals": "95% percentile bootstrap, resampling animals or paired donors; not simultaneous intervals",
            "tests": "two-sided independent-label permutation or paired donor sign-flip; Holm within stated families",
            "zeros": "reported zeros retained; no invented LOD or pseudocount",
            "strata": "collection week and analytical batch kept separate; no pooled longitudinal test",
            "interpretation": "No cross-compartment unity cutoff, kinetic calibration, external validation or functional KCC response is inferred.",
        },
        "consortium": consortium, "human_culture": culture, "human_donors": donors,
        "human_complete_sensitivity": culture_complete, "urine": families["S5"], "bladder": families["S4"],
        "germ_free_controls": gf, "acute": acute, "isolate_signals": isolate,
        "published_consortium_contrasts": [r for r in tables["S40"] if r["term"] == "Group" and r[".y."] == "BCPN"],
        "audits": audits,
        "kinetic_calibration": {"performed": False,
            "reason": "The acute table has unresolved protocol/count and published-statistic discrepancies; relative signals lack physical scale, and the existing mechanism requires structural revision. Experimental contrasts are estimated without fitting physiological rates."},
        "scope": {"EMD2": "Measured microbial transformation and community perturbation evidence",
                  "KCC1": "Metabolic activation-related evidence; not a new assay of electrophilicity",
                  "KCC6_KCC7_KCC8_KCC10": "Require corresponding functional host-response endpoints, not supplied here",
                  "KCC2": "Possible downstream consequence; requires genotoxicity evidence"},
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
        raise ValueError("Acute ID-prefix treatment mapping no longer corroborated by labelled source contrasts")
    return {
        "descriptive": descriptive, "published_contrasts": checks,
        "treatment_mapping": "A-prefixed ID interpreted as ABX/BBN; corroborated by 11/12 labelled source contrasts within 1 nM, not by an explicit raw group field",
        "resampling_unit": "terminal animal ID within time and inferred treatment; no longitudinal pairing",
        "limitations": [
            "Methods report five animals/time/group; the table contains five or six IDs per cell.",
            "Article text reports P=0.085 for the 1-h caecal comparison; S18 reports adjusted P=0.0108913, and its estimate differs slightly from raw mean difference.",
            "All first samples are at 1 h; concentrations in both compartments already present do not establish which source generated BCPN first.",
            "Reported zero values may be below detection; no assay LOD is supplied in this derived table.",
        ],
        "use": "Descriptive temporal comparison only; excluded from kinetic calibration and new significance testing pending source reconciliation.",
    }


def write_outputs(outdir: Path, *, data=None, figure=True) -> dict:
    result = analyse(data)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "emd2_empirical_summary.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if figure:
        from .empirical_figure import make_figure
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
