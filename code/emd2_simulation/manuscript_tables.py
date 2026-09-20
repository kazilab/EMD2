"""Emit the supplementary mechanistic-model tables as one JSON file.

    python -m emd2_simulation.manuscript_tables

Writes `figures/output/emd2_manuscript_tables.json`. Published-data results
are generated separately in emd2_empirical_summary.json. The Word generators
read these two explicitly separate outputs. Run emd2_simulation.run first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import identifiability as idf
from .conditions import ARMS, BY_KEY, CHECKS, RATIO_SEGMENT
from .ensemble import FREE_SIGMA, STRUCTURAL_CHECKS
from .model import (IDX, SEG_ABSORB, SEG_KTRANSIT, SEG_MICROBES, SEG_OXYGEN,
                    SEG_VOL, SEGMENTS, Params, lumen_plasma_ratio, simulate,
                    trace)

T_END, N_T = 720.0, 1441

# unit, and what the value rests on. "chosen" is said plainly wherever true.
PARAM_META = {
    "dose_rate": ("nmol h⁻¹", "Drinking-water mass balance, ~5 mL day⁻¹, "
                  "M = 174.244 g mol⁻¹ → 598 nmol h⁻¹. The source writes "
                  "\"0.05% BBN\", and v/v in at least one place; treated here "
                  "as w/v (0.5 mg mL⁻¹). A v/v reading would give 0.5ρ mg mL⁻¹, "
                  "so the two differ by the density of BBN — near but not "
                  "exactly 1 g mL⁻¹, i.e. a few per cent, against ±30% from "
                  "the drinking-volume estimate. These move the reported ratios "
                  "slightly rather than exactly zero: a maximum deviation of about 0.8% "
                  "over ±30%. The model's exact invariance is under a JOINT "
                  "rescaling of dose with both Vmax and both Km, which dose "
                  "uncertainty does not perform. See `dose_sensitivity`."),
    "V_plasma": ("mL", "Assigned illustrative mouse compartment volume."),
    "V_bladder": ("mL", "Assigned illustrative mouse compartment volume."),
    "V_kidney": ("mL", "Assigned illustrative volume of both mouse kidneys."),
    "Q_bile": ("mL h⁻¹", "Assigned illustrative bile flow; converts biliary flux to "
               "the concentration an assay would report on aspirated bile."),
    "ka_BBN": ("h⁻¹", "Chosen. Neutral lipophilic nitrosamine, absorbed readily."),
    "ka_BCPN": ("h⁻¹", "Assigned. Ionisation of the carboxylic-acid metabolite "
                "motivates a lower assumed absorption rate than parent; its "
                "magnitude is not established by ionisation alone."),
    "k_back": ("h⁻¹", "Chosen. Plasma → lumen back-diffusion of BCPN; the route "
               "present at baseline; independent of luminal absorption, with no enforced reciprocity."),
    "k_uptake": ("h⁻¹", "Chosen. Plasma → liver uptake of parent."),
    "k_ox_host": ("h⁻¹", "Chosen, and deliberately LOW relative to k_gluc: the "
                  "germ-free arm must produce little BCPN."),
    "k_gluc": ("h⁻¹", "Chosen. Hepatic glucuronidation."),
    "k_bile": ("h⁻¹", "Chosen. Biliary transfer of the glucuronide."),
    "k_bile_BCPN": ("h⁻¹", "Unmeasured structural assumption, zero at baseline. "
                    "Biliary BCPN delivery is swept separately in V7b. Zero "
                    "cannot be explored with multiplicative log-normal draws. "
                    "Exchange changes also defeat a universal unity boundary."),
    "Vmax_gus": ("nmol h⁻¹ mL⁻¹", "CHOSEN, not measured. Caecal-scale "
                 "β-glucuronidase capacity."),
    "Km_gus": ("nmol mL⁻¹", "CHOSEN. Stated as a pair with Vmax: the model is "
               "exactly homogeneous under joint rescaling with the dose."),
    "Vmax_ox": ("nmol h⁻¹ mL⁻¹", "CHOSEN, not measured. Microbial oxidation "
                "capacity; the parameter profiled for identifiability."),
    "Km_ox": ("nmol mL⁻¹", "CHOSEN. Keeps the luminal system sub-saturating at "
              "the assigned dose (caecal BBN ≈ 0.9 µmol mL⁻¹)."),
    "k_renal": ("h⁻¹", "Chosen. Plasma → kidney tissue."),
    "k_kidney_out": ("h⁻¹", "Chosen. Kidney → bladder lumen. Steady-state "
                     "neutral: k_kidney_out·A_k = k_renal·A_p at steady state."),
    "k_void": ("h⁻¹", "Chosen. Bladder emptying."),
    "k_uro": ("dimensionless", "CHOSEN, and the quantity it scales is an "
              "ASSUMED EXPOSURE INDEX rather than a tissue concentration: there "
              "is no urothelial compartment and nothing is retained. Under the "
              "implemented equations the index is exactly proportional to "
              "cumulative urinary excretion, AUC/U = k_uro/(V_bladder·k_void), "
              "in every arm, so it carries no information beyond urinary "
              "excretion and no host-response evidence. Earlier releases "
              "labelled this h⁻¹, which made the printed \"AUC\" "
              "dimensionally a concentration."),
    "k_clear_p": ("h⁻¹", "Chosen. Non-renal plasma loss of BCPN."),
    "k_clear_BBN": ("h⁻¹", "Chosen. Non-hepatic plasma loss of parent."),
}

STATE_GROUPS = [
    ("Gut lumen, per segment (6 segments × 4 species = 24 states)",
     [("A_BBN,i", "parent BBN in segment i"),
      ("A_gBBN,i", "biliary glucuronide in segment i"),
      ("A_BCPNᵐ,i", "luminal BCPN of MICROBIAL origin (latent)"),
      ("A_BCPNʰ,i", "luminal host-origin BCPN via exchange or biliary delivery (latent)")]),
    ("Systemic",
     [("A_BBN,p", "plasma parent"),
      ("A_BBN,L", "hepatic parent"),
      ("A_gBBN,bile", "glucuronide en route to the duodenum"),
      ("A_BCPNʰ,p", "plasma BCPN, host origin (latent)"),
      ("A_BCPNᵐ,p", "plasma BCPN, microbial origin (latent)")]),
    ("Elimination limb",
     [("A_BCPNʰ,k / A_BCPNᵐ,k", "kidney tissue, by provenance (latent)"),
      ("A_BCPNʰ,u / A_BCPNᵐ,u", "bladder lumen, by provenance (latent)")]),
    ("Cumulative",
     [("AUCʰ, AUCᵐ", "urothelial exposure by provenance"),
      ("Uʰ, Uᵐ", "urinary excretion by provenance")]),
]


def build(summary: dict) -> dict:
    p = Params()
    t = np.linspace(0.0, T_END, N_T)

    params = []
    for f in fields(Params):
        unit, basis = PARAM_META.get(f.name, ("", ""))
        params.append({"name": f.name, "value": float(getattr(p, f.name)),
                       "unit": unit, "basis": basis,
                       "sigma": FREE_SIGMA.get(f.name)})

    segments = [{"segment": s,
                 "microbes": float(SEG_MICROBES[i]),
                 "oxygen": float(SEG_OXYGEN[i]),
                 "volume_mL": float(SEG_VOL[i]),
                 "k_transit": float(SEG_KTRANSIT[i]),
                 "absorb": float(SEG_ABSORB[i])}
                for i, s in enumerate(SEGMENTS)]

    # per-segment lumen/plasma ratio: shows the headline does not depend on
    # which large-intestinal segment is quoted
    seg_ratio = []
    for s in SEGMENTS:
        row = {"segment": s,
               "conc_conventional": float(summary["arms"]["B"][f"c_BCPN_{s}"])}
        for k in ("B", "Germ-Free", "2Mem"):
            row[f"ratio_{k}"] = lumen_plasma_ratio(
                simulate(p, BY_KEY[k], t), p, s)
        seg_ratio.append(row)

    arms = [{"key": a.key, "label": a.label, "gus": a.gus, "ox": a.ox,
             "reference": a.reference,
             **{k: float(summary["arms"][a.key][k])
                for k in ("c_BCPN_cecum", "c_BCPN_plasma", "c_BCPN_kidney",
                          "c_BCPN_urine", "AUC_uro", "AUC_uro_micro",
                          "micro_fraction", "ratio_cecum")},
             "auc_pct_of_conventional":
                 100.0 * float(summary["arms"][a.key]["AUC_uro"])
                 / float(summary["arms"]["B"]["AUC_uro"])}
            for a in ARMS]

    systemic = []
    for lab, key in (("caecum (lumen)", "c_BCPN_cecum"), ("plasma", "c_BCPN_plasma"),
                     ("kidney", "c_BCPN_kidney"), ("urine", "c_BCPN_urine")):
        b, a_ = (float(summary["arms"]["B"][key]),
                 float(summary["arms"]["BA"][key]))
        systemic.append({"compartment": lab, "conventional": b, "antibiotic": a_,
                         "ratio": b / a_})

    audit = summary.get("ensemble", {}).get("audit", {})
    reasons = (summary.get("ensemble", {}).get("unaudited_reason")
               or dict.fromkeys(STRUCTURAL_CHECKS,
                                "not audited draw by draw"))
    checks = [{"cid": c.cid, "edge": c.edge, "statement": c.statement,
               "source": c.source,
               "passed": bool(summary["checks"].get(c.cid)),
               "robustness": audit.get(c.cid),
               # Present and non-null EXACTLY when `robustness` is absent, so a
               # consumer can never read "no audit" as "audited and scored 0".
               "unaudited_reason": (None if c.cid in audit
                                    else reasons.get(c.cid, "not audited"))}
              for c in CHECKS]

    ident = summary.get("identifiability", {})
    ident_rows = [{"set": idf.SET_LABELS[k],
                   "compartments": ", ".join(idf.SETS[k][0]),
                   "n_arms": len(idf.SETS[k][1]),
                   **ident["sets"][k]}
                  for k in idf.SETS] if ident else []

    return {
        "parameters": params,
        "segment_constants": segments,
        "segment_ratio": seg_ratio,
        "state_groups": STATE_GROUPS,
        "arms": arms,
        "systemic_contrast": systemic,
        "null": summary["null"],
        "null_host_cost": summary["null_host_cost"],
        "biliary_escape": summary["null"].get("biliary_escape", {}),
        "capacity_decomposition": summary.get("capacity_decomposition", {}),
        "dose_sensitivity": summary.get("dose_sensitivity", {}),
        "exposure_index": summary.get("exposure_index", {}),
        # Acquisition counts only; no biological n is inferred from filenames.
        "design_coverage": (summary.get("identifiability", {})
                            .get("facts", {}).get("coverage", {})),
        "identifiability": ident_rows,
        "identifiability_facts": ident.get("facts", {}),
        "ensemble": summary.get("ensemble", {}),
        "gus_sweep": summary.get("two_member_gus_sensitivity", {}),
        "scope": summary["scope"],
        "checks": checks,
        "design_counts": idf.load_design(),
        "ratio_segment": RATIO_SEGMENT,
        "n_states": len(IDX),
        "t_end_h": T_END,
    }


EXPECTED_CHECKS = {f"V{i}" for i in range(1, 13)} | {"V7b"}


def audit(summary: dict) -> list[str]:
    """Reasons this summary must not be turned into manuscript text.

    A partial run (``--no-identifiability``, ``--no-ensemble``) writes a
    perfectly well-formed summary in which whole blocks are simply ABSENT and
    skipped checks are dropped from the tally rather than failed -- so it still
    prints "N/N checks passed" and exits 0. Nothing downstream notices: the
    docx generators read this file and `build_docx.sh` does not rerun the
    simulation unless asked.

    This is not hypothetical. A `--no-identifiability --no-ensemble
    --no-figure` smoke test once overwrote a complete summary with one missing
    `ensemble`, `identifiability` and `two_member_gus_sensitivity`, and with
    `V10` absent from `checks` so the tally read 11/11. The gate therefore
    tests for MISSING BLOCKS and for the full check roster, not for
    ``all(passed)`` -- which would never have fired.
    """
    problems = []

    checks = summary.get("checks") or {}
    failed = sorted(cid for cid, ok in checks.items() if not ok)
    if failed:
        problems.append(f"validation check(s) did not pass: {', '.join(failed)}")
    missing = sorted(EXPECTED_CHECKS - set(checks))
    if missing:
        problems.append(
            f"check(s) absent from the summary, i.e. skipped rather than "
            f"passed: {', '.join(missing)} (rerun with no skip flags)")

    for block in ("arms", "null", "null_host_cost", "scope",
                  "identifiability", "ensemble", "two_member_gus_sensitivity",
                  "capacity_decomposition", "dose_sensitivity",
                  "exposure_index"):
        if summary.get(block) is None:
            problems.append(f"{block!r} block missing from the summary")

    # V7's scope statement is not optional prose. If the biliary-escape pricing
    # is absent the manuscript would assert the unconditional form of the
    # headline, which is the exact overclaim V7b exists to prevent.
    if not (summary.get("null") or {}).get("biliary_escape"):
        problems.append("null.biliary_escape missing; V7 would be stated "
                        "unconditionally (see FINDINGS.md Pass 6)")

    ident = summary.get("identifiability") or {}
    sets = ident.get("sets") or {}
    if sets and len(sets) != len(idf.SETS):
        problems.append(f"identifiability has {len(sets)} measurement sets, "
                        f"expected {len(idf.SETS)}")

    ens = summary.get("ensemble") or {}
    if ens and not ens.get("n_ok"):
        problems.append("ensemble.n_ok is 0 or absent; no draw integrated")
    if ens and ens.get("null_cost") is None:
        problems.append("ensemble.null_cost missing; V9 must be quoted as an "
                        "interval, not as a pass (--n-null-audit 0 was used)")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures/output")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write the tables even if the run was partial or "
                         "failed; for debugging only, never for a manuscript")
    args = ap.parse_args()
    out = Path(args.outdir)
    summary = json.loads((out / "emd2_simulation_summary.json").read_text())

    problems = audit(summary)
    if problems:
        print("REFUSING to write manuscript tables from this summary:")
        for why in problems:
            print(f"  - {why}")
        print(f"\n  The summary in {out / 'emd2_simulation_summary.json'} is "
              "from a partial or failed run.\n  Rerun "
              "`python -m emd2_simulation.run` with no skip flags, then retry.")
        if not args.allow_incomplete:
            return 1
        print("\n  --allow-incomplete given; writing anyway. DO NOT PUBLISH THIS.")

    tables = build(summary)
    path = out / "emd2_manuscript_tables.json"
    path.write_text(json.dumps(tables, indent=2, default=float))
    print(f"wrote {path}  ({len(tables['parameters'])} parameters, "
          f"{len(tables['checks'])} checks, {len(tables['identifiability'])} "
          f"measurement sets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
