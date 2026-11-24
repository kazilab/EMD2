"""Build supplementary model tables from a completed reference run."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from ..paths import OUTPUT_DIR
from ..analysis import identifiability as idf
from ..simulation.conditions import ARMS, BY_KEY, RATIO_SEGMENT
from ..analysis.checks import CHECKS
from ..simulation.ensemble import FREE_SIGMA
from ..analysis.checks import STRUCTURAL_CHECKS
from ..simulation.model import (IDX, SEG_ABSORB, SEG_KTRANSIT, SEG_MICROBES, SEG_OXYGEN,
                    SEG_VOL, SEGMENTS, Params, lumen_plasma_ratio, simulate,
                    trace)

T_END, N_T = 720.0, 1441

PARAM_META = {
    "dose_rate": ("nmol h⁻¹", "Drinking-water mass balance assumes 0.05% w/v and "
                  "5 mL/day. The source also reports v/v. The dose_sensitivity table "
                  "shows the effect of varying dose alone and jointly with Vmax and Km."),
    "V_plasma": ("mL", "Assigned plasma volume."),
    "V_bladder": ("mL", "Assigned bladder volume."),
    "V_kidney": ("mL", "Assigned combined kidney volume."),
    "Q_bile": ("mL h⁻¹", "Assigned bile flow for converting biliary flux to concentration."),
    "ka_BBN": ("h⁻¹", "Assigned absorption rate for the parent compound."),
    "ka_BCPN": ("h⁻¹", "Assigned metabolite absorption rate. Ionisation motivates "
                "the lower value relative to the parent compound."),
    "k_back": ("h⁻¹", "Assigned plasma-to-lumen BCPN transfer, independent of absorption."),
    "k_uptake": ("h⁻¹", "Assigned plasma-to-liver uptake of the parent compound."),
    "k_ox_host": ("h⁻¹", "Host oxidation is assumed to be slow relative to glucuronidation."),
    "k_gluc": ("h⁻¹", "Assigned hepatic glucuronidation rate."),
    "k_bile": ("h⁻¹", "Assigned biliary transfer of glucuronide."),
    "k_bile_BCPN": ("h⁻¹", "Biliary BCPN transfer is zero at baseline. Alternative "
                    "values are evaluated in the biliary_escape table."),
    "Vmax_gus": ("nmol h⁻¹ mL⁻¹", "Assigned caecal β-glucuronidase capacity."),
    "Km_gus": ("nmol mL⁻¹", "Assigned half-saturation concentration for β-glucuronidase."),
    "Vmax_ox": ("nmol h⁻¹ mL⁻¹", "Assigned microbial oxidation capacity profiled "
                "in the identifiability analysis."),
    "Km_ox": ("nmol mL⁻¹", "Assigned half-saturation concentration for oxidation."),
    "k_renal": ("h⁻¹", "Assigned plasma-to-kidney transfer."),
    "k_kidney_out": ("h⁻¹", "Assigned transfer from kidney to bladder. At steady state, "
                     "kidney outflow equals inflow."),
    "k_void": ("h⁻¹", "Assigned bladder emptying rate."),
    "k_uro": ("dimensionless", "Scale factor for the exposure index. In every arm, "
              "the index equals cumulative urinary excretion times k_uro/(V_bladder·k_void)."),
    "k_clear_p": ("h⁻¹", "Assigned non-renal clearance of plasma BCPN."),
    "k_clear_BBN": ("h⁻¹", "Assigned non-hepatic clearance of the parent compound."),
}

STATE_GROUPS = [
    ("Gut lumen, per segment (6 segments × 4 species = 24 states)",
     [("A_BBN,i", "parent BBN in segment i"),
      ("A_gBBN,i", "biliary glucuronide in segment i"),
      ("A_BCPNᵐ,i", "luminal BCPN of microbial origin (latent)"),
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
     [("AUCʰ, AUCᵐ", "cumulative urothelial exposure index by source"),
      ("Uʰ, Uᵐ", "urinary excretion by provenance")]),
]


def build(summary: dict) -> dict:
    metadata = summary.get("reproducibility")
    if metadata is not None and not metadata.get("reference_compatible", False):
        raise ValueError("These tables require the reference run. The CSV tables contain results for custom settings.")
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
    """List missing results, failed checks and incompatible reference settings."""
    problems = []
    metadata = summary.get("reproducibility")
    if metadata is not None and not metadata.get("reference_compatible", False):
        problems.append("settings differ from the reference run")

    checks = summary.get("checks") or {}
    failed = sorted(cid for cid, ok in checks.items() if not ok)
    if failed:
        problems.append(f"model checks failed: {', '.join(failed)}")
    missing = sorted(EXPECTED_CHECKS - set(checks))
    if missing:
        problems.append(f"Missing model checks: {', '.join(missing)}")

    for block in ("arms", "null", "null_host_cost", "scope",
                  "identifiability", "ensemble", "two_member_gus_sensitivity",
                  "capacity_decomposition", "dose_sensitivity",
                  "exposure_index"):
        if summary.get(block) is None:
            problems.append(f"{block!r} block missing from the summary")

    if not (summary.get("null") or {}).get("biliary_escape"):
        problems.append("Missing host-only biliary comparison (null.biliary_escape)")

    ident = summary.get("identifiability") or {}
    sets = ident.get("sets") or {}
    if sets and len(sets) != len(idf.SETS):
        problems.append(f"identifiability has {len(sets)} measurement sets, "
                        f"expected {len(idf.SETS)}")

    ens = summary.get("ensemble") or {}
    if ens and not ens.get("n_ok"):
        problems.append("No completed parameter draws recorded")
    if ens and ens.get("null_cost") is None:
        problems.append("Missing host-only sensitivity interval (ensemble.null_cost)")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(OUTPUT_DIR / "reproduced"))
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="attempt to export available tables after reporting missing results")
    args = ap.parse_args()
    out = Path(args.outdir)
    summary = json.loads((out / "emd2_simulation_summary.json").read_text())

    problems = audit(summary)
    if problems:
        print("The reference table checks found problems.")
        for why in problems:
            print(f"  - {why}")
        if not args.allow_incomplete:
            print(f"Run the full reference analysis in {out} before exporting these tables.")
            return 1

    tables = build(summary)
    path = out / "emd2_manuscript_tables.json"
    path.write_text(json.dumps(tables, indent=2, default=float, allow_nan=False))
    print(f"wrote {path}  ({len(tables['parameters'])} parameters, "
          f"{len(tables['checks'])} checks, {len(tables['identifiability'])} "
          f"measurement sets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
