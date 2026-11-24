"""Summarize systemic and urinary BCPN measurements and intestinal profiles.

S15 and S39 match tissues by animal ID. S6 and S7 compare batch-matched
cohort medians. Matrix-specific recovery and dilution are unresolved across
tissues. SI1, SI2 and SI3 are interpreted as proximal-to-distal thirds.

Running this module writes derived/observed_roje2024.json."""

from __future__ import annotations

import json
import hashlib
import statistics as st
from collections import defaultdict
from pathlib import Path

from ..paths import RAW_DATA_DIR, DERIVED_DIR
from .published import SOURCE_SHA256

DATA = RAW_DATA_DIR
XLSX = DATA / "41586_2024_7754_MOESM4_ESM.xlsx"
OUT = DERIVED_DIR / "observed_roje2024.json"

SEG = {"SI1": "duodenum", "SI2": "jejunum", "SI3": "ileum",
       "CEC": "cecum", "COL": "colon", "REC": "rectum"}
SOURCE = ("Roje B et al. Gut microbiota carcinogen metabolism causes distal "
          "tissue tumours. Nature 632:1137-1144 (2024), Supplementary Tables "
          "1-51 (41586_2024_7754_MOESM4_ESM.xlsx), DOI 10.1038/s41586-024-07754-w")


def _sheet(wb, name: str) -> list[dict]:
    rows = [r for r in wb[name].iter_rows(values_only=True)]
    hdr = [str(h).strip() if h is not None else "" for h in rows[0]]
    return [dict(zip(hdr, r)) for r in rows[1:] if any(c is not None for c in r)]


def _med(v):
    return st.median(v) if v else None


def _profile(rows, conc_key: str, group_key: str = "Group") -> dict:
    """Median BCPN per segment per group, in uM."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if str(r.get("Drug", "BCPN")).strip() != "BCPN":
            continue
        seg = SEG.get(str(r.get("Tissue", "")).strip())
        if seg is None or r.get(conc_key) is None:
            continue
        by[str(r[group_key])][seg].append(float(r[conc_key]))
    return {g: {s: _med(v) for s, v in sorted(d.items())} for g, d in by.items()}


def _same_animal_ratios(rows, conc_key: str, *, dg: bool) -> dict:
    """Pair caecum and plasma within animal, excluding conflicting measurements."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if str(r.get("Drug", "BCPN")).strip() != "BCPN":
            continue
        if dg and str(r.get("Deglucuronidation", "")).strip() != "BeforeDG":
            continue
        if r.get(conc_key) is None:
            continue
        by[(str(r["Group"]), str(r["ID"]))][str(r["Tissue"]).strip()].append(
            float(r[conc_key]))
    out = defaultdict(list)
    for (grp, _id), tis in by.items():
        cec_values, pl_values = set(tis.get("CEC", [])), set(tis.get("PL", []))
        if len(cec_values) != 1 or len(pl_values) != 1:
            continue
        cec, pl = next(iter(cec_values)), next(iter(pl_values))
        if cec is not None and pl:
            out[grp].append(round(cec / pl, 4))
    return {g: sorted(v) for g, v in out.items()}


def build() -> dict:
    if hashlib.sha256(XLSX.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source workbook checksum changed; review its layout before importing")
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    try:
        s5, s6, s7 = _sheet(wb, "S5"), _sheet(wb, "S6"), _sheet(wb, "S7")
        s13, s15, s39 = _sheet(wb, "S13"), _sheet(wb, "S15"), _sheet(wb, "S39")
    finally:
        wb.close()

    prof3 = _profile(s7, "uM")
    prof7 = _profile(s13, "uM")

    # Pairing does not resolve matrix-specific normalisation. Conflicting S15
    # caecal readings are excluded, so no GF/GFA ratio is produced.
    per_animal = {}
    per_animal.update(_same_animal_ratios(s15, "uM", dg=False))
    per_animal.update(_same_animal_ratios(s39, "uM_raw", dg=True))

    # S7 caecum and S6 plasma are matched by batch and treatment.
    PL = "nM (nmole/kg)"
    batch_ratio = defaultdict(dict)
    for grp in ("BBN", "ABX/BBN"):
        for b in sorted({str(r.get("Batch")) for r in s7}):
            cec = _med([float(r["uM"]) for r in s7
                        if str(r.get("Group")) == grp and str(r.get("Tissue")) == "CEC"
                        and str(r.get("Batch")) == b and str(r.get("Drug")) == "BCPN"
                        and r.get("uM") is not None])
            pl = _med([float(r[PL]) / 1000.0 for r in s6
                       if str(r.get("Group")) == grp and str(r.get("Tissue")) == "PL"
                       and str(r.get("Batch")) == b and r.get(PL) is not None])
            if cec is not None and pl:
                batch_ratio[grp][b] = {"cecum_uM": round(cec, 4),
                                       "plasma_uM": round(pl, 5),
                                       "ratio": round(cec / pl, 4)}

    systemic = {}
    for tis, lab in (("PL", "plasma"), ("KID", "kidney"), ("LVR", "liver")):
        vals = {}
        for grp in ("BBN", "ABX/BBN"):
            v = [float(r[PL]) for r in s6 if str(r.get("Group")) == grp
                 and str(r.get("Tissue")) == tis and r.get(PL) is not None]
            vals[grp] = _med(v)
        systemic[lab] = {"conventional": vals["BBN"], "antibiotic": vals["ABX/BBN"],
                         "ratio": (vals["BBN"] / vals["ABX/BBN"]
                                   if vals["ABX/BBN"] else None),
                         "unit": "nM" if tis == "PL" else "nmole/kg"}

    urine = {}
    for grp, lab in (("BBN", "conventional"), ("ABX/BBN", "antibiotic")):
        sel = [r for r in s5 if str(r.get("Group")) == grp
               and str(r.get("Drug")) == "BCPN" and str(r.get("Length")) == "3wk"]
        urine[lab] = {
            "nM": _med([float(r["nM"]) for r in sel if r.get("nM") is not None]),
            "nmole_24h": _med([float(r["nmole"]) for r in sel
                               if r.get("nmole") is not None]),
            "n": len(sel)}

    tests = {}
    try:
        from scipy.stats import mannwhitneyu
        pairs = [("Germ-Free", "2Mem"), ("Germ-Free", "3Mem"), ("2Mem", "3Mem"),
                 ("GF", "2Mem"), ("GF", "3Mem")]
        for a, b in pairs:
            if a in per_animal and b in per_animal:
                p = mannwhitneyu(per_animal[a], per_animal[b],
                                 alternative="two-sided").pvalue
                tests[f"{a}_vs_{b}"] = round(float(p), 4)
    except Exception:
        tests = {"note": "scipy unavailable; tests not computed"}

    return {
        "source": SOURCE,
        "tables_used": ["S5", "S6", "S7", "S13", "S15", "S39"],
        "segment_mapping": SEG,
        "comparability": 'Ratios are provisional source-table summaries. Same-animal pairing or batch matching does not cancel matrix-specific extraction, dilution, recovery or internal-standard factors. Original-compartment concentration normalisation has not been demonstrated here, so comparisons against a physical threshold of one are not validated. Conventional/antibiotic summaries are ratios of batch-matched cohort medians, not same-animal ratios. Cross-sheet animal-ID linkage has not been established by the importer. Reported zeros may be below detection, not biological absence. Absolute concentrations are not directly comparable across sheets; intraluminal profiles are used as shape rather than calibrated magnitude.',
        "profile_uM_3wk": prof3,
        "profile_uM_7wk": prof7,
        "ratio_per_animal": per_animal,
        "ratio_batch_matched": {k: dict(v) for k, v in batch_ratio.items()},
        "systemic_3wk": systemic,
        "urine_3wk": urine,
        "tests": tests,
    }


def main() -> int:
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    pa = out["ratio_per_animal"]
    print(f"wrote {OUT}")
    print("  same-animal caecum/plasma ratios (median [n]):")
    for g, v in sorted(pa.items()):
        print(f"    {g:<12} {st.median(v):.3f}  [n={len(v)}]  "
              f"range {min(v):.3f}-{max(v):.3f}")
    print("  batch-matched cohort ratios:")
    for g, d in out["ratio_batch_matched"].items():
        for b, r in d.items():
            print(f"    {g:<10} {b:<12} ratio={r['ratio']:.3f} "
                  f"(cec {r['cecum_uM']} uM / pl {r['plasma_uM']} uM)")
    print(f"  tests: {out['tests']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
