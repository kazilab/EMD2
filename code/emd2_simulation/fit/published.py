"""Import selected published experiments, retaining row provenance.

Optional rebuild: python -m emd2_simulation.fit.published
The main analysis uses the bundled JSON and does not need the source workbook
or openpyxl. Source-table units are retained; no cross-matrix conversion is
assumed. The checksum binds the explicitly selected source ranges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
WORKBOOK = DATA / "41586_2024_7754_MOESM4_ESM.xlsx"
OUTPUT = DATA / "published_roje2024.json"
SOURCE_SHA256 = "fe43970068026537c9f95b8132ded29d93397598cc28aab74e381147e341928c"
SOURCE_URL = "https://doi.org/10.1038/s41586-024-07754-w"
SHEETS = ("S4", "S5", "S15", "S16", "S23", "S32", "S37", "S38", "S39", "S40", "S42")


def build(path: Path = WORKBOOK) -> dict:
    import openpyxl

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Source workbook checksum changed; review its layout before importing")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    tables = {}
    for name in SHEETS:
        # S32 has a separate strain/source key in I:K, including a second
        # column named Strain. Never let that side table overwrite column D.
        iterator = wb[name].iter_rows(max_col=7 if name == "S32" else None, values_only=True)
        header = [str(v).strip() if v is not None else None for v in next(iterator)]
        named = [v for v in header if v is not None]
        if len(named) != len(set(named)):
            raise ValueError(f"Duplicate column names in selected {name} range")
        tables[name] = [
            {"source_row": n, **{k: v for k, v in zip(header, row) if k is not None}}
            for n, row in enumerate(iterator, 2) if any(v is not None for v in row)
        ]
    # S3 gives the organism-level outcome as FIVE separate experiments, with the
    # group label written once per ABX/BBN + BBN pair and left blank on the
    # second row. The blank is layout, not a missing value, so it is forward
    # filled into an explicit `experiment` key while `Group` keeps exactly what
    # the cell contained. Pooling the five is what the article reports; this
    # build analyses them stratified, so the structure has to survive import.
    rows = list(wb["S3"].iter_rows(values_only=True))
    header = [str(v).strip() if v is not None else None for v in rows[0]]
    experiment = None
    tables["S3"] = []
    for n, row in enumerate(rows[1:], 2):
        if not any(v is not None for v in row):
            continue
        record = {"source_row": n, **{k: v for k, v in zip(header, row) if k is not None}}
        if record.get("Group") is not None:
            experiment = str(record["Group"]).strip()
        record["experiment"] = experiment
        tables["S3"].append(record)

    # S18 has a spurious million-row used range. The checksum-reviewed data
    # occupy three side-by-side blocks within the first 135 rows.
    rows = list(wb["S18"].iter_rows(max_row=135, values_only=True))
    for name, lo, hi in (("S18_BCPN", 0, 6), ("S18_gBBN", 7, 12),
                         ("S18_contrasts", 13, 28)):
        header = [str(v).strip() for v in rows[0][lo:hi]]
        tables[name] = [
            {"source_row": n, **dict(zip(header, row[lo:hi]))}
            for n, row in enumerate(rows[1:], 2)
            if row[lo] is not None
        ]
    wb.close()
    return {
        "schema_version": 1,
        "source": {"citation": "Roje et al. Nature 632, 1137–1144 (2024)",
                   "url": SOURCE_URL, "workbook": path.name, "sha256": digest,
                   "selection": "S4, S5, S15, S16, S18, S23, S32, S37–S40"},
        "units": {
            "S39": "uM_raw: source-table µM; compare groups within the same matrix",
            "S23": "BCPN_nM in culture medium; technical replicates nested in donor",
            "S5": "nmole excreted over 24 h; retain collection week and batch",
            "S4": "nM source-table bladder measurement; retain week and batch",
            "S15": "uM source-table values; compare GF/GFA within matrix",
            "S18_BCPN": "BCPN_uM source-table values; tissue/plasma normalisation unresolved",
            "S32": "A. U. LC-MS signal, not concentration or exposure AUC",
            "S37": "wide tissue signals have no declared absolute unit; mixed SI1 scales need reconciliation",
            "S3": "animal counts by terminal bladder histology in five separate experiments (I-V); `experiment` is the forward-filled group label, `Group` is the raw cell",
            "S42": "animal counts by terminal bladder histology, EHBN agent, one experiment",
        },
        "tables": tables,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", type=Path, default=WORKBOOK)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    result = build(args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: " + ", ".join(f"{k}={len(v)} rows" for k, v in result["tables"].items()))


if __name__ == "__main__":
    main()
