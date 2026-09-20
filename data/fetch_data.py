#!/usr/bin/env python3
"""Fetch or verify source inputs. The core simulation needs no downloads.

`s_MTBLS3581.txt` is already bundled, because the run, the identifiability
analysis and two validation checks read it directly. Everything else is
optional: the MetaboLights assay and metabolite-assignment files are provenance
only, and the source paper's supplementary workbook is publisher content that
this deposit does not redistribute.

Run from any directory. `--verify` downloads nothing and exits non-zero on a
missing automatic input or any checksum mismatch. Manual inputs are optional
and their absence is reported rather than treated as a failure.
"""
import argparse
import csv
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DEST = HERE.parent / "code" / "emd2_simulation" / "data"
MANIFEST = HERE / "SOURCES.tsv"


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows():
    with MANIFEST.open(newline="") as fh:
        yield from csv.DictReader(fh, delimiter="\t")


def checked_copy(stream, destination: Path, expected: str) -> None:
    """Never install a partial or mismatched download as an analysis input."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as fh:
        staging = Path(fh.name)
        shutil.copyfileobj(stream, fh)
    got = sha256(staging)
    if got != expected:
        staging.unlink(missing_ok=True)
        raise SystemExit(
            f"checksum mismatch for {destination.name}\n"
            f"  expected {expected}\n  got      {got}\n"
            "The source has changed since this deposit was made. Nothing was "
            "installed; compare against the manifest before proceeding."
        )
    staging.replace(destination)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check existing files without downloading")
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help="destination directory (default: the package's data/)")
    args = ap.parse_args()
    dest = Path(args.dest).resolve()

    problems, fetched, ok, skipped = [], 0, 0, 0
    for r in rows():
        target = dest / r["path"]
        automatic = r["automatic"] == "1"

        if target.exists():
            got = sha256(target)
            if got == r["sha256"]:
                print(f"  ok        {r['path']}")
                ok += 1
            else:
                print(f"  MISMATCH  {r['path']}")
                problems.append(f"{r['path']}: checksum mismatch")
            continue

        if args.verify:
            # Only a missing BUNDLED file is a problem. Everything else is
            # optional provenance that this deposit deliberately does not
            # carry, so a clean checkout must verify cleanly.
            if r["bundled"] == "1":
                print(f"  MISSING   {r['path']}  (bundled file absent)")
                problems.append(f"{r['path']}: bundled file missing")
            else:
                kind = "fetchable" if automatic else "manual download"
                print(f"  absent    {r['path']}  (optional, {kind})")
                skipped += 1
            continue

        if not automatic:
            print(f"  skip      {r['path']}  (manual: {r['url']})")
            skipped += 1
            continue

        print(f"  fetching  {r['path']}")
        try:
            with urllib.request.urlopen(r["url"]) as stream:
                checked_copy(stream, target, r["sha256"])
        except SystemExit:
            raise
        except Exception as exc:                      # network, 404, timeout
            print(f"  FAILED    {r['path']}: {exc}")
            problems.append(f"{r['path']}: {exc}")
            continue
        fetched += 1

    print(f"\n  {ok} verified, {fetched} fetched, {skipped} manual/skipped, "
          f"{len(problems)} problem(s)")
    if problems:
        for p in problems:
            print(f"    - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
