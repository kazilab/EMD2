"""Read source measurements, descriptive summaries and acquisition metadata from SQLite."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from ..paths import DATABASE_PATH

SCHEMA_VERSION = 1
TABLES = frozenset({"metadata", "references", "provenance", "published_metadata",
                    "published_rows", "observed_components", "acquisition_records"})
TABLE_DESCRIPTIONS = {
    "published_rows": "Workbook measurements and counts with their sheet names and row numbers.",
    "published_metadata": "Workbook citation, file identity, schema and units.",
    "observed_components": "Descriptive summaries of gut, systemic and urinary measurements.",
    "acquisition_records": "LC–MS acquisition metadata for analytical coverage and tissue-label checks.",
    "references": "Citations, URLs and source terms.",
    "provenance": "Source files, checksums and extraction methods.",
    "metadata": "Database schema, scope, unit conventions and source manifest checksum.",
}


@contextmanager
def connect_readonly(database: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open an existing input database in SQLite read-only mode."""
    path = Path(database) if database is not None else DATABASE_PATH
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required EMD2 input database is missing: {path}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported EMD2 database schema {version}; expected {SCHEMA_VERSION}")
        yield connection
    finally:
        connection.close()


def read_table(name: str, database: str | Path | None = None):
    """Return a database table as a pandas DataFrame in row order."""
    if name not in TABLES:
        raise ValueError(f"Unknown EMD2 input table: {name!r}")
    import pandas as pd
    with connect_readonly(database) as connection:
        return pd.read_sql_query(f'SELECT * FROM "{name}" ORDER BY rowid', connection)


def source_checksum(artifact_id: str, database: str | Path | None = None) -> str:
    """Return an original source-artifact hash, distinct from the database hash."""
    with connect_readonly(database) as connection:
        row = connection.execute("SELECT sha256 FROM provenance WHERE artifact_id = ?",
                                 (artifact_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown EMD2 source artifact: {artifact_id}")
    return row[0]


def load_published(database: str | Path | None = None) -> dict:
    """Return workbook metadata and records grouped by source table."""
    with connect_readonly(database) as connection:
        result = {key: json.loads(value) for key, value in connection.execute(
            "SELECT key, value_json FROM published_metadata ORDER BY position")}
        tables = {}
        for sheet, payload in connection.execute(
                "SELECT sheet, record_json FROM published_rows ORDER BY sheet_order, record_order"):
            tables.setdefault(sheet, []).append(json.loads(payload))
    result["tables"] = tables
    return result


def published_table(sheet: str, database: str | Path | None = None):
    """Return a selected source sheet as a wide DataFrame without dropping rows."""
    import pandas as pd
    with connect_readonly(database) as connection:
        rows = connection.execute(
            "SELECT record_json FROM published_rows WHERE sheet = ? ORDER BY record_order", (sheet,)).fetchall()
    if not rows:
        raise KeyError(f"Unknown selected source table: {sheet}")
    return pd.DataFrame(json.loads(row[0]) for row in rows)


def load_observed(database: str | Path | None = None) -> dict:
    """Return the descriptive summaries derived from the source workbook."""
    with connect_readonly(database) as connection:
        return {key: json.loads(value) for key, value in connection.execute(
            "SELECT key, value_json FROM observed_components ORDER BY position")}


def load_acquisitions(database: str | Path | None = None) -> list[dict]:
    """Return LC–MS acquisition metadata, including standards and original tissue labels."""
    with connect_readonly(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT source_row, source_name, compartment, sample_type, arm, week "
            "FROM acquisition_records ORDER BY acquisition_index").fetchall()
    return [dict(row) for row in rows]


def table_catalog(database: str | Path | None = None):
    """Show row counts and meanings for every shipped database table."""
    import pandas as pd
    with connect_readonly(database) as connection:
        return pd.DataFrame({"table": name,
                             "rows": connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0],
                             "meaning": description}
                            for name, description in TABLE_DESCRIPTIONS.items())
