"""Import the selected workbook rows, descriptive summaries and acquisition metadata into SQLite."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from ..paths import DATABASE_PATH
from .published import sheet_selection

SCHEMA_VERSION = 1
SOURCE_FILES = ('published_roje2024.json', 'observed_roje2024.json', 's_MTBLS3581.txt')
EXPECTED_COUNTS = {
    'S4': 141, 'S5': 65, 'S15': 72, 'S16': 23, 'S23': 113,
    'S32': 368, 'S37': 144, 'S38': 72, 'S39': 512, 'S40': 72,
    'S42': 2, 'S3': 10, 'S18_BCPN': 134, 'S18_gBBN': 44, 'S18_contrasts': 48,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def source_directory(source_root: Path) -> Path:
    candidates = [source_root / 'src/emd2_simulation/data',
                  source_root / 'code/emd2_simulation/data', source_root / 'data', source_root]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in SOURCE_FILES):
            return candidate
    raise FileNotFoundError('Selected source tree lacks the three original EMD2 input files')


def build(source_root: Path, output: Path = DATABASE_PATH) -> dict[str, int]:
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    inputs = source_directory(source_root)
    manifest_path = source_root / 'data/SOURCES.tsv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Required original source manifest is missing: {manifest_path}')
    with manifest_path.open(newline='', encoding='utf-8') as stream:
        manifest = {row['path']: row for row in csv.DictReader(stream, delimiter='\t')}
    published = json.loads((inputs / SOURCE_FILES[0]).read_text())
    observed = json.loads((inputs / SOURCE_FILES[1]).read_text())
    if published.get('schema_version') != 1:
        raise ValueError('Unsupported original published-row schema')
    counts = {sheet: len(rows) for sheet, rows in published['tables'].items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f'Published source ranges changed; review selection: {counts}')
    published["source"]["selection"] = sheet_selection(published["tables"])
    if len(observed) != 11 or 'comparability' not in observed:
        raise ValueError('Unexpected descriptive source structure')
    workbook = manifest['41586_2024_7754_MOESM4_ESM.xlsx']
    if published['source']['sha256'] != workbook['sha256']:
        raise ValueError('Published extract and original workbook manifest checksums disagree')
    sample_manifest = manifest['s_MTBLS3581.txt']
    sample_path = inputs / 's_MTBLS3581.txt'
    if sha256(sample_path) != sample_manifest['sha256']:
        raise ValueError('Original MetaboLights acquisition table checksum changed')
    with sample_path.open(newline='', encoding='utf-8') as stream:
        source_rows = list(csv.reader(stream, delimiter='\t'))
    header, samples = source_rows[0], source_rows[1:]
    expected = {0: 'Source Name', 4: 'Characteristics[Organism part]',
                10: 'Characteristics[Sample type]', 18: 'Factor Value[Abx treatment]',
                21: 'Factor Value[Bbn treatment length]'}
    if any(header[index] != label for index, label in expected.items()) or len(samples) != 2957:
        raise ValueError('Acquisition-table layout/count changed; review original source')
    if any(len(row) <= 21 for row in samples):
        raise ValueError('Incomplete acquisition metadata row')

    references = [
        ('roje2024', 'Roje et al., Nature 632:1137–1144 (2024)',
         'Gut microbiota carcinogen metabolism causes distal tissue tumours',
         '10.1038/s41586-024-07754-w', published['source']['url'],
         'Selected workbook measurements and counts, attributed to Roje et al. under CC BY 4.0.'),
        ('mtbls3581', 'MetaboLights MTBLS3581, deposited LC-MS sample metadata',
         'MTBLS3581 study acquisition metadata', None,
         'https://www.ebi.ac.uk/metabolights/MTBLS3581',
         'Each row represents an LC–MS acquisition. Source names and recorded tissue labels support checks of analytical coverage.'),
    ]
    provenance = [
        ('published_extract', str((inputs / SOURCE_FILES[0]).relative_to(source_root)),
         sha256(inputs / SOURCE_FILES[0]), None, 'roje2024', 'selected workbook rows',
         'Imported 1,820 rows from 15 workbook ranges with their source row numbers, order, labels and units.'),
        ('observed_extract', str((inputs / SOURCE_FILES[1]).relative_to(source_root)),
         sha256(inputs / SOURCE_FILES[1]), None, 'roje2024', 'descriptive workbook summary',
         'Imported 11 components of the workbook summary. Cross-matrix concentration ratios have unresolved normalization.'),
        ('metabolights_sample_table', str(sample_path.relative_to(source_root)),
         sha256(sample_path), sample_manifest['url'], 'mtbls3581', 'LC–MS acquisition metadata',
         'Selected source name, organism part, sample type, antibiotic treatment and BBN treatment length for all 2,957 acquisitions. Source row numbers and tissue labels are preserved.'),
        ('source_workbook', workbook['path'], workbook['sha256'], workbook['url'],
         'roje2024', 'source workbook identity',
         'Workbook identity and SHA-256 from the source manifest and the workbook extract.'),
    ]
    metadata = [
        ('schema_version', str(SCHEMA_VERSION)),
        ('scope', 'Published measurements, descriptive summaries and LC–MS acquisition metadata for EMD2.'),
        ('published_rows', '1,820 records from 15 workbook ranges, in the source units.'),
        ('acquisition_unit', 'One LC–MS acquisition per record.'),
        ('legacy_observed', 'Descriptive summaries of intestinal, systemic and urinary measurements.'),
        ('cross_matrix_comparability', 'Interpreting a caecum-to-plasma ratio requires comparable recovery, dilution and internal-standard normalization across matrices.'),
        ('source_manifest_sha256', sha256(manifest_path)),
        ('builder', 'emd2_simulation.processing.build_database'),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='emd2-database-', dir=output.parent) as folder:
        temporary = Path(folder) / 'emd2.sqlite'
        with sqlite3.connect(temporary) as connection:
            connection.execute('PRAGMA page_size = 4096')
            connection.execute('PRAGMA foreign_keys = ON')
            connection.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            connection.executescript('''
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE "references" (
                    reference_id TEXT PRIMARY KEY, citation TEXT NOT NULL,
                    title TEXT NOT NULL, doi TEXT, url TEXT NOT NULL, notes TEXT NOT NULL);
                CREATE TABLE provenance (
                    artifact_id TEXT PRIMARY KEY, source_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL, url TEXT,
                    reference_id TEXT NOT NULL REFERENCES "references"(reference_id),
                    role TEXT NOT NULL, transformation_notes TEXT NOT NULL);
                CREATE TABLE published_metadata (
                    position INTEGER UNIQUE NOT NULL, key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    reference_id TEXT NOT NULL REFERENCES "references"(reference_id),
                    provenance_id TEXT NOT NULL REFERENCES provenance(artifact_id));
                CREATE TABLE published_rows (
                    sheet_order INTEGER NOT NULL, sheet TEXT NOT NULL,
                    record_order INTEGER NOT NULL, source_row INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    reference_id TEXT NOT NULL REFERENCES "references"(reference_id),
                    provenance_id TEXT NOT NULL REFERENCES provenance(artifact_id),
                    PRIMARY KEY(sheet, record_order), UNIQUE(sheet, source_row));
                CREATE TABLE observed_components (
                    position INTEGER UNIQUE NOT NULL, key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    reference_id TEXT NOT NULL REFERENCES "references"(reference_id),
                    provenance_id TEXT NOT NULL REFERENCES provenance(artifact_id));
                CREATE TABLE acquisition_records (
                    acquisition_index INTEGER PRIMARY KEY, source_row INTEGER UNIQUE NOT NULL,
                    source_name TEXT NOT NULL, compartment TEXT NOT NULL,
                    sample_type TEXT NOT NULL, arm TEXT NOT NULL, week TEXT NOT NULL,
                    reference_id TEXT NOT NULL REFERENCES "references"(reference_id),
                    provenance_id TEXT NOT NULL REFERENCES provenance(artifact_id));
            ''')
            connection.executemany('INSERT INTO "references" VALUES (?,?,?,?,?,?)', references)
            connection.executemany('INSERT INTO provenance VALUES (?,?,?,?,?,?,?)', provenance)
            connection.executemany('INSERT INTO metadata VALUES (?,?)', metadata)
            for position, (key, value) in enumerate(published.items()):
                if key != 'tables':
                    connection.execute('INSERT INTO published_metadata VALUES (?,?,?,?,?)',
                                       (position, key, encode(value), 'roje2024', 'published_extract'))
            for sheet_order, (sheet, rows) in enumerate(published['tables'].items()):
                for record_order, row in enumerate(rows):
                    connection.execute('INSERT INTO published_rows VALUES (?,?,?,?,?,?,?)',
                                       (sheet_order, sheet, record_order, row['source_row'], encode(row),
                                        'roje2024', 'published_extract'))
            for position, (key, value) in enumerate(observed.items()):
                connection.execute('INSERT INTO observed_components VALUES (?,?,?,?,?)',
                                   (position, key, encode(value), 'roje2024', 'observed_extract'))
            connection.executemany('INSERT INTO acquisition_records VALUES (?,?,?,?,?,?,?,?,?)',
                                   ((index, index + 2, row[0], row[4], row[10], row[18], row[21],
                                     'mtbls3581', 'metabolights_sample_table')
                                    for index, row in enumerate(samples)))
            if connection.execute('PRAGMA foreign_key_check').fetchall():
                raise ValueError('Database foreign-key check failed')
            if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Database integrity check failed')
            connection.commit()
            connection.execute('VACUUM')
        temporary.replace(output)
    return {'published_rows': sum(counts.values()), 'published_metadata': 3,
            'observed_components': len(observed), 'acquisition_records': len(samples),
            'references': len(references), 'provenance': len(provenance), 'metadata': len(metadata)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path,
                        help='Source archive containing the JSON extracts, LC–MS sample table and data/SOURCES.tsv')
    parser.add_argument('--output', type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    counts = build(args.source_root, args.output)
    print(f'Built {args.output.resolve()} ({args.output.stat().st_size:,} bytes)')
    for name, count in counts.items():
        print(f'  {name}: {count}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
