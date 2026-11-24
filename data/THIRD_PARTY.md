# EMD2 source data

The experimental measurements come from the supplementary workbook for Roje et al., *Gut microbiota carcinogen metabolism causes distal tissue tumours*, *Nature* 632, 1137–1144 (2024). The compartmental simulation uses assigned kinetics. [Source article and supplementary tables](https://doi.org/10.1038/s41586-024-07754-w)

## Database

The notebook reads `emd2.sqlite` (schema version 1). The `published_rows` table contains 1,820 records from 15 workbook ranges. Each record retains the sheet name, row number, source labels, units and values. In Python, `emd2_simulation.processing.store.published_table("S39")` returns one sheet as a data frame. The `references` and `provenance` tables identify the source files and their checksums.

The `acquisition_records` table contains 2,957 LC–MS acquisitions, including standards. The records describe analytical coverage. Biological replication comes from the workbook's animal or donor identifiers. Sixteen of 48 two-member consortium acquisitions labelled colon have a liver filename. Both labels are retained. The deposited metabolite-assignment file lacks quantitative per-sample abundances. [MetaboLights MTBLS3581](https://www.ebi.ac.uk/metabolights/MTBLS3581)

The 11 entries in `observed_components` summarize gut profiles, urinary excretion and systemic measurements. The model maps the source segments SI1, SI2 and SI3 to duodenum, jejunum and ileum as a convention. Caecum-to-plasma ratios are provisional because matrix-specific extraction, recovery and dilution are unresolved, even for paired animals. Ratios from S6 and S7 compare batch-matched cohort medians.

## Measurements and eligibility

| Source tables | Measurements | Treatment in the analysis |
| --- | --- | --- |
| S3, S42 | Animal counts by terminal bladder histology | S3 compares BBN treatment groups across five experiments using the Cochran–Mantel–Haenszel test stratified by experiment. S42 compares EHBN groups in one experiment using a two-sided Fisher exact test. |
| S4, S5 | Bladder concentration (nM) and 24-hour urinary excretion (nmol) | Each comparison uses animals within one of five collection-week and batch strata. |
| S15 | Germ-free groups, source concentrations (µM) | Plasma, bladder, liver and kidney have six IDs per arm. Each of 12 animals has two conflicting, unlabelled caecal readings, which precludes a caecal contrast. |
| S18 | Acute BCPN concentrations (µM), a glucuronide block and published contrasts | Descriptive only. The A-prefix treatment assignment reproduces 11 of 12 labelled contrasts within 1 nM. Some cells have six IDs although the Methods specify five. For the one-hour caecal contrast, the article gives P = 0.085 and S18 gives adjusted P = 0.0108913. Sampling begins at one hour, after product is present in both compartments. |
| S23 | Culture BCPN at 24 hours (nM) | 113 technical readings yield 30 donor-by-oxygen means for ten paired donors. A sensitivity analysis uses the nine donors with complete technical replicates. |
| S32, S37 | Strain LC–MS signal (arbitrary units) and tissue signals | S32 measures technical replicates. S37 lacks absolute units, mixes scales for SI1 and has two unresolved normalization records. Both tables are descriptive. |
| S39 | BCPN before deglucuronidation (`uM_raw`) | Nine within-compartment comparisons use eight animal IDs per arm. Sixteen duplicate plasma rows are counted once per animal and measurement. |
| S16, S18 contrasts, S38, S40 | Published statistical results | Retained with their source labels. |

For metabolite contrasts, the analysis reports arithmetic mean differences and within-matrix ratios of means. Pointwise 95% bootstrap intervals resample animals or paired donors. Two-sided permutation or paired sign-flip tests use Holm adjustment within each stated family. Reported zeros are retained. Detection limits and cage-level dependence cannot be estimated from these tables.

## Reuse and source processing

The workbook rows are redistributed under CC BY 4.0 with attribution to Roje et al. Selected ranges were converted to JSON records in SQLite. For S3, the experiment label is carried forward within each pair of treatment rows. Original MetaboLights metadata retain the EMBL-EBI terms. Derived analyses and model results are available under CC BY 4.0 with attribution to the accompanying manuscript and source studies. The software is licensed under MIT. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [MIT license](../LICENSE)

To retrieve acquisition metadata from the EMD2 directory, run `python data/fetch_data.py`. Use `python data/fetch_data.py --verify` to check local files. The supplementary workbook is available through the article. Source URLs, filenames and SHA-256 checksums are in the [source manifest](SOURCES.tsv).

To extract the workbook ranges, run from `src/`. The parser verifies the workbook checksum and writes `derived/published_roje2024.json` under EMD2.

```bash
python -m pip install -r emd2_simulation/requirements-source.txt
python -m emd2_simulation.processing.published --workbook /path/to/41586_2024_7754_MOESM4_ESM.xlsx
```

To rebuild the SQLite database, supply the source archive containing `published_roje2024.json`, `observed_roje2024.json`, `s_MTBLS3581.txt` and `data/SOURCES.tsv`. Run from EMD2.

```bash
python data/build_database.py --source-root /path/to/EMD2-zenodo --output emd2-rebuilt.sqlite
```
