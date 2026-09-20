# EMD2: minimal reproducibility deposit

Exploratory secondary analysis of published BBN-to-BCPN transformation and microbial-intervention experiments from [Roje et al., Nature 632:1137–1144 (2024)](https://doi.org/10.1038/s41586-024-07754-w), with a separate uncalibrated compartmental model.

## Reproduce the measured-data results offline

Use Python 3.11 or later. From this archive's root:

```bash
python -m pip install -r code/emd2_simulation/requirements.txt
cd code
python -m emd2_simulation.run --empirical-only --outdir ../reproduced
python -m pytest emd2_simulation/tests -q -p no:cacheprovider
```

The first command generates `emd2_empirical_summary.json`, the main measured-data figure `figure2d_emd2_simulation.*` and supplementary measured controls `figureS_emd2_empirical_controls.*`. It takes seconds, uses the bundled row extract and needs no downloads. The historical main-figure filename is retained for compatibility; its panels now contain measured data. `--no-figure` generates JSON only.

The main caecal community contrast has a ratio of means of approximately 3.49 (pointwise 95% bootstrap interval 2.37–5.06; eight animals per group; Holm P = 0.0028 across nine compartments). This is a new statistical analysis of published observations, not an independent validation experiment. All compartments, culture comparisons, collection strata and audit decisions are reported, including non-significant and opposing results. No physiological rate is fitted.

## Reproduce the supplementary model

From `code/`:

```bash
python -m emd2_simulation.run --outdir ../reproduced
python -m emd2_simulation.manuscript_tables --outdir ../reproduced
```

The full run includes 400 seeded parameter draws, a 40-draw null-cost audit and five conditional synthetic profiles. Allow tens of minutes. The model figure is `figureS_emd2_mechanistic.*`, separate from the measured-data main figure. Passing internal checks is not experimental validation. Diagnostic flags `--no-identifiability --n-ensemble 8 --n-null-audit 0` require a different output directory; partial results cannot generate publication model tables.

Compare the three generated JSON files with `outputs/`. Floating-point results may vary slightly across environments. Reviewed software: Python 3.13.9, NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.1, pytest 8.4.2. `requirements-tested.txt` pins those packages; the core requirements specify minimum versions. The analysis uses 20,000 percentile bootstrap draws and exact or 50,000 Monte Carlo permutations. A bladder result close to the adjusted 0.05 threshold is explicitly treated as Monte Carlo-sensitive.

The reviewed standalone package passed 54 tests. The complete supplementary run passed 13/13 internal checks, with 400/400 integrations and 40 null-cost audit draws. The empirical JSON was byte-identical across the full CLI, isolated deposit and executed notebook.

## Contents and provenance

| Path | Purpose |
|---|---|
| `code/emd2_simulation/empirical.py` | Unit-level secondary analysis and source eligibility audit |
| `code/emd2_simulation/empirical_figure.py` | Main measured figure and supplementary measured controls |
| `code/emd2_simulation/data/published_roje2024.json` | Selected original source rows with workbook checksum and row provenance |
| `code/emd2_simulation/fit/published.py` | Optional checksum-guarded workbook importer; requires openpyxl only for rebuilding |
| Remaining model modules | Assigned-kinetics model, null comparisons, synthetic profiles and sensitivity analyses |
| `code/emd2_simulation/fit/PROVENANCE.md` | Units, duplicate handling, eligibility, discrepancy audit and interpretation limits |
| `outputs/` | Empirical summary and two complete supplementary-model reference JSONs |
| `data/` | Source manifest, reuse attribution and optional metadata downloader |
| `MANIFEST.sha256` | Checksums for every deposit file except the manifest itself |

To rebuild the row extract, retrieve the Supplementary Tables workbook from the article, place it at `code/emd2_simulation/data/41586_2024_7754_MOESM4_ESM.xlsx`, install openpyxl, and run `python -m emd2_simulation.fit.published` from `code/`. A changed workbook checksum causes an error rather than silently accepting a changed layout. The approximately 20 MB workbook is omitted to keep the deposit small. The article and selected source rows are attributed under the original CC BY 4.0 terms; MetaboLights metadata retain their own terms.

The deposit excludes publication images, notebook outputs, Word files/generators, raw instrument files, historical reviews and caches. Figures regenerate from code. The notebook and Word toolchain remain in the working manuscript project and are unnecessary to reproduce the numerical analysis.

## Evidentiary limits

- The main analysis supports microbial transformation through measured products and community intervention. It does not infer functionality from taxonomic shifts.
- Exact S39 plasma duplicates collapse; conflicting S15 caecal readings are excluded. Culture wells are averaged before donor-level inference. Urine and bladder collection strata stay separate.
- Acute S18 data remain descriptive because treatment labels are inferred and counts/statistics need reconciliation. S37 normalisation is unsuitable for new quantitative inference. No kinetic calibration or temporal source attribution is claimed.
- Within-matrix group ratios do not validate a physical lumen/plasma cutoff. Model exchange and biliary counterexamples demonstrate the ambiguity of such a cutoff under assigned kinetics.
- KCC1-related metabolism does not establish the conditional KCC6 home or KCC7/8/10 links. These require their corresponding functional host-response evidence; KCC2 requires genotoxicity evidence. The exposure index supplies none.

Software: MIT. New derived results: CC BY 4.0. Third-party source rows retain attribution and original terms. See `LICENSE`, `LICENSE-DATA` and `data/THIRD_PARTY.md`.

`code/emd2_simulation/fit/EVIDENCE_MAP.md` records why the EMD2 domain is assumed and which published exemplar carries each typed EMD2–KCC edge, with full citations. Identifiers it completed rather than inherited are tagged; verify those before reuse.
