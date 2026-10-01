# EMD2: exploratory secondary analysis of published measurements

An exploratory secondary analysis of published measurements of BBN-to-BCPN
microbial transformation and microbial-community intervention, with
*N*-butyl-*N*-(4-hydroxybutyl)nitrosamine (BBN) in mouse as the exemplar exposure
([Roje et al., *Nature* 632:1137–1144, 2024](https://doi.org/10.1038/s41586-024-07754-w)).
It addresses the proposed typed links with home KCC1 and conditional home KCC6,
and downstream KCC2, KCC7, KCC8 and KCC10. A separate uncalibrated 37-state
compartmental model examines source ambiguity under assigned kinetics, with a
400-draw sensitivity ensemble and conditional synthetic profiles.

**This deposit contains two different kinds of object, and they carry different
weight.** The measured-data analysis estimates experimental effects from
published observations with animal- and donor-level uncertainty; it is a new
statistical analysis of existing measurements, not an independent validation
experiment. The compartmental model is supplementary and **remains uncalibrated**:
no physiological rate is fitted anywhere, and its parameters are assigned rather
than estimated.

Internal checks and regression tests establish that the code computes what it
claims, not that the model is biologically correct. They were written alongside
the analysis and revised during development; they are neither preregistered nor
independent experimental validation. Model output is not an independent
key-characteristic or EMD positive, and the typed EMD2–KCC links remain
conditional on the functional evidence cited in the manuscript.

## Reproduce offline

Use Python 3.11 or later. From this archive's root:

```bash
python -m pip install -r code/emd2_simulation/requirements.txt
cd code
python -m emd2_simulation.run --empirical-only --outdir ../reproduced
python -m pytest emd2_simulation/tests -q -p no:cacheprovider
```

The first command generates `emd2_empirical_summary.json`, the main measured-data
figure `figure2d_emd2_simulation.*` and supplementary measured controls
`figureS_emd2_empirical_controls.*`. It takes seconds, uses the bundled row
extract and needs no downloads. The historical main-figure filename is retained
for compatibility; its panels now contain measured data. `--no-figure` generates
JSON only.

The deposited run passed **13/13** internal checks, with 400/400 integrations and
40 null-cost audit draws, and the standalone package passed **60** tests. The
empirical JSON was byte-identical across the full CLI, the isolated deposit and
the executed notebook. Byte-identical regeneration demonstrates computational
reproducibility, not biological validity. Reviewed software: Python 3.13.9,
NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.1, pytest 8.4.2.
`requirements-tested.txt` pins those packages; the core requirements specify
minimum versions. Floating-point results may vary slightly across environments.

## Reproduce the supplementary model (optional)

From `code/`:

```bash
python -m emd2_simulation.run --outdir ../reproduced
python -m emd2_simulation.manuscript_tables --outdir ../reproduced
```

The full run includes 400 seeded parameter draws, a 40-draw null-cost audit and
five conditional synthetic profiles. Allow tens of minutes. The model figure is
`figureS_emd2_mechanistic.*`, separate from the measured-data main figure.
Diagnostic flags `--no-identifiability --n-ensemble 8 --n-null-audit 0` require a
different output directory; partial results cannot generate publication model
tables. Compare the three generated JSON files with `outputs/`.

## What the analysis shows

**Microbial transformation is supported by measured products and a
reconstructed-community intervention.** The main caecal community contrast has a
ratio of means of approximately 3.49 (pointwise 95% bootstrap interval
2.37–5.06; eight animals per group; Holm P = 0.0028 across nine compartments).
Function is not inferred from taxonomic composition: no abundance table enters
any estimate reported here. All compartments, culture comparisons, collection
strata and audit decisions are reported, including non-significant and opposing
results. The analysis uses 20,000 percentile bootstrap draws and exact or 50,000
Monte Carlo permutations; a bladder result close to the adjusted 0.05 threshold
is explicitly treated as Monte Carlo-sensitive.

**Record counts are acquisitions, not animals.** Deposition rows are LC-MS
acquisitions: each aliquot appears once per ionisation mode and once before and
once after deglucuronidation, so record counts overstate biological replication.
Compartment coverage is reported with `biological_n_verified: false`; the
animal-level counts stated for the community contrast above are the verified
ones.

**No lumen/plasma cutoff is established, for a data reason and a model reason.**
Pairing by animal cancels a per-animal scale factor but not matrix-specific
extraction, recovery or internal-standard normalisation, so cross-matrix ratios
remain provisional. Independently, two model counterexamples raise a host-only
ratio above one — increased exchange, and biliary delivery at a few per cent of
plasma clearance — because influx and efflux coefficients vary independently and
no reciprocity is enforced. Within-matrix treatment contrasts are a different
estimand and do not bear on a cutoff either way.

**The model supplies no independent information about the host-response links.**
The exposure index is exactly proportional to cumulative urinary excretion, so it
carries no independent tissue or host-response information. The summary marks the
host-response, genotoxicity and KCC10-response quantities as unidentifiable in
this build, leaving KCC6, KCC7, KCC8, KCC2 and KCC10 unconstrained by it.

## Contents and provenance

| Path | Purpose |
|---|---|
| `code/emd2_simulation/empirical.py` | Unit-level secondary analysis and source eligibility audit |
| `code/emd2_simulation/empirical_figure.py` | Main measured figure and supplementary measured controls |
| `code/emd2_simulation/conditions.py` | The 13 checks V1–V12 and V7b, each with its evidentiary basis |
| `code/emd2_simulation/data/published_roje2024.json` | Selected original source rows with workbook checksum and row provenance |
| `code/emd2_simulation/fit/published.py` | Optional checksum-guarded workbook importer; requires openpyxl only for rebuilding |
| Remaining model modules | Assigned-kinetics model, null comparisons, synthetic profiles and sensitivity analyses |
| `code/emd2_simulation/fit/PROVENANCE.md` | Units, duplicate handling, eligibility, discrepancy audit and interpretation limits |
| `code/emd2_simulation/fit/EVIDENCE_MAP.md` | Why the EMD2 domain is assumed, and which published exemplar carries each typed EMD2–KCC edge |
| `outputs/` | Empirical summary and two complete supplementary-model reference JSONs |
| `data/` | Source manifest, reuse attribution and optional metadata downloader |
| `MANIFEST.sha256` | Checksums for every deposit file except the manifest itself |

To rebuild the row extract, retrieve the Supplementary Tables workbook from the
article, place it at
`code/emd2_simulation/data/41586_2024_7754_MOESM4_ESM.xlsx`, install openpyxl,
and run `python -m emd2_simulation.fit.published` from `code/`. A changed
workbook checksum causes an error rather than silently accepting a changed
layout. The approximately 20 MB workbook is omitted to keep the deposit small.

The deposit excludes publication images, notebook outputs, Word files and their
generators, raw instrument files, historical reviews and caches. Figures
regenerate from code. The notebook and Word toolchain remain in the working
manuscript project and are unnecessary to reproduce the numerical analysis.

`EVIDENCE_MAP.md` tags identifiers it completed rather than inherited; verify
those before reuse.

## Evidentiary limits

- **Nothing in the compartmental model is fitted.** Kinetics are assigned, not
  estimated, and no physiological rate or kinetic calibration is claimed
  anywhere in the deposit.
- **The measured-data analysis is secondary.** It estimates effects from
  published observations and is not an independent validation experiment.
- **Data handling is exclusionary where sources conflict.** Exact S39 plasma
  duplicates collapse; conflicting S15 caecal readings are excluded, which
  removes every germ-free and GFA caecal value, so no germ-free caecum/plasma
  ratio is computable and earlier summaries built on one are withdrawn. Culture
  wells are averaged within donor before paired donor-level inference. Urine and
  bladder collection strata are analysed separately and never pooled.
- **Acute data remain descriptive.** The earliest samples are at 1 h with BCPN
  already present in both compartments, so they cannot support source
  attribution; treatment labels are inferred, per-cell counts disagree with the
  stated protocol, and published statistics disagree with the table. S37 mixes
  normalisation scales and is excluded from new quantitative inference. No
  temporal source attribution is claimed.
- **KCC1-related metabolism does not establish the other typed links.** The
  conditional KCC6 home and the KCC7, KCC8 and KCC10 links each require their own
  functional host-response endpoint, and KCC2 requires genotoxicity evidence.
  None is imported here, and the exposure index supplies none of it.

## Archive identity and citation

This directory is the complete EMD2 reproducibility archive. For peer review,
provide the whole directory as the submission's code/data archive or through a
reviewer-accessible repository; the general hKCC repository link in
`CITATION.cff` does not by itself identify this exact snapshot. When a
DOI-minting repository record is created, add that DOI to `CITATION.cff` and
cite it in the manuscript without changing the archived version.

Software: MIT. New derived results: CC BY 4.0. The article and selected source
rows are attributed under the original CC BY 4.0 terms; MetaboLights metadata
retain their own terms. Third-party source rows retain attribution and original
terms. See `LICENSE`, `LICENSE-DATA` and `data/THIRD_PARTY.md`. Cite using
`CITATION.cff`.
