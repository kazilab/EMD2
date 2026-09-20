# EMD2 input provenance and analysis decisions

Source: Roje B et al., *Gut microbiota carcinogen metabolism causes distal tissue tumours*, Nature 632:1137–1144 (2024), https://doi.org/10.1038/s41586-024-07754-w. Original acquisition metadata: https://www.ebi.ac.uk/metabolights/MTBLS3581.

## Primary empirical input

`data/published_roje2024.json` is a selected row-level extract of the source Supplementary Tables workbook. `fit/published.py` preserves source table names, original headers, source row numbers and values. It verifies workbook SHA-256 `fe43970068026537c9f95b8132ded29d93397598cc28aab74e381147e341928c` before parsing. The workbook itself is not bundled in the minimal deposit. Optional rebuilding requires openpyxl:

```bash
python -m emd2_simulation.fit.published --workbook emd2_simulation/data/41586_2024_7754_MOESM4_ESM.xlsx
```

Run from the package parent. The ordinary analysis uses the bundled JSON offline and does not require Excel software or openpyxl. Its summary records both the file SHA-256 and a canonical-content hash. Supplied custom data do not inherit the bundled file hash.

| Source | Use and unit | Eligibility decision |
|---|---|---|
| S39 | Nine within-matrix BCPN comparisons; `uM_raw`, before deglucuronidation | Eight animal IDs/arm. Sixteen surplus plasma rows are exact duplicates; collapse, retain all row provenance. |
| S23 | Culture BCPN nM, 24 h | 113 technical readings → 30 donor/oxygen means → ten paired donors. G03 incomplete; separate nine-complete-donor sensitivity. |
| S5 | Urinary `nmole`, 24-h collection | All five collection-week/batch strata; animal IDs within strata; no cross-week pooling. |
| S4 | Bladder `nM`, source label | All five week/batch strata; no pooled longitudinal test. |
| S15 | Germ-free/GF-plus-ABX `uM` | PL, BLD, LVR and KID: six IDs/arm. Twelve caecal keys have conflicting unlabelled readings and are excluded, never silently averaged. |
| S18 | Acute source-table `BCPN_uM` | Descriptive only. A-prefixed IDs inferred to mean ABX/BBN; 11/12 labelled contrasts agree within 1 nM. Several cells have six IDs despite Methods stating five. Article text P = 0.085 vs table adjusted P = 0.0108913 at one-hour caecum. No new significance testing or kinetic fitting. |
| S32 | Type-strain LC-MS `A. U.` | Technical signals only, no biological n. Import A:G only; an unrelated side table also has a Strain column. |
| S37 | Wide monocolonisation tissue signals, no declared absolute unit | Mixed SI1 scales (some one, some raw signals), two missing/non-reconcilable normalisation rows. Retained for audit, excluded from new quantitative inference. |
| S16, S18 contrast block, S38, S40 | Originally reported tests | Preserved as source information; not relabelled as our tests. |

S18 has a spurious million-row worksheet range; its three reviewed blocks lie within the first 135 rows. The bounded import is protected by the workbook checksum.

`empirical.py` defines the analysis: arithmetic mean differences, same-matrix ratios of means, 20,000 unit-level bootstrap draws, paired sign flips or exact/50,000 Monte Carlo independent-label permutations, and Holm adjustment within explicit families. This is exploratory, not preregistered. Confidence intervals are pointwise. Zero values remain zero; detection limits are not invented. A zero bootstrap denominator makes a ratio interval unavailable. Resampling assumes recorded animal units are independent; cage dependence cannot be modelled from these tables. Independent donors, not technical wells, define human culture n.

## Supplementary model inputs

`data/s_MTBLS3581.txt` is original acquisition metadata, used only for model measurement-set coverage. Its rows are LC-MS acquisitions, not independent animals. The metabolite-assignment file contains no quantitative per-sample abundances. The published workbook is therefore the empirical source. Model rates and community capacities remain assigned and are not calibrated to the extracted data.

Sixteen of 48 two-member records labelled colon have `_LVR_` filenames. The existing code flags these metadata conflicts without silently changing anatomy. A complete biological specimen manifest cannot be reconstructed from filenames. SI1/SI2/SI3-to-duodenum/jejunum/ileum mapping is a modelling convention; empirical figures preserve SI1/SI2/SI3 labels.

`data/observed_roje2024.json` and `fit/observed.py` retain older exploratory descriptive summaries for traceability. They are not the primary empirical input and are not overlaid on the current main or default model figure. GF/GFA caecum/plasma ratios have been removed because conflicting S15 readings cannot define one animal/one assay. Exact S39 duplicates collapse. The legacy unadjusted consortium cross-matrix ratio test (P about 0.13) addresses a different estimand from the new within-caeca treatment effect; it establishes neither separation nor equivalence.

## Scope and reuse

Source concentration labels do not demonstrate original-compartment normalisation across tissues. Pairing does not cancel matrix-specific recovery, extraction or dilution. No physical lumen/plasma unity cutoff is validated. Acute observations begin at one hour and do not establish which compartment first generated the product. No genotoxicity, inflammatory, immune, receptor, proliferation/cell-death/nutrient-supply or tumour outcome is imported into this analysis; that does not assert these assays are absent from the whole original study.

The Nature article is open access under CC BY 4.0. Selected source rows are redistributed with attribution under the source terms; column selection and JSON conversion are the changes. New analysis outputs are separately identified. Original MetaboLights metadata retain EMBL-EBI terms. See the minimal deposit's `data/THIRD_PARTY.md`, `LICENSE-DATA` and `data/SOURCES.tsv`.
