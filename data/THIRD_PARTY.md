# Third-party data attribution

Source study: Roje B et al., **Gut microbiota carcinogen metabolism causes distal tissue tumours**, Nature 632:1137–1144 (2024), https://doi.org/10.1038/s41586-024-07754-w. The article is open access under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); source material retains its authors' attribution. The EMD2 software authors are not the authors of those experiments.

`code/emd2_simulation/data/published_roje2024.json` reproduces selected source-workbook rows from S3, S4, S5, S15, S16, S18, S23, S32, S37–S40 and S42, converted to JSON with explicit row provenance. Changes are table/range selection and JSON conversion; original values and labels are preserved. Analyses, duplicate decisions, exclusions and new estimates are separate in `empirical.py` and `outputs/emd2_empirical_summary.json`. These are adaptations and secondary analyses, not additional experiments. The approximately 20 MB workbook is not included; retrieve it through the article's Supplementary Tables link. Its SHA-256 is recorded in `SOURCES.tsv` and the importer.

`observed_roje2024.json` is an older descriptive summary from S5, S6, S7, S13, S15 and S39. It is retained for traceability and legacy code tests, not used in the current main analysis or default figure. Conflicting GF caecal readings no longer produce a paired ratio. Its cross-matrix summaries remain provisional; pairing does not cancel matrix-specific assay factors.

`s_MTBLS3581.txt` is original study acquisition metadata from [MetaboLights MTBLS3581](https://www.ebi.ac.uk/metabolights/MTBLS3581), redistributed unchanged under the study's stated EMBL-EBI Terms of Use. It describes 2,957 LC-MS acquisitions, not 2,957 independent animals. It is used only for supplementary model coverage. Known liver-filename/colon-label conflicts are flagged, with original metadata retained. The metabolite-assignment file has no quantitative per-sample abundances and is not an input to the empirical analysis.

`SOURCES.tsv` records original-source URLs, checksums and optional-download status. `fetch_data.py` downloads only entries marked automatic=1 and verifies their checksums. Publisher workbook retrieval is manual; the main analysis requires no downloads. To verify optional downloaded source files:

```bash
python data/fetch_data.py --verify
```

See `code/emd2_simulation/fit/PROVENANCE.md` for experiment-specific units, resampling units, source discrepancies and eligibility decisions.
