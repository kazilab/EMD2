# EMD2 microbial metabolism of BBN

This notebook reproduces Figures 3, 4 and S5 using published measurements and a compartmental model of N-butyl-N-(4-hydroxybutyl)nitrosamine (BBN). It examines whether urinary concentrations of its metabolite, N-butyl-N-(3-carboxypropyl)nitrosamine (BCPN), distinguish microbial from host metabolism. [Open the notebook](EMD2_simulation.ipynb).

## Run the notebook

From the EMD2 directory, use Python 3.13.9 for the tested environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m jupyter lab EMD2_simulation.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate`. In Jupyter, select the environment’s Python kernel and choose **Run All Cells**. The notebook runs offline after installation. For an unpinned environment, install `requirements.txt` instead.

| Notebook mode | Calculation |
| --- | --- |
| `reference` (default) | Published-data analysis with 20,000 bootstrap resamples, six model arms, 400 parameter draws, 40 host-only sensitivity draws and five synthetic profiles |
| `quick` | Published-data analysis and 40 parameter draws. Omits the host-only sensitivity draws and synthetic profiles. |
| `empirical` | Published-data analysis only |

## Results

The default run writes figures, captions, tables and numerical summaries to `outputs/reproduced/`. The other modes use `outputs/quick/` and `outputs/empirical/`. Choose a new `RUN_NAME` to keep results from different runs.

| Figure | Content | Output filename stem |
| --- | --- | --- |
| 3 | Published BCPN measurements in animals and human donor cultures | `emd2_empirical` |
| 4 | Model comparison of the anatomical source of BCPN | `emd2_mechanistic` |
| S5 | Supporting measurements and controls | `emd2_empirical_controls` |

Bootstrap confidence intervals estimate uncertainty in the experimental contrasts by resampling animals or donors. Model sensitivity ranges describe variation across assigned parameter values. The source studies, units, exclusions and data reuse terms are in the [data notes](data/THIRD_PARTY.md). The `tables/table_catalog.csv` file describes the exported tables.

## Command line and tests

After installing the dependencies, run the following commands from EMD2.

```bash
cd src
python -m emd2_simulation --outdir ../outputs/reproduced
python -m pytest emd2_simulation/tests -q
```

Use `--empirical-only` to run the published-data analysis alone or `--help` to list the settings. Software is distributed under the [MIT license](LICENSE).
