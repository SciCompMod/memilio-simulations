# Federated Learning with Differential Privacy for Infectious Disease Dynamics

This directory contains the code used for the simulations and figures in the paper:

Raouf Kerkouche, Henrik Zunker, Mario Fritz, and Martin J. Kühn.
**Differentially private federated learning for localized control of infectious disease dynamics.**
arXiv:2509.14024 [cs.LG], 2026.
https://arxiv.org/abs/2509.14024




The main focus is county-level COVID-19 forecasting in Germany with and without client-level differential privacy. The repository also contains the synthetic community-level pipeline used for the exploratory fine-resolution analysis.

## What This README Covers

This README is intended to make reproduction straightforward. It explains:

1. which files are required,
2. where they must be placed,
3. which scripts produce the county-level paper results,
4. how to generate the main plots, and
5. which parts of the code require editing hard-coded parameters.

## Important Conventions

Before running anything, keep these three points in mind:

1. Run all commands from the repository root.

2. Most scripts do not expose command-line arguments. They are configured by editing constants near the top of the file.

3. Most scripts resolve input paths relative to the current working directory and therefore expect a local directory named `casedata/` inside the repository root.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `county_dp.py` | Main county-level DP-FL experiments used for the paper |
| `community_dp.py` | Community-level DP-FL experiments on synthetic fine-resolution data |
| `RDP_moment_accountant.py` | Privacy accounting utilities |
| `plots/` | Figure-generation scripts |
| `data_generation/` | Scripts for building the synthetic community-level dataset |

## Software Requirements

The code was written for Python and uses PyTorch, pandas, NumPy, scikit-learn, matplotlib, seaborn, and tqdm. Reading the community population table also requires Excel support via `openpyxl`.

A minimal setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn tqdm torch openpyxl jupyter
```

If you want to download the public case data automatically rather than preparing the files manually, you will additionally need the MEmilio-Epidata tool for RKI data access and preprocessing. For more information about that tool and how to call the `get_case_data.py` file, see:
https://memilio.readthedocs.io/en/latest/python/m-epidata.html

The same holds for the county-level population file. If you want to download it directly rather than preparing the file manually, you can use the `get_population_data.py` script from the MEmilio-Epidata package.

## Required Data Files

Create the directory `casedata/` and place the following files there.

### Minimum files for the revised county-level manuscript results

| File | Used by | Notes |
| --- | --- | --- |
| `cases_all_county_ma7_trailing.json` | `county_dp.py` | Trailing 7-day moving average county time series. This is the key input for the revised manuscript. |
| `county_population.json` | `county_dp.py` | County population file used for unscaling and population-stratified analysis. |

### Additional useful county-level files

| File | Used by | Notes |
| --- | --- | --- |
| `cases_all_county.json` | plotting or preprocessing | Raw county-level time series without moving average |
| `cases_all_county_ma7.json` | comparisons | Centered moving average version from earlier experiments |
| `county_current_population.json` | some plotting scripts | Current county population file used by certain descriptive plots |

### Additional files for the synthetic community-level pipeline

| File | Used by | Source |
| --- | --- | --- |
| `12411-02-03-5.xlsx` | `data_generation/transform_data_resolved.py`, `community_dp.py` | Community-level population table from German regional statistics |
| `CaseDataFull.json` | `data_generation/transform_data_resolved.py` | Detailed case data exported from the public RKI data via MEmilio Epidata |

## Recommended Directory Layout

The cleanest setup is:

```text
root/
├── casedata/
│   ├── cases_all_county_ma7_trailing.json
│   ├── county_population.json
│   ├── county_current_population.json
│   ├── cases_all_county.json
│   ├── cases_all_county_ma7.json
│   ├── 12411-02-03-5.xlsx
│   └── CaseDataFull.json
├── county_dp.py
├── community_dp.py
├── data_generation/
└── plots/
```

## Step 1: Prepare the County-Level Inputs

The county-level paper results are based on public German county case data and county population metadata. The revised manuscript uses the trailing-smoothed county file:

```text
casedata/cases_all_county_ma7_trailing.json
```

At minimum, ensure that the following two files exist before running `county_dp.py`:

```text
casedata/cases_all_county_ma7_trailing.json
casedata/county_population.json
```

If you are maintaining both old and revised runs, keep the centered and raw variants as well:

```text
casedata/cases_all_county.json
casedata/cases_all_county_ma7.json
```

## Step 2: Prepare the Synthetic Community-Level Inputs

The community-level analysis is not based on directly available public community case time series. Instead, the repository builds a synthetic fine-resolution dataset from:

1. a population table by county/community/age/gender, and
2. detailed public case records.

Place the raw input files here:

```text
casedata/12411-02-03-5.xlsx
casedata/CaseDataFull.json
```

The script `data_generation/transform_data_resolved.py` reads its inputs from `data_generation/`, not from `casedata/`. The simplest workflow is therefore:

```bash
cp casedata/12411-02-03-5.xlsx data_generation/
cp casedata/CaseDataFull.json data_generation/
python data_generation/transform_data_resolved.py
cp data_generation/cases_agg_2022.csv casedata/
cp data_generation/cases_individual_2022.csv casedata/
```

After that, create the full daily panel with explicit zero rows:

```bash
python data_generation/extend_case_data.py
```

Then smooth the community series:

```bash
python data_generation/smooth_cases.py
```

Important:

1. `transform_data_resolved.py` is currently hard-coded for a specific date range and writes `cases_agg_2022.csv` and `cases_individual_2022.csv`.
2. `smooth_cases.py` is controlled by the constants `year`, `extended`, and `center` at the top of the file.
3. For the revised manuscript logic, use `center = False` in `smooth_cases.py` so that the script produces a trailing moving average file.

## Step 3: Reproduce the Main County-Level DP-FL Results

The main experimental script is `county_dp.py`.

Before each run, check the following settings near the top of the file:

| Variable | Meaning | Recommended value for revised manuscript |
| --- | --- | --- |
| `year` | Epidemic phase to run | `2020` or `2022` |
| `trailing` | Choose trailing vs. centered smoothing | `True` |
| `scale_to_relative` | Population scaling | `False` |
| `EPOCHS` | Local epochs per round | `30` |
| `FED_rounds` | Number of federated rounds | `75` |
| `Nbr_selected_Counties` | Expected clients per round | `100` |
| `DELTA` | DP failure probability | `1e-5` |

At the bottom of the script, also check:

| Variable | Meaning | Value |
| --- | --- | --- |
| `num_runs` | Number of repeated runs | `15` |
| `fine_grid` | Whether to run the extended epsilon grid | `False` for the main 5-epsilon results, `True` for the privacy-utility sweep |

### Main 5-epsilon results

To reproduce the standard county-level results used for scatter plots and the main result tables:

1. Edit `county_dp.py` and set:
   - `year = 2020`
   - `trailing = True`
   - `fine_grid = False`
2. Run:

   ```bash
   python county_dp.py
   ```

3. Repeat the same process with:
   - `year = 2022`
   - `trailing = True`
   - `fine_grid = False`

This produces files of the form:

```text
year-2020_county_predictions_scaled-False_runs-15_rounds-75_ma-trailing.csv
year-2020_loss_curves_scaled-False_runs-15_rounds-75_ma-trailing.csv
year-2020_county_predictions_by_population_scaled-False_runs-15_rounds-75_ma-trailing.csv

year-2022_county_predictions_scaled-False_runs-15_rounds-75_ma-trailing.csv
year-2022_loss_curves_scaled-False_runs-15_rounds-75_ma-trailing.csv
year-2022_county_predictions_by_population_scaled-False_runs-15_rounds-75_ma-trailing.csv
```

### Extended epsilon grid for the privacy-utility curves

To reproduce the finer epsilon sweep used for the privacy-utility analysis:

1. Edit `county_dp.py` and set:
   - `fine_grid = True`
   - `trailing = True`
2. Run once for `year = 2020`.
3. Run once for `year = 2022`.

This produces files with the suffix `_fine-eps`, for example:

```text
year-2020_county_predictions_scaled-False_runs-15_rounds-75_ma-trailing_fine-eps.csv
year-2020_loss_curves_scaled-False_runs-15_rounds-75_ma-trailing_fine-eps.csv
```

## Step 4: Generate the Main Plots

### Prediction plots

To generate the scatter plots and MAPE distributions from the county-level prediction CSV:

1. Open `plots/plot_prediction_results.py`.
2. Set `year = 2020` or `year = 2022`.
3. Run:

   ```bash
   python plots/plot_prediction_results.py
   ```

The script reads the trailing-moving-average output file:

```text
year-{year}_county_predictions_scaled-False_runs-15_rounds-75_ma-trailing.csv
```

and writes plots to:

```text
plots/plots_{year}/
```

### Loss-curve plots

To generate the training-vs-test loss figures:

```bash
python plots/plot_loss_curves.py
```

This script expects:

```text
year-2020_loss_curves_scaled-False_runs-15_rounds-75_ma-trailing.csv
year-2022_loss_curves_scaled-False_runs-15_rounds-75_ma-trailing.csv
```

and writes figures under:

```text
plots/loss_curves/
```

### Community sparsity plots

To visualize how sparse the community-level data is:

```bash
python plots/plot_community_zero_entries.py
```

Before running it, check the constants at the top of the script, especially:

| Variable | Meaning |
| --- | --- |
| `data_year` | Which year to analyze |
| `ma` | Whether to use the smoothed community file |
| `extended` | Whether to use the zero-filled full panel |

## Step 5: Run the Community-Level Experiments

The exploratory community-level experiments are executed with:

```bash
python community_dp.py
```

Before running, inspect the constants at the top of the file:

| Variable | Meaning |
| --- | --- |
| `data_year` | Year of the community dataset |
| `fill_gaps` | Whether to use the zero-filled version |
| `mavg` | Moving-average window |
| `Nbr_selected_LHA` | Expected clients per round |
| `FED_rounds` | Number of federated rounds |
| `EPOCHS` | Local epochs |

The script expects the community input CSV to already exist in `casedata/`, for example:

```text
casedata/cases_agg_2022_extended_ma7_trailing.csv
```

The exact filename used by `community_dp.py` depends on the hard-coded values of `data_year`, `fill_gaps`, and `mavg`.

## Typical Reproduction Order for the Paper

If your goal is to reproduce the revised county-level manuscript results as directly as possible, use this order:

1. Prepare `casedata/cases_all_county_ma7_trailing.json` and `casedata/county_population.json`.
2. Run `county_dp.py` for `year = 2020`, `trailing = True`, `fine_grid = False`.
3. Run `county_dp.py` for `year = 2022`, `trailing = True`, `fine_grid = False`.
4. Run `plots/plot_prediction_results.py` for 2020 and 2022.
5. Run `plots/plot_loss_curves.py`.
6. If needed, rerun `county_dp.py` for both years with `fine_grid = True` to reproduce the extended privacy-utility curves.

## Citation

If you use this code or reproduce the experiments, please cite the corresponding paper and, where appropriate, the MEmilio project for data access and preprocessing infrastructure.

[1] Raouf Kerkouche, Henrik Zunker, Mario Fritz, and Martin J. Kühn.
**Differentially private federated learning for localized control of infectious disease dynamics.**
arXiv:2509.14024 [cs.LG], 2026.
https://arxiv.org/abs/2509.14024

[2] Julia Bicker, Carlotta Gerstein, David Kerkmann, Sascha Korf, René Schmieding, Anna Wendler, Henrik Zunker, et al.
**MEmilio -- A high performance Modular EpideMIcs simuLatIOn software for multi-scale and comparative simulations of infectious disease dynamics.**
arXiv:2602.11381 [q-bio.PE], 2026.
https://arxiv.org/abs/2602.11381 
