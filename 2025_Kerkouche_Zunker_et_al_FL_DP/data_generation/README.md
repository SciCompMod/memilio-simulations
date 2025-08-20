# Data generation

This directory contains scripts to prepare, extend, and smooth COVID‑19 case data at county and community level. Typical pipeline:

1. Generate synthetic fine‑grained (county/community + single age year) data from RKI style aggregated inputs (`transform_data_resolved.py`).
2. Fill explicit zero rows for (date, county, community) combinations without reported cases (`extend_case_data.py`).
3. Smooth time series with a centered moving average (`smooth_cases.py`).

---
## Script Overview

### 1. `transform_data_resolved.py`
Purpose: Transform RKI case records (by age group & county) into (a) an aggregated community level dataset and (b) a simulated individual‑level dataset with exact age years and inferred community assignment. Produces:
- `cases_agg_2022.csv`  (Date, ID_County, ID_Community, AgeGroup, Count)
- `cases_individual_2022.csv` (Date, ID_County, ID_Community, AgeYear, Gender)

Data needed:
- Date range: `2022-03-01` to `2022-04-01`.
- Input files (expected in this directory):
  - `12411-02-03-5.xlsx` (population by gender & fine age intervals; from: https://www.regionalstatistik.de/genesis//online?operation=table&code=12411-02-03-5).
  - `CaseDataFull.json` (case data with fields: `Refdatum`, `Altersgruppe`, `Geschlecht`, `AnzahlFall`, `IdLandkreis`). Obtained from the MEmilio Epidata package (see https://memilio.readthedocs.io/en/latest/python/m-epidata.html)

Outputs:
- `cases_agg_2022.csv`: columns `Date,ID_County,ID_Community,Age,Count` (Age = original group like A35-A59).
- `cases_individual_2022.csv`: columns `Date,ID_County,ID_Community,Age,Gender` (Age = exact year). Row count equals total cases in range.

### 2. `extend_case_data.py`
Ensure a complete daily panel for every (County, Community) by inserting explicit zero counts where no cases were reported.

### 3. `smooth_cases.py`
Apply centered moving average smoothing to counts per (County, Community).
Output: `cases_agg_<year>[_extended]_ma<window>.csv` with smoothed `Count`.

---
## Recommended Workflow (example for 2022)
1. Download/Create `12411-02-03-5.xlsx` and `CaseDataFull.json`
2. Edit date range & year in `transform_data_resolved.py`; run it.
3. Move or write `cases_agg_2022.csv` into `casedata/` (or adjust script paths).
4. Run `extend_case_data.py` → `cases_agg_2022_extended.csv`.
5. Set `year = 2022` in `smooth_cases.py`; run → `cases_agg_2022_extended_ma7.csv`.
6. Use smoothed panel for downstream modeling/analysis.
