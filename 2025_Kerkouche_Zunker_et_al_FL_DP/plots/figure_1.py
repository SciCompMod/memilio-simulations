import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogLocator
from plot_utility import map_county_name_to_id, get_old_county_names
import geopandas as gpd

# --- Paths / configuration ---
cwd = os.getcwd()
COUNTY_JSON_PATH = os.path.join(cwd, 'casedata', 'cases_all_county.json')
COMMUNITY_CSV_PATH = os.path.join(cwd, 'casedata', 'cases_agg.csv')
PATH_DATA_MA7 = os.path.join(cwd, 'casedata', 'cases_all_county_ma7.json')
POPULATION_PATH = os.path.join(
    cwd, 'casedata', 'county_current_population.json')
# Get shape file for germany from: https://github.com/isellsoap/deutschlandGeoJSON/
SHAPE_PATH = os.path.join(cwd, 'shape_files_kreise', '2_hoch_updated.json')
OUTPUT_DIR = os.path.join(cwd, "plots")

# --- Load static data ---
shapes = gpd.read_file(SHAPE_PATH)
population_df = pd.read_json(POPULATION_PATH)[['ID_County', 'Population']]

# --- Plot style constants ---
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_county_data(path: str, year: int) -> pd.DataFrame:
    """Load county data and compute daily new cases for the specified year and month range."""
    df = pd.read_json(path)
    df['Date'] = pd.to_datetime(df['Date'])

    if year == 2022:
        # March 2022 (including Feb 28 baseline)
        baseline_date = pd.to_datetime('2022-02-28')
        start_date = pd.to_datetime('2022-03-01')
        end_date = pd.to_datetime('2022-03-31')
    elif year == 2020:
        # November 2020 (including Oct 31 baseline)
        baseline_date = pd.to_datetime('2020-10-31')
        start_date = pd.to_datetime('2020-11-01')
        end_date = pd.to_datetime('2020-11-30')
    else:
        raise ValueError("Year must be either 2020 or 2022")

    temp = df[(df['Date'] == baseline_date) | (
        (df['Date'] >= start_date) & (df['Date'] <= end_date))]
    temp = temp.sort_values(['ID_County', 'Date'])
    temp['Daily_Cases'] = temp.groupby(
        'ID_County')['Confirmed'].diff().fillna(0)
    df_county = temp[temp['Date'] >= start_date]
    return df_county


def load_community_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    df = df.groupby(['Date', 'ID_County', 'ID_Community'],
                    as_index=False)['Count'].sum()
    return df


def plot_county_population_distribution(df_county: pd.DataFrame, legend_size=30, label_size=26) -> None:
    """Visualize the distribution of county population sizes for the given subset (or all if empty)."""
    county_pop_df = population_df.copy()
    if not df_county.empty:
        county_ids = df_county['ID_County'].unique()
        county_pop_df = county_pop_df[county_pop_df['ID_County'].isin(
            county_ids)]

    missing_mask = county_pop_df['Population'].isnull()
    if missing_mask.any():
        missing_ids = county_pop_df[missing_mask]['ID_County'].tolist()
        print(f"Missing population data for county IDs: {missing_ids}")

    numeric_pop_data = county_pop_df[~missing_mask].copy()

    median_pop = numeric_pop_data['Population'].median()
    mean_pop = numeric_pop_data['Population'].mean()

    plt.figure(figsize=(12, 7))
    sns.histplot(data=numeric_pop_data, x='Population',
                 log_scale=True, bins=100, kde=False)

    # Add median and mean as subtle vertical lines
    plt.axvline(median_pop, color='darkred', linestyle='-', linewidth=3, alpha=0.7,
                label=f'Median: {median_pop:,.0f}')
    plt.axvline(mean_pop, color='darkorange', linestyle='-', linewidth=3, alpha=0.7,
                label=f'Mean: {mean_pop:,.0f}')
    plt.legend(fontsize=legend_size, loc='upper right',
               frameon=True, fancybox=True, shadow=True)

    plt.title('Distribution of county population sizes',
              fontsize=legend_size)
    plt.xlabel('Population', fontsize=label_size)
    plt.ylabel('Number of counties', fontsize=label_size)

    # Custom X ticks for readability on log scale
    population_ticks = [30000, 50000, 100000,
                        200000, 500000, 1_000_000, 3_000_000]
    tick_labels = [f'{x/1_000_000:.1f}M' if x >=
                   1_000_000 else f'{int(x/1000)}k' for x in population_ticks]
    plt.xticks(population_ticks, tick_labels, fontsize=label_size)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(LogLocator(
        base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    plt.yticks(fontsize=label_size)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'county_population_distribution.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"County population distribution plot saved to: {plot_path}")


def plot_community_population_distribution(df_comm, legend_size=30, label_size=26, only_communities=False):
    path_pop_data = "/localdata1/code_fl/casedata/12411-02-03-5.xlsx"
    pop_data = pd.read_excel(
        path_pop_data, sheet_name='12411-02-03-5', header=5)
    pop_data = pop_data.iloc[3:13922, [1, 20]]
    pop_data.columns = ['ID', 'Population']

    if only_communities:
        # Filter for valid community IDs (length >= 6)
        pop_data = pop_data[pop_data['ID'] >= 100000].reset_index(drop=True)
    else:
        # Filter for IDs which are used in the community data
        ids_avail_df = []
        for row_df in df_comm.itertuples():
            county = str(int(row_df.ID_County))
            comm = str(int(row_df.ID_Community)).zfill(3)
            if comm == '000':
                comm = ""
            # Hamburg
            if county == '2000':
                county = '2'
            # Berlin
            elif county == '11000':
                county = '11'
            # create the key for the population dictionary
            key = f"{county}{comm}"
            ids_avail_df.append(key)
        # delete all doubled entries
        ids_avail_df = list(set(ids_avail_df))
        # Filter the population data to only include IDs that are in the community data
        pop_data = pop_data[pop_data['ID'].astype(
            str).isin(ids_avail_df)].reset_index(drop=True)

        if len(pop_data) != len(ids_avail_df):
            print(
                f"Warning: Population data contains {len(pop_data)} entries, but community data has {len(ids_avail_df)} unique IDs.")

    total_entries = len(pop_data)

    missing_pop_mask = pop_data['Population'] == '-'
    num_missing = missing_pop_mask.sum()

    # print which IDs are missing
    if num_missing > 0:
        missing_ids = pop_data[missing_pop_mask]['ID'].tolist()
        print(f"Missing population data for IDs: {missing_ids}")

    numeric_pop_data = pop_data[~missing_pop_mask].copy()

    # Convert the 'Population' column in the new DataFrame to a numeric type
    # remove entries with value '.' or '-'
    numeric_pop_data = numeric_pop_data[numeric_pop_data['Population'] != '.']
    numeric_pop_data = numeric_pop_data[numeric_pop_data['Population']
                                        != '-'].reset_index(drop=True)
    numeric_pop_data['Population'] = pd.to_numeric(
        numeric_pop_data['Population'])

    median_pop = numeric_pop_data['Population'].median()
    mean_pop = numeric_pop_data['Population'].mean()

    plt.figure(figsize=(12, 7))

    sns.histplot(data=numeric_pop_data, x='Population',
                 log_scale=True, bins=100, kde=False)

    # Add median and mean as subtle vertical lines
    plt.axvline(median_pop, color='darkred', linestyle='-', linewidth=3, alpha=0.7,
                label=f'Median: {median_pop:,.0f}')
    plt.axvline(mean_pop, color='darkorange', linestyle='-', linewidth=3, alpha=0.7,
                label=f'Mean: {mean_pop:,.0f}')
    plt.legend(fontsize=legend_size, loc='upper right',
               frameon=True, fancybox=True, shadow=True)

    plt.title('Distribution of community population sizes',
              fontsize=legend_size)
    plt.xlabel('Population', fontsize=label_size)
    plt.ylabel('Number of communities', fontsize=label_size)

    # Custom X ticks for readability on log scale (adapted for community scale)
    population_ticks = [10, 100, 1000, 10000, 100000, 1_000_000]
    tick_labels = []
    for x in population_ticks:
        if x >= 1_000_000:
            tick_labels.append(f'{x/1_000_000:.1f}M')
        elif x >= 1000:
            tick_labels.append(f'{int(x/1000)}k')
        else:
            tick_labels.append(str(x))
    plt.xticks(population_ticks, tick_labels, fontsize=label_size)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(LogLocator(
        base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    plt.yticks(fontsize=label_size)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(
        OUTPUT_DIR, 'community_population_distribution.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot of population distribution saved to: {plot_path}")


if __name__ == "__main__":
    # county
    df_county = load_county_data(COUNTY_JSON_PATH, 2022)
    plot_county_population_distribution(df_county)

    # community
    df_comm = load_community_data(COMMUNITY_CSV_PATH)
    plot_community_population_distribution(df_comm)
