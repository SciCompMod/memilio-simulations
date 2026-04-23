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


def plot_county_daily(path_county, path_ma7, year, legend_size=22, label_size=22):
    # Load data for the entire dataset (including 2021 for baseline)
    df = pd.read_json(path_county)
    df_ma7 = pd.read_json(path_ma7)

    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'])
    df_ma7['Date'] = pd.to_datetime(df_ma7['Date'])

    # Calculate daily cases FIRST for ALL available data
    # This ensures we have baseline data (e.g., Dec 31, 2021) for proper diff() calculation
    groups = []
    for county, grp in df.groupby('ID_County'):
        grp = grp.copy().sort_values('Date')
        grp['Daily_Cases'] = grp['Confirmed'].diff().fillna(0)
        groups.append(grp)
    df_all = pd.concat(groups)

    groups_ma7 = []
    for county, grp in df_ma7.groupby('ID_County'):
        grp = grp.copy().sort_values('Date')
        grp['Daily_Cases'] = grp['Confirmed'].diff().fillna(0)
        groups_ma7.append(grp)
    df_ma7_all = pd.concat(groups_ma7)

    # NOW filter to year data after calculating daily cases
    df_county_year = df_all[df_all['Date'].dt.year == year]
    df_ma7_year = df_ma7_all[df_ma7_all['Date'].dt.year == year]

    if year == 2022:
        # filter to Jan - April 2022
        df_county_year = df_county_year[(df_county_year['Date'] >=
                                        pd.to_datetime('2022-01-01')) & (df_county_year['Date'] <= pd.to_datetime('2022-04-30'))].reset_index(drop=True)
        df_ma7_year = df_ma7_year[(df_ma7_year['Date'] >=
                                   pd.to_datetime('2022-01-01')) & (df_ma7_year['Date'] <= pd.to_datetime('2022-04-30'))].reset_index(drop=True)
    elif year == 2020:
        # filter to Sep - Dec 2020
        df_county_year = df_county_year[(df_county_year['Date'] >=
                                        pd.to_datetime('2020-09-01')) & (df_county_year['Date'] <= pd.to_datetime('2020-12-31'))].reset_index(drop=True)
        df_ma7_year = df_ma7_year[(df_ma7_year['Date'] >=
                                   pd.to_datetime('2020-09-01')) & (df_ma7_year['Date'] <= pd.to_datetime('2020-12-31'))].reset_index(drop=True)
    else:
        raise ValueError("Year must be either 2020 or 2022")

    # Aggregate for the plot
    agg = df_county_year.groupby('Date')['Daily_Cases'].sum().reset_index()
    agg_ma7 = df_ma7_year.groupby('Date')['Daily_Cases'].sum().reset_index()

    # Create the plot
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=agg, x='Date', y='Daily_Cases',
                 label='Daily cases', alpha=0.6, linewidth=2.0)
    sns.lineplot(data=agg_ma7, x='Date', y='Daily_Cases',
                 label='7-day moving average', linewidth=3)

    # plt.title('Daily new cases - 2022', fontsize=LEGEND_SIZE)
    plt.xlabel('Date', fontsize=label_size)
    plt.ylabel('Cases', fontsize=label_size)

    # Set x-ticks to show monthly markers
    month_ticks = pd.date_range(
        start=agg['Date'].min(), end=agg['Date'].max(), freq='MS')
    plt.xticks(month_ticks, [d.strftime('%b %Y')
               for d in month_ticks], fontsize=label_size, style='italic')

    plt.yticks(fontsize=label_size)
    plt.legend(fontsize=legend_size, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'county_daily_cases_{year}.png'))
    plt.close()


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

    info_text = (
        f"Median population: {numeric_pop_data['Population'].median():,.0f}\n"
        f"Mean population: {numeric_pop_data['Population'].mean():,.0f}\n"
        f"Smallest county: {numeric_pop_data['Population'].min():,.0f}\n"
        f"Largest county: {numeric_pop_data['Population'].max():,.0f}\n"
        f"Total population: {numeric_pop_data['Population'].sum():,.0f}"
    )

    plt.figure(figsize=(12, 7))
    sns.histplot(data=numeric_pop_data, x='Population',
                 log_scale=True, bins=100, kde=False)
    plt.title('Distribution of county population sizes',
              fontsize=legend_size + 6)
    plt.xlabel('Population', fontsize=label_size + 6)
    plt.ylabel('Number of counties', fontsize=label_size + 6)

    # Custom X ticks for readability on log scale
    population_ticks = [30000, 50000, 100000,
                        200000, 500000, 1_000_000, 3_000_000]
    tick_labels = [f'{x/1_000_000:.1f}M' if x >=
                   1_000_000 else f'{int(x/1000)}k' for x in population_ticks]
    plt.xticks(population_ticks, tick_labels, fontsize=label_size + 4)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(LogLocator(
        base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    plt.yticks(fontsize=label_size + 4)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    props = dict(boxstyle='round,pad=0.9', facecolor='white',
                 edgecolor='black', linewidth=2, alpha=1.0)
    plt.gca().text(0.97, 0.935, info_text, transform=plt.gca().transAxes,
                   fontsize=19, fontweight='bold', va='top', ha='right', bbox=props)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'county_population_distribution.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"County population distribution plot saved to: {plot_path}")


def plot_county_map(df_county: pd.DataFrame, shapes_gdf: gpd.GeoDataFrame, year: int, legend_size=30, label_size=26) -> None:
    """Plot normalized daily new cases per 100k for four representative dates."""
    df = df_county.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Define dates based on year
    if year == 2022:
        # Four weekly snapshots in March 2022
        selected_dates = [pd.Timestamp('2022-03-01'), pd.Timestamp('2022-03-08'),
                          pd.Timestamp('2022-03-15'), pd.Timestamp('2022-03-22')]
    elif year == 2020:
        # Four weekly snapshots in November 2020
        selected_dates = [pd.Timestamp('2020-11-01'), pd.Timestamp('2020-11-08'),
                          pd.Timestamp('2020-11-15'), pd.Timestamp('2020-11-22')]
    else:
        raise ValueError("Year must be either 2020 or 2022")

    df_selected = df[df['Date'].isin(selected_dates)]

    # Normalize per 100k population
    df_selected = df_selected.merge(population_df, on='ID_County', how='left')
    if df_selected['Population'].isnull().any():
        missing = df_selected[df_selected['Population'].isnull()
                              ]['ID_County'].unique()
        print(f"Warning: Missing population for counties: {missing}")
    df_selected['Normalized_Cases'] = df_selected['Daily_Cases'] / \
        df_selected['Population'] * 100000

    df_selected['ID_County_Str'] = df_selected['ID_County'].astype(
        str).str.zfill(5)
    map_data = shapes_gdf.copy()
    map_data['ARS'] = map_data['ARS'].astype(str)

    # Prepare federal state boundaries
    state_geoms = map_data.copy()
    state_geoms['STATE_CODE'] = state_geoms['ARS']
    states = state_geoms.dissolve(by='STATE_CODE')

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.flatten()

    # Adjust spacing between subplots to bring them closer together
    plt.subplots_adjust(wspace=-0.65, hspace=0)

    # Year-specific logarithmic normalization
    if year == 2022:
        norm = LogNorm(vmin=100, vmax=1000)  # Higher values for Omicron wave
    elif year == 2020:
        # Lower values for autumn 2020 wave
        norm = LogNorm(vmin=1, vmax=100)

    # Custom color map: green -> yellow -> orange -> red -> purple
    custom_cmap = LinearSegmentedColormap.from_list(
        'cases_map', [
            '#00b050',  # dark green
            '#92d050',  # light green
            '#ffff00',  # yellow
            '#ffa500',  # orange
            '#ff0000',  # red
            '#800080'   # purple
        ]
    )

    for i, date in enumerate(selected_dates):
        date_data = df_selected[df_selected['Date'] == date]
        merged_data = map_data.merge(
            date_data, left_on='ARS', right_on='ID_County_Str', how='left')
        ax = axes[i]
        merged_data.plot(
            column='Normalized_Cases',
            ax=ax,
            legend=False,
            cmap=custom_cmap,
            norm=norm,
            edgecolor='black',
            linewidth=0.15,
            missing_kwds={'color': 'lightgrey', 'label': 'Missing data'}
        )
        states.boundary.plot(ax=ax, color='black', linewidth=0.5)
        ax.set_title(f'{date.strftime("%d %b %Y")}', fontsize=legend_size)
        ax.set_axis_off()

    # Colorbar below plots - made narrower and more compact
    # [left, bottom, width, height]
    cax = fig.add_axes([0.25, 0.08, 0.5, 0.06])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=label_size-2)
    cbar.set_label('Daily cases per 100,000 individuals',
                   fontsize=label_size-2)

    # Keep space at bottom for colorbar axis with adjusted spacing
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'county_map_{year}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 2022 plots
    df_county_2022 = load_county_data(COUNTY_JSON_PATH, 2022)
    plot_county_daily(COUNTY_JSON_PATH, PATH_DATA_MA7, 2022)
    plot_county_map(df_county_2022, shapes, 2022)

    # 2020 plots
    df_county_2020 = load_county_data(COUNTY_JSON_PATH, 2020)
    plot_county_daily(COUNTY_JSON_PATH, PATH_DATA_MA7, 2020)
    plot_county_map(df_county_2020, shapes, 2020)
