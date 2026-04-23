import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import json
import os


# --- Configuration ---

cwd = os.getcwd()
COUNTY_JSON_PATH = os.path.join(cwd, 'casedata', 'cases_all_county.json')
COMMUNITY_CSV_PATH = os.path.join(cwd, 'casedata', 'cases_agg.csv')
PATH_DATA_MA7 = os.path.join(cwd, 'casedata', 'cases_all_county_ma7.json')
POPULATION_PATH = os.path.join(
    cwd, 'casedata', 'county_current_population.json')
# Get shape file for germany from: https://github.com/isellsoap/deutschlandGeoJSON/
OUTPUT_DIR = os.path.join(cwd, "plots")

SHAPE_PATH_COUNTY = os.path.join(
    cwd, 'shape_files_kreise', '1_sehr_hoch.geo.json')
SHAPE_PATH_COMMUNITY = os.path.join(
    cwd, 'shape_files_community', 'vg250_01-01.tm32.shape.ebenen', 'vg250_ebenen_0101', 'VG250_GEM.shp')
SHAPE_PATH_FEDERAL = os.path.join(
    cwd, 'shapes_files_bundesländer', '1_sehr_hoch.geo.json')

TRANSFORM_COMMUNITY_ID = {
    '5558004': 'Ascheberg',
    '5558008': 'Billerbeck, Stadt',
    '5558012': 'Coesfeld, Stadt',
    '5558016': 'Dülmen, Stadt',
    '5558020': 'Havixbeck',
    '5558024': 'Lüdinghausen, Stadt',
    '5558028': 'Nordkirchen',
    '5558032': 'Nottuln',
    '5558036': 'Olfen, Stadt',
    '5558040': 'Rosendahl',
    '5558044': 'Senden'
}

# --- Helper Functions ---


def create_output_directory(directory_name):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name

# --- Plotting Functions ---


def plot_germany_highlighted(county_id: str | None = None):
    county_gdf = gpd.read_file(SHAPE_PATH_COUNTY)
    federal_gdf = gpd.read_file(SHAPE_PATH_FEDERAL)

    fig, ax = plt.subplots(figsize=(10, 10))
    # Base: all German counties in light gray
    county_gdf.plot(ax=ax, color='#F0F0F0', edgecolor='black', linewidth=0.3)

    # NRW highlighted in orange
    state_counties = county_gdf[county_gdf['NAME_1'] == 'Nordrhein-Westfalen']
    state_counties.plot(ax=ax, color='orange',
                        edgecolor='black', linewidth=0.3)

    # Optional: highlight given county in red using community layer (robust id handling)
    if county_id:
        try:
            community_gdf = gpd.read_file(SHAPE_PATH_COMMUNITY)
            county_comms = community_gdf[community_gdf['ARS'].str.startswith(
                str(county_id))]
            if not county_comms.empty:
                dissolved = county_comms.dissolve()
                if hasattr(dissolved, 'crs') and dissolved.crs is not None and dissolved.crs != county_gdf.crs:
                    dissolved = dissolved.to_crs(county_gdf.crs)
                dissolved.plot(ax=ax, color='red',
                               edgecolor='black', linewidth=0.8)
        except Exception:
            pass

    # Federal state borders on top
    federal_gdf.boundary.plot(ax=ax, color='black', linewidth=2.0)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_deutschland_markiert.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_county_with_communities(county_id, county_name):
    """Plot NRW filled in orange and highlight the given county in red."""
    community_gdf = gpd.read_file(SHAPE_PATH_COMMUNITY)

    fig, ax = plt.subplots(figsize=(10, 10))
    state_communities = community_gdf[community_gdf['SN_L'] == '05']
    # Fill entire NRW like in plot_germany_highlighted (orange)
    state_communities.plot(ax=ax, color='orange',
                           edgecolor='black', linewidth=0.3)

    # Highlight the selected county (e.g., Coesfeld) in red
    county_communities = community_gdf[community_gdf['ARS'].str.startswith(
        county_id)]
    county_communities.plot(
        ax=ax, color='red', edgecolor='black', linewidth=0.8)

    # County borders (draw last to sit on top)
    state_communities_copy = state_communities.copy()
    state_communities_copy['COUNTY_ID'] = state_communities_copy['ARS'].str.slice(
        0, 5)
    dissolved_counties = state_communities_copy[[
        'COUNTY_ID', state_communities_copy.geometry.name]].dissolve(by='COUNTY_ID')
    dissolved_counties.boundary.plot(ax=ax, edgecolor='black', linewidth=2.0)

    ax.set_axis_off()
    plt.tight_layout()
    filename = f"5_{county_name.lower().replace(' ', '_')}_with_communities.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_county_and_communities_cases(county_id, county_name,
                                      county_json_path, community_csv_path,
                                      output_dir="plots", date_range=None):
    """
    Plot cases for a specific county and all its communities with dashed lines.
    """
    create_output_directory(output_dir)

    with open(county_json_path, 'r') as f:
        county_data = json.load(f)

    county_df = pd.DataFrame(county_data)
    county_df['Date'] = pd.to_datetime(county_df['Date'])

    county_cases = county_df[county_df['ID_County'] == int(county_id)]

    if county_cases.empty:
        return

    community_df = pd.read_csv(community_csv_path)
    community_df['Date'] = pd.to_datetime(community_df['Date'])

    county_communities = community_df[community_df['ID_County'] == int(
        county_id)]

    if not county_communities.empty:
        if date_range is None:
            date_range = (county_communities['Date'].min(
            ), county_communities['Date'].max())
    else:
        return

    if date_range:
        start_date, end_date = date_range
        prev_day = start_date - pd.Timedelta(days=1)
        county_cases_extended = county_cases[(county_cases['Date'] >= prev_day) &
                                             (county_cases['Date'] <= end_date)]
        county_communities = county_communities[(county_communities['Date'] >= start_date) &
                                                (county_communities['Date'] <= end_date)]
    else:
        county_cases_extended = county_cases

    # Set publication quality style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'grid.linewidth': 0.8
    })

    plt.figure(figsize=(12, 8))

    county_agg = county_cases_extended.groupby(
        'Date')['Confirmed'].first().reset_index()
    county_agg['Daily_New_Cases'] = county_agg['Confirmed'].diff().fillna(
        county_agg['Confirmed'].iloc[0])

    if date_range:
        county_agg = county_agg[county_agg['Date'] >= start_date]

    plt.plot(county_agg['Date'], county_agg['Daily_New_Cases'],
             linewidth=3, label=f'{county_name} County',
             color='red', alpha=0.8)

    if not county_communities.empty:
        community_colors = plt.cm.tab10(
            range(len(county_communities['ID_Community'].unique())))

        for i, community_id in enumerate(county_communities['ID_Community'].unique()):
            community_data = county_communities[county_communities['ID_Community'] == community_id]
            community_agg = community_data.groupby(
                'Date')['Count'].sum().reset_index()

            if len(community_agg) != 32:
                all_dates = pd.date_range(start=start_date, end=end_date)
                community_agg = community_agg.set_index(
                    'Date').reindex(all_dates).fillna(0).reset_index()
                community_agg.rename(columns={'index': 'Date'}, inplace=True)

            community_id_str = str(community_id).zfill(3)
            community_name = TRANSFORM_COMMUNITY_ID[str(
                int(county_id)) + community_id_str]

            plt.plot(community_agg['Date'], community_agg['Count'],
                     linestyle='--', linewidth=1.5, alpha=0.7,
                     color=community_colors[i % len(community_colors)],
                     label=community_name)

    plt.xlabel('Date', fontsize=26, labelpad=12)
    plt.ylabel('Number of Cases', fontsize=26, labelpad=12)
    plt.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    plt.gca().set_axisbelow(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=25)
    plt.yscale('log')

    plt.gca().tick_params(axis='both', which='major', labelsize=20, width=1.5, length=6)
    plt.gca().tick_params(axis='both', which='minor', width=1.0, length=3)
    plt.xticks(rotation=45)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    filename = f"{county_name.lower().replace(' ', '_')}_cases_with_communities.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    plt.rcdefaults()
    plt.close()


def plot_custom_county_cases(county_id, county_name):
    """Generic function to plot cases for any county."""
    output_dir = create_output_directory(OUTPUT_DIR)

    # Create map plots
    plot_germany_highlighted(county_id)
    plot_county_with_communities(county_id, county_name)

    # Create case plots
    plot_county_and_communities_cases(
        county_id=county_id,
        county_name=county_name,
        county_json_path=COUNTY_JSON_PATH,
        community_csv_path=COMMUNITY_CSV_PATH,
        output_dir=output_dir
    )


if __name__ == "__main__":
    plot_custom_county_cases("05558", "Coesfeld")
