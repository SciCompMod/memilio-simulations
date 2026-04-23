from matplotlib.ticker import LogLocator, ScalarFormatter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
import os

matplotlib.use('Agg')

# --- Paths / configuration ---
cwd = os.getcwd()
data_year = 2022
ma = False
extended = True
extended_appendix = "_extended" if extended else ""
ma_appendix = "_ma7" if ma else ""
COUNTY_JSON_PATH = os.path.join(cwd, 'casedata', 'cases_all_county.json')
COMMUNITY_CSV_PATH = os.path.join(
    cwd, 'casedata', f'cases_agg_{data_year}{extended_appendix}{ma_appendix}.csv')
PATH_DATA_MA7 = os.path.join(cwd, 'casedata', 'cases_all_county_ma7.json')
POPULATION_PATH = os.path.join(
    cwd, 'casedata', 'county_current_population.json')
POPULATION_COMMUNITY_XLS_PATH = os.path.join(
    cwd, 'casedata', '12411-02-03-5.xlsx')
OUTPUT_DIR = os.path.join(cwd, 'plots')

# --- Parameters ---
DAY_F = 7
DAY_BEFORE = 10


# --- Data loading & preparation ---
def load_community_agg(csv_path: str) -> pd.DataFrame:
    """Load and aggregate community cases per day, county, and community."""
    df = pd.read_csv(csv_path)
    df = df.groupby(['Date', 'ID_County', 'ID_Community'],
                    as_index=False)['Count'].sum()
    df = df.sort_values(by=['ID_County', 'ID_Community'])
    return df


def extend_with_population(df_comm: pd.DataFrame, pop_xls_path: str) -> pd.DataFrame:
    """Attach population by building merge keys like <county><community>."""
    pop = pd.read_excel(
        pop_xls_path, sheet_name='12411-02-03-5', header=5, usecols=[1, 20])
    pop.columns = ['ID', 'Population']
    pop['Population'] = pd.to_numeric(pop['Population'], errors='coerce')
    pop.dropna(subset=['Population'], inplace=True)
    pop['Population'] = pop['Population'].astype(int)
    pop.loc[pop['ID'] == 'DG', 'ID'] = 0
    pop['ID'] = pop['ID'].astype(int)

    dfc = df_comm.copy()
    county_str = dfc['ID_County'].astype(
        str).replace({'2000': '2', '11000': '11'})
    comm_str = dfc['ID_Community'].astype(str).str.zfill(3).replace('000', '')
    dfc['Merge_ID'] = (county_str + comm_str).astype(int)

    out = pd.merge(dfc, pop, left_on='Merge_ID', right_on='ID', how='left')
    out.drop(columns=['Merge_ID', 'ID'], inplace=True)
    out.dropna(subset=['Population'], inplace=True)
    return out


def preprocess_dataset(a: np.ndarray, day_f: int, day_before: int):
    """Build sliding windows (history H=day_before, predict P=day_f)."""
    X, Y, X_dates = [], [], []
    dates = np.array([datetime.strptime(str(d), "%Y-%m-%d") for d in a[:, 0]])
    for k in range(day_before, len(a) - day_f):
        tmp, ok = [], True
        for i in range(day_before):
            cur, prev = k - i, k - i - 1
            if (dates[cur] - dates[prev]).days != 1 or a[cur, 1] != a[k, 1] or a[cur, 2] != a[k, 2]:
                ok = False
                break
            tmp.append(float(a[cur, 3]))
        if not ok:
            continue
        target_idx = k + day_f
        if (dates[target_idx] - dates[k]).days != day_f or a[target_idx, 1] != a[k, 1] or a[target_idx, 2] != a[k, 2]:
            continue
        X.append([int(a[k, 1]), int(a[k, 2]), int(
            dates[k].month), int(a[k, 4]), *reversed(tmp)])
        Y.append(float(a[target_idx, 3]))
        X_dates.append(dates[k])
    return np.array(X), np.array(Y), np.array(X_dates)


def build_zero_stats(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """Compute zero counts across history+target and carry population for plotting."""
    zero_counts, populations = [], []
    for i in range(len(X)):
        zeros_hist = np.sum(X[i][4:] == 0)
        zeros_target = 1 if Y[i] == 0 else 0
        zero_counts.append(int(zeros_hist + zeros_target))
        populations.append(int(X[i][3]))
    return pd.DataFrame({'Zero_Count': zero_counts, 'Population': populations})


# --- Plots ---
def plot_zero_entry_distribution(
    plot_df: pd.DataFrame,
    out_dir: str,
    prefix: str,
    label_size: int = 30,
    tick_size: int = 26,
    annot_size: int = 18,
    figsize: tuple = (12, 7),
) -> str:
    """Bar chart: how many samples contain N zeros (N=0..11).
    Sizes are configurable via function arguments.
    """
    os.makedirs(out_dir, exist_ok=True)
    max_zeros = 11
    counts = plot_df['Zero_Count'].value_counts().to_dict()
    labels = [str(i) for i in range(0, max_zeros + 1)]
    values = [counts.get(i, 0) for i in range(0, max_zeros + 1)]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color='royalblue', alpha=0.8)
    ax.set_xlabel('Number of zero entries', fontsize=label_size, labelpad=14)
    ax.set_ylabel('Frequency', fontsize=label_size, labelpad=16)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for b in bars:
        yv = b.get_height()
        if yv > 0:
            ax.text(
                b.get_x() + b.get_width()/2.0,
                yv,
                f"{int(yv):,}",
                va='bottom',
                ha='center',
                fontsize=annot_size,
            )
    path = os.path.join(out_dir, f"{prefix}_entry_distribution.png")
    fig.tight_layout()
    fig.subplots_adjust(left=0.16, bottom=0.18)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_population_by_zero(
    plot_df: pd.DataFrame,
    out_dir: str,
    prefix: str,
    label_size: int = 30,
    tick_size: int = 26,
    figsize: tuple = (12, 7),
) -> str:
    """Boxplot: community population per zero-count bucket (log y).
    Sizes are configurable via function arguments.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    order = sorted(plot_df['Zero_Count'].unique())
    palette = sns.color_palette('viridis', len(order))
    sns.boxplot(data=plot_df, x='Zero_Count', y='Population', order=order, palette=palette,
                showfliers=False, width=0.55, linewidth=1.2, ax=ax)
    ax.set_yscale('log')
    ax.set_xlabel('Number of zero entries', fontsize=label_size)
    ax.set_ylabel('Population', fontsize=label_size)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 5)))
    ax.yaxis.set_minor_locator(LogLocator(
        base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_label_coords(-0.15, 0.5)
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.grid(True, which='major', axis='y',
            linestyle='--', linewidth=0.7, alpha=0.85)
    ax.grid(True, which='minor', axis='y',
            linestyle=':', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_population_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


# --- Main ---
def main():
    df = load_community_agg(COMMUNITY_CSV_PATH)
    df_ext = extend_with_population(df, POPULATION_COMMUNITY_XLS_PATH)
    a = df_ext.to_numpy()
    X, Y, _ = preprocess_dataset(a, DAY_F, DAY_BEFORE)
    plot_df = build_zero_stats(X, Y)
    prefix = f"year-{data_year}_extended_{extended}_ma-{ma}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path1 = plot_zero_entry_distribution(plot_df, OUTPUT_DIR, prefix)
    path2 = plot_population_by_zero(plot_df, OUTPUT_DIR, prefix)
    print(f"Saved: {path1}")
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
