import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import ast

# --- Configuration ---
EPSILON_ORDER = ['0.3', '0.5', '1.0', '2.0', 'Non-DP']

# --- Helper Functions ---


def load_data(filepath):
    """Loads the CSV data and converts it to a pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        # Convert string representations of lists back to lists
        for col in ['MAPE_Individuals', 'y_pred', 'y_true', 'test_ids', 'inputs']:
            if col in df.columns:
                # Using ast.literal_eval to safely evaluate the string
                df[col] = df[col].apply(ast.literal_eval)

        # Ensure Epsilon is a string before converting to categorical
        df['Epsilon'] = df['Epsilon'].astype(str)
        # Ensure Epsilon is a categorical type to maintain order in plots
        df['Epsilon'] = pd.Categorical(
            df['Epsilon'], categories=EPSILON_ORDER, ordered=True)
        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'.")
        print("Please run the training script first to generate the predictions.")
        return None


def create_output_directory(directory_name):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name

# --- Plotting Functions ---


def plot_error_distribution(df, output_dir, cap_vals=True, use_log=False):
    """
    Generates a box plot or violin plot of the MAPE distribution for each epsilon value.
    This shows the spread and central tendency of prediction errors.
    """
    print("Generating error distribution plot...")

    # Set the style for publication quality
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'font.family': 'sans-serif',  # geändert von serif auf sans-serif
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'grid.linewidth': 0.8
    })

    all_mape_data = []
    for _, row in df.iterrows():
        # Filter out infinite or very high MAPE values for better visualization
        mape_values = np.array(row['MAPE_Individuals'])
        mape_values = mape_values[np.isfinite(mape_values)]

        if cap_vals:
            mape_values = mape_values[mape_values < 500]

        for mape in mape_values:
            epsilon_label = "non-DP" if row['Epsilon'] == "non-DP" else f"ε = {row['Epsilon']}"
            all_mape_data.append(
                {'Epsilon': epsilon_label, 'MAPE (%)': mape, 'Run': row['run']})

    if not all_mape_data:
        print("Warning: No valid MAPE data to plot for error distribution.")
        return

    mape_df = pd.DataFrame(all_mape_data)

    # Create ordered categories with proper labels
    epsilon_labels = []
    for eps in EPSILON_ORDER:
        if eps == "Non-DP":
            epsilon_labels.append("non-DP")
        else:
            epsilon_labels.append(f"ε = {eps}")

    mape_df['Epsilon'] = pd.Categorical(
        mape_df['Epsilon'], categories=epsilon_labels, ordered=True)

    # Professional color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']

    plt.figure(figsize=(14, 8))

    # Create boxplot with professional styling
    box_plot = sns.boxplot(x='Epsilon', y='MAPE (%)', data=mape_df,
                           palette=colors, linewidth=1.5,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(linewidth=2, color='white'),
                           showfliers=cap_vals)  # Hide outliers when not capping

    # plt.title('Distribution of Individual MAPE per Privacy Level (Across All Runs)',
    #           fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Privacy Level', fontsize=22, fontweight='bold', labelpad=12)
    plt.ylabel('MAPE (%)',
               fontsize=22, fontweight='bold', labelpad=12)

    # Enhanced grid
    plt.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
    plt.gca().set_axisbelow(True)

    # Use log scale if requested
    if use_log:
        plt.gca().set_yscale('log')
        plt.ylabel('MAPE (%)',
                   fontsize=22, fontweight='bold', labelpad=12)
    else:
        # Set a reasonable y-limit for linear scale
        max_mape = mape_df['MAPE (%)'].max()
        plt.ylim(0, max_mape * 1.02)

    # Enhanced tick formatting - larger font sizes
    plt.gca().tick_params(axis='both', which='major', labelsize=20, width=1.5, length=6)
    plt.gca().tick_params(axis='both', which='minor', width=1.0, length=3)

    # Make x-axis labels (epsilon values) even larger and bold
    for label in plt.gca().get_xticklabels():
        label.set_fontsize(22)
        label.set_fontweight('bold')

    # Add spines styling
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # Generate filename
    fn = "mape_distribution"
    if cap_vals:
        fn += "_capped"
    else:
        fn += "_uncapped"
    if use_log:
        fn += "_log"

    output_path = os.path.join(output_dir, f'{fn}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    # Reset matplotlib parameters
    plt.rcdefaults()
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_scatter_predictions(df, output_dir):
    """
    Creates individual scatter plots of true vs. predicted values for each epsilon,
    averaged over all runs, saved as separate PNG files.
    """
    print("Generating individual true vs. predicted scatter plots...")

    # Set the style for publication quality
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        # Schriftart zurück auf Standard (sans-serif)
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'grid.linewidth': 0.8
    })

    # Define a professional color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']

    for i, eps in enumerate(EPSILON_ORDER):
        # Create individual figure for each epsilon
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        eps_df = df[df['Epsilon'] == eps]

        if eps_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=18)
            if eps == "Non-DP":
                title = "non-DP"
                filename = "scatter_no_dp.png"
            else:
                title = f'ε = {eps}'
                filename = f"scatter_epsilon_{eps}.png"
            ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        else:
            # Concatenate all predictions and true values for this epsilon from all runs
            all_preds = np.concatenate(eps_df['y_pred'].values)
            all_trues = np.concatenate(eps_df['y_true'].values)

            # To avoid overplotting, sample a subset of points if there are too many
            if len(all_preds) > 2000:
                sample_indices = np.random.choice(
                    len(all_preds), 2000, replace=False)
                all_preds = all_preds[sample_indices]
                all_trues = all_trues[sample_indices]

            # Use professional styling for scatter plot
            color = colors[i % len(colors)]
            ax.scatter(all_trues, all_preds, alpha=0.6, s=25,
                       color=color, edgecolors='white', linewidths=0.5,
                       rasterized=True)

            if eps == "Non-DP":
                title = "non-DP"
                filename = "scatter_no_dp.png"
            else:
                title = f'ε = {eps}'
                filename = f"scatter_epsilon_{eps}.png"

            ax.set_title(title, fontsize=30, fontweight='bold', pad=20)

            # Individual axis scaling
            x_min, x_max = np.min(all_trues), np.max(all_trues)
            y_min, y_max = np.min(all_preds), np.max(all_preds)
            # Optional: some padding
            x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1
            y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

            # Reference line y=x (only in visible range)
            min_diag = max(x_min - x_pad, y_min - y_pad)
            max_diag = min(x_max + x_pad, y_max + y_pad)
            ax.plot([min_diag, max_diag], [min_diag, max_diag],
                    'k-', alpha=0.8, linewidth=2, zorder=0, label='Perfect Prediction')

        # Common styling for all plots
        ax.set_xlabel('True Values', fontsize=22,
                      fontweight='bold', labelpad=10)
        ax.set_ylabel('Predicted Values', fontsize=22,
                      fontweight='bold', labelpad=10)

        # Enhanced grid
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_axisbelow(True)

        # Enhanced tick formatting
        ax.tick_params(axis='both', which='major',
                       labelsize=20, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=3)

        # Use scientific notation for tick labels
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

        # Adjust the font size of the scientific notation
        ax.xaxis.get_offset_text().set_fontsize(18)
        ax.yaxis.get_offset_text().set_fontsize(18)

        # Add spines styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

        # Save individual plot
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved {title} plot to {output_path}")

    # Reset matplotlib parameters
    plt.rcdefaults()
    print("All individual scatter plots saved.")


if __name__ == "__main__":
    cwd = os.getcwd()
    # load results from county_dp file:
    year = 2020  # 2020
    input_file = os.path.join(
        cwd, f"year-{year}_county_predictions_scaled-False_runs-15_rounds-75.csv")

    # Create outout directory
    output_dir = create_output_directory(os.path.join(
        cwd, "plots", f"plots_{year}"))

    print(f"--- Starting Result Plot Generation from {input_file} ---")

    df = load_data(input_file)
    plot_error_distribution(df, output_dir, False,
                            True)    # Uncapped, log scale

    plot_scatter_predictions(df, output_dir)

    print(f"All plots have been saved to the '{output_dir}' directory.")
