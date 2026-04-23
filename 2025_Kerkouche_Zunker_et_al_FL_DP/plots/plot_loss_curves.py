import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

EPSILON_ORDER = ['0.3', '0.5', '1.0', '2.0', 'Non-DP']
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']

RCPARAMS = {
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
    'grid.linewidth': 0.8,
}


def load_loss_curves(filepath):
    df = pd.read_csv(filepath)
    df['epsilon'] = df['epsilon'].astype(str)
    return df


def plot_loss_curves(df, output_dir, year, eps_subset=None):
    """Plot mean ± std of train/test MSE vs. round for each epsilon."""
    plt.style.use('default')
    plt.rcParams.update(RCPARAMS)

    epsilons = eps_subset if eps_subset is not None else EPSILON_ORDER
    epsilons = [e for e in epsilons if e in df['epsilon'].unique()]

    n = len(epsilons)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, eps, color in zip(axes, epsilons, COLORS):
        sub = df[df['epsilon'] == eps]
        grouped = sub.groupby('round')

        rounds = sorted(sub['round'].unique())
        train_mean = grouped['train_mse'].mean().values
        train_std = grouped['train_mse'].std().values
        test_mean = grouped['test_mse'].mean().values
        test_std = grouped['test_mse'].std().values

        ax.plot(rounds, train_mean, color=color,
                linewidth=2, label='Train MSE')
        ax.fill_between(rounds,
                        np.maximum(0, train_mean - train_std),
                        train_mean + train_std,
                        color=color, alpha=0.2)

        ax.plot(rounds, test_mean, color=color, linewidth=2,
                linestyle='--', label='Test MSE')
        ax.fill_between(rounds,
                        np.maximum(0, test_mean - test_std),
                        test_mean + test_std,
                        color=color, alpha=0.1)

        title = 'non-DP' if eps == 'Non-DP' else f'ε = {eps}'
        ax.set_title(title, fontsize=20, fontweight='bold', pad=10)
        ax.set_xlabel('FL Round', fontsize=18, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=18, fontweight='bold')
        ax.legend(fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    fig.suptitle(f'Training vs. Test Loss - {year}', fontsize=22,
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    eps_tag = '_'.join(epsilons).replace('.', '').replace('Non-DP', 'nodp')
    out = os.path.join(output_dir, f'loss_curves_{year}_{eps_tag}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.rcdefaults()
    plt.close()
    print(f'Saved: {out}')
    return out


if __name__ == '__main__':
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, 'plots', 'loss_curves')

    for year in [2020, 2022]:
        fp = os.path.join(
            cwd,
            f'year-{year}_loss_curves_scaled-False_runs-15_rounds-75_ma-trailing.csv'
        )
        df = load_loss_curves(fp)

        # All epsilons
        plot_loss_curves(df, output_dir, year)

        # Key subset: Non-DP and ε=2.0 (most relevant for overfitting argument)
        plot_loss_curves(df, output_dir, year, eps_subset=['2.0', 'Non-DP'])

    print('Done.')
