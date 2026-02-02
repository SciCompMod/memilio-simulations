import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys


# Load plot settings from parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from plotting_settings import set_fontsize, dpi, colors as pt_colors


# Load measurement data
csv_path = 'measurements_gnn_vs_ode.csv'
data = pd.read_csv(csv_path)

# MAPE values (90-day model reference)
mape_values = {1: 8.95, 10: 9.02, 100: 9.15}

# Plot Setup (use paper settings)
set_fontsize()
base_fontsize = 17
base_tick = int(0.8 * base_fontsize)
legend_size = int(0.8 * base_fontsize)

# Match layout from test.py
fig = plt.figure(figsize=(8, 5), dpi=dpi)
panel = [0.2, 0.2, 0.78, 0.75]
ax1 = fig.add_axes(panel)

# Tick params update
ax1.tick_params(axis='both', which='both', labelsize=base_tick)

# Colors from plotting_settings
color_cpp = pt_colors.get('Blue', '#155489')
color_r = pt_colors.get('Orange', '#E89A63')
color_py = pt_colors.get('Green', '#5D8A2B')
colors = [color_cpp, color_r, color_py]

# Speedup plotting
batch_sizes = sorted(data['Num_Pred'].unique())

for i, days in enumerate([30, 60, 90]):
    subset = data[data['Days'] == days].sort_values('Num_Pred')
    
    # ODE Line (Dashed)
    ax1.plot(subset['Num_Pred'], subset['Mean_Time_ODE'], 
             color=colors[i], linestyle='--', marker='o', markersize=4, alpha=0.7,
             label=f'Graph-ODE {days}d')

    # GNN Line (Solid)
    line, = ax1.plot(subset['Num_Pred'], subset['Mean_Time_GNN'], 
                     color=colors[i], linestyle='-', marker='s', markersize=4,
                     label=f'GNN {days}d')
    
    # Annotate Speedups (style from test.py)
    if days == 90:
        connector_color = colors[i]  # Use loop color (Green for 90d)
        for idx, row in subset.iterrows():
            n_pred = row['Num_Pred']
            if n_pred in [1, 8, 32, 128]:  # Select a few points
                y_gnn = row['Mean_Time_GNN']
                y_ode = row['Mean_Time_ODE']
                speedup = row['Speedup']

                lower = min(y_gnn, y_ode)
                upper = max(y_gnn, y_ode)

                # Draw triangles along the vertical line
                n_points = 12
                y_points = np.logspace(
                    np.log10(lower), np.log10(upper), n_points)
                x_points = np.full(n_points, n_pred)

                ax1.plot(
                    x_points,
                    y_points,
                    color=connector_color,
                    linewidth=0,
                    linestyle='None',
                    marker='^',
                    markersize=5,
                    markerfacecolor=connector_color,
                    markeredgecolor=connector_color,
                    alpha=0.45,
                    zorder=1.5,
                )

                # Label placement (shifted x)
                y_mid = np.sqrt(lower * upper)

                # Consistent placement on the right
                x_text = n_pred * 1.1
                
                ax1.text(
                    x_text,
                    y_mid,
                    f'{int(speedup):,}x'.replace(',', '\u2009'),
                    ha='left', va='center',
                    fontsize=12, color=connector_color, 
                    zorder=4
                )

ax1.set_ylabel('Runtime [s]')
ax1.set_yscale('log')
ax1.set_xscale('log', base=2) # Base 2 matches batch sizes
ax1.set_xticks(batch_sizes)
ax1.set_xticklabels([str(b) for b in batch_sizes])
ax1.set_xlabel('Simulations [#]')
# Extend x-axis to fit the rightmost annotation
ax1.set_xlim(right=256)

ax1.grid(True, which="major", ls="-", alpha=0.3)
ax1.grid(True, which="minor", ls=":", alpha=0.1)

# Custom legend to handle lines
lines, labels = ax1.get_legend_handles_labels()
# Reorder to [ODE30, ODE60, ODE90, GNN30, GNN60, GNN90]
# With ncol=2 (col-fill), this creates rows: (ODE30, GNN30), (ODE60, GNN60)...
reorder = [0, 2, 4, 1, 3, 5]
lines = [lines[i] for i in reorder]
labels = [labels[i] for i in reorder]

ax1.legend(lines, labels, loc='upper left', fontsize=legend_size, frameon=True, shadow=True, ncol=2)

# Save figures
output_dir = r'PATH\gnn_plot'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# plt.tight_layout() # conflicts with add_axes
plt.savefig(os.path.join(output_dir, 'gnn_scaling.pdf'), dpi=dpi)
plt.savefig(os.path.join(output_dir, 'gnn_scaling.png'), dpi=dpi)

print(f"Figures saved to {output_dir}.")
