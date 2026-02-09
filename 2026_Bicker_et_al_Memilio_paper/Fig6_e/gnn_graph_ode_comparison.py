import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Plot settings from paper
from plotting_settings import set_fontsize, dpi, colors as pt_colors


# 1. Daten aus der Messung laden
csv_path = 'measurements_gnn_vs_ode.csv'
data = pd.read_csv(csv_path)

# MAPE Werte (Stabilität der Vorhersage)
# Wir nehmen hier die Werte für das 90-Tage Modell als Referenz,
# da diese die höchste statistische Signifikanz haben.
mape_values = {1: 8.95, 10: 9.02, 100: 9.15}

# Plot Setup (use paper settings)
set_fontsize()
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=dpi)

# Farben aus plotting_settings (wie in test.py)
color_cpp = pt_colors.get('Blue', '#155489')
color_r = pt_colors.get('Orange', '#E89A63')
color_py = pt_colors.get('Green', '#5D8A2B')
colors = [color_cpp, color_r, color_py]

# 2. Speedup Plotting
# Bestimme Batch-Größen aus den Daten (robust gegenüber fehlenden Kombinationen)
batch_sizes = sorted(data['Num_Pred'].unique())
x = np.arange(len(batch_sizes))
width = 0.25

for i, days in enumerate([30, 60, 90]):
    subset = data[data['Days'] == days]
    subset = subset.set_index('Num_Pred')

    # Werte entsprechend der globalen batch_sizes anordnen (fehlende -> NaN)
    speeds = np.array([subset['Speedup'].get(b, np.nan)
                      for b in batch_sizes], dtype=float)

    bars = ax1.bar(x + (i-1)*width, speeds, width,
                   label=f'{days} Days', color=colors[i],
                   edgecolor='black', alpha=0.85, linewidth=0.8)

    # Annotations für die 90-Tage Balken (nur vorhandene Werte)
    if days == 90:
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                         f'{int(height):,}x', ha='center', va='bottom',
                         fontsize=10, color=colors[i])

ax1.set_ylabel(
    'Speedup [MPM/GNN]')
ax1.set_yscale('log')
ax1.set_xticks(x)
ax1.set_xticklabels([str(b) for b in batch_sizes])
ax1.set_xlabel('Simulations [#]')
ax1.set_ylim(10, 10**5)
ax1.grid(True, which="both", axis='y', ls="--", alpha=0.4)

# Legende (nur Speedup-Balken)
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True)

# Save
output_dir = r'C:\Users\zunk_he\Downloads\gnn_plot'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gnn_scaling.pdf'), dpi=dpi)
plt.savefig(os.path.join(output_dir, 'gnn_scaling.png'), dpi=dpi)

print(f"Grafiken erfolgreich in {output_dir} gespeichert.")
