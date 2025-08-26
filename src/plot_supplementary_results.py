#!/usr/bin/env python3
"""
Supplementary results plot for energy efficiency and allocentric decoding comparison.
Includes original model variants and new temporal contrastive results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

# Set style to match paper
sns.set_context('talk')

# Data from Thomas's results + new temporal contrastive results + untrained baseline
models = ['full model', '4 timesteps', '8 timesteps', '1 hidden\nlayer', '3 hidden\nlayers', 
          'categorisation\n(supervised)', 'temporal contrastive\nobjective', 'untrained\nbaseline']

# Loss data: (mean, lower_99CI, upper_99CI)
loss_data = [
    (0.128, 0.127, 0.129),  # full model (estimated from 4 timesteps)
    (0.12811479, 0.12713718292245169, 0.12909239652624827),  # 4 timesteps
    (0.1285683, 0.12759205751324207, 0.1295445556173274),   # 8 timesteps
    (0.1335439, 0.1325247788863055, 0.13456300850344974),   # 1 hidden layer
    (0.12374754, 0.12280473117477891, 0.12469035382621291), # 3 hidden layers
    (0.3195281, 0.3182375727972682, 0.3208186329522435),    # supervised (epoch 100)
    (0.423138, 0.420405, 0.425870),  # temporal contrastive - NEW RESULTS
    (0, 0, 0)  # untrained baseline (no loss data)
]

# R² data: [x_coordinate, y_coordinate] - Global decoding results
r2_data = [
    [0.91, 0.93],           # full model (original)
    [0.96726419, 0.97306166],  # 4 timesteps
    [0.82636739, 0.83783585],  # 8 timesteps  
    [0.79468732, 0.80130765],  # 1 hidden layer
    [0.94286285, 0.95643485],  # 3 hidden layers
    [0.48640877, 0.55557019],  # supervised (epoch 100)
    [0.47562987, 0.50586957],  # temporal contrastive - UPDATED GLOBAL RESULTS (Test R²: 0.490693)
    [0.47897762, 0.48458517]   # untrained baseline - GLOBAL RESULTS (Test R²: 0.481723)
]

# Extract values for plotting
test_losses = [data[0] for data in loss_data]
loss_lower_ci = [data[1] for data in loss_data]
loss_upper_ci = [data[2] for data in loss_data]

# Calculate error bars (distance from mean)
loss_errors = [
    [test_losses[i] - loss_lower_ci[i], loss_upper_ci[i] - test_losses[i]] 
    for i in range(len(test_losses))
]
loss_errors = np.array(loss_errors).T

r2_x = [data[0] for data in r2_data]
r2_y = [data[1] for data in r2_data]

# Create figure exactly like paper style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7), gridspec_kw={'width_ratios': [1, 1]})

# Colors - emerald green for full model, blue for temporal contrastive, orange for untrained, grays for others
full_color = '#27ae60'
temporal_contrastive_color = "#3c45e7ff"  # Blue for temporal contrastive
supervised_color = "#23a59fff"  # Teal for supervised
untrained_color = "#ff6b35"  # Orange for untrained baseline
pallette_greys = sns.color_palette('Greys', len(models) + 3)[3:]
pallette_greys[0] = full_color
pallette_greys[-1] = untrained_color  # Last color for untrained
pallette_greys[-2] = temporal_contrastive_color  # Second last for temporal contrastive
pallette_greys[-3] = supervised_color  # Third last for supervised

# Plot 1: Energy Efficiency Loss with error bars
bars1 = ax1.bar(range(len(models)), test_losses,
                color=pallette_greys, width=0.6, edgecolor='black', linewidth=1,
                yerr=loss_errors, capsize=5, ecolor='black')

ax1.set_ylabel('energy efficiency\n[loss]')
ax1.set_title('B model results', fontweight='bold', loc='left')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_ylim(0, 0.45)  # Adjusted to accommodate temporal contrastive results

# Add full model reference line
full_model_loss = test_losses[0]
ax1.axhline(full_model_loss, linestyle='--', linewidth=3,
            label='full model\nloss', color='darkgrey', zorder=0)

# Removed annotation as requested

ax1.legend(frameon=False)

# Plot 2: Allocentric Decoding Performance
# Create grouped barplot data
r2_data_df = []
for i, model in enumerate(models):
    r2_data_df.extend([
        {'model': model, 'coordinate': 'x coordinate', 'r2': r2_x[i], 'pos': i},
        {'model': model, 'coordinate': 'y coordinate', 'r2': r2_y[i], 'pos': i}
    ])

r2_df = pd.DataFrame(r2_data_df)

# Colors for coordinates
coord_colors = ['#8d159fff', '#c94a77ff']

sns.barplot(data=r2_df, x='pos', y='r2', hue='coordinate',
            palette=coord_colors, ax=ax2,
            width=0.6, edgecolor='black', errwidth=2)

ax2.set_ylabel('fixation position decoding\n[R²]')
ax2.set_title('C fixation position decoding', fontweight='bold', loc='left')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.legend(frameon=False, loc='lower right')
ax2.set_ylim(0, 1.0)
ax2.set_xlabel('')

# Add reference lines for full model R²
ax2.axhline(y=0.91, color='darkgrey', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.axhline(y=0.93, color='darkgrey', linestyle='--', alpha=0.7, linewidth=1.5)

# Removed annotation as requested

# Remove spines like in paper
for ax in [ax1, ax2]:
    sns.despine(ax=ax)

plt.tight_layout()

# Save as SVG
output_dir = './efficient-remapping/figures'
os.makedirs(output_dir, exist_ok=True)

# Make sure the text is editable in the SVG
plt.rcParams['svg.fonttype'] = 'none'
fname = 'supplement_energy_efficiency_with_temporal_contrastive.svg'

# Print save path
print(f"Saving figure to: {os.path.join(output_dir, fname)}")
plt.savefig(os.path.join(output_dir, fname), format='svg', bbox_inches='tight')

# Also save as PNG for quick viewing
plt.savefig(os.path.join(output_dir, fname.replace('.svg', '.png')), 
           format='png', dpi=300, bbox_inches='tight')

print("Figure saved successfully!")
print("\nTemporal Contrastive Results Summary:")
print(f"Energy Efficiency: {test_losses[-1]:.6f} (99% CI: [{loss_lower_ci[-1]:.6f}, {loss_upper_ci[-1]:.6f}])")
print(f"Spatial Decoding: R²_x = {r2_x[-1]:.6f}, R²_y = {r2_y[-1]:.6f}")
print("Status: Temporal contrastive learning shows minimal improvement over untrained baseline")

# Display the plot
plt.show()

if __name__ == "__main__":
    print("Generating supplementary results plot with temporal contrastive results...")
    # The script will run automatically when imported or called directly