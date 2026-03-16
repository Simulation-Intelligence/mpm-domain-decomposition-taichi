import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ────────────────────────────────────────────────────────────────
with open('useful_results/test1_batch/integrated_error_data.json') as f:
    single = json.load(f)
with open('useful_results/test1_schwarz_2/integrated_error_data.json') as f:
    dual = json.load(f)

output_dir = 'useful_results/test1_comparison'
os.makedirs(output_dir, exist_ok=True)

dx_values = np.array(single['dx_values'])   # same for both: [0.02, 0.01, 0.00667, 0.005]

# ── Plot 1: Convergence rate (log-log) comparison ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for data, color, label in [
    (single, 'tab:red',  'Single domain'),
    (dual,   'tab:blue', 'Dual domain'),
]:
    dx  = np.array(data['dx_values'])
    err = np.array(data['integrated_errors_total'])

    ax.loglog(dx, err, 'o-', color=color, linewidth=2, markersize=8, label=label)

    log_dx  = np.log(dx)
    log_err = np.log(err)
    coeffs  = np.polyfit(log_dx, log_err, 1)
    dx_fit  = np.array([dx.min(), dx.max()])
    err_fit = np.exp(coeffs[1]) * dx_fit ** coeffs[0]
    ax.loglog(dx_fit, err_fit, '--', color=color, linewidth=1.5, alpha=0.6,
              label=f'Slope = {coeffs[0]:.2f}')

ax.set_xlabel('Grid Spacing $dx$ (m)', fontsize=12)
ax.set_ylabel('Normalized Integrated Error (Pa)', fontsize=12)
ax.set_title('Convergence Rate (log-log scale)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_comparison.pdf'), bbox_inches='tight')
plt.close()
print('Saved convergence_comparison.pdf')

# ── Plot 2: Stacked CPU time + Error (secondary axis) ────────────────────────
n     = len(dx_values)
x     = np.arange(n)
bar_w = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Single domain bars (left)
s_solve = np.array(single['solve_time'])
s_other = np.array(single['other_time'])
ax1.bar(x - bar_w / 2, s_solve, bar_w,
        label='Single: Solve',  color='tab:orange')
ax1.bar(x - bar_w / 2, s_other, bar_w, bottom=s_solve,
        label='Single: Other',  color='#e07b39', alpha=0.6)

# Dual domain bars (right)
d_big   = np.array(dual['big_domain_solve_time'])
d_small = np.array(dual['small_domain_solve_time'])
d_other = np.array(dual['other_time'])
ax1.bar(x + bar_w / 2, d_big,   bar_w,
        label='Dual: Big domain',   color='tab:blue')
ax1.bar(x + bar_w / 2, d_small, bar_w, bottom=d_big,
        label='Dual: Small domain', color='tab:cyan')
ax1.bar(x + bar_w / 2, d_other, bar_w, bottom=d_big + d_small,
        label='Dual: Other',        color='#2ca02c', alpha=0.6)

ax1.set_xlabel('Grid Spacing $dx$ (m)', fontsize=12)
ax1.set_ylabel('CPU Time (s)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{v:.4g}' for v in dx_values])
ax1.grid(True, alpha=0.3, axis='y')

# Error on secondary axis
ax2 = ax1.twinx()
err_single = np.array(single['integrated_errors_total'])
err_dual   = np.array(dual['integrated_errors_total'])

ax2.plot(x - bar_w / 2, err_single, 'o--', color='tab:orange', linewidth=2,
         markersize=9, markerfacecolor='white', markeredgewidth=2,
         label='Single: L2 Error', zorder=5)
ax2.plot(x + bar_w / 2, err_dual,   's--', color='tab:blue',   linewidth=2,
         markersize=9, markerfacecolor='white', markeredgewidth=2,
         label='Dual: L2 Error',   zorder=5)
ax2.set_ylabel('Normalized Integrated Error (Pa)', fontsize=12)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

ax1.set_title('CPU Time and Error Comparison: Single vs Dual Domain', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_error_comparison.pdf'), bbox_inches='tight')
plt.close()
print('Saved time_error_comparison.pdf')
