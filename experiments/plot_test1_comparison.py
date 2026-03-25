import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.plot_style import apply_cmame_style
apply_cmame_style()


def load_experiment_data(exp_dir):
    """Load error + timing data from an experiment directory.

    Error data comes from integrated_error_data.json.
    Timing data is loaded per-grid from grid_<N>/timing.json,
    iterated in grid_sizes order.
    """
    with open(os.path.join(exp_dir, 'integrated_error_data.json')) as f:
        data = json.load(f)

    for gs in data['grid_sizes']:
        t_path = os.path.join(exp_dir, f'grid_{gs}', 'timing.json')
        if not os.path.exists(t_path):
            continue
        with open(t_path) as f:
            t = json.load(f)
        for key in ('solve_time', 'other_time', 'total_time',
                    'big_domain_solve_time', 'small_domain_solve_time'):
            if key in t:
                data.setdefault(key, []).append(t[key])

    return data


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Plot single vs dual domain comparison')
parser.add_argument('--single-dir', default='useful_results/test1_batch',
                    help='Single-domain experiment directory')
parser.add_argument('--dual-dir',   default='useful_results/test1_schwarz_2',
                    help='Dual-domain experiment directory')
parser.add_argument('--output-dir', default='useful_results/test1_comparison',
                    help='Output directory for PDFs')
args = parser.parse_args()

# ── Load data ────────────────────────────────────────────────────────────────
single = load_experiment_data(args.single_dir)
dual   = load_experiment_data(args.dual_dir)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

dx_values = np.array(single['dx_values'])   # same for both: [0.02, 0.01, 0.00667, 0.005]

# ── Plot 1: Convergence rate (log-log) comparison ────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))

for data, color, label in [
    (single, '#b2182b', 'Single domain'),
    (dual,   '#3182bd', 'Dual domain'),
]:
    dx  = np.array(data['dx_values'])
    err = np.array(data['integrated_errors_total'])

    ax.loglog(dx, err, 'o-', color=color, linewidth=2, markersize=7,
              markerfacecolor='none', markeredgewidth=1.5, label=label)

    log_dx  = np.log(dx)
    log_err = np.log(err)
    coeffs  = np.polyfit(log_dx, log_err, 1)
    dx_fit  = np.array([dx.min(), dx.max()])
    err_fit = np.exp(coeffs[1]) * dx_fit ** coeffs[0]
    ax.loglog(dx_fit, err_fit, '--', color=color, linewidth=1.5, alpha=0.6,
              label=rf'Slope $= {coeffs[0]:.2f}$')

ax.set_xlabel(r'Grid spacing $dx$ (m)')
ax.set_ylabel(r'Normalized integrated error (Pa)')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
ax.minorticks_on()
plt.savefig(os.path.join(output_dir, 'convergence_comparison.pdf'))
plt.close()
print('Saved convergence_comparison.pdf')

# ── Plot 2: Stacked CPU time + Error (secondary axis) ────────────────────────
n     = len(dx_values)
x     = np.arange(n)
bar_w = 0.35

fig, ax1 = plt.subplots(figsize=(8, 5))

# Single domain bars (left) — warm tones
s_solve = np.array(single['solve_time'])
s_other = np.array(single['other_time'])
ax1.bar(x - bar_w / 2, s_solve, bar_w,
        label='Single: solve',  color='#b2182b')
ax1.bar(x - bar_w / 2, s_other, bar_w, bottom=s_solve,
        label='Single: other',  color='#e5735c', alpha=0.8)

# Dual domain bars (right) — cool tones
d_big   = np.array(dual['big_domain_solve_time'])
d_small = np.array(dual['small_domain_solve_time'])
d_other = np.array(dual['other_time'])
ax1.bar(x + bar_w / 2, d_big,   bar_w,
        label='Dual: big domain',   color='#3182bd')
ax1.bar(x + bar_w / 2, d_small, bar_w, bottom=d_big,
        label='Dual: small domain', color='#6baed6')
ax1.bar(x + bar_w / 2, d_other, bar_w, bottom=d_big + d_small,
        label='Dual: other',        color='#2ca02c', alpha=0.8)

ax1.set_xlabel(r'Grid spacing $dx$ (m)')
ax1.set_ylabel(r'CPU time (s)')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{v:.4g}' for v in dx_values])
ax1.grid(True, alpha=0.3, axis='y')

# Error on secondary axis
ax2 = ax1.twinx()
err_single = np.array(single['integrated_errors_total'])
err_dual   = np.array(dual['integrated_errors_total'])

ax2.plot(x - bar_w / 2, err_single, 'o--', color='#b2182b', linewidth=2,
         markersize=7, markerfacecolor='none', markeredgewidth=1.5,
         label='Single: L2 error', zorder=5)
ax2.plot(x + bar_w / 2, err_dual,   's--', color='#3182bd',   linewidth=2,
         markersize=7, markerfacecolor='none', markeredgewidth=1.5,
         label='Dual: L2 error',   zorder=5)
ax2.set_ylabel(r'Normalized integrated error (Pa)')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.0, 0.85))

plt.savefig(os.path.join(output_dir, 'time_error_comparison.pdf'))
plt.close()
print('Saved time_error_comparison.pdf')
