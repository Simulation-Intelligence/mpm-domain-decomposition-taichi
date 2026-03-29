import argparse
import json
import os
import sys

import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.plot_style import apply_cmame_style
apply_cmame_style()


def _load_config(path):
    with open(path) as f:
        return json.load(f)


def _get_d2_outer_circle(config):
    """Extract Domain2's first add-ellipse center and semi-axes (global coords)."""
    d2_cfg = config.get('Domain2', {})
    offset = np.array(d2_cfg.get('offset', [0.0, 0.0]))
    for shape in d2_cfg.get('shapes', []):
        if shape['type'] == 'ellipse' and shape['operation'] == 'add':
            center = np.array(shape['params']['center']) + offset
            semi_axes = shape['params']['semi_axes']
            return center, semi_axes
    return None, None


def _load_grid_data(grid_dir, use_schwarz, config=None):
    """Load positions and stresses for one grid directory.

    For Schwarz data, points inside Domain2's outer boundary ellipse come from
    Domain2 only; points outside come from Domain1 only.
    If config is None it is loaded from grid_dir/config_backup.json.
    """
    if use_schwarz:
        pos1 = np.load(os.path.join(grid_dir, 'domain1_positions.npy'))
        str1 = np.load(os.path.join(grid_dir, 'domain1_stresses.npy'))
        pos2 = np.load(os.path.join(grid_dir, 'domain2_positions.npy'))
        str2 = np.load(os.path.join(grid_dir, 'domain2_stresses.npy'))

        if config is None:
            config = _load_config(os.path.join(grid_dir, 'config_backup.json'))

        circle_center, semi_axes = _get_d2_outer_circle(config)
        if circle_center is not None:
            a, b = semi_axes[0], semi_axes[1]
            dx = pos1[:, 0] - circle_center[0]
            dy = pos1[:, 1] - circle_center[1]
            inside = (dx / a) ** 2 + (dy / b) ** 2 <= 1.0
            pos1 = pos1[~inside]
            str1 = str1[~inside]

        pos    = np.concatenate([pos1, pos2], axis=0)
        stress = np.concatenate([str1, str2], axis=0)
    else:
        pos    = np.load(os.path.join(grid_dir, 'positions.npy'))
        stress = np.load(os.path.join(grid_dir, 'stresses.npy'))
    return pos, stress


def _extract_circle_params(config, use_schwarz):
    """Parse circle boundaries from config_backup.json."""
    params = {
        'center': None, 'radius': None,
        'd1_hole_center': None, 'd1_hole_radius': None,
        'd2_boundary_center': None, 'd2_boundary_radius': None,
    }
    if use_schwarz:
        domain2_cfg = config.get('Domain2', {})
        domain1_cfg = config.get('Domain1', {})
        d2_offset = np.array(domain2_cfg.get('offset', [0.0, 0.0]))
        d1_offset = np.array(domain1_cfg.get('offset', [0.0, 0.0]))

        for shape in domain2_cfg.get('shapes', []):
            if shape['type'] == 'ellipse' and shape['operation'] == 'add' and params['d2_boundary_center'] is None:
                params['d2_boundary_center'] = np.array(shape['params']['center']) + d2_offset
                params['d2_boundary_radius'] = shape['params']['semi_axes'][0]
            if shape['type'] == 'ellipse' and shape['operation'] == 'change' and params['center'] is None:
                params['center'] = np.array(shape['params']['center']) + d2_offset
                params['radius'] = shape['params']['semi_axes'][0]

        for shape in domain1_cfg.get('shapes', []):
            if shape['type'] == 'ellipse' and shape['operation'] == 'subtract' and params['d1_hole_center'] is None:
                params['d1_hole_center'] = np.array(shape['params']['center']) + d1_offset
                params['d1_hole_radius'] = shape['params']['semi_axes'][0]
    else:
        offset = np.array([0.0, 0.0])
        for shape in config.get('shapes', []):
            if shape['type'] == 'ellipse' and shape['operation'] == 'change' and params['center'] is None:
                params['center'] = np.array(shape['params']['center']) + offset
                params['radius'] = shape['params']['semi_axes'][0]
    return params


def _plot_stress_panel(ax, pos, vals, levels, vmin, vmax, clabel, params):
    """Draw one stress field subplot."""
    plot_xmin, plot_xmax = 0.3, 0.7
    plot_ymin, plot_ymax = 0.3, 0.7

    tcf = ax.tricontourf(pos[:, 0], pos[:, 1], vals,
                         levels=levels, cmap='turbo',
                         vmin=vmin, vmax=vmax, extend='both')
    for c in tcf.collections:
        c.set_edgecolor('face')
    cbar = plt.colorbar(tcf, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(clabel)

    if params.get('center') is not None:
        ax.add_patch(plt.Circle(params['center'], params['radius'],
                                fill=False, linestyle='-', linewidth=1.0,
                                edgecolor='black', zorder=6))
    if params.get('d1_hole_center') is not None:
        ax.add_patch(plt.Circle(params['d1_hole_center'], params['d1_hole_radius'],
                                fill=False, linestyle='-', linewidth=1.0,
                                edgecolor='black', zorder=6))
    if params.get('d2_boundary_center') is not None:
        ax.add_patch(plt.Circle(params['d2_boundary_center'], params['d2_boundary_radius'],
                                fill=False, linestyle='--', linewidth=1.0,
                                edgecolor='black', zorder=6))

    ax.set_xlim(plot_xmin, plot_xmax)
    ax.set_ylim(plot_ymin, plot_ymax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')

    plot_span = plot_xmax - plot_xmin
    raw = plot_span / 4.0
    mag = 10 ** np.floor(np.log10(raw))
    tick_interval = round(raw / mag) * mag
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))


def create_stress_comparison_plots(single_dir, dual_dir, output_dir):
    """Generate 4 groups of 2×3 comparison figures (dual on top, single below).

    Each group corresponds to one resolution level (low → high).
    Within each group, each stress component (σ_xx, σ_yy, σ_xy) shares the
    same colour scale: vmin/vmax taken as the global min/max across both domains.
    """
    plot_xmin, plot_xmax = 0.3, 0.7
    plot_ymin, plot_ymax = 0.3, 0.7

    def _crop(pos):
        return ((pos[:, 0] >= plot_xmin) & (pos[:, 0] <= plot_xmax) &
                (pos[:, 1] >= plot_ymin) & (pos[:, 1] <= plot_ymax))

    components = [
        ((0, 0), r'$\sigma_{xx}$ (Pa)', r'$\sigma_{xx}$'),
        ((1, 1), r'$\sigma_{yy}$ (Pa)', r'$\sigma_{yy}$'),
        ((0, 1), r'$\sigma_{xy}$ (Pa)', r'$\sigma_{xy}$'),
    ]

    dual_grids = sorted([
        int(d.replace('grid_', ''))
        for d in os.listdir(dual_dir)
        if d.startswith('grid_') and os.path.isdir(os.path.join(dual_dir, d))
    ])
    single_grids = sorted([
        int(d.replace('grid_', ''))
        for d in os.listdir(single_dir)
        if d.startswith('grid_') and os.path.isdir(os.path.join(single_dir, d))
    ])

    if len(dual_grids) != len(single_grids):
        print(f'Warning: dual has {len(dual_grids)} grids, single has {len(single_grids)} grids; pairing by sorted order')

    print(f'\nCreating stress comparison plots...')
    print(f'  Dual grids:   {dual_grids}')
    print(f'  Single grids: {single_grids}')

    for dual_gs, single_gs in zip(dual_grids, single_grids):
        dual_grid_dir   = os.path.join(dual_dir,   f'grid_{dual_gs}')
        single_grid_dir = os.path.join(single_dir, f'grid_{single_gs}')

        d_cfg = _load_config(os.path.join(dual_grid_dir,   'config_backup.json'))
        d_pos, d_stress = _load_grid_data(dual_grid_dir,   use_schwarz=True,  config=d_cfg)
        s_pos, s_stress = _load_grid_data(single_grid_dir, use_schwarz=False)

        d_mask = _crop(d_pos)
        s_mask = _crop(s_pos)
        d_pos_c, d_stress_c = d_pos[d_mask], d_stress[d_mask]
        s_pos_c, s_stress_c = s_pos[s_mask], s_stress[s_mask]
        s_cfg = _load_config(os.path.join(single_grid_dir, 'config_backup.json'))
        d_params = _extract_circle_params(d_cfg, use_schwarz=True)
        s_params = _extract_circle_params(s_cfg, use_schwarz=False)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))

        for col_idx, ((i, j), clabel, sym) in enumerate(components):
            d_vals = d_stress_c[:, i, j]
            s_vals = s_stress_c[:, i, j]

            vmin = float(min(d_vals.min(), s_vals.min()))
            vmax = float(max(d_vals.max(), s_vals.max()))
            levels = np.linspace(vmin, vmax, 13)

            ax_dual   = axes[0, col_idx]
            ax_single = axes[1, col_idx]

            _plot_stress_panel(ax_dual,   d_pos_c, d_vals, levels, vmin, vmax, clabel, d_params)
            _plot_stress_panel(ax_single, s_pos_c, s_vals, levels, vmin, vmax, clabel, s_params)

            ax_dual.set_title(f'Dual-domain, {sym}')
            ax_single.set_title(f'Single-domain, {sym}')

        fig.suptitle(f'Stress distribution (dual grid {dual_gs} / single grid {single_gs})')
        plt.tight_layout()
        fname = f'stress_comparison_grid{dual_gs}_{single_gs}.pdf'
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f'  Saved {fname}')


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

# ── Plot 2: Stacked CPU time + Error (secondary axis, broken y-axis) ─────────
BREAK_LOW  = 1000   # lower bound of the removed region
BREAK_HIGH = 4500   # upper bound of the removed region

n     = len(dx_values)
x     = np.arange(n)
bar_w = 0.35

# Single domain bars data
s_solve = np.array(single['solve_time'])
s_other = np.array(single['other_time'])

# Dual domain bars data
d_big   = np.array(dual['big_domain_solve_time'])
d_small = np.array(dual['small_domain_solve_time'])
d_other = np.array(dual['other_time'])

max_time = max((s_solve + s_other).max(), (d_big + d_small + d_other).max())

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 5),
    height_ratios=[1, 3],
    gridspec_kw={'hspace': 0.1},
)

def _draw_bars(ax):
    ax.bar(x - bar_w / 2, s_solve, bar_w,
           label='Single: solve',  color='#b2182b')
    ax.bar(x - bar_w / 2, s_other, bar_w, bottom=s_solve,
           label='Single: other',  color='#e5735c', alpha=0.8)
    ax.bar(x + bar_w / 2, d_big,   bar_w,
           label='Dual: big domain',   color='#3182bd')
    ax.bar(x + bar_w / 2, d_small, bar_w, bottom=d_big,
           label='Dual: small domain', color='#6baed6')
    ax.bar(x + bar_w / 2, d_other, bar_w, bottom=d_big + d_small,
           label='Dual: other',        color='#2ca02c', alpha=0.8)

_draw_bars(ax_top)
_draw_bars(ax_bot)

ax_top.set_ylim(BREAK_HIGH, max_time * 1.08)
ax_bot.set_ylim(0, BREAK_LOW)

# Hide the spines at the cut
ax_top.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)
ax_bot.spines['top'].set_linewidth(0)
ax_top.tick_params(bottom=False, labelbottom=False)
ax_bot.tick_params(top=False, labeltop=False)

# Diagonal break markers

ax_bot.set_xlabel(r'Grid spacing $dx$ (m)')
ax_bot.set_xticks(x)
ax_bot.set_xticklabels([f'{v:.4g}' for v in dx_values])
ax_bot.grid(True, alpha=0.3, axis='y')
ax_top.grid(True, alpha=0.3, axis='y')
fig.text(0.04, 0.5, r'CPU time (s)', va='center', rotation='vertical')

# Error on secondary axis (bottom panel only)
ax2 = ax_bot.twinx()
err_single = np.array(single['integrated_errors_total'])
err_dual   = np.array(dual['integrated_errors_total'])

ax2.plot(x - bar_w / 2, err_single, 'o--', color='#d6604d', linewidth=2,
         markersize=7, markeredgewidth=1.5,
         label='Single: L2 error', zorder=5)
ax2.plot(x + bar_w / 2, err_dual,   's--', color='#4393c3',   linewidth=2,
         markersize=7, markeredgewidth=1.5,
         label='Dual: L2 error',   zorder=5)
ax2.set_ylabel(r'Normalized integrated error (Pa)')

# Combined legend
lines1, labels1 = ax_bot.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_bot.legend(lines1 + lines2, labels1 + labels2,
              loc='upper left', bbox_to_anchor=(0.0, 0.95))

plt.savefig(os.path.join(output_dir, 'time_error_comparison.pdf'))
plt.close()
print('Saved time_error_comparison.pdf')

# ── Plot 3: Stress distribution comparison (dual vs single, per resolution) ──
create_stress_comparison_plots(args.single_dir, args.dual_dir, output_dir)
