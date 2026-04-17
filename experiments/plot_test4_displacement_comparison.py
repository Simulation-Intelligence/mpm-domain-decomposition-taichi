#!/usr/bin/env python3
"""Plot cantilever displacement profiles for single- and dual-domain runs."""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.plot_style import COLOR_DUAL, COLOR_MPM, apply_cmame_style

apply_cmame_style()


DEFAULT_SINGLE_DIR = 'useful_results/test4_displacement_test/test_4_single'
DEFAULT_DUAL_DIR = 'useful_results/test4_displacement_test/test_4_schwarz'
DEFAULT_OUTPUT = 'useful_results/test4_displacement_test/displacement_comparison_x_over_L.pdf'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot normalized cantilever displacement profiles for single and dual domains.'
    )
    parser.add_argument(
        '--single-dir',
        default=DEFAULT_SINGLE_DIR,
        help='Directory containing single-domain positions.npy',
    )
    parser.add_argument(
        '--dual-dir',
        default=DEFAULT_DUAL_DIR,
        help='Directory containing dual-domain domain1_positions.npy and domain2_positions.npy',
    )
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help='Output figure path',
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=80,
        help='Number of x bins used to extract the lowest-point y profile in each bin',
    )
    parser.add_argument(
        '--x-axis',
        choices=('normalized', 'physical'),
        default='normalized',
        help='Use normalized x/L or physical x coordinate on the horizontal axis',
    )
    parser.add_argument(
        '--x-min',
        type=float,
        default=None,
        help='Optional lower bound for shared binning range',
    )
    parser.add_argument(
        '--x-max',
        type=float,
        default=None,
        help='Optional upper bound for shared binning range',
    )
    return parser.parse_args()


def load_single_positions(single_dir):
    path = os.path.join(single_dir, 'positions.npy')
    return np.load(path)


def load_dual_positions(dual_dir):
    path1 = os.path.join(dual_dir, 'domain1_positions.npy')
    path2 = os.path.join(dual_dir, 'domain2_positions.npy')
    pos1 = np.load(path1)
    pos2 = np.load(path2)
    return np.concatenate([pos1, pos2], axis=0), pos1, pos2


def resolve_x_range(single_positions, dual_positions, x_min=None, x_max=None):
    inferred_min = min(single_positions[:, 0].min(), dual_positions[:, 0].min())
    inferred_max = max(single_positions[:, 0].max(), dual_positions[:, 0].max())
    x_min = inferred_min if x_min is None else x_min
    x_max = inferred_max if x_max is None else x_max
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError('x-range must be finite')
    if x_max <= x_min:
        raise ValueError(f'Invalid x-range: x_min={x_min}, x_max={x_max}')
    return float(x_min), float(x_max)


def compute_binned_bottom_curve(positions, x_min, x_max, bins):
    if bins <= 0:
        raise ValueError('--bins must be positive')

    in_range = (positions[:, 0] >= x_min) & (positions[:, 0] <= x_max)
    filtered = positions[in_range]
    if filtered.size == 0:
        raise ValueError('No particles remain inside the requested x-range')

    edges = np.linspace(x_min, x_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.clip(np.digitize(filtered[:, 0], edges) - 1, 0, bins - 1)

    counts = np.bincount(bin_ids, minlength=bins)
    min_y = np.full(bins, np.inf, dtype=np.float64)
    np.minimum.at(min_y, bin_ids, filtered[:, 1])
    bottom_y = np.full(bins, np.nan, dtype=np.float64)
    valid = counts > 0
    bottom_y[valid] = min_y[valid]

    return centers, bottom_y, valid, counts


def normalize_displacement(profile_y, valid_mask):
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size < 2:
        raise ValueError('Need at least two non-empty bins to normalize displacement')

    left_idx = valid_idx[0]
    right_idx = valid_idx[-1]
    y_left = profile_y[left_idx]
    y_right = profile_y[right_idx]
    denom = abs(y_right - y_left)
    if denom <= 1e-14:
        raise ValueError('Right-end and left-end profile y are identical; cannot normalize displacement')

    disp = (profile_y - y_left) / denom
    return disp, float(y_left), float(y_right)


def format_x_values(centers, x_min, x_max, axis_mode):
    if axis_mode == 'physical':
        return centers

    length = x_max - x_min
    if length <= 0.0:
        raise ValueError('Beam length must be positive for normalized x-axis')
    return (centers - x_min) / length


def plot_curves(single_x, single_disp, single_valid, dual_x, dual_disp, dual_valid, output_path, axis_mode):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.plot(
        single_x[single_valid],
        single_disp[single_valid],
        color=COLOR_MPM,
        linewidth=2.2,
        label='Single domain',
    )
    ax.plot(
        dual_x[dual_valid],
        dual_disp[dual_valid],
        color=COLOR_DUAL,
        linewidth=2.2,
        label='Dual domain',
    )

    ax.set_xlabel(r'$x/L$' if axis_mode == 'normalized' else r'$x$')
    ax.set_ylabel(r'$\bar{w}$')
    ax.minorticks_on()
    ax.grid(True, which='major', alpha=0.25, linewidth=0.6)
    ax.legend()

    ymin = min(np.nanmin(single_disp[single_valid]), np.nanmin(dual_disp[dual_valid]))
    ax.set_ylim(ymin - 0.05, 0.05)
    if axis_mode == 'normalized':
        ax.set_xlim(0.0, 1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def summarize_curve(name, x_values, disp, valid_mask, y_left, y_right, counts):
    valid_x = x_values[valid_mask]
    valid_disp = disp[valid_mask]
    print(
        f'{name}: bins={valid_mask.sum()}/{len(valid_mask)}, '
        f'x_range=[{valid_x[0]:.6f}, {valid_x[-1]:.6f}], '
        f'left_y={y_left:.6f}, right_y={y_right:.6f}, '
        f'disp_start={valid_disp[0]:.6f}, disp_end={valid_disp[-1]:.6f}, '
        f'min_bin_count={counts[valid_mask].min()}, max_bin_count={counts[valid_mask].max()}'
    )


def main():
    args = parse_args()

    single_positions = load_single_positions(args.single_dir)
    dual_positions, dual_domain1, dual_domain2 = load_dual_positions(args.dual_dir)

    x_min, x_max = resolve_x_range(
        single_positions,
        dual_positions,
        x_min=args.x_min,
        x_max=args.x_max,
    )

    single_centers, single_profile_y, single_valid, single_counts = compute_binned_bottom_curve(
        single_positions, x_min, x_max, args.bins
    )
    dual_centers, dual_profile_y, dual_valid, dual_counts = compute_binned_bottom_curve(
        dual_positions, x_min, x_max, args.bins
    )

    single_disp, single_left_y, single_right_y = normalize_displacement(single_profile_y, single_valid)
    dual_disp, dual_left_y, dual_right_y = normalize_displacement(dual_profile_y, dual_valid)

    single_x = format_x_values(single_centers, x_min, x_max, args.x_axis)
    dual_x = format_x_values(dual_centers, x_min, x_max, args.x_axis)

    plot_curves(
        single_x, single_disp, single_valid,
        dual_x, dual_disp, dual_valid,
        args.output, args.x_axis,
    )

    print(
        f'Loaded particles: single={len(single_positions)}, '
        f'dual={len(dual_positions)} (domain1={len(dual_domain1)} + domain2={len(dual_domain2)})'
    )
    print(f'Using shared x-range [{x_min:.6f}, {x_max:.6f}] with {args.bins} bins')
    summarize_curve('Single domain', single_x, single_disp, single_valid, single_left_y, single_right_y, single_counts)
    summarize_curve('Dual domain', dual_x, dual_disp, dual_valid, dual_left_y, dual_right_y, dual_counts)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
