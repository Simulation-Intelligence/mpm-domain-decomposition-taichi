#!/usr/bin/env python3
"""
比较 test2 MPM 模拟结果与理论值。

用法示例:
    python experiments/compare_test2_analytical.py
    python experiments/compare_test2_analytical.py --y-center 0.05555 --strip-width 0.002
    python experiments/compare_test2_analytical.py --n-bins 60
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.plot_style import apply_cmame_style
apply_cmame_style()


# ─────────────────────────────────────────────────────────────────────────────
# .rpt 文件解析
# ─────────────────────────────────────────────────────────────────────────────

def read_rpt_file(filepath):
    """读取 Abaqus .rpt 文件，返回 (x_m, values_pa)。

    .rpt 文件格式：
        - 跳过以 'X' 或空白开头的标题行
        - 数据行：两列，用空白分隔
        - X 坐标单位：mm → 除以 1000 转换为 m
        - Stress 单位：MPa → 乘以 1e6 转换为 Pa
    """
    x_vals, v_vals = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 跳过标题行（以字母或非数字字符开头）
            if line[0].isalpha() or line[0] == '-':
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x_vals.append(float(parts[0]))
                v_vals.append(float(parts[1]))
            except ValueError:
                continue

    x_arr = np.array(x_vals) / 1000.0 +0.0055555556  # mm → m
    v_arr = np.array(v_vals) * 1e6       # MPa → Pa
    return x_arr, v_arr


# ─────────────────────────────────────────────────────────────────────────────
# Von Mises 应力计算（参考 tools/visualize_stress.py）
# ─────────────────────────────────────────────────────────────────────────────

def compute_von_mises_2d(stress):
    """2D von Mises 应力：√(σ_xx² + σ_yy² - σ_xx·σ_yy + 3·σ_xy²)

    Parameters
    ----------
    stress : ndarray, shape (N, 2, 2)

    Returns
    -------
    von_mises : ndarray, shape (N,)
    """
    s00 = stress[:, 0, 0]
    s11 = stress[:, 1, 1]
    s01 = stress[:, 0, 1]
    return np.sqrt(s00**2 + s11**2 - s00 * s11 + 3 * s01**2)


# ─────────────────────────────────────────────────────────────────────────────
# MPM 结果加载与过滤
# ─────────────────────────────────────────────────────────────────────────────

def load_mpm_xaxis(result_dir, y_center=None, strip_width=None, x_offset=None):
    """从结果目录加载 stresses.npy / positions.npy，
    筛选 y ∈ [y_center - strip_width, y_center + strip_width] 的粒子，
    按 x 坐标排序后返回应力分量。

    Parameters
    ----------
    result_dir : str
    y_center : float | None
        若为 None，自动取 positions y 坐标的中位数（中线）
    strip_width : float | None
        若为 None，自动取 y 坐标范围的 2%
    x_offset : float | None
        从 x 坐标中减去的偏移量，使 x 从 0 开始。
        若为 None，自动取 positions 中 x 的最小值。

    Returns
    -------
    x : ndarray (M,)
    s11 : ndarray (M,)    σ_xx
    s22 : ndarray (M,)    σ_yy
    von_mises : ndarray (M,)
    n_total : int         过滤前粒子总数
    """
    stresses  = np.load(os.path.join(result_dir, 'stresses.npy'))
    positions = np.load(os.path.join(result_dir, 'positions.npy'))
    n_total = len(positions)

    y_vals = positions[:, 1]

    if y_center is None:
        y_center = float(np.median(y_vals))

    if strip_width is None:
        strip_width = (y_vals.max() - y_vals.min()) * 0.02

    mask = np.abs(y_vals - y_center) <= strip_width
    pos_f  = positions[mask]
    str_f  = stresses[mask]

    if len(pos_f) == 0:
        print(f"  警告: y_center={y_center:.5f}, strip_width={strip_width:.5f} 内无粒子，"
              f"y 范围 [{y_vals.min():.5f}, {y_vals.max():.5f}]")
        return None, None, None, None, n_total

    # 按 x 排序
    idx = np.argsort(pos_f[:, 0])
    x      = pos_f[idx, 0]
    str_f  = str_f[idx]

    # 减去 x 偏移，使 x 从 0 开始
    if x_offset is None:
        x_offset = float(positions[:, 0].min())
    x = x - x_offset

    s11       = str_f[:, 0, 0]
    s22       = str_f[:, 1, 1]
    von_mises = compute_von_mises_2d(str_f)

    print(f"  筛选到 {len(x)}/{n_total} 个粒子 "
          f"(y_center={y_center:.5f}, strip={strip_width:.5f}, "
          f"y∈[{pos_f[idx,1].min():.5f},{pos_f[idx,1].max():.5f}], "
          f"x_offset={x_offset:.6f})")
    return x, s11, s22, von_mises, n_total


# ─────────────────────────────────────────────────────────────────────────────
# Bin 平均
# ─────────────────────────────────────────────────────────────────────────────

def bin_average(x, y, n_bins, x_range=None):
    """将 (x, y) 按 x 方向分 n_bins 个 bin，返回 bin 中心和平均值。

    Returns
    -------
    centers : ndarray (n_bins,)
    means   : ndarray (n_bins,)   空 bin 为 NaN
    counts  : ndarray (n_bins,)
    """
    if x_range is None:
        x_range = (x.min(), x.max())
    edges   = np.linspace(x_range[0], x_range[1], n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    idx     = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    means   = np.full(n_bins, np.nan)
    counts  = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = idx == i
        counts[i] = m.sum()
        if counts[i] > 0:
            means[i] = y[m].mean()
    return centers, means, counts


# ─────────────────────────────────────────────────────────────────────────────
# Grid stress 加载（网格节点应力）
# ─────────────────────────────────────────────────────────────────────────────

def get_domain2_xlim(grid_dir, x_offset=0.0):
    """从 config_backup.json 读取 x 显示范围（减去 x_offset）。

    双域：取 Domain2 的 offset + domain_width。
    单域：无 Domain2 配置，返回 None（让绘图自动用解析解范围）。

    Returns
    -------
    (x_min, x_max) or None if config not found / single domain
    """
    cfg_path = os.path.join(grid_dir, 'config_backup.json')
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    d2 = cfg.get('Domain2', {})
    off_x = d2.get('offset', [0.0, 0.0])[0]
    width = d2.get('domain_width', 0.0)
    if width == 0.0:
        # 单域：无 Domain2，返回 None
        return None
    return (off_x - x_offset, off_x + width - x_offset)


def detect_domain_mode(grid_dir):
    """检测 grid_dir 是单域还是双域结果。

    Returns
    -------
    'single' | 'dual' | None (无法判断)
    """
    if not os.path.isdir(grid_dir):
        return None
    for entry in os.listdir(grid_dir):
        if entry.startswith('single_domain_'):
            return 'single'
        if entry.startswith('schwarz_'):
            return 'dual'
    return None


def find_latest_frame_dir(grid_dir):
    """在 grid_dir/schwarz_* 或 single_domain_*/stress_data/ 中找到编号最大的 frame_* 目录。

    Returns
    -------
    (frame_num, frame_path, is_single) or (None, None, None) if not found
      is_single: True 表示单域（single_domain_*），False 表示双域（schwarz_*）
    """
    for entry in sorted(os.listdir(grid_dir)):
        if entry.startswith('single_domain_'):
            is_single = True
        elif entry.startswith('schwarz_'):
            is_single = False
        else:
            continue
        stress_data_dir = os.path.join(grid_dir, entry, 'stress_data')
        if not os.path.isdir(stress_data_dir):
            continue
        frames = []
        for d in os.listdir(stress_data_dir):
            if d.startswith('frame_'):
                try:
                    frames.append((int(d.split('_')[1]),
                                   os.path.join(stress_data_dir, d)))
                except ValueError:
                    pass
        if frames:
            frame_num, frame_path = max(frames, key=lambda t: t[0])
            return frame_num, frame_path, is_single
    return None, None, None


def load_grid_stress_xline(grid_dir, y_index=None, x_offset=None):
    """从 grid_dir 加载 domain1/domain2 网格应力，提取水平行并合并。

    数据路径: grid_dir/schwarz_*/stress_data/frame_latest/domain{1,2}_grid_stress.npy

    Parameters
    ----------
    grid_dir : str
    y_index : int | None
        提取的行号 j（纵向）。None → 取各 domain 的 ny//2（中线）。
    x_offset : float | None
        从 x 坐标中减去的偏移。None → 自动取 positions.npy 中粒子 x 最小值（与粒子模式一致）。

    Returns
    -------
    x : ndarray (M,)
    s11 : ndarray (M,)   σ_xx
    s22 : ndarray (M,)   σ_yy
    vm  : ndarray (M,)   von Mises
    frame_num : int
    """
    frame_num, frame_dir, is_single = find_latest_frame_dir(grid_dir)
    if frame_dir is None:
        print(f'  警告: {grid_dir} 中未找到 stress_data/frame_* 目录')
        return None, None, None, None, None

    domain_type = '单域' if is_single else '双域'
    print(f'  检测到{domain_type}结果')

    # 自动从 positions.npy 推断 x_offset（与粒子模式行为一致）
    if x_offset is None:
        pos_file = os.path.join(grid_dir, 'positions.npy')
        if os.path.exists(pos_file):
            x_offset = float(np.load(pos_file)[:, 0].min())
            print(f'  grid stress x_offset 自动设为粒子 x 最小值: {x_offset:.6f}')
        else:
            x_offset = 0.0
            print(f'  未找到 positions.npy，x_offset=0')

    all_x, all_s11, all_s22, all_vm = [], [], [], []

    # 单域：文件无 domain 前缀；双域：只取 domain2（对应解析解 x 范围）
    domain_prefixes = [None] if is_single else ['domain2']
    for domain_prefix in domain_prefixes:
        if domain_prefix is None:
            meta_file   = os.path.join(frame_dir, 'grid_stress_meta.json')
            stress_file = os.path.join(frame_dir, 'grid_stress.npy')
            mass_file   = os.path.join(frame_dir, 'grid_mass.npy')
        else:
            meta_file   = os.path.join(frame_dir, f'{domain_prefix}_grid_stress_meta.json')
            stress_file = os.path.join(frame_dir, f'{domain_prefix}_grid_stress.npy')
            mass_file   = os.path.join(frame_dir, f'{domain_prefix}_grid_mass.npy')

        if not os.path.exists(stress_file):
            continue
        if not os.path.exists(meta_file):
            label = domain_prefix if domain_prefix else '单域'
            print(f'  警告: 找不到 {meta_file}，跳过 {label}')
            continue

        with open(meta_file) as f:
            meta = json.load(f)

        nx     = meta['nx']
        ny     = meta['ny']
        off_x  = meta['offset'][0]
        dx_x   = meta['dx_x']

        j = (ny // 2) if y_index is None else int(y_index)
        label = domain_prefix if domain_prefix else '单域'
        if j < 0 or j >= ny:
            print(f'  警告: y_index={j} 超出 {label} 范围 [0, {ny-1}]，跳过')
            continue

        grid_stress = np.load(stress_file)    # (nx, ny, 2, 2)
        stress_row  = grid_stress[:, j, :, :]  # (nx, 2, 2)

        # valid mask from grid mass
        valid_mask = np.ones(nx, dtype=bool)
        if os.path.exists(mass_file):
            grid_mass  = np.load(mass_file)     # (nx, ny)
            valid_mask = grid_mass[:, j] > 1e-10

        x_coords = off_x + np.arange(nx) * dx_x
        if x_offset is not None:
            x_coords = x_coords - x_offset

        x_v  = x_coords[valid_mask] + 0.0021
        s_v  = stress_row[valid_mask] 

        all_x.append(x_v)
        all_s11.append(s_v[:, 0, 0])
        all_s22.append(s_v[:, 1, 1])
        all_vm.append(compute_von_mises_2d(s_v))

        print(f'  {label}: nx={nx}, ny={ny}, j={j}, '
              f'有效点 {valid_mask.sum()}/{nx}, '
              f'x=[{x_v.min():.5f}, {x_v.max():.5f}]')

    if not all_x:
        return None, None, None, None, frame_num

    x_arr  = np.concatenate(all_x)
    s11_arr = np.concatenate(all_s11)
    s22_arr = np.concatenate(all_s22)
    vm_arr  = np.concatenate(all_vm)

    idx = np.argsort(x_arr)
    return x_arr[idx], s11_arr[idx], s22_arr[idx], vm_arr[idx], frame_num


# ─────────────────────────────────────────────────────────────────────────────
# 主绘图逻辑
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ['#3182bd', '#2ca02c', '#8c510a', '#756bb1',
          '#e5735c', '#636363', '#1f77b4']

COMPONENT_CFG = [
    # (key,      ylabel,                    s11_col, s22_col, vm_col)
    ('s11',      r'$\sigma_{11}$ (Pa)',    True,  False, False),
    ('s22',      r'$\sigma_{22}$ (Pa)',    False, True,  False),
    ('vonmises', r'von Mises stress (Pa)', False, False, True),
]


def plot_comparison(ax, analytical_x, analytical_y,
                    mpm_data_list, component_key, n_bins, xlim=None):
    """在 ax 上绘制解析解 + 各网格 MPM 结果。

    mpm_data_list : list of (grid_size, x, s11, s22, vm)
    xlim : (x_min, x_max) | None  若提供则限制 bin 范围并设置坐标轴
    """
    ax.plot(analytical_x, analytical_y, color='#b2182b', linewidth=2.0, alpha=0.9,
            label='Analytical', zorder=10)

    for i, (gs, x, s11, s22, vm) in enumerate(mpm_data_list):
        if x is None:
            continue

        if component_key == 's11':
            y_mpm = s11
        elif component_key == 's22':
            y_mpm = s22
        else:
            y_mpm = vm

        color = COLORS[i % len(COLORS)]
        label = f'MPM grid={gs}'

        if n_bins and n_bins > 0:
            x_range = xlim if xlim else (analytical_x.min(), analytical_x.max())
            bx, by, bc = bin_average(x, y_mpm, n_bins, x_range=x_range)
            valid = bc > 0
            ax.plot(bx[valid], by[valid], '-o', color=color,
                    linewidth=1.5, markersize=5, markerfacecolor='none',
                    markeredgewidth=1.2, alpha=0.9, label=label, zorder=5)
        else:
            ax.plot(x, y_mpm, '-o', color=color,
                    linewidth=1.5, markersize=5, markerfacecolor='none',
                    markeredgewidth=1.2, alpha=0.9, label=label, zorder=5)

    if xlim:
        ax.set_xlim(*xlim)


def make_plots(grid_data, ana, output_dir, n_bins, separate, xlim=None):
    """生成图像文件。

    Parameters
    ----------
    grid_data : list of (grid_size, x, s11, s22, von_mises)
    ana : dict with keys 's11', 's22', 'vonmises' → (x_arr, y_arr)
    separate : bool  若 True 则每个分量单独保存一张图
    xlim : (x_min, x_max) | None  横坐标显示范围（默认 Domain2 x 范围）
    """
    os.makedirs(output_dir, exist_ok=True)

    components = [
        ('s11',      r'$\sigma_{11}$ (Pa)'),
        ('s22',      r'$\sigma_{22}$ (Pa)'),
        ('vonmises', r'von Mises stress (Pa)'),
    ]

    available = [(key, ylabel) for key, ylabel in components if key in ana]

    if separate:
        for key, ylabel in available:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.set_xlabel(r'$x$ (m)')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            ax_x, ax_y = ana[key]
            plot_comparison(ax, ax_x, ax_y, grid_data, key, n_bins, xlim=xlim)

            ax.legend()
            plt.tight_layout()
            out = os.path.join(output_dir, f'test2_compare_{key}.pdf')
            plt.savefig(out)
            plt.close()
            print(f'Saved: {out}')
    else:
        n = len(available)
        if n == 0:
            return
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n + 1, 4.5))
        if n == 1:
            axes = [axes]

        for ax, (key, ylabel) in zip(axes, available):
            ax.set_xlabel(r'$x$ (m)')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            ax_x, ax_y = ana[key]
            plot_comparison(ax, ax_x, ax_y, grid_data, key, n_bins, xlim=xlim)
            ax.legend()

        plt.tight_layout()
        out = os.path.join(output_dir, 'test2_compare_all.pdf')
        plt.savefig(out)
        plt.close()
        print(f'Saved: {out}')


def make_grid_stress_plots(grid_data, ana, output_dir, n_bins, separate, xlim=None):
    """生成网格节点应力对比图像文件。

    Parameters
    ----------
    grid_data : list of (grid_size, x, s11, s22, von_mises, frame_num)
    ana : dict with keys 's11', 's22', 'vonmises' → (x_arr, y_arr)
    separate : bool
    xlim : (x_min, x_max) | None
    """
    os.makedirs(output_dir, exist_ok=True)

    components = [
        ('s11',      r'$\sigma_{11}$ (Pa)'),
        ('s22',      r'$\sigma_{22}$ (Pa)'),
        ('vonmises', r'von Mises stress (Pa)'),
    ]

    # Repack to the (gs, x, s11, s22, vm) format expected by plot_comparison
    plot_data = [(gs, x, s11, s22, vm) for gs, x, s11, s22, vm, _ in grid_data]

    available = [(key, ylabel) for key, ylabel in components if key in ana]

    if separate:
        for key, ylabel in available:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.set_xlabel(r'$x$ (m)')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            ax_x, ax_y = ana[key]
            plot_comparison(ax, ax_x, ax_y, plot_data, key, n_bins, xlim=xlim)

            ax.legend()
            plt.tight_layout()
            out = os.path.join(output_dir, f'test2_grid_stress_compare_{key}.pdf')
            plt.savefig(out)
            plt.close()
            print(f'Saved: {out}')
    else:
        n = len(available)
        if n == 0:
            return
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n + 1, 4.5))
        if n == 1:
            axes = [axes]

        for ax, (key, ylabel) in zip(axes, available):
            ax.set_xlabel(r'$x$ (m)')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            ax_x, ax_y = ana[key]
            plot_comparison(ax, ax_x, ax_y, plot_data, key, n_bins, xlim=xlim)
            ax.legend()

        plt.tight_layout()
        out = os.path.join(output_dir, 'test2_grid_stress_compare_all.pdf')
        plt.savefig(out)
        plt.close()
        print(f'Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='比较 test2 MPM 结果与理论值（s11/s22/von Mises）')

    parser.add_argument('--results-dir', default='useful_results/test2_40-100',
                        help='MPM 结果根目录，包含 grid_40/ grid_60/ 等子目录')
    parser.add_argument('--analytical-dir', default='useful_results/test2_analytical',
                        help='双域理论值目录，含 s11.rpt / s22.rpt / vonmises.rpt')
    parser.add_argument('--analytical-dir-single', default=None,
                        help='单域理论值目录（未指定则与 --analytical-dir 相同）')
    parser.add_argument('--output-dir', default='experiments/test2_comparison',
                        help='图像输出目录')

    parser.add_argument('--grid-sizes', nargs='+', type=int, default=None,
                        help='指定要比较的网格大小（默认自动扫描所有 grid_* 子目录）')

    parser.add_argument('--y-center', type=float, default=None,
                        help='X 轴对应的 y 坐标（m）。'
                             '默认自动取粒子 y 坐标中位数（约 0.05555 m）')
    parser.add_argument('--strip-width', type=float, default=None,
                        help='X 轴附近筛选半宽（m）。'
                             '默认取 y 坐标范围的 2%%')

    parser.add_argument('--n-bins', type=int, default=None,
                        help='按 x 方向分 bin 并计算平均值（默认：直接散点绘制）')
    parser.add_argument('--x-offset', type=float, default=None,
                        help='从 MPM x 坐标中减去的固定偏移（m）。'
                             '默认自动取粒子 x 最小值（≈0.006108）')
    parser.add_argument('--separate', action='store_true',
                        help='为每个应力分量单独生成一张图（默认：三合一图）')

    parser.add_argument('--grid-stress', action='store_true',
                        help='额外生成网格节点应力对比图（读取 domain1/2_grid_stress.npy）')
    parser.add_argument('--grid-y-index', type=int, default=None,
                        help='从网格应力中提取的行号 j（纵向索引）。'
                             '默认取各 domain 的 ny//2（中线行）')

    args = parser.parse_args()

    # ── 发现网格目录 ────────────────────────────────────────────────────────
    results_dir = args.results_dir
    if args.grid_sizes:
        grid_dirs = {gs: os.path.join(results_dir, f'grid_{gs}') for gs in args.grid_sizes}
    else:
        grid_dirs = {}
        for d in sorted(os.listdir(results_dir)):
            if d.startswith('grid_'):
                try:
                    gs = int(d.split('_')[1])
                    grid_dirs[gs] = os.path.join(results_dir, d)
                except ValueError:
                    continue

    if not grid_dirs:
        raise RuntimeError(f'在 {results_dir} 中未找到任何 grid_* 子目录')

    print(f'发现网格: {sorted(grid_dirs.keys())}')

    # ── 确定 x_offset（统一用于粒子模式和 grid stress 模式）────────────────────
    x_offset = args.x_offset
    if x_offset is None:
        first_dir = grid_dirs[sorted(grid_dirs.keys())[0]]
        pos_file = os.path.join(first_dir, 'positions.npy')
        if os.path.exists(pos_file):
            x_offset = float(np.load(pos_file)[:, 0].min())
            print(f'x_offset 自动设为粒子 x 最小值: {x_offset:.6f} m')
        else:
            x_offset = 0.0

    # ── 检测单/双域，选择对应解析解目录 ─────────────────────────────────────
    first_dir = grid_dirs[sorted(grid_dirs.keys())[0]]
    domain_mode = detect_domain_mode(first_dir)
    if domain_mode == 'single':
        ana_dir = args.analytical_dir_single or args.analytical_dir
        print(f'检测到单域结果，使用解析解目录: {ana_dir}')
    else:
        ana_dir = args.analytical_dir
        if domain_mode == 'dual':
            print(f'检测到双域结果，使用解析解目录: {ana_dir}')
        else:
            print(f'无法检测域类型，使用默认解析解目录: {ana_dir}')

    # ── 读取理论值 ──────────────────────────────────────────────────────────
    print('读取理论值...')
    ana = {}
    for key, fname in [('s11', 's11.rpt'), ('s22', 's22.rpt'), ('vonmises', 'vonmises.rpt')]:
        fpath = os.path.join(ana_dir, fname)
        if not os.path.exists(fpath):
            print(f'  跳过 {fname}（文件不存在）')
            continue
        x_a, y_a = read_rpt_file(fpath)
        ana[key] = (x_a, y_a)
        print(f'  {fname}: {len(x_a)} 个点, x=[{x_a.min():.4f}, {x_a.max():.4f}] m, '
              f'val=[{y_a.min():.2e}, {y_a.max():.2e}] Pa')

    # ── 确定 Domain2 x 显示范围 ──────────────────────────────────────────────
    domain2_xlim = get_domain2_xlim(first_dir, x_offset)
    if domain2_xlim:
        print(f'Domain2 x 范围（xlim）: [{domain2_xlim[0]:.5f}, {domain2_xlim[1]:.5f}] m')

    # ── 加载 MPM 数据 ────────────────────────────────────────────────────────
    grid_data = []  # list of (grid_size, x, s11, s22, von_mises)
    for gs in sorted(grid_dirs.keys()):
        d = grid_dirs[gs]
        if not os.path.isdir(d):
            print(f'跳过 grid_{gs}: 目录不存在 ({d})')
            continue
        print(f'加载 grid_{gs}: {d}')
        x, s11, s22, vm, _ = load_mpm_xaxis(
            d,
            y_center=args.y_center,
            strip_width=args.strip_width,
            x_offset=x_offset,
        )
        grid_data.append((gs, x, s11, s22, vm))

    # ── 粒子应力绘图 ──────────────────────────────────────────────────────────
    make_plots(
        grid_data=grid_data,
        ana=ana,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        separate=args.separate,
        xlim=domain2_xlim,
    )

    # ── 网格节点应力绘图（可选）────────────────────────────────────────────────
    if args.grid_stress:
        print('\n加载 grid stress 数据（网格节点应力）...')
        grid_stress_data = []  # list of (grid_size, x, s11, s22, vm, frame_num)
        for gs in sorted(grid_dirs.keys()):
            d = grid_dirs[gs]
            if not os.path.isdir(d):
                continue
            print(f'加载 grid_{gs} grid stress: {d}')
            x, s11, s22, vm, fn = load_grid_stress_xline(
                d,
                y_index=args.grid_y_index,
                x_offset=x_offset,
            )
            grid_stress_data.append((gs, x, s11, s22, vm, fn))

        make_grid_stress_plots(
            grid_data=grid_stress_data,
            ana=ana,
            output_dir=args.output_dir,
            n_bins=args.n_bins,
            separate=args.separate,
            xlim=domain2_xlim,
        )


if __name__ == '__main__':
    main()
