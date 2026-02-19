#!/usr/bin/env python3
"""
实验2: 域应力分析实验
基于 experiment_1，但不进行解析解对比，专注于保存和分析Domain2的应力分布。
额外保存Domain2最中间沿着X轴和Y轴的应力分布。
"""

import json
import numpy as np
import sys
import os
import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid segfault with Taichi
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulators.implicit_mpm import ImplicitMPM
from Util.Config import Config
import taichi as ti

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path):
    """保存配置文件"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def modify_config_grid_size(config, grid_size, use_schwarz=False):
    """
    修改配置文件的网格分辨率

    参数:
        config: 配置字典
        grid_size: 新的网格大小
        use_schwarz: 是否为Schwarz双域配置

    返回:
        修改后的配置字典
    """
    import copy
    new_config = copy.deepcopy(config)

    if use_schwarz:
        # Schwarz模式：只修改Domain2的网格大小
        if 'Domain2' in new_config:
            original_nx = new_config['Domain2'].get('grid_nx', 30)
            original_ny = new_config['Domain2'].get('grid_ny', 30)
            aspect_ratio = original_ny / original_nx if original_nx > 0 else 1.0

            new_config['Domain2']['grid_nx'] = grid_size
            new_config['Domain2']['grid_ny'] = int(grid_size * aspect_ratio)

            print(f"修改Schwarz配置: Domain2网格大小设为 {grid_size}x{int(grid_size * aspect_ratio)} (比例: {aspect_ratio:.3f})")
        else:
            print("警告: Schwarz配置中未找到Domain2")
    else:
        # 原有单域逻辑
        # 获取原始网格大小和宽高比
        original_nx = new_config.get('grid_nx', 100)
        original_ny = new_config.get('grid_ny', 100)
        aspect_ratio = original_ny / original_nx if original_nx > 0 else 1.0

        # 设置新的网格大小
        new_config['grid_nx'] = grid_size
        new_config['grid_ny'] = int(grid_size * aspect_ratio)

        print(f"修改配置: 网格大小设为 {grid_size}x{int(grid_size * aspect_ratio)} (比例: {aspect_ratio:.3f})")

    return new_config

def run_simulation(config_path, use_schwarz=False):
    """
    通过subprocess运行模拟，避免Taichi和matplotlib冲突
    """
    import subprocess
    import glob

    print(f"运行模拟: {config_path}")

    if use_schwarz:
        # 调用 Schwarz 模拟器
        cmd = [
            sys.executable,  # 使用当前Python解释器
            "simulators/implicit_mpm_schwarz.py",
            "--config", config_path,
        ]
        print(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        # 退出码-11是segfault，通常发生在模拟完成、数据保存后的绘图阶段
        # 只要数据保存成功就继续
        if result.returncode == -11:
            print("警告: 进程以segfault结束（退出码-11），这通常发生在性能统计绘图时")
            print("检查数据文件是否已保存...")
        elif result.returncode != 0:
            print("模拟失败！")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Schwarz simulation failed with code {result.returncode}")

        print("加载结果数据...")

        # 查找最新的实验结果目录
        experiment_dirs = glob.glob("experiment_results/schwarz_*")
        if not experiment_dirs:
            print("错误: 未找到模拟结果目录")
            if result.returncode == -11:
                print("segfault发生在数据保存之前")
            raise RuntimeError("未找到模拟结果目录")

        latest_dir = max(experiment_dirs, key=os.path.getmtime)
        print(f"从目录加载数据: {latest_dir}")

        # 查找最新的帧目录
        frame_dirs = glob.glob(f"{latest_dir}/stress_data/frame_*")
        if not frame_dirs:
            raise RuntimeError(f"未找到应力数据: {latest_dir}/stress_data/")

        latest_frame_dir = max(frame_dirs, key=lambda x: int(x.split('_')[-1]))
        print(f"加载帧数据: {latest_frame_dir}")

        # 加载Domain1和Domain2数据
        positions1 = np.load(f"{latest_frame_dir}/domain1_positions.npy")
        stresses1 = np.load(f"{latest_frame_dir}/domain1_stress.npy")
        positions2 = np.load(f"{latest_frame_dir}/domain2_positions.npy")
        stresses2 = np.load(f"{latest_frame_dir}/domain2_stress.npy")

        print(f"Domain1: {len(positions1)} particles")
        print(f"Domain2: {len(positions2)} particles")

        # Domain2位置已经是全局坐标（implicit_mpm_schwarz.py中已处理）
        # 但为了确保，我们从配置中读取offset并验证
        cfg = load_config(config_path)
        domain2_offset = np.array(cfg['Domain2']['offset'])

        # positions2 已经在保存时加上了offset，所以这里不需要再加
        # 验证一下positions2的范围是否合理
        print(f"Domain2位置范围: x=[{positions2[:, 0].min():.3f}, {positions2[:, 0].max():.3f}], y=[{positions2[:, 1].min():.3f}, {positions2[:, 1].max():.3f}]")

        return (positions1, stresses1, positions2, stresses2)
    else:
        # 单域模式：调用 implicit_mpm.py
        cmd = [
            sys.executable,
            "simulators/implicit_mpm.py",
            "--config", config_path
        ]
        print(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("模拟失败！")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Simulation failed with code {result.returncode}")

        print("模拟完成，加载结果数据...")

        # 单域模式需要从experiment_results/single_domain_<timestamp>加载
        # 找到最新的实验目录
        exp_results_dir = "experiment_results"
        single_domain_dirs = [d for d in os.listdir(exp_results_dir)
                             if d.startswith("single_domain_") and
                             os.path.isdir(os.path.join(exp_results_dir, d))]

        if not single_domain_dirs:
            raise RuntimeError("未找到单域模式的实验结果目录")

        # 按时间戳排序，获取最新的
        single_domain_dirs.sort()
        latest_exp_dir = os.path.join(exp_results_dir, single_domain_dirs[-1])
        print(f"加载实验结果: {latest_exp_dir}")

        # 查找stress_data目录中的最后一帧
        stress_data_dir = os.path.join(latest_exp_dir, "stress_data")
        if not os.path.exists(stress_data_dir):
            raise RuntimeError(f"未找到应力数据目录: {stress_data_dir}")

        # 获取所有frame目录
        frame_dirs = [d for d in os.listdir(stress_data_dir)
                     if d.startswith("frame_") and
                     os.path.isdir(os.path.join(stress_data_dir, d))]

        if not frame_dirs:
            raise RuntimeError(f"未找到帧数据目录: {stress_data_dir}")

        # 按帧号排序，获取最后一帧
        frame_numbers = [int(d.replace("frame_", "")) for d in frame_dirs]
        latest_frame = max(frame_numbers)
        latest_frame_dir = os.path.join(stress_data_dir, f"frame_{latest_frame}")

        print(f"加载帧数据: {latest_frame_dir}")

        # 加载positions和stress数据
        positions = np.load(os.path.join(latest_frame_dir, "positions.npy"))
        stresses = np.load(os.path.join(latest_frame_dir, "stress.npy"))

        print(f"加载了 {len(positions)} 个粒子的数据")

        return (positions, stresses)

def merge_schwarz_domains(positions1, stresses1, positions2, stresses2, config):
    """
    合并Schwarz双域数据

    参数:
        positions1, stresses1: Domain1的位置和应力
        positions2, stresses2: Domain2的位置和应力
        config: 配置字典

    返回:
        tuple: (positions, stresses) 合并后的数据
    """
    # 获取Domain2的范围
    domain2_config = config.get('Domain2', {})
    domain2_offset = np.array(domain2_config.get('offset', [0.35, 0.35]))
    domain2_width = domain2_config.get('domain_width', 0.3)
    domain2_height = domain2_config.get('domain_height', 0.3)

    # Domain2的全局范围
    d2_xmin, d2_xmax = domain2_offset[0], domain2_offset[0] + domain2_width
    d2_ymin, d2_ymax = domain2_offset[1], domain2_offset[1] + domain2_height

    # 从Domain1中排除Domain2范围内的粒子
    mask_outside_d2 = ~((positions1[:, 0] >= d2_xmin) & (positions1[:, 0] <= d2_xmax) &
                        (positions1[:, 1] >= d2_ymin) & (positions1[:, 1] <= d2_ymax))

    positions1_filtered = positions1[mask_outside_d2]
    stresses1_filtered = stresses1[mask_outside_d2]

    # 合并Domain1(过滤后) + Domain2
    positions = np.vstack([positions1_filtered, positions2])
    stresses = np.vstack([stresses1_filtered, stresses2])

    return positions, stresses

def extract_domain2_cross_sections(positions2, stresses2, config, use_schwarz=True):
    """
    提取Domain2（或单域）中间沿X轴和Y轴的应力分布

    参数:
        positions2: Domain2的粒子位置（全局坐标）或单域的粒子位置
        stresses2: Domain2的应力数据或单域的应力数据
        config: 配置字典
        use_schwarz: 是否为Schwarz模式

    返回:
        dict: 包含X轴和Y轴截面数据的字典
    """
    # 获取配置
    if use_schwarz:
        domain_config = config.get('Domain2', {})
        domain_offset = np.array(domain_config.get('offset', [0.3, 0.3]))
        domain_width = domain_config.get('domain_width', 0.4)
        domain_height = domain_config.get('domain_height', 0.4)
        grid_nx = domain_config.get('grid_nx', 40)
        grid_ny = domain_config.get('grid_ny', 40)
        domain_name = "Domain2"
    else:
        domain_config = config
        domain_offset = np.array([0.0, 0.0])
        domain_width = domain_config.get('domain_width', 1.0)
        domain_height = domain_config.get('domain_height', 0.4)
        grid_nx = domain_config.get('grid_nx', 100)
        grid_ny = domain_config.get('grid_ny', 40)
        domain_name = "Domain"

    # 计算域的中心（全局坐标）
    center_x = domain_offset[0] + domain_width / 2.0
    center_y = domain_offset[1] + domain_height / 2.0

    print(f"{domain_name}中心: ({center_x:.3f}, {center_y:.3f})")

    # 计算网格间距和截面宽度
    dx = domain_width / grid_nx
    dy = domain_height / grid_ny
    strip_width_x = 0.2 * dx  # X方向截面宽度（垂直于Y轴）
    strip_width_y = 0.2 * dy  # Y方向截面宽度（垂直于X轴）

    # 提取沿X轴的截面（Y = center_y，水平线）
    mask_x_axis = np.abs(positions2[:, 1] - center_y) < strip_width_y
    x_axis_positions = positions2[mask_x_axis]
    x_axis_stresses = stresses2[mask_x_axis]

    # 按X坐标排序
    sort_idx_x = np.argsort(x_axis_positions[:, 0])
    x_axis_x = x_axis_positions[sort_idx_x, 0]
    x_axis_y = x_axis_positions[sort_idx_x, 1]
    x_axis_sig_xx = x_axis_stresses[sort_idx_x, 0, 0]
    x_axis_sig_yy = x_axis_stresses[sort_idx_x, 1, 1]
    x_axis_sig_xy = x_axis_stresses[sort_idx_x, 0, 1]

    print(f"X轴截面: {len(x_axis_x)} 个粒子")

    # 提取沿Y轴的截面（X = center_x，垂直线）
    mask_y_axis = np.abs(positions2[:, 0] - center_x) < strip_width_x
    y_axis_positions = positions2[mask_y_axis]
    y_axis_stresses = stresses2[mask_y_axis]

    # 按Y坐标排序
    sort_idx_y = np.argsort(y_axis_positions[:, 1])
    y_axis_x = y_axis_positions[sort_idx_y, 0]
    y_axis_y = y_axis_positions[sort_idx_y, 1]
    y_axis_sig_xx = y_axis_stresses[sort_idx_y, 0, 0]
    y_axis_sig_yy = y_axis_stresses[sort_idx_y, 1, 1]
    y_axis_sig_xy = y_axis_stresses[sort_idx_y, 0, 1]

    print(f"Y轴截面: {len(y_axis_y)} 个粒子")

    return {
        'center': (center_x, center_y),
        'x_axis': {
            'x': x_axis_x,
            'y': x_axis_y,
            'sig_xx': x_axis_sig_xx,
            'sig_yy': x_axis_sig_yy,
            'sig_xy': x_axis_sig_xy
        },
        'y_axis': {
            'x': y_axis_x,
            'y': y_axis_y,
            'sig_xx': y_axis_sig_xx,
            'sig_yy': y_axis_sig_yy,
            'sig_xy': y_axis_sig_xy
        }
    }

def analyze_and_plot(positions_or_tuple, stresses_or_none, config, output_dir, grid_size=None, use_schwarz=False):
    """
    分析和绘制结果（不包含解析解对比）

    参数:
        positions_or_tuple: 单域模式下为positions数组，Schwarz模式下为(pos1, stress1, pos2, stress2)元组
        stresses_or_none: 单域模式下为stresses数组，Schwarz模式下为None
        config: 配置字典
        output_dir: 输出目录
        grid_size: 网格大小（用于标注），可选
        use_schwarz: 是否为Schwarz模式

    返回:
        dict: 包含plot_path, positions, stresses等信息
    """
    # 处理Schwarz双域数据
    if use_schwarz:
        positions1, stresses1, positions2, stresses2 = positions_or_tuple
        positions, stresses = merge_schwarz_domains(positions1, stresses1, positions2, stresses2, config)

        # 清理原始域数据，只保留合并后的数据
        del positions1, stresses1
        gc.collect()

        # 提取Domain2的截面数据
        cross_sections = extract_domain2_cross_sections(positions2, stresses2, config, use_schwarz=True)

        # 清理positions2和stresses2，已经提取了截面数据
        del positions2, stresses2
        gc.collect()
    else:
        positions = positions_or_tuple
        stresses = stresses_or_none

        # 单域模式也提取截面数据
        cross_sections = extract_domain2_cross_sections(positions, stresses, config, use_schwarz=False)

    print(f"总粒子数: {len(positions)}")
    print(f"应力范围: σ_xx [{np.min(stresses[:, 0, 0]):.2e}, {np.max(stresses[:, 0, 0]):.2e}], σ_yy [{np.min(stresses[:, 1, 1]):.2e}, {np.max(stresses[:, 1, 1]):.2e}]")

    # 绘制全局应力分布
    plt.figure(figsize=(16, 6))
    title_suffix = f" (Grid {grid_size}x{grid_size})" if grid_size else ""

    # 左图: Sigma_XX
    plt.subplot(1, 3, 1)
    plt.title(f"Stress $\sigma_{{xx}}${title_suffix}")
    scatter = plt.scatter(positions[:, 0], positions[:, 1], c=stresses[:, 0, 0],
                         s=5, cmap='coolwarm', vmin=-1e5, vmax=1e5)
    plt.colorbar(scatter, label='$\sigma_{xx}$ (Pa)')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.axis('equal')

    # 中图: Sigma_YY
    plt.subplot(1, 3, 2)
    plt.title(f"Stress $\sigma_{{yy}}${title_suffix}")
    scatter = plt.scatter(positions[:, 0], positions[:, 1], c=stresses[:, 1, 1],
                         s=5, cmap='coolwarm', vmin=-1e5, vmax=1e5)
    plt.colorbar(scatter, label='$\sigma_{yy}$ (Pa)')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.axis('equal')

    # 右图: Sigma_XY
    plt.subplot(1, 3, 3)
    plt.title(f"Shear Stress $\sigma_{{xy}}${title_suffix}")
    scatter = plt.scatter(positions[:, 0], positions[:, 1], c=stresses[:, 0, 1],
                         s=5, cmap='coolwarm', vmin=-5e4, vmax=5e4)
    plt.colorbar(scatter, label='$\sigma_{xy}$ (Pa)')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.axis('equal')

    # 保存图形
    save_path = os.path.join(output_dir, 'stress_distribution.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {save_path}")

    # 绘制截面数据（单域和Schwarz模式都绘制）
    if cross_sections is not None:
        plot_cross_sections(cross_sections, output_dir, grid_size, use_schwarz)
        plot_individual_cross_sections(cross_sections, output_dir, grid_size, use_schwarz)

    # 保存数据
    np.save(os.path.join(output_dir, 'positions.npy'), positions)
    np.save(os.path.join(output_dir, 'stresses.npy'), stresses)

    result = {
        'plot_path': save_path,
        'positions': positions,
        'stresses': stresses
    }

    if cross_sections is not None:
        # 保存截面数据
        domain_label = "domain2" if use_schwarz else "domain"
        np.save(os.path.join(output_dir, f'{domain_label}_x_axis_positions.npy'),
                np.column_stack([cross_sections['x_axis']['x'], cross_sections['x_axis']['y']]))
        np.save(os.path.join(output_dir, f'{domain_label}_x_axis_stresses.npy'),
                np.column_stack([cross_sections['x_axis']['sig_xx'],
                                cross_sections['x_axis']['sig_yy'],
                                cross_sections['x_axis']['sig_xy']]))

        np.save(os.path.join(output_dir, f'{domain_label}_y_axis_positions.npy'),
                np.column_stack([cross_sections['y_axis']['x'], cross_sections['y_axis']['y']]))
        np.save(os.path.join(output_dir, f'{domain_label}_y_axis_stresses.npy'),
                np.column_stack([cross_sections['y_axis']['sig_xx'],
                                cross_sections['y_axis']['sig_yy'],
                                cross_sections['y_axis']['sig_xy']]))

        result['cross_sections'] = cross_sections
        print(f"截面数据已保存")

    return result

def plot_cross_sections(cross_sections, output_dir, grid_size=None, use_schwarz=False):
    """
    绘制X轴和Y轴截面应力分布（组合图）

    参数:
        cross_sections: 截面数据字典
        output_dir: 输出目录
        grid_size: 网格大小（用于标注），可选
        use_schwarz: 是否为Schwarz模式
    """
    x_axis = cross_sections['x_axis']
    y_axis = cross_sections['y_axis']
    center = cross_sections['center']

    title_suffix = f" (Grid {grid_size}x{grid_size})" if grid_size else ""

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # X轴截面（水平线，Y=center_y）
    # 相对于中心的X坐标
    x_rel = x_axis['x'] - center[0]

    # σ_xx 沿X轴
    axes[0, 0].plot(x_rel, x_axis['sig_xx'], 'o-', markersize=4, linewidth=1.5)
    axes[0, 0].set_title(f"$\sigma_{{xx}}$ along X-axis (Y={center[1]:.3f}){title_suffix}")
    axes[0, 0].set_xlabel('X position (relative to center)')
    axes[0, 0].set_ylabel('$\sigma_{xx}$ (Pa)')
    axes[0, 0].grid(True, alpha=0.3)

    # σ_yy 沿X轴
    axes[0, 1].plot(x_rel, x_axis['sig_yy'], 'o-', markersize=4, linewidth=1.5, color='green')
    axes[0, 1].set_title(f"$\sigma_{{yy}}$ along X-axis (Y={center[1]:.3f}){title_suffix}")
    axes[0, 1].set_xlabel('X position (relative to center)')
    axes[0, 1].set_ylabel('$\sigma_{yy}$ (Pa)')
    axes[0, 1].grid(True, alpha=0.3)

    # σ_xy 沿X轴
    axes[0, 2].plot(x_rel, x_axis['sig_xy'], 'o-', markersize=4, linewidth=1.5, color='orange')
    axes[0, 2].set_title(f"$\sigma_{{xy}}$ along X-axis (Y={center[1]:.3f}){title_suffix}")
    axes[0, 2].set_xlabel('X position (relative to center)')
    axes[0, 2].set_ylabel('$\sigma_{xy}$ (Pa)')
    axes[0, 2].grid(True, alpha=0.3)

    # Y轴截面（垂直线，X=center_x）
    # 相对于中心的Y坐标
    y_rel = y_axis['y'] - center[1]

    # σ_xx 沿Y轴
    axes[1, 0].plot(y_rel, y_axis['sig_xx'], 's-', markersize=4, linewidth=1.5)
    axes[1, 0].set_title(f"$\sigma_{{xx}}$ along Y-axis (X={center[0]:.3f}){title_suffix}")
    axes[1, 0].set_xlabel('Y position (relative to center)')
    axes[1, 0].set_ylabel('$\sigma_{xx}$ (Pa)')
    axes[1, 0].grid(True, alpha=0.3)

    # σ_yy 沿Y轴
    axes[1, 1].plot(y_rel, y_axis['sig_yy'], 's-', markersize=4, linewidth=1.5, color='green')
    axes[1, 1].set_title(f"$\sigma_{{yy}}$ along Y-axis (X={center[0]:.3f}){title_suffix}")
    axes[1, 1].set_xlabel('Y position (relative to center)')
    axes[1, 1].set_ylabel('$\sigma_{yy}$ (Pa)')
    axes[1, 1].grid(True, alpha=0.3)

    # σ_xy 沿Y轴
    axes[1, 2].plot(y_rel, y_axis['sig_xy'], 's-', markersize=4, linewidth=1.5, color='orange')
    axes[1, 2].set_title(f"$\sigma_{{xy}}$ along Y-axis (X={center[0]:.3f}){title_suffix}")
    axes[1, 2].set_xlabel('Y position (relative to center)')
    axes[1, 2].set_ylabel('$\sigma_{xy}$ (Pa)')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存截面图
    domain_label = "domain2" if use_schwarz else "domain"
    cross_section_path = os.path.join(output_dir, f'{domain_label}_cross_sections.png')
    plt.savefig(cross_section_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"截面组合图已保存: {cross_section_path}")

def plot_individual_cross_sections(cross_sections, output_dir, grid_size=None, use_schwarz=False):
    """
    分别绘制并保存X轴和Y轴截面的应力变化图

    参数:
        cross_sections: 截面数据字典
        output_dir: 输出目录
        grid_size: 网格大小（用于标注），可选
        use_schwarz: 是否为Schwarz模式
    """
    x_axis = cross_sections['x_axis']
    y_axis = cross_sections['y_axis']
    center = cross_sections['center']

    title_suffix = f" (Grid {grid_size}x{grid_size})" if grid_size else ""

    # ========== X轴截面（水平线，Y=center_y）==========
    fig_x, axes_x = plt.subplots(1, 3, figsize=(18, 5))
    x_rel = x_axis['x'] - center[0]

    # σ_xx 沿X轴
    axes_x[0].plot(x_rel, x_axis['sig_xx'], 'o-', markersize=5, linewidth=2, color='blue')
    axes_x[0].set_title(f"$\sigma_{{xx}}$ along X-axis (Y={center[1]:.3f}){title_suffix}", fontsize=14)
    axes_x[0].set_xlabel('X position (relative to center)', fontsize=12)
    axes_x[0].set_ylabel('$\sigma_{xx}$ (Pa)', fontsize=12)
    axes_x[0].grid(True, alpha=0.3)
    axes_x[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes_x[0].axvline(x=0, color='r', linestyle=':', alpha=0.3, label='Center')
    axes_x[0].legend()

    # σ_yy 沿X轴
    axes_x[1].plot(x_rel, x_axis['sig_yy'], 'o-', markersize=5, linewidth=2, color='green')
    axes_x[1].set_title(f"$\sigma_{{yy}}$ along X-axis (Y={center[1]:.3f}){title_suffix}", fontsize=14)
    axes_x[1].set_xlabel('X position (relative to center)', fontsize=12)
    axes_x[1].set_ylabel('$\sigma_{yy}$ (Pa)', fontsize=12)
    axes_x[1].grid(True, alpha=0.3)
    axes_x[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes_x[1].axvline(x=0, color='r', linestyle=':', alpha=0.3, label='Center')
    axes_x[1].legend()

    # σ_xy 沿X轴
    axes_x[2].plot(x_rel, x_axis['sig_xy'], 'o-', markersize=5, linewidth=2, color='orange')
    axes_x[2].set_title(f"$\sigma_{{xy}}$ along X-axis (Y={center[1]:.3f}){title_suffix}", fontsize=14)
    axes_x[2].set_xlabel('X position (relative to center)', fontsize=12)
    axes_x[2].set_ylabel('$\sigma_{xy}$ (Pa)', fontsize=12)
    axes_x[2].grid(True, alpha=0.3)
    axes_x[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes_x[2].axvline(x=0, color='r', linestyle=':', alpha=0.3, label='Center')
    axes_x[2].legend()

    plt.tight_layout()

    # 保存X轴截面图
    domain_label = "domain2" if use_schwarz else "domain"
    x_axis_path = os.path.join(output_dir, f'{domain_label}_x_axis_stress.png')
    plt.savefig(x_axis_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"X轴截面应力图已保存: {x_axis_path}")

    # ========== Y轴截面（垂直线，X=center_x）==========
    fig_y, axes_y = plt.subplots(1, 3, figsize=(18, 5))
    y_rel = y_axis['y'] - center[1]

    # σ_xx 沿Y轴
    axes_y[0].plot(y_rel, y_axis['sig_xx'], 's-', markersize=5, linewidth=2, color='blue')
    axes_y[0].set_title(f"$\sigma_{{xx}}$ along Y-axis (X={center[0]:.3f}){title_suffix}", fontsize=14)
    axes_y[0].set_xlabel('Y position (relative to center)', fontsize=12)
    axes_y[0].set_ylabel('$\sigma_{xx}$ (Pa)', fontsize=12)
    axes_y[0].grid(True, alpha=0.3)
    axes_y[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes_y[0].axvline(x=0, color='r', linestyle=':', alpha=0.3, label='Center')
    axes_y[0].legend()

    # σ_yy 沿Y轴
    axes_y[1].plot(y_rel, y_axis['sig_yy'], 's-', markersize=5, linewidth=2, color='green')
    axes_y[1].set_title(f"$\sigma_{{yy}}$ along Y-axis (X={center[0]:.3f}){title_suffix}", fontsize=14)
    axes_y[1].set_xlabel('Y position (relative to center)', fontsize=12)
    axes_y[1].set_ylabel('$\sigma_{yy}$ (Pa)', fontsize=12)
    axes_y[1].grid(True, alpha=0.3)
    axes_y[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes_y[1].axvline(x=0, color='r', linestyle=':', alpha=0.3, label='Center')
    axes_y[1].legend()

    # σ_xy 沿Y轴
    axes_y[2].plot(y_rel, y_axis['sig_xy'], 's-', markersize=5, linewidth=2, color='orange')
    axes_y[2].set_title(f"$\sigma_{{xy}}$ along Y-axis (X={center[0]:.3f}){title_suffix}", fontsize=14)
    axes_y[2].set_xlabel('Y position (relative to center)', fontsize=12)
    axes_y[2].set_ylabel('$\sigma_{xy}$ (Pa)', fontsize=12)
    axes_y[2].grid(True, alpha=0.3)
    axes_y[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes_y[2].axvline(x=0, color='r', linestyle=':', alpha=0.3, label='Center')
    axes_y[2].legend()

    plt.tight_layout()

    # 保存Y轴截面图
    y_axis_path = os.path.join(output_dir, f'{domain_label}_y_axis_stress.png')
    plt.savefig(y_axis_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Y轴截面应力图已保存: {y_axis_path}")

def run_single_experiment(config_path, grid_size, output_dir, use_schwarz=False):
    """
    运行单个实验

    参数:
        config_path: 配置文件路径
        grid_size: 网格大小（可选，None表示使用配置文件中的值）
        output_dir: 输出目录
        use_schwarz: 是否使用Schwarz求解器

    返回:
        dict: 实验结果
    """
    print(f"\n{'='*60}")
    if grid_size:
        mode = "Schwarz" if use_schwarz else "Single"
        print(f"运行实验 ({mode}模式) - 网格大小: {grid_size}x{grid_size}")
    else:
        print(f"运行实验 - 使用配置文件中的网格大小")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    config = load_config(config_path)

    # 运行模拟
    sim_results = run_simulation(config_path, use_schwarz)

    # 分析和绘图
    if use_schwarz:
        results = analyze_and_plot(sim_results, None, config, output_dir, grid_size, use_schwarz=True)
    else:
        positions, stresses = sim_results
        results = analyze_and_plot(positions, stresses, config, output_dir, grid_size, use_schwarz=False)

    # 保存配置备份
    config_backup_path = os.path.join(output_dir, 'config_backup.json')
    save_config(config, config_backup_path)

    print(f"实验完成! 结果保存到: {output_dir}")

    return results

def run_batch_experiments(args):
    """
    运行批量实验

    参数:
        args: 命令行参数对象

    返回:
        dict: 批量实验结果汇总
    """
    print("=" * 80)
    print("开始批量网格分辨率实验")
    print("=" * 80)

    # 生成网格大小列表
    grid_sizes = list(range(args.grid_start, args.grid_end + 1, args.grid_step))
    print(f"网格大小列表: {grid_sizes}")
    print(f"总共 {len(grid_sizes)} 个实验")

    # 加载基础配置
    base_config = load_config(args.config)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 记录实验结果
    results = {
        'base_config': args.config,
        'grid_sizes': grid_sizes,
        'successful_runs': [],
        'failed_runs': [],
        'result_dirs': {},
        'experiment_data': {}
    }

    for i, grid_size in enumerate(grid_sizes):
        print(f"\n{'='*20} 实验 {i+1}/{len(grid_sizes)}: 网格 {grid_size}x{grid_size} {'='*20}")

        try:
            # 1. 修改配置
            modified_config = modify_config_grid_size(base_config, grid_size, use_schwarz=args.schwarz)

            # 2. 创建临时配置文件
            temp_config_path = os.path.join(args.output_dir, f"temp_config_grid{grid_size}.json")
            save_config(modified_config, temp_config_path)

            # 3. 创建输出子目录
            grid_output_dir = os.path.join(args.output_dir, f"grid_{grid_size}")

            # 4. 运行单个实验
            exp_results = run_single_experiment(temp_config_path, grid_size, grid_output_dir, use_schwarz=args.schwarz)

            # 5. 记录结果（只保存必要的元数据，不保存大型numpy数组）
            results['successful_runs'].append(grid_size)
            results['result_dirs'][grid_size] = grid_output_dir

            # 只保存 plot_path，不保存大型数组
            results['experiment_data'][grid_size] = {
                'plot_path': exp_results.get('plot_path', '')
            }

            print(f"✓ 实验 {i+1} 成功完成")

            # 清理临时配置文件
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

            # 清理内存：删除所有大型临时变量
            del exp_results
            del modified_config
            gc.collect()

        except Exception as e:
            print(f"✗ 实验 {i+1} 失败: {e}")
            import traceback
            traceback.print_exc()
            results['failed_runs'].append(grid_size)

    # 保存实验汇总
    from datetime import datetime
    results['completed_time'] = datetime.now().isoformat()
    summary_file = os.path.join(args.output_dir, "experiment_summary.json")

    # 移除不能序列化的numpy数组
    summary_results = {k: v for k, v in results.items() if k != 'experiment_data'}
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=4)

    print(f"\n{'='*80}")
    print("批量实验完成!")
    print(f"成功: {len(results['successful_runs'])}/{len(grid_sizes)} 个实验")
    print(f"成功的网格大小: {results['successful_runs']}")
    if results['failed_runs']:
        print(f"失败的网格大小: {results['failed_runs']}")
    print(f"实验汇总保存到: {summary_file}")
    print("=" * 80)

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='实验2: 域应力分析实验 - 支持批量网格分辨率实验')

    # 基础参数
    parser.add_argument('--config', default="config/config_2d_test2_new.json",
                       help='配置文件路径')
    parser.add_argument('--output-dir', default="experiments/test2",
                       help='输出目录')

    # Schwarz模式
    parser.add_argument('--schwarz', action='store_true',
                       help='使用Schwarz域分解求解器')

    # 批量模式参数
    parser.add_argument('--batch-mode', action='store_true',
                       help='启用批量模式：运行多个不同网格分辨率的实验')
    parser.add_argument('--grid-start', type=int, default=40,
                       help='批量模式：起始网格大小 (默认: 20)')
    parser.add_argument('--grid-end', type=int, default=100,
                       help='批量模式：结束网格大小 (默认: 80)')
    parser.add_argument('--grid-step', type=int, default=20,
                       help='批量模式：网格大小步长 (默认: 20)')

    args = parser.parse_args()

    # 如果使用Schwarz模式但没有指定配置文件，使用Schwarz默认配置
    if args.schwarz and args.config == "config/config_2d_test2_new.json":
        args.config = "config/schwarz_2d_test2_new.json"
        print(f"Schwarz模式：使用默认配置 {args.config}")

    if args.batch_mode:
        # 批量模式
        print("=" * 80)
        mode_str = "Schwarz域分解" if args.schwarz else "单域"
        print(f"批量{mode_str}网格分辨率实验模式")
        print(f"网格范围: {args.grid_start} - {args.grid_end}, 步长: {args.grid_step}")
        print("=" * 80)

        results = run_batch_experiments(args)

        print(f"\n批量实验全部完成!")
        print(f"结果目录: {args.output_dir}")

    else:
        # 单次运行模式（原始功能）
        print("=" * 80)
        mode_str = "Schwarz域分解" if args.schwarz else "单域"
        print(f"{mode_str}单次实验模式")
        print("=" * 80)

        run_single_experiment(args.config, None, args.output_dir, use_schwarz=args.schwarz)

if __name__ == "__main__":
    main()
