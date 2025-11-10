#!/usr/bin/env python3
"""
网格分辨率批处理实验脚本
基于配置文件模板，自动修改网格分辨率并运行单域或双域MPM模拟
支持批量测试不同网格大小对模拟结果的影响
"""

import json
import numpy as np
import sys
import os
import subprocess
import glob
import shutil
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入质量计算函数
try:
    from experiments.experiment_3_hertz_contact import get_actual_volume_force_mass
    MASS_CALCULATION_AVAILABLE = True
except ImportError:
    MASS_CALCULATION_AVAILABLE = False
    print("警告: 无法导入质量计算函数，将使用理论质量")

def load_config(config_path):
    """加载配置文件"""
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
        # 双域配置：只修改Domain2的网格大小
        if 'Domain2' in new_config:
            # 获取Domain2的原始网格大小比例
            original_nx = new_config['Domain2'].get('grid_nx', 64)
            original_ny = new_config['Domain2'].get('grid_ny', 64)
            aspect_ratio = original_ny / original_nx if original_nx > 0 else 1.0

            # 设置新的网格大小
            new_config['Domain2']['grid_nx'] = grid_size
            new_config['Domain2']['grid_ny'] = int(grid_size * aspect_ratio)

            print(f"修改双域配置: 仅修改Domain2网格大小 nx={grid_size}, ny={int(grid_size * aspect_ratio)} (比例: {aspect_ratio:.3f})")
        else:
            print("警告: 双域配置中未找到Domain2")
    else:
        # 单域配置：修改顶层网格大小
        new_config['grid_nx'] = grid_size
        new_config['grid_ny'] = grid_size

        print(f"修改单域配置: 网格大小设为 {grid_size}x{grid_size}")

    return new_config

def run_simulation(config_path, use_schwarz=False, timeout=None):
    """
    运行MPM模拟

    参数:
        config_path: 配置文件路径
        use_schwarz: 是否使用Schwarz双域求解器
        timeout: 超时时间（秒）

    返回:
        bool: 模拟是否成功
    """
    if use_schwarz:
        cmd = [
            "python", "simulators/implicit_mpm_schwarz.py",
            "--config", config_path
        ]
        print(f"运行双域模拟: {config_path}")
    else:
        cmd = [
            "python", "simulators/implicit_mpm.py",
            "--config", config_path
        ]
        print(f"运行单域模拟: {config_path}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode == 0:
            print(f"  模拟完成")
            return True
        else:
            print(f"  模拟失败，返回码: {result.returncode}")
            if result.stderr:
                print(f"  错误信息: {result.stderr[-500:]}")  # 显示最后500字符
            return True

    except subprocess.TimeoutExpired:
        print(f"  模拟超时（超过{timeout}秒）")
        return False
    except Exception as e:
        print(f"  模拟过程出错: {e}")
        return False

def find_latest_result_dir(use_schwarz=False):
    """查找最新的结果目录"""
    if use_schwarz:
        pattern = "experiment_results/schwarz_*"
    else:
        pattern = "experiment_results/single_domain_*"

    dirs = glob.glob(pattern)
    if not dirs:
        return None

    # 按修改时间排序，返回最新的
    latest_dir = max(dirs, key=os.path.getctime)
    return latest_dir

def backup_result_dir(source_dir, grid_size, output_base_dir, use_schwarz=False):
    """备份结果目录到指定位置"""
    if source_dir is None or not os.path.exists(source_dir):
        print(f"警告: 源目录不存在: {source_dir}")
        return None

    # 创建目标目录名
    solver_type = "schwarz" if use_schwarz else "single"
    target_name = f"{solver_type}_grid{grid_size}"
    target_dir = os.path.join(output_base_dir, target_name)

    # 如果目标目录已存在，先删除
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # 复制结果目录
    shutil.copytree(source_dir, target_dir)
    print(f"  结果已备份到: {target_dir}")

    return target_dir

def run_batch_grid_study(base_config_path, grid_sizes, use_schwarz=False,
                        output_dir="experiment_results/grid_study", timeout=3600,
                        y_filter_min=0.0, y_filter_max=0.01):
    """
    运行批量网格研究

    参数:
        base_config_path: 基础配置文件路径
        grid_sizes: 网格大小列表
        use_schwarz: 是否使用Schwarz双域求解器
        output_dir: 输出目录
        timeout: 单个模拟的超时时间

    返回:
        dict: 实验结果汇总
    """
    print("=" * 80)
    solver_type = "双域Schwarz" if use_schwarz else "单域"
    print(f"开始{solver_type}网格分辨率批处理实验")
    print(f"基础配置: {base_config_path}")
    print(f"网格大小: {grid_sizes}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载基础配置
    base_config = load_config(base_config_path)

    # 记录实验结果
    results = {
        'base_config': base_config_path,
        'use_schwarz': use_schwarz,
        'grid_sizes': grid_sizes,
        'successful_runs': [],
        'failed_runs': [],
        'result_dirs': {}
    }

    for i, grid_size in enumerate(grid_sizes):
        print(f"\n{'='*20} 实验 {i+1}/{len(grid_sizes)}: 网格 {grid_size}x{grid_size} {'='*20}")

        # 1. 修改配置文件
        modified_config = modify_config_grid_size(base_config, grid_size, use_schwarz)

        # 创建临时配置文件
        temp_config_path = os.path.join(output_dir, f"temp_config_grid{grid_size}.json")
        save_config(modified_config, temp_config_path)

        # 2. 运行模拟
        success = run_simulation(temp_config_path, use_schwarz, timeout)

        if success:
            # 3. 备份结果
            latest_result_dir = find_latest_result_dir(use_schwarz)
            backup_dir = backup_result_dir(latest_result_dir, grid_size, output_dir, use_schwarz)

            # 4. 复制配置文件到结果目录
            if backup_dir:
                config_backup_path = os.path.join(backup_dir, "config.json")
                shutil.copy2(temp_config_path, config_backup_path)
                print(f"  配置文件已保存到: {config_backup_path}")

            results['successful_runs'].append(grid_size)
            if backup_dir:
                results['result_dirs'][grid_size] = backup_dir

            print(f"  实验 {i+1} 成功完成")
        else:
            results['failed_runs'].append(grid_size)
            print(f"  实验 {i+1} 失败")

        # 清理临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    # 保存实验汇总
    results['completed_time'] = datetime.now().isoformat()
    summary_file = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)

    print(f"\n{'='*80}")
    print("批量网格研究完成!")
    print(f"成功: {len(results['successful_runs'])}/{len(grid_sizes)} 个实验")
    print(f"成功的网格大小: {results['successful_runs']}")
    if results['failed_runs']:
        print(f"失败的网格大小: {results['failed_runs']}")
    print(f"实验汇总保存到: {summary_file}")
    print(f"结果目录: {output_dir}")
    print("=" * 80)

    # 添加应力分析和解析解对比
    if results['successful_runs']:
        print(f"\n{'='*80}")
        print("开始应力分析和解析解对比...")
        analyze_stress_convergence(results, base_config_path, output_dir, use_schwarz, y_filter_min, y_filter_max)

    return results

def extract_hertz_parameters_from_config(config, actual_mass=None):
    """从配置文件提取Hertz接触问题相关参数"""
    # 检查是否是Schwarz配置（有Domain1/Domain2）还是单域配置
    if 'Domain1' in config:
        # Schwarz配置：从Domain1获取参数
        domain_config = config['Domain1']
        print("检测到Schwarz域分解配置，使用Domain1参数")
    else:
        # 单域配置：直接使用顶层配置
        domain_config = config
        print("检测到单域配置")

    # 材料参数
    material = domain_config['material_params'][0]
    E = material['E'] / 2.0  # 按要求E除以2
    nu = material['nu']
    rho = material['rho']

    # 获取基础的椭圆形状参数
    ellipse_shape = None
    for shape in domain_config['shapes']:
        if shape['type'] == 'ellipse' and shape['operation'] == 'add':
            ellipse_shape = shape
            break

    if ellipse_shape is None:
        raise ValueError("未找到椭圆形状")

    center = ellipse_shape['params']['center']
    # 对于椭圆，使用semi_axes作为等效半径（通常取较小的半轴）
    semi_axes = ellipse_shape['params']['semi_axes']
    radius = min(semi_axes)  # 使用较小的半轴作为等效半径

    # 获取体积力并计算等效接触压力
    volume_forces = domain_config.get('volume_forces', [])
    if not volume_forces:
        raise ValueError("未找到体积力配置")

    volume_force = volume_forces[0]
    force_magnitude = abs(volume_force['force'][1])  # 取y方向力的绝对值

    # 载荷区域几何
    force_range = volume_force['params']['range']  # range在params里面
    load_width = force_range[0][1] - force_range[0][0]
    load_height = force_range[1][1] - force_range[1][0]

    # 计算质量：如果提供了实际质量则使用，否则计算理论质量
    if actual_mass is not None:
        mass_to_use = actual_mass
        print(f"使用实际质量: {actual_mass:.6f} kg")
    else:
        mass_to_use = load_width * load_height * rho  # 2D情况，假设厚度为1
        print(f"使用理论质量: {mass_to_use:.6f} kg")

    # 计算等效接触压力：体力 × 质量 / 载荷宽度
    total_force = force_magnitude * mass_to_use
    applied_pressure = total_force / load_width

    # 计算分析区域
    if 'Domain1' in config:
        # 双域配置：寻找rectangle类型的shape来获取domain range
        domain_shape = None
        for shape in domain_config['shapes']:
            if shape['type'] == 'rectangle':
                domain_shape = shape
                break

        if domain_shape is not None:
            domain1_range = domain_shape['params']['range']
            x_range = domain1_range[0]
            y_range = domain1_range[1]
        else:
            # 如果没有找到rectangle，使用默认域大小
            x_range = [0, domain_config.get('domain_width', 1.0)]
            y_range = [0, domain_config.get('domain_height', 1.0)]
    else:
        # 单域配置：使用网格域尺寸
        grid_domain_width = domain_config.get('domain_width', 1.0)
        grid_domain_height = domain_config.get('domain_height', 1.0)
        x_range = [0, grid_domain_width]
        y_range = [0, grid_domain_height]

    return {
        'E': E, 'nu': nu, 'rho': rho,
        'center': center, 'radius': radius,
        'applied_pressure': applied_pressure,
        'x_range': x_range, 'y_range': y_range,
        'config': config
    }

def calculate_hertz_analytical_solution(config_path, actual_mass=None, y_filter_min=None, y_filter_max=None):
    """计算Hertz接触问题的解析解"""
    from analytical_solutions.test3 import calculate_hertz_pressure_at_y_zero, calculate_contact_width

    params = extract_hertz_parameters_from_config(load_config(config_path), actual_mass=actual_mass)

    print("椭圆接触问题参数:")
    print(f"材料: E={params['E']:.0e} Pa (E/2), nu={params['nu']}, rho={params['rho']}")
    print(f"椭圆等效半径: {params['radius']} m (较小半轴)")
    print(f"等效接触压力: {params['applied_pressure']:.2f} Pa")
    print(f"分析区域: x∈{params['x_range']}")
    if y_filter_min is not None or y_filter_max is not None:
        print(f"Y高度过滤: {y_filter_min} <= y <= {y_filter_max}")
    print()

    # 计算接触宽度
    contact_width = calculate_contact_width(
        params['radius'], params['E'], params['nu'], params['applied_pressure'])
    print(f"接触半宽度: {contact_width:.4f} m")

    # 创建x坐标数组（只需要x坐标，因为只计算y=0）
    x_range = params['x_range']
    nx = 1000  # 高分辨率
    x_coords = np.linspace(x_range[0], x_range[1], nx)

    # 计算y=0位置的压力分布
    x_relative = x_coords - params['center'][0]  # 相对于圆柱中心的x坐标
    pressure_y_zero = calculate_hertz_pressure_at_y_zero(
        x_relative, params['radius'], params['E'], params['nu'], params['applied_pressure'])

    # 转换为应力（压应力为负）
    stress_yy = -pressure_y_zero

    return {
        'x_coords': x_coords,
        'stress_yy': stress_yy,
        'params': params,
        'contact_width': contact_width
    }

def load_mpm_stress_results(result_dir, use_schwarz=False, y_filter_min=None, y_filter_max=None):
    """加载MPM模拟的应力结果"""
    print(f"加载结果目录: {result_dir}")

    if use_schwarz:
        # 双域MPM：加载Domain2的结果（包含底部边界粒子）
        stress_files = [f for f in os.listdir(result_dir)
                       if f.startswith('domain2_stress_frame_') and f.endswith('.npy')]
        if not stress_files:
            print("No Domain2 stress data files found")
            return None

        stress_file = stress_files[0]
        frame_num = stress_file.split('_')[3].split('.')[0]

        stress_data = np.load(os.path.join(result_dir, f'domain2_stress_frame_{frame_num}.npy'))
        positions = np.load(os.path.join(result_dir, f'domain2_positions_frame_{frame_num}.npy'))
        boundary_flags = np.load(os.path.join(result_dir, f'domain2_boundary_flags_frame_{frame_num}.npy'))

        print(f"Loaded {len(positions)} Domain2 particles from frame {frame_num}")
    else:
        # 单域MPM：加载常规结果
        stress_files = [f for f in os.listdir(result_dir)
                       if f.startswith('stress_frame_') and f.endswith('.npy')]
        if not stress_files:
            print("No stress data files found")
            return None

        stress_file = stress_files[0]
        frame_num = stress_file.split('_')[2].split('.')[0]

        stress_data = np.load(os.path.join(result_dir, f'stress_frame_{frame_num}.npy'))
        positions = np.load(os.path.join(result_dir, f'positions_frame_{frame_num}.npy'))
        boundary_flags = np.load(os.path.join(result_dir, f'boundary_flags_frame_{frame_num}.npy'))

        print(f"Loaded {len(positions)} particles from frame {frame_num}")

    try:
        # 过滤边界粒子
        y_min = positions[:, 1].min()
        y_max = positions[:, 1].max()
        boundary_mask = boundary_flags == 1  # 边界粒子标志为1

        # 按Y高度过滤（同时支持max和min）
        if y_filter_min is not None or y_filter_max is not None:
            if y_filter_min is not None and y_filter_max is not None:
                y_filter_mask = (positions[:, 1] >= y_filter_min) & (positions[:, 1] <= y_filter_max)
                print(f"应用Y坐标过滤: {y_filter_min} <= y <= {y_filter_max}")
            elif y_filter_min is not None:
                y_filter_mask = positions[:, 1] >= y_filter_min
                print(f"应用Y坐标过滤: y >= {y_filter_min}")
            else:  # y_filter_max is not None
                y_filter_mask = positions[:, 1] <= y_filter_max
                print(f"应用Y坐标过滤: y <= {y_filter_max}")
            boundary_mask = boundary_mask & y_filter_mask

        boundary_positions = positions[boundary_mask]
        boundary_stress = stress_data[boundary_mask]

        solver_type = "Domain2" if use_schwarz else "Single-domain"
        print(f"{solver_type} 粒子y坐标范围: {y_min:.4f} - {y_max:.4f}")

        if y_filter_min is not None or y_filter_max is not None:
            boundary_y_min = boundary_positions[:, 1].min() if len(boundary_positions) > 0 else 0
            boundary_y_max = boundary_positions[:, 1].max() if len(boundary_positions) > 0 else 0
            print(f"Found {len(boundary_positions)} filtered boundary particles (y: {boundary_y_min:.4f}-{boundary_y_max:.4f}, total: {len(positions)})")
        else:
            print(f"Found {len(boundary_positions)} boundary particles (total: {len(positions)})")

        # 直接使用所有边界粒子，按x坐标排序
        sort_indices = np.argsort(boundary_positions[:, 0])
        sorted_positions = boundary_positions[sort_indices]
        sorted_stress = boundary_stress[sort_indices]

        print(f"Using all {len(sorted_positions)} boundary particles directly")

        return {
            'positions': sorted_positions,
            'stress': sorted_stress,
            'y_range': [y_min, y_max],
            'boundary_particle_count': len(boundary_positions),
            'results_dir': result_dir
        }

    except Exception as e:
        print(f"Error loading MPM results: {e}")
        return None

def compare_stress_with_analytical(analytical_results, mpm_results, grid_size, output_dir):
    """对比MPM结果和解析解"""
    from analytical_solutions.test3 import calculate_contact_width

    params = analytical_results['params']
    contact_width = analytical_results['contact_width']

    # 创建图形
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 解析解 - y=0位置的压力分布
    x_coords = analytical_results['x_coords']
    stress_yy = analytical_results['stress_yy']
    center_x = params['center'][0]

    # 只在接触区域内显示
    contact_indices = (x_coords >= center_x - contact_width*1.2) & (x_coords <= center_x + contact_width*1.2)
    x_analytical = x_coords[contact_indices]
    pressure_analytical = -stress_yy[contact_indices]/1000

    # MPM结果
    mpm_x = mpm_results['positions'][:, 0]
    mpm_stress_yy = mpm_results['stress'][:, 1, 1]  # σ_yy分量
    mpm_pressure = -mpm_stress_yy/1000  # 转换为正的压力值

    # 只显示接触区域内的MPM数据
    mpm_contact_indices = (mpm_x >= center_x - contact_width*1.2) & (mpm_x <= center_x + contact_width*1.2)
    x_mpm_contact = mpm_x[mpm_contact_indices]
    pressure_mpm_contact = mpm_pressure[mpm_contact_indices]

    # 绘制结果
    ax.plot(x_analytical, pressure_analytical, 'r-', linewidth=2, label='Analytical (Hertz)', alpha=0.8)

    # MPM标签
    mpm_label = f'MPM (Grid {grid_size})'
    ax.scatter(x_mpm_contact, pressure_mpm_contact, c='blue', s=20, alpha=0.6, label=mpm_label)

    # 标记接触边界
    ax.axvline(x=center_x - contact_width, color='red', linestyle='--', alpha=0.7,
               label=f'Contact boundary (±{contact_width:.6f}m)')
    ax.axvline(x=center_x + contact_width, color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('Pressure (kPa)')
    ax.set_title(f'Ellipse Contact: MPM vs Analytical Solution (Grid {grid_size})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(center_x - contact_width*1.2, center_x + contact_width*1.2)

    plt.tight_layout()

    # 保存图片
    plot_file = os.path.join(output_dir, f"stress_comparison_grid{grid_size}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"应力对比图保存到: {plot_file}")
    return plot_file

def create_summary_plot(analytical_results, all_mpm_results, output_dir):
    """创建汇总图，将所有网格大小的结果绘制在一张图上"""
    from analytical_solutions.test3 import calculate_contact_width

    params = analytical_results['params']
    contact_width = analytical_results['contact_width']

    # 创建大图形
    _, ax = plt.subplots(1, 1, figsize=(14, 10))

    # 解析解 - y=0位置的压力分布
    x_coords = analytical_results['x_coords']
    stress_yy = analytical_results['stress_yy']
    center_x = params['center'][0]

    # 只在接触区域内显示
    contact_indices = (x_coords >= center_x - contact_width*1.2) & (x_coords <= center_x + contact_width*1.2)
    x_analytical = x_coords[contact_indices]
    pressure_analytical = -stress_yy[contact_indices]/1000

    # 绘制解析解
    ax.plot(x_analytical, pressure_analytical, 'r-', linewidth=3, label='Analytical (Hertz)', alpha=0.9, zorder=10)

    # 定义颜色列表
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

    # 绘制所有网格大小的MPM结果
    for i, (grid_size, mpm_results) in enumerate(all_mpm_results.items()):
        if mpm_results is None:
            continue

        # MPM结果
        mpm_x = mpm_results['positions'][:, 0]
        mpm_stress_yy = mpm_results['stress'][:, 1, 1]  # σ_yy分量
        mpm_pressure = -mpm_stress_yy/1000  # 转换为正的压力值

        # 只显示接触区域内的MPM数据
        mpm_contact_indices = (mpm_x >= center_x - contact_width*1.2) & (mpm_x <= center_x + contact_width*1.2)
        x_mpm_contact = mpm_x[mpm_contact_indices]
        pressure_mpm_contact = mpm_pressure[mpm_contact_indices]

        # 使用不同颜色和标记
        color = colors[i % len(colors)]

        # 标签
        label = f'MPM (Grid {grid_size})'
        ax.scatter(x_mpm_contact, pressure_mpm_contact, c=color, s=15, alpha=0.7,
                  label=label, zorder=5)

    # 标记接触边界
    ax.axvline(x=center_x - contact_width, color='red', linestyle='--', alpha=0.7,
               label=f'Contact boundary (±{contact_width:.6f}m)', zorder=8)
    ax.axvline(x=center_x + contact_width, color='red', linestyle='--', alpha=0.7, zorder=8)

    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('Pressure (kPa)', fontsize=12)
    ax.set_title('Ellipse Contact: Grid Convergence Study\nMPM vs Analytical Solution (All Grid Sizes)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(center_x - contact_width*1.2, center_x + contact_width*1.2)

    plt.tight_layout()

    # 保存汇总图
    summary_plot_file = os.path.join(output_dir, "stress_comparison_summary_all_grids.png")
    plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"汇总对比图保存到: {summary_plot_file}")

    return summary_plot_file

def analyze_stress_convergence(batch_results, base_config_path, output_dir, use_schwarz=False, y_filter_min=0.0, y_filter_max=0.01):
    """分析不同网格分辨率下的应力收敛性"""
    print("开始应力收敛性分析...")
    print(f"Y坐标过滤范围: {y_filter_min} <= y <= {y_filter_max}")

    comparison_plots = []
    all_mpm_results = {}  # 存储所有网格的MPM结果用于汇总图
    all_analytical_results = {}  # 存储每个网格对应的解析解
    grid_sizes = batch_results['successful_runs']

    for grid_size in grid_sizes:
        if grid_size not in batch_results['result_dirs']:
            print(f"跳过网格 {grid_size}: 未找到结果目录")
            continue

        result_dir = batch_results['result_dirs'][grid_size]
        print(f"\n分析网格 {grid_size}: {result_dir}")

        # 加载MPM结果
        mpm_results = load_mpm_stress_results(result_dir, use_schwarz=use_schwarz,
                                             y_filter_min=y_filter_min, y_filter_max=y_filter_max)

        if mpm_results is None:
            print(f"无法加载网格 {grid_size} 的结果")
            continue

        # 获取实际质量并计算对应的解析解
        try:
            config_path = os.path.join(result_dir, "config.json")
            if not os.path.exists(config_path):
                print(f"  警告: 未找到配置文件 {config_path}，使用理论质量")
                actual_mass = None
            elif MASS_CALCULATION_AVAILABLE:
                actual_mass = get_actual_volume_force_mass(config_path, mpm_instance=None, results_dir=result_dir)
                print(f"  获取实际质量: {actual_mass:.6f} kg")
            else:
                print(f"  警告: 质量计算函数不可用，使用理论质量")
                actual_mass = None

            # 使用实际质量计算解析解
            analytical_results = calculate_hertz_analytical_solution(config_path, actual_mass=actual_mass)
            all_analytical_results[grid_size] = analytical_results

        except Exception as e:
            print(f"  警告: 计算实际质量失败: {e}")
            # 使用基础配置和理论质量作为fallback
            analytical_results = calculate_hertz_analytical_solution(base_config_path)
            all_analytical_results[grid_size] = analytical_results

        # 存储MPM结果用于汇总图
        all_mpm_results[grid_size] = mpm_results

        # 对比和绘图（单个网格）
        plot_file = compare_stress_with_analytical(analytical_results, mpm_results, grid_size, output_dir)
        comparison_plots.append(plot_file)

    # 创建汇总图（所有网格在一张图上）
    if all_mpm_results:
        print(f"\n创建汇总对比图（包含 {len(all_mpm_results)} 个网格大小）...")
        # 使用第一个网格的解析解作为参考（因为实际质量不同，每个网格的解析解也不同）
        first_grid = list(all_analytical_results.keys())[0]
        reference_analytical = all_analytical_results[first_grid]
        summary_plot = create_summary_plot(reference_analytical, all_mpm_results, output_dir)
        comparison_plots.append(summary_plot)

    print(f"\n应力分析完成! 生成了 {len(comparison_plots)} 个对比图")
    print("对比图文件:")
    for plot in comparison_plots:
        print(f"  - {plot}")

    return comparison_plots

def load_existing_results(experiment_dir, base_config_path):
    """从现有的实验结果目录加载batch_results信息"""
    print(f"正在加载现有实验结果: {experiment_dir}")

    if not os.path.exists(experiment_dir):
        raise ValueError(f"实验目录不存在: {experiment_dir}")

    # 尝试加载实验汇总文件
    summary_file = os.path.join(experiment_dir, "experiment_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            batch_results = json.load(f)

        # 修正result_dirs的键格式：从字符串转换为整数
        if 'result_dirs' in batch_results:
            fixed_result_dirs = {}
            for key, value in batch_results['result_dirs'].items():
                try:
                    int_key = int(key)
                    fixed_result_dirs[int_key] = value
                except ValueError:
                    # 如果转换失败，保持原样
                    fixed_result_dirs[key] = value
            batch_results['result_dirs'] = fixed_result_dirs

        print(f"从汇总文件加载了 {len(batch_results.get('successful_runs', []))} 个成功实验")
        print(f"结果目录映射: {list(batch_results['result_dirs'].keys())}")
        return batch_results

    # 如果没有汇总文件，从目录结构推断
    print("未找到汇总文件，正在从目录结构推断实验结果...")

    batch_results = {
        'base_config': base_config_path,
        'successful_runs': [],
        'failed_runs': [],
        'result_dirs': {}
    }

    # 扫描目录中的网格结果
    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)
        if not os.path.isdir(item_path):
            continue

        # 匹配格式: single_grid64, schwarz_grid128 等
        if '_grid' in item:
            try:
                grid_size = int(item.split('_grid')[1])
                batch_results['successful_runs'].append(grid_size)
                batch_results['result_dirs'][grid_size] = item_path
                print(f"  发现网格 {grid_size}: {item}")
            except ValueError:
                continue

    if not batch_results['successful_runs']:
        raise ValueError(f"在目录 {experiment_dir} 中未找到有效的网格实验结果")

    # 排序网格大小
    batch_results['successful_runs'].sort()

    print(f"成功加载 {len(batch_results['successful_runs'])} 个网格结果: {batch_results['successful_runs']}")
    return batch_results

def run_analysis_only(experiment_dir, base_config_path, output_dir=None, use_schwarz=False, y_filter_min=0.0, y_filter_max=0.01):
    """仅运行应力分析，不进行新的模拟"""
    print("=" * 80)
    print("运行应力分析模式（不进行新模拟）")
    print(f"实验目录: {experiment_dir}")
    print(f"基础配置: {base_config_path}")
    print("=" * 80)

    # 加载现有结果
    batch_results = load_existing_results(experiment_dir, base_config_path)

    # 确定输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        solver_type = "schwarz" if use_schwarz else "single"
        output_dir = f"experiment_results/analysis_only_{solver_type}_{timestamp}"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"分析结果将保存到: {output_dir}")

    # 运行应力分析
    comparison_plots = analyze_stress_convergence(
        batch_results, base_config_path, output_dir, use_schwarz, y_filter_min, y_filter_max
    )

    print(f"\n应力分析完成! 输出目录: {output_dir}")
    return comparison_plots

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='网格分辨率批处理实验')
    parser.add_argument('--use-schwarz', action='store_true',
                       help='使用Schwarz双域求解器（默认使用单域）')
    parser.add_argument('--grid-range', nargs=2, type=int, default=[64, 160],
                       help='网格大小范围 [开始, 结束] (默认: 64 160)')
    parser.add_argument('--grid-step', type=int, default=16,
                       help='网格大小步长 (默认: 16)')
    parser.add_argument('--grid-sizes', nargs='+', type=int, default=None,
                       help='直接指定网格大小列表，覆盖range和step参数')
    parser.add_argument('--output-dir', default=None,
                       help='输出目录（默认自动生成）')
    parser.add_argument('--timeout', type=int, default=None,
                       help='单个模拟超时时间（秒，默认1小时）')
    parser.add_argument('--y-filter-min', type=float, default=0.4,
                       help='Y坐标过滤最小值（默认: 0.4）')
    parser.add_argument('--y-filter-max', type=float, default=0.6,
                       help='Y坐标过滤最大值（默认: 0.6）')

    # 分析模式选项
    parser.add_argument('--analyze-only', type=str, default=None,
                       help='仅运行应力分析，指定现有实验结果目录路径')
    parser.add_argument('--analysis-output-dir', type=str, default=None,
                       help='分析结果输出目录（用于--analyze-only模式）')

    args = parser.parse_args()

    # 确定配置文件路径
    if args.use_schwarz:
        base_config_path = "config/schwarz_2d_test3_1.json"
        if not os.path.exists(base_config_path):
            # 备选配置文件
            base_config_path = "config/schwarz_2d_test3.json"
        solver_name = "schwarz"
    else:
        base_config_path = "config/config_2d_test3_1.json"
        if not os.path.exists(base_config_path):
            # 备选配置文件
            base_config_path = "config/config_2d_test3.json"
        solver_name = "single"

    if not os.path.exists(base_config_path):
        print(f"错误: 找不到基础配置文件: {base_config_path}")
        print("请确保配置文件存在")
        return

    # 检查是否为仅分析模式
    if args.analyze_only:
        print("检测到仅分析模式")
        try:
            results = run_analysis_only(
                experiment_dir=args.analyze_only,
                base_config_path=base_config_path,
                output_dir=args.analysis_output_dir,
                use_schwarz=args.use_schwarz,
                y_filter_min=args.y_filter_min,
                y_filter_max=args.y_filter_max
            )
            print(f"分析完成，生成了 {len(results)} 个对比图")
            return
        except Exception as e:
            print(f"分析失败: {e}")
            return

    # 确定网格大小
    if args.grid_sizes:
        grid_sizes = args.grid_sizes
    else:
        grid_start, grid_end = args.grid_range
        grid_sizes = list(range(grid_start, grid_end + 1, args.grid_step))

    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grid_range_str = f"grid{min(grid_sizes)}-{max(grid_sizes)}"
        output_dir = f"experiment_results/grid_study_{solver_name}_{grid_range_str}_{timestamp}"

    print(f"配置文件: {base_config_path}")
    print(f"网格大小: {grid_sizes}")
    print(f"求解器: {'Schwarz双域' if args.use_schwarz else '单域'}")
    print(f"输出目录: {output_dir}")

    # 运行批量实验
    results = run_batch_grid_study(
        base_config_path=base_config_path,
        grid_sizes=grid_sizes,
        use_schwarz=args.use_schwarz,
        output_dir=output_dir,
        timeout=args.timeout,
        y_filter_min=args.y_filter_min,
        y_filter_max=args.y_filter_max
    )

    return results

if __name__ == "__main__":
    main()