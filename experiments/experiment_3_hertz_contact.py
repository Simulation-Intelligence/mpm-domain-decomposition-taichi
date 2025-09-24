#!/usr/bin/env python3
"""
实验3: Hertz接触问题分析 - 垂直力
基于config/config_2d_test3.json，分析圆柱体与弹性基础的接触应力
重点关注垂直方向的接触压力分布
"""

import json
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def extract_parameters_from_config(config, actual_max_y=None):
    """从配置文件提取所有必要参数"""
    # 材料参数
    material = config['material_params'][0]
    E = material['E']
    nu = material['nu']
    rho = material['rho']

    # 几何参数 - 圆柱半径
    for shape in config['shapes']:
        if shape['type'] == 'ellipse' and shape['operation'] == 'add':
            center = shape['params']['center']
            radius = shape['params']['semi_axes'][0]
            break

    # 载荷参数 - 垂直体力
    volume_force = config['volume_forces'][0]
    force_magnitude = abs(volume_force['force'][1])  # 只关心垂直方向

    # 载荷区域 - 如果提供了实际最大y，则更新载荷高度
    force_range = volume_force['params']['range']
    load_height = force_range[1][1] - force_range[1][0]

    # 计算等效接触压力：体力 × 材料密度 × 载荷高度
    applied_pressure = force_magnitude * rho * load_height

    # 应力输出区域
    if 'stress_output_regions' in config:
        output_region = config['stress_output_regions'][0]['params']['range']
        x_range = output_region[0]
        y_range = output_region[1]
    else:
        # 默认分析底部区域
        x_range = [0, 1]
        y_range = [0, 0.1]

    return {
        'E': E, 'nu': nu, 'rho': rho,
        'center': center, 'radius': radius,
        'applied_pressure': applied_pressure,
        'load_height': load_height,
        'x_range': x_range, 'y_range': y_range,
        'config': config
    }

def calculate_hertz_stress_field(config_path, actual_max_y=None):
    """计算y=0位置的Hertz接触压力分布"""
    from analytical_solutions.test3 import calculate_hertz_pressure_at_y_zero, calculate_contact_width

    params = extract_parameters_from_config(load_config(config_path), actual_max_y)

    print("Hertz接触问题参数:")
    print(f"材料: E={params['E']:.0e} Pa, nu={params['nu']}, rho={params['rho']}")
    print(f"圆柱半径: {params['radius']} m")
    print(f"等效接触压力: {params['applied_pressure']:.2f} Pa")
    print(f"分析区域: x∈{params['x_range']}")
    print()

    # 计算接触宽度
    contact_width = calculate_contact_width(
        params['radius'], params['E'], params['nu'], params['applied_pressure'])
    print(f"接触半宽度: {contact_width:.4f} m")

    # 网格参数
    config = params['config']
    grid_size = config['grid_size']
    particles_per_grid = config['particles_per_grid']
    effective_resolution = grid_size * int(np.sqrt(particles_per_grid)) * 100

    # 创建x坐标数组（只需要x坐标，因为只计算y=0）
    x_range = params['x_range']
    nx = int(effective_resolution * (x_range[1] - x_range[0]))
    x_coords = np.linspace(x_range[0], x_range[1], nx)

    print(f"网格分辨率: {nx} 个点")

    # 计算y=0位置的压力分布
    x_relative = x_coords - params['center'][0]  # 相对于圆柱中心的x坐标
    pressure_y_zero = calculate_hertz_pressure_at_y_zero(
        x_relative, params['radius'], params['E'], params['nu'], params['applied_pressure'])

    # 转换为应力（压应力为负）
    stress_yy = -pressure_y_zero

    return {
        'x_coords': x_coords,
        'stress_yy': stress_yy,  # 一维数组
        'params': params
    }

def visualize_results(results, output_dir, save_image=True):
    """Visualize contact pressure distribution"""
    from analytical_solutions.test3 import calculate_contact_width

    x_coords = results['x_coords']
    stress_yy = results['stress_yy']
    params = results['params']

    # Calculate contact width
    contact_width = calculate_contact_width(
        params['radius'], params['E'], params['nu'], params['applied_pressure'])

    # Create figure with single plot
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Pressure distribution at y=0 (now stress_yy is 1D array)
    center_x = params['center'][0]

    # Convert to positive pressure in kPa
    pressure_contact = -stress_yy/1000
    x_contact = x_coords

    ax.plot(x_contact, pressure_contact, 'b-', linewidth=2, label='Hertz contact pressure')
    ax.axvline(x=center_x - contact_width, color='red', linestyle='--', alpha=0.7, label=f'Contact boundary (±{contact_width:.9f}m)')
    ax.axvline(x=center_x + contact_width, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Pressure (kPa)')
    ax.set_title('Hertz Contact Pressure Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(center_x - contact_width*1.2, center_x + contact_width*1.2)

    plt.tight_layout()

    if save_image:
        save_path = os.path.join(output_dir, 'hertz_contact_pressure.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")

    plt.show()

def save_results(results, output_dir):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存应力场数据（现在只有1D数据）
    np.savez(os.path.join(output_dir, 'hertz_contact_pressure_y0.npz'),
             x_coords=results['x_coords'],
             stress_yy=results['stress_yy'])

    # 保存参数
    params_to_save = {
        'material': {k: v for k, v in results['params'].items() if k != 'config'},
    }

    with open(os.path.join(output_dir, 'hertz_params.json'), 'w') as f:
        json.dump(params_to_save, f, indent=2)

    print(f"结果已保存到: {output_dir}")

def run_mpm_simulation(config_path):
    """运行MPM模拟"""
    import taichi as ti
    from Util.Config import Config
    from simulators.implicit_mpm import ImplicitMPM

    # 初始化taichi
    cfg = Config(config_path)
    float_type = ti.f32 if cfg.get("float_type", "f32") == "f32" else ti.f64
    arch = cfg.get("arch", "cpu")
    if arch == "cuda":
        arch = ti.cuda
    elif arch == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, default_fp=float_type, device_memory_GB=20)

    print("Initializing MPM simulator...")
    mpm = ImplicitMPM(cfg)

    # 运行模拟
    i = 0
    while mpm.gui.running:
        mpm.step()
        mpm.render()
        i += 1

        # 自动停止条件
        if i >= mpm.recorder.max_frames:
            break

    # 记录最终帧的应力数据
    print("Recording final frame stress data...")
    mpm.save_stress_strain_data(i)

    if mpm.recorder is None:
        exit()
    print("Simulation completed.")

def load_mpm_bottom_layer_stress(thickness=0.01):
    """加载MPM模拟结果中最底层的应力数据"""
    stress_output_dir = "stress_strain_output"

    if not os.path.exists(stress_output_dir):
        print(f"MPM output directory not found: {stress_output_dir}")
        return None

    # 找到最新的子目录（按修改时间排序）
    subdirs = []
    for d in os.listdir(stress_output_dir):
        dir_path = os.path.join(stress_output_dir, d)
        if os.path.isdir(dir_path):
            subdirs.append((d, os.path.getmtime(dir_path)))

    if not subdirs:
        print("No MPM result subdirectories found")
        return None

    # 按修改时间排序，取最新的
    latest_dir = sorted(subdirs, key=lambda x: x[1])[-1][0]
    result_dir = os.path.join(stress_output_dir, latest_dir)

    print(f"Loading MPM results from: {result_dir}")

    try:
        # 找到应力文件
        stress_files = [f for f in os.listdir(result_dir)
                       if f.startswith('stress_frame_') and f.endswith('.npy')]
        if not stress_files:
            print("No stress data files found")
            return None

        stress_file = stress_files[0]
        frame_num = stress_file.split('_')[2].split('.')[0]

        stress_data = np.load(os.path.join(result_dir, f'stress_frame_{frame_num}.npy'))
        positions = np.load(os.path.join(result_dir, f'positions_frame_{frame_num}.npy'))

        print(f"Loaded {len(positions)} particles from frame {frame_num}")

        # 找到最底层的粒子和最大y坐标
        y_min = positions[:, 1].min()
        y_max = positions[:, 1].max()
        bottom_mask = positions[:, 1] <= (y_min + thickness)

        bottom_positions = positions[bottom_mask]
        bottom_stress = stress_data[bottom_mask]

        print(f"粒子y坐标范围: {y_min:.4f} - {y_max:.4f}")
        print(f"Found {len(bottom_positions)} particles in bottom layer (y <= {y_min + thickness:.3f})")

        # 按x坐标分组并平均应力
        x_coords = bottom_positions[:, 0]

        # 创建x坐标的分组，使用固定的分辨率
        x_min, x_max = x_coords.min(), x_coords.max()
        num_bins = min(100, len(bottom_positions))  # 最多100个bins或粒子数
        x_edges = np.linspace(x_min, x_max, num_bins + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2

        # 将粒子分配到x坐标bins
        bin_indices = np.digitize(x_coords, x_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # 为每个bin计算平均应力
        averaged_stress = []
        averaged_positions = []

        for i in range(num_bins):
            mask = bin_indices == i
            if np.any(mask):
                # 计算该bin内粒子的平均应力
                avg_stress = np.mean(bottom_stress[mask], axis=0)
                avg_y = np.mean(bottom_positions[mask, 1])
                averaged_stress.append(avg_stress)
                averaged_positions.append([x_centers[i], avg_y])

        if averaged_stress:
            averaged_stress = np.array(averaged_stress)
            averaged_positions = np.array(averaged_positions)

            # 按x坐标排序
            sort_indices = np.argsort(averaged_positions[:, 0])
            averaged_positions = averaged_positions[sort_indices]
            averaged_stress = averaged_stress[sort_indices]

            print(f"Averaged to {len(averaged_positions)} x-coordinate bins")
        else:
            averaged_positions = bottom_positions
            averaged_stress = bottom_stress

        return {
            'positions': averaged_positions,
            'stress': averaged_stress,
            'y_range': [y_min, y_min + thickness],
            'y_max': y_max  # 添加最大y坐标
        }

    except Exception as e:
        print(f"Error loading MPM results: {e}")
        return None

def compare_with_mpm(analytical_results, mpm_results, output_dir, save_image=True):
    """对比解析解和MPM结果"""
    from analytical_solutions.test3 import calculate_contact_width

    params = analytical_results['params']

    # 计算接触宽度
    contact_width = calculate_contact_width(
        params['radius'], params['E'], params['nu'], params['applied_pressure'])

    # 创建图形
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 解析解 - y=0位置的压力分布
    x_coords = analytical_results['x_coords']
    stress_yy = analytical_results['stress_yy']  # 现在是1D数组
    center_x = params['center'][0]

    # 只在接触区域内显示
    contact_indices = (x_coords >= center_x - contact_width) & (x_coords <= center_x + contact_width)
    x_analytical = x_coords[contact_indices]
    pressure_analytical = -stress_yy[contact_indices]/1000

    ax.plot(x_analytical, pressure_analytical, 'r-', linewidth=2, label='Analytical (Hertz theory)')

    # MPM结果
    mpm_x = mpm_results['positions'][:, 0]
    mpm_stress_yy = mpm_results['stress'][:, 1, 1]  # σ_yy分量
    mpm_pressure = -mpm_stress_yy/1000  # 转换为正的压力值

    # 只显示接触区域内的MPM数据
    mpm_contact_mask = (mpm_x >= center_x - contact_width) & (mpm_x <= center_x + contact_width)
    mpm_x_contact = mpm_x[mpm_contact_mask]
    mpm_pressure_contact = mpm_pressure[mpm_contact_mask]

    ax.scatter(mpm_x_contact, mpm_pressure_contact, c='blue', s=20, alpha=0.7, label='MPM simulation')

    # 标记接触边界
    ax.axvline(x=center_x - contact_width, color='red', linestyle='--', alpha=0.7,
              label=f'Contact boundary (±{contact_width:.3f}m)')
    ax.axvline(x=center_x + contact_width, color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('Pressure (kPa)')
    ax.set_title('Hertz Contact Pressure: Analytical vs MPM Simulation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(center_x - contact_width*1.2, center_x + contact_width*1.2)

    plt.tight_layout()

    if save_image:
        save_path = os.path.join(output_dir, 'hertz_comparison_analytical_vs_mpm.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison image saved to: {save_path}")

    plt.show()

    # 计算统计误差
    print("\nComparison statistics:")
    print(f"Analytical pressure range: {pressure_analytical.min():.2f} - {pressure_analytical.max():.2f} kPa")
    print(f"MPM pressure range: {mpm_pressure_contact.min():.2f} - {mpm_pressure_contact.max():.2f} kPa")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实验3: Hertz接触垂直应力分析')
    parser.add_argument('--config', default="config/config_2d_test3.json",
                       help='配置文件路径')
    parser.add_argument('--output-dir', default="experiments/experiment_3_results",
                       help='输出目录')
    parser.add_argument('--no-save-image', action='store_true',
                       help='不保存图像')
    parser.add_argument('--analytical-only', action='store_true',
                       help='只计算解析解，不运行MPM模拟')
    parser.add_argument('--bottom-thickness', type=float, default=0.005,
                       help='底层粒子厚度 (默认: 0.005)')

    args = parser.parse_args()

    print("实验3: Hertz接触垂直应力分析")
    print("=" * 50)

    if args.analytical_only:
        # 只计算解析解模式
        print("\n1. 计算Hertz接触解析解")
        analytical_results = calculate_hertz_stress_field(args.config)
        save_results(analytical_results, args.output_dir)

        print(f"解析解垂直应力统计:")
        print(f"σ_yy: 最小值={analytical_results['stress_yy'].min():.2e} Pa")
        print(f"σ_yy: 最大值={analytical_results['stress_yy'].max():.2e} Pa")

        print("\n2. 可视化解析解")
        visualize_results(analytical_results, args.output_dir, save_image=not args.no_save_image)
        print(f"\n只计算解析解模式完成! 结果保存在: {args.output_dir}")
        return

    # 完整流程：先模拟，再根据模拟结果计算解析解
    # 1. 运行MPM模拟
    print("\n1. 运行MPM模拟")
    run_mpm_simulation(args.config)

    # 2. 加载MPM应力数据，获取实际粒子分布
    print(f"\n2. 加载MPM底层应力数据 (厚度: {args.bottom_thickness})")
    mpm_results = load_mpm_bottom_layer_stress(args.bottom_thickness)

    if mpm_results is None:
        print("MPM结果加载失败，无法进行对比")
        return

    # 3. 根据实际粒子最大y计算解析解
    actual_max_y = mpm_results['y_max']
    print(f"\n3. 根据实际粒子分布计算Hertz接触解析解 (y_max={actual_max_y:.4f})")
    analytical_results = calculate_hertz_stress_field(args.config, actual_max_y)
    save_results(analytical_results, args.output_dir)

    print(f"解析解垂直应力统计:")
    print(f"σ_yy: 最小值={analytical_results['stress_yy'].min():.2e} Pa")
    print(f"σ_yy: 最大值={analytical_results['stress_yy'].max():.2e} Pa")

    # 4. 对比分析
    print("\n4. 对比解析解与MPM结果")
    compare_with_mpm(analytical_results, mpm_results, args.output_dir, save_image=not args.no_save_image)
    print(f"\n实验3完成! 结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()