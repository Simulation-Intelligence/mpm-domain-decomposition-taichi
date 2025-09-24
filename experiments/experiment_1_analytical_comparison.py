#!/usr/bin/env python3
"""
实验1: 模拟结果与解析解对比
从config/config_2d_test1.json读取参数，使用analytical_solutions/test1.py计算解析解
运行MPM模拟并对比结果
"""

import json
import numpy as np
import sys
import os
import subprocess
import time
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analytical_solutions.test1 import calculate_eshelby_stress

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def extract_material_params(config):
    """从配置中提取材料参数"""
    materials = config['material_params']

    # 默认材料 (基体)
    base_material = next(m for m in materials if m['id'] == 0)
    E1 = base_material['E']
    nu1 = base_material['nu']
    rho1 = base_material['rho']

    # 软材料 (夹杂)
    inclusion_material = next(m for m in materials if m['id'] == 1)
    E2 = inclusion_material['E']
    nu2 = inclusion_material['nu']
    rho2 = inclusion_material['rho']

    return E1, nu1, rho1, E2, nu2, rho2

def extract_geometry_params(config):
    """从配置中提取几何参数"""
    # 找到椭圆夹杂
    for shape in config['shapes']:
        if shape['type'] == 'ellipse' and shape['operation'] == 'change':
            center = shape['params']['center']
            semi_axes = shape['params']['semi_axes']
            # 假设是圆形夹杂，取第一个半轴作为半径
            radius = semi_axes[0]
            return center, radius

    raise ValueError("未找到椭圆形夹杂定义")

def extract_loading_conditions(config):
    """从配置中提取载荷条件"""
    # 从体力中推断远场应力
    volume_forces = config['volume_forces']

    # 获取基本参数
    materials = config['material_params']
    base_material = next(m for m in materials if m['id'] == 0)
    rho = base_material['rho']  # 密度

    stress_contributions_x = []
    stress_contributions_y = []

    print("体力到远场应力转换:")
    print("-" * 40)

    for force_config in volume_forces:
        if force_config['type'] == 'rectangle':
            range_x = force_config['params']['range'][0]
            range_y = force_config['params']['range'][1]
            force = force_config['force']

            # 计算区域面积
            width = range_x[1] - range_x[0]
            height = range_y[1] - range_y[0]
            region_area = width * height

            # 计算区域内总质量
            total_mass = rho * region_area

            # 计算总力 = 区域内总质量 × 体力加速度
            total_force_x = total_mass * force[0]  # N/m (2D情况)
            total_force_y = total_mass * force[1]  # N/m (2D情况)

            # 根据力的方向判断边界并计算等效应力
            if force[0] != 0:  # X方向有力，影响σ_xx
                boundary_length = height  # Y方向长度作为边界长度
                stress_contribution = total_force_x / boundary_length
                stress_contributions_x.append(abs(stress_contribution))  # 取绝对值
                force_direction = "左边界" if force[0] < 0 else "右边界"
                print(f"{force_direction}: 区域面积={region_area:.4f}, 总力={total_force_x:.4f} N/m, 边界长度={boundary_length:.2f}, 应力贡献={stress_contribution:.4f} Pa")

            if force[1] != 0:  # Y方向有力，影响σ_yy
                boundary_length = width  # X方向长度作为边界长度
                stress_contribution = total_force_y / boundary_length
                stress_contributions_y.append(abs(stress_contribution))  # 取绝对值
                force_direction = "下边界" if force[1] < 0 else "上边界"
                print(f"{force_direction}: 区域面积={region_area:.4f}, 总力={total_force_y:.4f} N/m, 边界长度={boundary_length:.2f}, 应力贡献={stress_contribution:.4f} Pa")

    # 计算等效远场应力：取各边界应力的平均值
    sigma_inf_xx = sum(stress_contributions_x) / len(stress_contributions_x) if stress_contributions_x else 0.0
    sigma_inf_yy = sum(stress_contributions_y) / len(stress_contributions_y) if stress_contributions_y else 0.0
    sigma_inf_xy = 0.0

    # 暂时假设无剪切载荷
    sigma_inf_xy = 0.0

    print(f"最终远场应力: σ_xx={sigma_inf_xx:.4f} Pa, σ_yy={sigma_inf_yy:.4f} Pa, σ_xy={sigma_inf_xy:.4f} Pa")
    print()

    return sigma_inf_xx, sigma_inf_yy, sigma_inf_xy

def calculate_analytical_solution(config_path):
    """计算解析解"""
    # 加载配置
    config = load_config(config_path)

    # 提取参数
    E1, nu1, rho1, E2, nu2, rho2 = extract_material_params(config)
    center, radius = extract_geometry_params(config)
    sigma_inf_xx, sigma_inf_yy, sigma_inf_xy = extract_loading_conditions(config)

    print("材料参数:")
    print(f"基体材料: E1={E1:.0e} Pa, nu1={nu1}, rho1={rho1}")
    print(f"夹杂材料: E2={E2:.0e} Pa, nu2={nu2}, rho2={rho2}")
    print(f"夹杂中心: {center}")
    print(f"夹杂半径: {radius}")
    print(f"远场应力: σ_xx={sigma_inf_xx:.0e} Pa, σ_yy={sigma_inf_yy}, σ_xy={sigma_inf_xy}")
    print()

    # 计算网格点
    grid_size = config['grid_size']
    scale = config['scale']
    particles_per_grid = config['particles_per_grid']

    # 考虑粒子密度的有效分辨率
    effective_resolution = grid_size * int(np.sqrt(particles_per_grid))

    # 创建计算网格 (归一化坐标)
    x_coords = np.linspace(0, scale, effective_resolution)
    y_coords = np.linspace(0, scale, effective_resolution)

    # 存储结果
    stress_xx = np.zeros((effective_resolution, effective_resolution))
    stress_yy = np.zeros((effective_resolution, effective_resolution))
    stress_xy = np.zeros((effective_resolution, effective_resolution))

    # 计算每个网格点的应力
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # 转换到以夹杂中心为原点的坐标系
            x_rel = x - center[0]
            y_rel = y - center[1]

            # 计算解析解
            sigma_xx, sigma_yy, sigma_xy = calculate_eshelby_stress(
                x_rel, y_rel, E1, nu1, E2, nu2, radius,
                sigma_inf_xx, sigma_inf_yy, sigma_inf_xy
            )

            stress_xx[j, i] = sigma_xx
            stress_yy[j, i] = sigma_yy
            stress_xy[j, i] = sigma_xy

    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'stress_xx': stress_xx,
        'stress_yy': stress_yy,
        'stress_xy': stress_xy,
        'config': config,
        'material_params': {
            'E1': E1, 'nu1': nu1, 'rho1': rho1,
            'E2': E2, 'nu2': nu2, 'rho2': rho2
        },
        'geometry': {'center': center, 'radius': radius},
        'loading': {'sigma_inf_xx': sigma_inf_xx, 'sigma_inf_yy': sigma_inf_yy, 'sigma_inf_xy': sigma_inf_xy}
    }

def save_results(results, output_dir):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存应力场数据
    np.savez(os.path.join(output_dir, 'analytical_stress_field.npz'),
             x_coords=results['x_coords'],
             y_coords=results['y_coords'],
             stress_xx=results['stress_xx'],
             stress_yy=results['stress_yy'],
             stress_xy=results['stress_xy'])

    # 保存参数信息
    params = {
        'material_params': results['material_params'],
        'geometry': results['geometry'],
        'loading': results['loading']
    }

    with open(os.path.join(output_dir, 'analytical_params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    print(f"解析解结果已保存到: {output_dir}")

def run_simulation_and_compare(config_path, output_dir):
    """运行模拟并与解析解对比"""
    import taichi as ti
    from Util.Config import Config
    from simulators.implicit_mpm import ImplicitMPM

    # 初始化taichi，参照implicit_mpm.py的main函数
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

    print("初始化MPM模拟器...")
    mpm = ImplicitMPM(cfg)

    # 运行静力学求解（参照implicit_mpm.py）
    if mpm.static_solve:
        print("运行静力学求解...")
        mpm.run_static_solve()
        print("静力学求解完成")

        # 获取模拟结果
        simulation_results = load_simulation_results()
        return simulation_results
    else:
        # 原有的动态求解模式
        i = 0
        while mpm.gui.running:
            mpm.step()
            
            mpm.render()

            i += 1

            # 自动停止条件
            if i >= mpm.recorder.max_frames:
                break

        # 在最后一帧记录应力和应变数据
        print("记录最终帧的应力和应变数据...")
        mpm.save_stress_strain_data(i)

        if mpm.recorder is None:
            exit()
        print("Playback finished.")
        # 播放录制动画
        mpm.recorder.play(loop=True, fps=60)

def load_simulation_results():
    """加载模拟结果"""
    stress_output_dir = "stress_strain_output"

    if not os.path.exists(stress_output_dir):
        print(f"未找到模拟输出目录: {stress_output_dir}")
        return None

    # 找到最新的子目录
    subdirs = [d for d in os.listdir(stress_output_dir)
               if os.path.isdir(os.path.join(stress_output_dir, d))]

    if not subdirs:
        print("未找到模拟结果子目录")
        return None

    latest_dir = sorted(subdirs)[-1]
    result_dir = os.path.join(stress_output_dir, latest_dir)

    print(f"加载模拟结果: {result_dir}")

    try:
        stress_files = [f for f in os.listdir(result_dir) if f.startswith('stress_frame_') and f.endswith('.npy')]
        if not stress_files:
            print("未找到应力数据文件")
            return None

        stress_file = stress_files[0]
        frame_num = stress_file.split('_')[2].split('.')[0]

        stress_data = np.load(os.path.join(result_dir, f'stress_frame_{frame_num}.npy'))
        strain_data = np.load(os.path.join(result_dir, f'strain_frame_{frame_num}.npy'))
        positions = np.load(os.path.join(result_dir, f'positions_frame_{frame_num}.npy'))

        with open(os.path.join(result_dir, f'stress_strain_stats_frame_{frame_num}.json'), 'r') as f:
            stats = json.load(f)

        return {
            'stress': stress_data,
            'strain': strain_data,
            'positions': positions,
            'stats': stats,
            'result_dir': result_dir
        }

    except Exception as e:
        print(f"加载模拟结果时出错: {e}")
        return None

def compare_results(analytical_results, simulation_results, output_dir):
    """对比解析解和模拟结果"""
    print("对比解析解和模拟结果...")

    if simulation_results is None:
        print("无法对比：模拟结果为空")
        return

    sim_positions = simulation_results['positions']
    sim_stress = simulation_results['stress']

    # 从解析解结果中提取参数
    E1 = analytical_results['material_params']['E1']
    nu1 = analytical_results['material_params']['nu1']
    E2 = analytical_results['material_params']['E2']
    nu2 = analytical_results['material_params']['nu2']
    center = analytical_results['geometry']['center']
    radius = analytical_results['geometry']['radius']
    sigma_inf_xx = analytical_results['loading']['sigma_inf_xx']
    sigma_inf_yy = analytical_results['loading']['sigma_inf_yy']
    sigma_inf_xy = analytical_results['loading']['sigma_inf_xy']

    print(f"在 {len(sim_positions)} 个模拟粒子位置计算解析解...")

    analytical_stress_at_particles = []
    for pos in sim_positions:
        x_rel = pos[0] - center[0]
        y_rel = pos[1] - center[1]

        sigma_xx, sigma_yy, sigma_xy = calculate_eshelby_stress(
            x_rel, y_rel, E1, nu1, E2, nu2, radius,
            sigma_inf_xx, sigma_inf_yy, sigma_inf_xy
        )

        analytical_stress_at_particles.append([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])

    analytical_stress_at_particles = np.array(analytical_stress_at_particles)

    # 计算差异和相对误差
    stress_diff = sim_stress - analytical_stress_at_particles
    analytical_magnitude = np.sqrt(analytical_stress_at_particles[:, 0, 0]**2 +
                                   analytical_stress_at_particles[:, 1, 1]**2 +
                                   analytical_stress_at_particles[:, 0, 1]**2)
    sim_magnitude = np.sqrt(sim_stress[:, 0, 0]**2 + sim_stress[:, 1, 1]**2 + sim_stress[:, 0, 1]**2)

    relative_error = np.where(analytical_magnitude > 1e-10,
                             np.abs(sim_magnitude - analytical_magnitude) / analytical_magnitude,
                             0.0)

    # 保存对比结果
    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    np.savez(os.path.join(comparison_dir, 'comparison_data.npz'),
             sim_positions=sim_positions,
             sim_stress=sim_stress,
             analytical_stress=analytical_stress_at_particles,
             stress_diff=stress_diff,
             relative_error=relative_error)

    # 统计结果
    comparison_stats = {
        'n_points': len(sim_positions),
        'relative_error_stats': {
            'mean': float(np.mean(relative_error)),
            'std': float(np.std(relative_error)),
            'max': float(np.max(relative_error)),
            'percentiles': {
                '50': float(np.percentile(relative_error, 50)),
                '90': float(np.percentile(relative_error, 90)),
                '95': float(np.percentile(relative_error, 95))
            }
        }
    }

    with open(os.path.join(comparison_dir, 'comparison_stats.json'), 'w') as f:
        json.dump(comparison_stats, f, indent=2)

    print(f"\n对比统计结果:")
    print(f"对比点数: {comparison_stats['n_points']}")
    print(f"相对误差 - 平均: {comparison_stats['relative_error_stats']['mean']:.3%}")
    print(f"相对误差 - 最大: {comparison_stats['relative_error_stats']['max']:.3%}")
    print(f"相对误差 - 95%分位: {comparison_stats['relative_error_stats']['percentiles']['95']:.3%}")

    return comparison_stats

def visualize_analytical_results(analytical_results, output_dir, save_image=True):
    """可视化解析解结果"""
    from tools.visualize_stress import visualize_stress_2d, compute_von_mises_stress, compute_hydrostatic_pressure

    # 从解析解结果中构造粒子位置和应力数据
    x_coords = analytical_results['x_coords']
    y_coords = analytical_results['y_coords']
    stress_xx = analytical_results['stress_xx']
    stress_yy = analytical_results['stress_yy']
    stress_xy = analytical_results['stress_xy']

    # 创建粒子位置数组 (N, 2)
    X, Y = np.meshgrid(x_coords, y_coords)
    positions = np.column_stack([X.ravel(), Y.ravel()])

    # 创建应力张量数组 (N, 2, 2)
    n_points = len(positions)
    stress_tensor = np.zeros((n_points, 2, 2))
    stress_tensor[:, 0, 0] = stress_xx.ravel()  # σ_xx
    stress_tensor[:, 1, 1] = stress_yy.ravel()  # σ_yy
    stress_tensor[:, 0, 1] = stress_xy.ravel()  # σ_xy
    stress_tensor[:, 1, 0] = stress_xy.ravel()  # σ_yx = σ_xy

    # 计算von Mises应力和静水压力
    von_mises = compute_von_mises_stress(stress_tensor)
    pressure = compute_hydrostatic_pressure(stress_tensor)

    # 保存图像路径
    save_path = None
    if save_image:
        save_path = os.path.join(output_dir, 'analytical_stress_visualization.png')

    # 调用可视化函数
    visualize_stress_2d(positions, von_mises, pressure,
                       frame_number="Analytical", save_path=save_path)

    print(f"\n解析解可视化完成!")
    if save_image:
        print(f"图像已保存到: {save_path}")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实验1: 模拟结果与解析解对比')
    parser.add_argument('--analytical-only', action='store_true',
                       help='仅计算和可视化解析解，不运行MPM模拟')
    parser.add_argument('--config', default="config/config_2d_test1.json",
                       help='配置文件路径 (默认: config/config_2d_test1.json)')
    parser.add_argument('--output-dir', default="experiments/experiment_1_results",
                       help='输出目录 (默认: experiments/experiment_1_results)')
    parser.add_argument('--no-save-image', action='store_true',
                       help='不保存可视化图像，仅显示')

    args = parser.parse_args()

    config_path = args.config
    output_dir = args.output_dir

    print("实验1: 模拟结果与解析解对比")
    print("=" * 50)

    # 1. 计算解析解
    print("\n1. 计算解析解")
    analytical_results = calculate_analytical_solution(config_path)
    save_results(analytical_results, output_dir)

    print(f"解析解应力场统计:")
    print(f"σ_xx: 最小值={analytical_results['stress_xx'].min():.2e}, 最大值={analytical_results['stress_xx'].max():.2e}")
    print(f"σ_yy: 最小值={analytical_results['stress_yy'].min():.2e}, 最大值={analytical_results['stress_yy'].max():.2e}")

    # 2. 可视化解析解
    print("\n2. 可视化解析解")
    visualize_analytical_results(analytical_results, output_dir, save_image=not args.no_save_image)

    # 3. 根据参数决定是否运行模拟
    if args.analytical_only:
        print(f"\n仅解析解模式完成! 结果保存在: {output_dir}")
        return

    # 4. 运行模拟并对比
    print("\n3. 运行MPM模拟")
    simulation_results = run_simulation_and_compare(config_path, output_dir)

    if simulation_results is not None:
        print(f"模拟结果: {len(simulation_results['positions'])} 个粒子")

        # 5. 对比结果
        print("\n4. 对比分析")
        comparison_stats = compare_results(analytical_results, simulation_results, output_dir)

        print(f"\n实验完成! 结果保存在: {output_dir}")
    else:
        print("模拟失败，无法进行对比")

if __name__ == "__main__":
    main()