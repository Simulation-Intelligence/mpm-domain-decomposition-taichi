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

# 延迟导入Taichi相关模块，避免在不需要时初始化
try:
    import taichi as ti
    from Util.Config import Config
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def get_actual_volume_force_mass(config_path, mpm_instance=None):
    """获取实际的体积力对应的粒子总质量"""
    if mpm_instance is None:
        return None

    # 获取体积力对应的粒子总质量
    # 对于ImplicitMPM，需要通过solver属性访问
    if hasattr(mpm_instance, 'solver'):
        solver = mpm_instance.solver
    elif hasattr(mpm_instance, 'get_volume_force_masses'):
        solver = mpm_instance
    else:
        print("警告: 无法找到体积力质量获取方法")
        return None

    volume_force_masses = solver.get_volume_force_masses()

    if volume_force_masses:
        print(f"体积力实际粒子总质量: {volume_force_masses}")
        return volume_force_masses[0]  # 假设只有一个体积力
    return None

def extract_parameters_from_config(config, actual_mass=None):
    """从配置文件提取所有必要参数"""

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
    E = material['E']
    nu = material['nu']
    rho = material['rho']

    # 几何参数 - 圆柱半径
    for shape in domain_config['shapes']:
        if shape['type'] == 'ellipse' and shape['operation'] == 'add':
            center = shape['params']['center']
            radius = shape['params']['semi_axes'][0]
            break

    # 力参数 - 检查是否存在volume_forces
    applied_pressure = 100.0  # 默认值
    load_height = 0.005  # 默认值

    if 'volume_forces' in domain_config and domain_config['volume_forces']:
        volume_force = domain_config['volume_forces'][0]
        force_magnitude = abs(volume_force['force'][1])  # 只关心垂直方向

        if actual_mass is not None:
            # 使用实际粒子总质量计算等效压力
            # 总力 = 体力 × 实际总质量
            total_force = force_magnitude * actual_mass

            # 载荷区域长度计算
            force_range = volume_force['params']['range']
            load_width = force_range[0][1] - force_range[0][0]  

            # 等效接触压力 = 总力 / 载荷区域长度
            applied_pressure = total_force / load_width

            print(f"使用实际粒子质量计算:")
            print(f"  体力大小: {force_magnitude} N/kg")
            print(f"  实际粒子总质量: {actual_mass:.6f} kg")
            print(f"  总力: {total_force:.6f} N")
            print(f"  载荷宽度: {load_width} m")
            print(f"  等效接触压力: {applied_pressure:.2f} Pa")
        else:
            # 载荷区域
            force_range = volume_force['params']['range']
            load_height = force_range[1][1] - force_range[1][0]

            # 计算等效接触压力：体力 × 材料密度 × 载荷高度
            applied_pressure = force_magnitude * rho * load_height
            print(f"使用理论密度计算:")
            print(f"  体力大小: {force_magnitude} N/kg")
            print(f"  材料密度: {rho} kg/m³")
            print(f"  载荷高度: {load_height} m")
            print(f"  等效接触压力: {applied_pressure:.2f} Pa")
    else:
        print("未找到volume_forces，使用默认压力值")

    # 应力输出区域
    if 'stress_output_regions' in domain_config:
        output_region = domain_config['stress_output_regions'][0]['params']['range']
        x_range = output_region[0]
        y_range = output_region[1]
    else:
        # 默认分析底部区域
        x_range = [0, 1]
        y_range = [0, 0.1]

    # 网格参数
    if 'Domain1' in config:
        # Schwarz配置：从Domain1获取网格参数
        grid_size = config['Domain1']['grid_size']
        particles_per_grid = config['Domain1']['particles_per_grid']
    else:
        # 单域配置：从顶层获取
        grid_size = domain_config['grid_size']
        particles_per_grid = domain_config['particles_per_grid']

    return {
        'E': E, 'nu': nu, 'rho': rho,
        'center': center, 'radius': radius,
        'applied_pressure': applied_pressure,
        'load_height': load_height,
        'x_range': x_range, 'y_range': y_range,
        'grid_size': grid_size,
        'particles_per_grid': particles_per_grid,
        'config': config
    }

def calculate_hertz_stress_field(config_path, actual_max_y=None, actual_mass=None):
    """计算y=0位置的Hertz接触压力分布"""
    from analytical_solutions.test3 import calculate_hertz_pressure_at_y_zero, calculate_contact_width

    params = extract_parameters_from_config(load_config(config_path), actual_mass)

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

    # 网格参数 - 从参数字典中获取
    grid_size = params['grid_size']
    particles_per_grid = params['particles_per_grid']
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
    """运行单域MPM模拟"""
    if not TAICHI_AVAILABLE:
        raise ImportError("Taichi is not available. Cannot run MPM simulation.")

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

    print("Initializing single-domain MPM simulator...")
    mpm = ImplicitMPM(cfg)

    # 获取实际体积力粒子质量
    actual_mass = get_actual_volume_force_mass(config_path, mpm)

    # 运行模拟
    i = 0
    while mpm.gui.running:
        mpm.step()
        mpm.render()
        i += 1

        # 自动停止条件
        if i >= mpm.recorder.max_frames:
            break

    # 记录最终帧的应力数据（模拟器会自动创建统一目录结构）
    print("Recording final frame stress data...")
    mpm.save_stress_strain_data(i)

    if mpm.recorder is None:
        exit()
    print("Single-domain simulation completed.")

    # 返回最新创建的目录（通过查找最新时间戳目录）和实际质量
    import glob
    single_dirs = glob.glob("experiment_results/single_domain_*")
    if single_dirs:
        return max(single_dirs), actual_mass
    return None, actual_mass

def run_schwarz_simulation(config_path):
    """运行Schwarz域分解MPM模拟"""
    if not TAICHI_AVAILABLE:
        raise ImportError("Taichi is not available. Cannot run Schwarz simulation.")

    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulators'))

    # 导入必要的模块
    from simulators.implicit_mpm_schwarz import MPM_Schwarz

    print(f"使用Schwarz配置运行模拟: {config_path}")

    # 读取配置文件
    cfg = Config(path=config_path)
    float_type = ti.f32 if cfg.get("float_type", "f32") == "f32" else ti.f64
    arch = cfg.get("arch", "cpu")
    if arch == "cuda":
        arch = ti.cuda
    elif arch == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, default_fp=float_type, device_memory_GB=20)

    # 创建Schwarz域分解MPM实例
    mpm = MPM_Schwarz(cfg)

    # 获取Domain1的实际体积力粒子质量（假设体积力应用在Domain1）
    actual_mass = None
    if hasattr(mpm, 'domain1') and mpm.domain1:
        actual_mass = get_actual_volume_force_mass(config_path, mpm.domain1)

    frame_count = 0
    max_frames = cfg.get("record_frames", 60)

    # 运行模拟
    while frame_count < max_frames:
        mpm.step()
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"完成帧 {frame_count}/{max_frames}")

    # 记录最终帧的应力和应变数据（模拟器会自动创建统一目录结构）
    print("记录最终帧的应力和应变数据...")
    mpm.save_stress_strain_data(frame_count)

    print("Schwarz模拟完成")

    # 返回最新创建的目录（通过查找最新时间戳目录）和实际质量
    import glob
    schwarz_dirs = glob.glob("experiment_results/schwarz_*")
    if schwarz_dirs:
        return max(schwarz_dirs), actual_mass
    return None, actual_mass


def load_mpm_bottom_layer_stress(thickness=0.005, use_schwarz=False):
    """加载MPM模拟结果中最底层的应力数据"""
    if use_schwarz:
        # Schwarz域分解：查找最新的时间戳目录
        import glob
        import re
        from datetime import datetime

        # 查找统一输出目录下的Schwarz结果
        base_output_dir = "experiment_results"
        if not os.path.exists(base_output_dir):
            print(f"Experiment results directory not found: {base_output_dir}")
            return None

        schwarz_pattern = os.path.join(base_output_dir, "schwarz_*")
        schwarz_dirs = glob.glob(schwarz_pattern)

        if not schwarz_dirs:
            print("No Schwarz output directories found")
            return None

        # 从目录名中提取时间戳并排序，选择最新的
        def extract_timestamp(dirname):
            # 提取路径中的最后一部分（目录名）
            basename = os.path.basename(dirname)
            match = re.search(r'schwarz_(\d{8}_\d{6})', basename)
            if match:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            return datetime.min

        # 按时间戳排序，选择最新的
        latest_dir = max(schwarz_dirs, key=extract_timestamp)
        result_dir = latest_dir

        print(f"Found {len(schwarz_dirs)} Schwarz output directories, using latest: {latest_dir}")

        # 寻找Domain2的应力文件
        domain2_stress_files = [f for f in os.listdir(result_dir)
                               if f.startswith('domain2_stress_frame_') and f.endswith('.npy')]
        if not domain2_stress_files:
            print("No Domain2 stress data files found")
            return None

        stress_file = domain2_stress_files[0]
        frame_num = stress_file.split('_')[3].split('.')[0]

        stress_data = np.load(os.path.join(result_dir, f'domain2_stress_frame_{frame_num}.npy'))
        positions = np.load(os.path.join(result_dir, f'domain2_positions_frame_{frame_num}.npy'))

        print(f"Loaded {len(positions)} particles from Domain2, frame {frame_num}")
    else:
        # 单域MPM：查找最新的时间戳目录
        import glob
        import re
        from datetime import datetime

        # 查找统一输出目录下的单域MPM结果
        base_output_dir = "experiment_results"
        if not os.path.exists(base_output_dir):
            print(f"Experiment results directory not found: {base_output_dir}")
            return None

        mpm_pattern = os.path.join(base_output_dir, "single_domain_*")
        mpm_dirs = glob.glob(mpm_pattern)

        if not mpm_dirs:
            print("No single-domain MPM output directories found")
            return None

        # 从目录名中提取时间戳并排序，选择最新的
        def extract_timestamp_single(dirname):
            # 提取路径中的最后一部分（目录名）
            basename = os.path.basename(dirname)
            match = re.search(r'single_domain_(\d{8}_\d{6})', basename)
            if match:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            return datetime.min

        # 按时间戳排序，选择最新的
        latest_dir = max(mpm_dirs, key=extract_timestamp_single)
        result_dir = latest_dir

        print(f"Found {len(mpm_dirs)} single-domain output directories, using latest: {latest_dir}")

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

        print(f"Loaded {len(positions)} particles from frame {frame_num}")

    try:

        # 找到最底层的粒子和最大y坐标
        y_min = positions[:, 1].min()
        y_max = positions[:, 1].max()
        bottom_mask = positions[:, 1] <= (y_min + thickness)

        bottom_positions = positions[bottom_mask]
        bottom_stress = stress_data[bottom_mask]

        solver_type = "Domain2" if use_schwarz else "Single-domain"
        print(f"{solver_type} 粒子y坐标范围: {y_min:.4f} - {y_max:.4f}")
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
    contact_indices = (x_coords >= center_x - contact_width*1.2) & (x_coords <= center_x + contact_width*1.2)
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
    parser.add_argument('--output-dir', default="experiment_results/analytical",
                       help='输出目录')
    parser.add_argument('--no-save-image', action='store_true',
                       help='不保存图像')
    parser.add_argument('--analytical-only', action='store_true',
                       help='只计算解析解，不运行MPM模拟')
    parser.add_argument('--bottom-thickness', type=float, default=0.001,
                       help='底层粒子厚度 (默认: 0.001)')
    parser.add_argument('--use-schwarz', action='store_true',
                       help='使用Schwarz域分解方法运行和比较')
    parser.add_argument('--schwarz-config', default="config/schwarz_2d_test3.json",
                       help='Schwarz域分解配置文件路径')

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
    if args.use_schwarz:
        # Schwarz域分解模式
        print("\n1. 运行Schwarz域分解模拟")
        _, actual_mass = run_schwarz_simulation(args.schwarz_config)

        # 2. 加载Domain2应力数据，获取实际粒子分布
        print(f"\n2. 加载Domain2底层应力数据 (厚度: {args.bottom_thickness})")
        mpm_results = load_mpm_bottom_layer_stress(args.bottom_thickness, use_schwarz=True)

        if mpm_results is None:
            print("Domain2结果加载失败，无法进行对比")
            return

        # 3. 根据实际粒子最大y和实际质量计算解析解
        actual_max_y = mpm_results['y_max']
        print(f"\n3. 根据Domain2粒子分布和实际质量计算Hertz接触解析解 (y_max={actual_max_y:.4f})")
        analytical_results = calculate_hertz_stress_field(args.schwarz_config, actual_max_y, actual_mass)
        save_results(analytical_results, args.output_dir)

        print(f"解析解垂直应力统计:")
        print(f"σ_yy: 最小值={analytical_results['stress_yy'].min():.2e} Pa")
        print(f"σ_yy: 最大值={analytical_results['stress_yy'].max():.2e} Pa")

        # 4. 对比分析
        print("\n4. 对比解析解与Domain2结果")
        compare_with_mpm(analytical_results, mpm_results, args.output_dir, save_image=not args.no_save_image)
        print(f"\nSchwarz域分解实验3完成! 结果保存在: {args.output_dir}")

    else:
        # 单域MPM模式
        # 1. 运行MPM模拟
        print("\n1. 运行MPM模拟")
        _, actual_mass = run_mpm_simulation(args.config)

        # 2. 加载MPM应力数据，获取实际粒子分布
        print(f"\n2. 加载MPM底层应力数据 (厚度: {args.bottom_thickness})")
        mpm_results = load_mpm_bottom_layer_stress(args.bottom_thickness, use_schwarz=False)

        if mpm_results is None:
            print("MPM结果加载失败，无法进行对比")
            return

        # 3. 根据实际粒子最大y和实际质量计算解析解
        actual_max_y = mpm_results['y_max']
        print(f"\n3. 根据实际粒子分布和实际质量计算Hertz接触解析解 (y_max={actual_max_y:.4f})")
        analytical_results = calculate_hertz_stress_field(args.config, actual_max_y, actual_mass)
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