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

def get_actual_volume_force_mass(config_path, mpm_instance=None, results_dir=None):
    """获取实际的体积力对应的粒子总质量"""

    # 如果提供了results_dir，尝试从保存的文件中读取
    if results_dir is not None:
        import glob

        # 检查是单域还是双域
        config = load_config(config_path)
        if 'Domain1' in config:
            # Schwarz配置，读取Domain1的actual mass
            mass_files = glob.glob(f"{results_dir}/domain1_actual_masses_frame_*.npy")
            if mass_files:
                # 取最新的文件（按文件名排序）
                latest_file = sorted(mass_files)[-1]
                actual_masses = np.load(latest_file)
                print(f"从文件读取Domain1实际质量: {actual_masses}")
                return actual_masses[0] if len(actual_masses) > 0 else None
        else:
            # 单域配置
            mass_files = glob.glob(f"{results_dir}/actual_masses_frame_*.npy")
            if mass_files:
                # 取最新的文件（按文件名排序）
                latest_file = sorted(mass_files)[-1]
                actual_masses = np.load(latest_file)
                print(f"从文件读取实际质量: {actual_masses}")
                return actual_masses[0] if len(actual_masses) > 0 else None

    # 如果没有results_dir或文件不存在，使用原来的方法
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
        grid_nx = config['Domain1'].get('grid_nx', 16)
        grid_ny = config['Domain1'].get('grid_ny', 16)
        grid_size = max(grid_nx, grid_ny)  # 兼容性计算
        particles_per_grid = config['Domain1']['particles_per_grid']
    else:
        # 单域配置：从顶层获取，支持新旧格式
        if 'grid_nx' in domain_config and 'grid_ny' in domain_config:
            grid_nx = domain_config['grid_nx']
            grid_ny = domain_config['grid_ny']
            grid_size = max(grid_nx, grid_ny)
        else:
            grid_size = domain_config.get('grid_size', 16)
            grid_nx = grid_ny = grid_size
        particles_per_grid = domain_config['particles_per_grid']

    return {
        'E': E, 'nu': nu, 'rho': rho,
        'center': center, 'radius': radius,
        'applied_pressure': applied_pressure,
        'load_height': load_height,
        'x_range': x_range, 'y_range': y_range,
        'grid_size': grid_size,
        'grid_nx': grid_nx, 'grid_ny': grid_ny,
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
    grid_nx = params.get('grid_nx', grid_size)
    grid_ny = params.get('grid_ny', grid_size)
    particles_per_grid = params['particles_per_grid']
    # 使用x方向网格数量来计算分辨率，因为我们主要分析x方向的压力分布
    effective_resolution = grid_nx * int(np.sqrt(particles_per_grid)) * 100

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
    """运行单域MPM模拟（使用子进程）"""
    import subprocess

    print("运行单域MPM模拟（子进程模式）...")

    # 使用子进程调用单域模拟器
    mpm_cmd = [
        "python", "simulators/implicit_mpm.py",
        "--config", config_path
    ]

    print(f"执行命令: {' '.join(mpm_cmd)}")

    try:
        # 不捕获输出，让模拟进度实时显示
        result = subprocess.run(mpm_cmd, timeout=7200)

        if result.returncode == 0:
            print("单域MPM模拟完成")
        else:
            print(f"单域MPM模拟失败，返回码: {result.returncode}")
            return None, None

    except subprocess.TimeoutExpired:
        print("单域MPM模拟超时（超过2小时）")
        return None, None
    except Exception as e:
        print(f"单域MPM模拟过程出错: {e}")
        return None, None

    # 获取实际体积力粒子质量（从保存的结果中读取）
    import glob
    single_dirs = glob.glob("experiment_results/single_domain_*")
    if single_dirs:
        latest_dir = max(single_dirs)
        actual_mass = get_actual_volume_force_mass(config_path, results_dir=latest_dir)
        return latest_dir, actual_mass

    return None, None

def run_schwarz_simulation(config_path):
    """运行Schwarz域分解MPM模拟（使用子进程）"""
    import subprocess

    print(f"运行Schwarz域分解MPM模拟（子进程模式）: {config_path}")

    # 使用子进程调用Schwarz模拟器
    mpm_cmd = [
        "python", "simulators/implicit_mpm_schwarz.py",
        "--config", config_path
    ]

    print(f"执行命令: {' '.join(mpm_cmd)}")

    try:
        # 不捕获输出，让模拟进度实时显示
        result = subprocess.run(mpm_cmd, timeout=7200)

        if result.returncode == 0:
            print("Schwarz域分解MPM模拟完成")
        else:
            print(f"Schwarz域分解MPM模拟失败，返回码: {result.returncode}")
            return None, None

    except subprocess.TimeoutExpired:
        print("Schwarz域分解MPM模拟超时（超过2小时）")
        return None, None
    except Exception as e:
        print(f"Schwarz域分解MPM模拟过程出错: {e}")
        return None, None

    # 获取实际体积力粒子质量（从保存的结果中读取）
    import glob
    schwarz_dirs = glob.glob("experiment_results/schwarz_*")
    if schwarz_dirs:
        latest_dir = max(schwarz_dirs)
        actual_mass = get_actual_volume_force_mass(config_path, results_dir=latest_dir)
        return latest_dir, actual_mass

    return None, None


def load_mpm_boundary_particle_stress(use_schwarz=False,  max_y_threshold=0.1):
    """加载MPM模拟结果中边界粒子的应力数据，可选择性过滤上边界粒子"""
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
        boundary_flags = np.load(os.path.join(result_dir, f'domain2_boundary_flags_frame_{frame_num}.npy'))

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
        boundary_flags = np.load(os.path.join(result_dir, f'boundary_flags_frame_{frame_num}.npy'))

        print(f"Loaded {len(positions)} particles from frame {frame_num}")

    try:

        # 过滤边界粒子
        y_min = positions[:, 1].min()
        y_max = positions[:, 1].max()
        boundary_mask = boundary_flags == 1  # 边界粒子标志为1

        # 如果提供了y阈值，进一步过滤掉y坐标太大的边界粒子
        if max_y_threshold is not None:
            y_filter_mask = positions[:, 1] <= max_y_threshold
            boundary_mask = boundary_mask & y_filter_mask
            print(f"应用y坐标过滤: y <= {max_y_threshold}")

        boundary_positions = positions[boundary_mask]
        boundary_stress = stress_data[boundary_mask]

        solver_type = "Domain2" if use_schwarz else "Single-domain"
        print(f"{solver_type} 粒子y坐标范围: {y_min:.4f} - {y_max:.4f}")

        if max_y_threshold is not None:
            boundary_y_max = boundary_positions[:, 1].max() if len(boundary_positions) > 0 else 0
            print(f"Found {len(boundary_positions)} filtered boundary particles (y <= {max_y_threshold}, max_y: {boundary_y_max:.4f}, total: {len(positions)})")
        else:
            print(f"Found {len(boundary_positions)} boundary particles (total: {len(positions)})")

        # 直接使用所有边界粒子，按x坐标排序
        sort_indices = np.argsort(boundary_positions[:, 0])
        sorted_positions = boundary_positions[sort_indices]
        sorted_stress = boundary_stress[sort_indices]

        print(f"Using all {len(sorted_positions)} boundary particles directly (no averaging)")

        # 注意：Schwarz模拟器保存的坐标数据已经包含了offset（全局坐标），无需再次添加

        return {
            'positions': sorted_positions,
            'stress': sorted_stress,
            'y_range': [y_min, y_max],
            'y_max': y_max,  # 添加最大y坐标
            'boundary_particle_count': len(boundary_positions),
            'results_dir': result_dir  # 添加结果目录路径
        }

    except Exception as e:
        print(f"Error loading MPM results: {e}")
        return None

def compare_with_mpm(analytical_results, mpm_results, output_dir, save_image=True, grid_size=None):
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

    # MPM结果
    mpm_x = mpm_results['positions'][:, 0]
    mpm_stress_yy = mpm_results['stress'][:, 1, 1]  # σ_yy分量
    mpm_pressure = -mpm_stress_yy/1000  # 转换为正的压力值

    # 只显示接触区域内的MPM数据
    mpm_contact_mask = (mpm_x >= center_x - contact_width) & (mpm_x <= center_x + contact_width)
    mpm_x_contact = mpm_x[mpm_contact_mask]
    mpm_pressure_contact = mpm_pressure[mpm_contact_mask]

    # 直接使用原始压力值，不进行平移
    ax.plot(x_analytical, pressure_analytical, 'r-', linewidth=2, label='Analytical (Hertz theory)')
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
        if grid_size is not None:
            filename = f'hertz_comparison_analytical_vs_mpm_grid{grid_size}.png'
        else:
            filename = 'hertz_comparison_analytical_vs_mpm.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison image saved to: {save_path}")

    # 在批量模式下不显示图形窗口，避免资源问题
    try:
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            plt.show()
        else:
            print("在批量模式下跳过图形显示")
    except:
        pass

    # 计算统计误差
    print("\nComparison statistics:")
    print(f"Analytical pressure range: {pressure_analytical.min():.2f} - {pressure_analytical.max():.2f} kPa")
    print(f"MPM pressure range: {mpm_pressure_contact.min():.2f} - {mpm_pressure_contact.max():.2f} kPa")

def save_grid_study_summary_data(analytical_results, all_mpm_results, all_grid_sizes, output_dir, contact_width, center_x):
    """保存网格研究汇总数据到文件"""
    import json

    # 提取解析解数据
    x_coords = analytical_results['x_coords']
    stress_yy = analytical_results['stress_yy']
    params = analytical_results['params']

    # 只保存接触区域内的解析解数据
    contact_indices = (x_coords >= center_x - contact_width*1.2) & (x_coords <= center_x + contact_width*1.2)
    x_analytical = x_coords[contact_indices]
    pressure_analytical = -stress_yy[contact_indices]/1000

    # 使用原始解析解压力值，不进行平移

    # 收集所有MPM数据
    mpm_data_by_grid = {}

    for i, (mpm_results, grid_size) in enumerate(zip(all_mpm_results, all_grid_sizes)):
        mpm_x = mpm_results['positions'][:, 0]
        mpm_stress_yy = mpm_results['stress'][:, 1, 1]
        mpm_pressure = -mpm_stress_yy/1000

        # 只保存接触区域内的MPM数据
        mpm_contact_mask = (mpm_x >= center_x - contact_width*1.2) & (mpm_x <= center_x + contact_width*1.2)
        mpm_x_contact = mpm_x[mpm_contact_mask]
        mpm_pressure_contact = mpm_pressure[mpm_contact_mask]

        # 存储数据
        mpm_data_by_grid[f"grid_{grid_size}"] = {
            "x_coords": mpm_x_contact.tolist(),
            "pressure_kPa": mpm_pressure_contact.tolist(),
            "particle_count": len(mpm_x_contact),
            "pressure_range": {
                "min": float(mpm_pressure_contact.min()),
                "max": float(mpm_pressure_contact.max())
            }
        }

    # 保存为NumPy格式 - 解析解数据
    np.savez(os.path.join(output_dir, 'grid_study_analytical_data.npz'),
             x_coords=x_analytical,
             pressure_kPa=pressure_analytical)

    # 保存为NumPy格式 - 所有MPM数据
    mpm_arrays = {}
    for grid_key, grid_data in mpm_data_by_grid.items():
        mpm_arrays[f"{grid_key}_x"] = np.array(grid_data["x_coords"])
        mpm_arrays[f"{grid_key}_pressure"] = np.array(grid_data["pressure_kPa"])

    np.savez(os.path.join(output_dir, 'grid_study_mpm_data.npz'), **mpm_arrays)

    # 保存为JSON格式 - 汇总信息
    summary_data = {
        "experiment_info": {
            "grid_sizes_tested": all_grid_sizes,
            "contact_width_m": float(contact_width),
            "center_x_m": float(center_x),
            "analysis_region": {
                "x_min": float(center_x - contact_width*1.2),
                "x_max": float(center_x + contact_width*1.2)
            }
        },
        "analytical_solution": {
            "pressure_range_kPa": {
                "min": float(pressure_analytical.min()),
                "max": float(pressure_analytical.max())
            },
            "data_points": len(x_analytical),
            "parameters": {k: v for k, v in params.items() if k != 'config'}
        },
        "mpm_results": mpm_data_by_grid
    }

    with open(os.path.join(output_dir, 'grid_study_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"Summary data saved to:")
    print(f"  - {os.path.join(output_dir, 'grid_study_analytical_data.npz')}")
    print(f"  - {os.path.join(output_dir, 'grid_study_mpm_data.npz')}")
    print(f"  - {os.path.join(output_dir, 'grid_study_summary.json')}")

def create_grid_study_summary_plot(analytical_results, all_mpm_results, all_grid_sizes, output_dir):
    """创建网格研究汇总对比图：一条理论曲线和所有MPM结果"""
    from analytical_solutions.test3 import calculate_contact_width
    import os

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    params = analytical_results['params']

    # 计算接触宽度
    contact_width = calculate_contact_width(
        params['radius'], params['E'], params['nu'], params['applied_pressure'])

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # 解析解 - y=0位置的压力分布
    x_coords = analytical_results['x_coords']
    stress_yy = analytical_results['stress_yy']
    center_x = params['center'][0]

    # 只在接触区域内显示
    contact_indices = (x_coords >= center_x - contact_width*1.2) & (x_coords <= center_x + contact_width*1.2)
    x_analytical = x_coords[contact_indices]
    pressure_analytical = -stress_yy[contact_indices]/1000

    # 绘制解析解
    ax.plot(x_analytical, pressure_analytical, 'r-', linewidth=3,
            label='Analytical (Hertz theory)', zorder=10)

    # 为不同网格大小定义颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_mpm_results)))

    # 绘制所有MPM结果
    for i, (mpm_results, grid_size) in enumerate(zip(all_mpm_results, all_grid_sizes)):
        mpm_x = mpm_results['positions'][:, 0]
        mpm_stress_yy = mpm_results['stress'][:, 1, 1]  # σ_yy分量
        mpm_pressure = -mpm_stress_yy/1000  # 转换为正的压力值

        # 只显示接触区域内的MPM数据
        mpm_contact_mask = (mpm_x >= center_x - contact_width*1.2) & (mpm_x <= center_x + contact_width*1.2)
        mpm_x_contact = mpm_x[mpm_contact_mask]
        mpm_pressure_contact = mpm_pressure[mpm_contact_mask]

        # 绘制MPM结果
        ax.scatter(mpm_x_contact, mpm_pressure_contact,
                  c=[colors[i]], s=30, alpha=0.7,
                  label=f'MPM (Grid {grid_size})', zorder=5)

    # 标记接触边界
    ax.axvline(x=center_x - contact_width, color='red', linestyle='--', alpha=0.7,
              label=f'Contact boundary (±{contact_width:.3f}m)', zorder=1)
    ax.axvline(x=center_x + contact_width, color='red', linestyle='--', alpha=0.7, zorder=1)

    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('Pressure (kPa)', fontsize=12)
    ax.set_title('Grid Study Summary: Hertz Contact Pressure\nAnalytical vs MPM Simulation', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(center_x - contact_width*1.2, center_x + contact_width*1.2)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(output_dir, 'grid_study_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grid study summary plot saved to: {save_path}")

    # 在批量模式下不显示图形窗口，避免资源问题
    try:
        import matplotlib
        if matplotlib.get_backend() != 'Agg':
            plt.show()
        else:
            print("在批量模式下跳过图形显示")
    except:
        pass

    # 保存汇总数据
    save_grid_study_summary_data(analytical_results, all_mpm_results, all_grid_sizes, output_dir, contact_width, center_x)

    # 打印统计信息
    print(f"\nGrid Study Summary Statistics:")
    print(f"Analytical pressure range: {pressure_analytical.min():.2f} - {pressure_analytical.max():.2f} kPa")
    print(f"Grid sizes tested: {all_grid_sizes}")
    for i, grid_size in enumerate(all_grid_sizes):
        mpm_results = all_mpm_results[i]
        mpm_stress_yy = mpm_results['stress'][:, 1, 1]
        mpm_pressure = -mpm_stress_yy/1000
        mpm_contact_mask = (mpm_results['positions'][:, 0] >= center_x - contact_width) & \
                          (mpm_results['positions'][:, 0] <= center_x + contact_width)
        mpm_pressure_contact = mpm_pressure[mpm_contact_mask]
        print(f"  Grid {grid_size}: {mpm_pressure_contact.min():.2f} - {mpm_pressure_contact.max():.2f} kPa "
              f"({len(mpm_pressure_contact)} particles)")

def run_summary_only_mode(args):
    """仅汇总模式：从现有结果目录中加载数据并生成汇总图"""
    import glob
    import re
    from datetime import datetime

    grid_start, grid_end = args.grid_range
    grid_step = args.grid_step
    grid_sizes = list(range(grid_start, grid_end + 1, grid_step))

    print(f"搜索网格大小: {grid_sizes}")
    print(f"使用配置: {args.schwarz_config if args.use_schwarz else args.config}")

    # 收集现有结果
    all_mpm_results = []
    all_grid_sizes = []
    analytical_result_for_summary = None

    for grid_size in grid_sizes:
        print(f"\n搜索 grid_size={grid_size} 的结果...")

        # 构造对应的配置文件路径
        if args.use_schwarz:
            config_file = f"config/schwarz_2d_test3_grid{grid_size}.json"
        else:
            config_file = f"config/config_2d_test3_grid{grid_size}.json"

        if not os.path.exists(config_file):
            print(f"  配置文件不存在: {config_file}")
            continue

        # 加载MPM结果
        try:
            if args.use_schwarz:
                mpm_results = load_mpm_boundary_particle_stress(
                    use_schwarz=True,
                    config_path=config_file,
                    max_y_threshold=args.max_y_threshold
                )
            else:
                mpm_results = load_mpm_boundary_particle_stress(
                    use_schwarz=False,
                    config_path=config_file,
                    max_y_threshold=args.max_y_threshold
                )

            if mpm_results is None:
                print(f"  无法加载 grid_size={grid_size} 的MPM结果")
                continue

            print(f"  成功加载 grid_size={grid_size} 的结果: {len(mpm_results['positions'])} 个粒子")

            # 计算对应的解析解（如果还没有的话）
            if analytical_result_for_summary is None:
                print(f"  计算解析解...")
                results_dir = mpm_results.get('results_dir')
                actual_mass = get_actual_volume_force_mass(config_file, results_dir=results_dir)
                actual_max_y = mpm_results['y_max']
                analytical_result_for_summary = calculate_hertz_stress_field(config_file, actual_max_y, actual_mass)

            all_mpm_results.append(mpm_results)
            all_grid_sizes.append(grid_size)

        except Exception as e:
            print(f"  加载 grid_size={grid_size} 时出错: {e}")
            continue

    if not all_mpm_results:
        print("❌ 未找到任何有效的结果数据")
        print("请确保:")
        print("  1. 已运行过批量网格研究 (--batch-grid-study)")
        print("  2. 配置文件存在于 config/ 目录")
        print("  3. 模拟结果存在于 experiment_results/ 目录")
        return

    if analytical_result_for_summary is None:
        print("❌ 无法计算解析解")
        return

    print(f"\n✅ 成功加载 {len(all_grid_sizes)} 个网格大小的结果: {all_grid_sizes}")

    # 生成汇总图和保存数据
    print(f"\n生成汇总对比图和保存汇总数据...")
    summary_output_dir = "experiment_results/analytical_summary"
    create_grid_study_summary_plot(analytical_result_for_summary, all_mpm_results, all_grid_sizes, summary_output_dir)
    print(f"\n✅ 汇总模式完成! 结果保存在: {summary_output_dir}/")

def run_batch_grid_study(args):
    """运行批量网格研究：生成不同domain2-grid-size的配置并运行实验"""
    import subprocess

    # 设置matplotlib为非交互式后端，避免在批量模式下显示窗口
    import matplotlib
    matplotlib.use('Agg')
    print("设置matplotlib为非交互式模式（Agg后端）")

    grid_start, grid_end = args.grid_range
    grid_step = args.grid_step

    print(f"网格大小范围: {grid_start} - {grid_end}, 步长: {grid_step}")
    print("=" * 60)

    # 生成网格大小列表
    grid_sizes = list(range(grid_start, grid_end + 1, grid_step))

    print(f"将测试以下网格大小: {grid_sizes}")
    print()

    # 收集所有结果用于汇总图
    all_mpm_results = []
    all_grid_sizes = []
    analytical_result_for_summary = None

    for i, grid_size in enumerate(grid_sizes):
        if args.use_schwarz:
            print(f"\n{'='*20} 实验 {i+1}/{len(grid_sizes)}: Domain2 Grid Size = {grid_size} {'='*20}")
        else:
            print(f"\n{'='*20} 实验 {i+1}/{len(grid_sizes)}: Single Domain Grid Size = {grid_size} {'='*20}")

        # 1. 生成配置文件
        if args.use_schwarz:
            config_file = f"config/schwarz_2d_test3_grid{grid_size}.json"
            print(f"1. 生成Schwarz配置文件: {config_file}")
            # 调用配置生成脚本，使用--schwarz参数
            generate_cmd = [
                "python", "experiments/generate_test3_schwarz_config.py",
                "--domain2-grid-size", str(grid_size),
                "--domain1-grid-size", "64",  # 保持Domain1网格大小固定
                "--output", config_file,
                "--implicit",
                "--no-gui",  # 批量模式下禁用GUI
                "--schwarz"  # 添加schwarz参数
            ]
        else:
            config_file = f"config/config_2d_test3_grid{grid_size}.json"
            print(f"1. 生成单域配置文件: {config_file}")
            # 调用配置生成脚本，不使用--schwarz参数，直接保存单域配置到目标文件
            generate_cmd = [
                "python", "experiments/generate_test3_schwarz_config.py",
                "--domain1-grid-size", str(grid_size),  # 单域模式下设置domain1网格大小
                "--domain2-grid-size", "32",  # domain2大小不重要，但需要设置
                "--output", config_file,  # 直接输出到目标文件
                "--implicit",
                "--no-gui"
                # 不添加--schwarz参数，所以只保存单域配置
            ]

        try:
            result = subprocess.run(generate_cmd, capture_output=True, text=True, check=True)
            if args.use_schwarz:
                print(f"   Schwarz配置生成成功: {config_file}")
            # 不需要额外的复制操作，generate_test3_schwarz_config.py已经根据--schwarz参数直接保存到正确位置
        except subprocess.CalledProcessError as e:
            print(f"   配置生成失败: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            continue

        # 2. 运行MPM模拟（使用子进程）
        if args.use_schwarz:
            print(f"2. 运行Schwarz模拟 (grid_size={grid_size})")
            # 使用子进程调用Schwarz模拟器
            mpm_cmd = [
                "python", "simulators/implicit_mpm_schwarz.py",
                "--config", config_file,
                "--no-gui"
            ]
        else:
            print(f"2. 运行单域MPM模拟 (grid_size={grid_size})")
            # 使用子进程调用单域模拟器
            mpm_cmd = [
                "python", "simulators/implicit_mpm.py",
                "--config", config_file,
                "--no-gui"
            ]

        print(f"   执行命令: {' '.join(mpm_cmd)}")

        try:
            # 不捕获输出，让模拟进度实时显示
            result = subprocess.run(mpm_cmd, capture_output=True,timeout=7200)

            # 检查返回码，0表示成功
            if result.returncode == 0:
                print(f"   模拟完成，开始分析结果")
            else:
                print(f"   模拟失败，返回码: {result.returncode}")
                continue

        except subprocess.TimeoutExpired:
            print(f"   模拟超时（超过2小时），跳过此网格大小")
            continue
        except Exception as e:
            print(f"   模拟过程出错: {e}")
            # continue

        # 3. 加载结果并进行分析
        if args.use_schwarz:
            print(f"3. 加载Domain2边界粒子应力数据")
            mpm_results = load_mpm_boundary_particle_stress(
                use_schwarz=True,
                max_y_threshold=args.max_y_threshold
            )
        else:
            print(f"3. 加载单域边界粒子应力数据")
            mpm_results = load_mpm_boundary_particle_stress(
                use_schwarz=False,
                max_y_threshold=args.max_y_threshold
            )

        if mpm_results is None:
            print(f"   无法加载grid_size={grid_size}的模拟结果")
            continue

        # 4. 计算解析解（从保存的结果中获取actual_mass）
        print(f"4. 计算对应的Hertz接触解析解")
        actual_max_y = mpm_results['y_max']
        results_dir = mpm_results.get('results_dir')
        actual_mass = get_actual_volume_force_mass(config_file, results_dir=results_dir)
        analytical_results = calculate_hertz_stress_field(config_file, actual_max_y, actual_mass)

        # 5. 创建专用输出目录并保存结果
        output_dir = f"experiment_results/analytical_grid{grid_size}"
        save_results(analytical_results, output_dir)

        print(f"解析解垂直应力统计 (grid_size={grid_size}):")
        print(f"σ_yy: 最小值={analytical_results['stress_yy'].min():.2e} Pa")
        print(f"σ_yy: 最大值={analytical_results['stress_yy'].max():.2e} Pa")

        # 6. 对比分析并保存专用图片
        if args.use_schwarz:
            print(f"6. 对比解析解与Domain2结果")
        else:
            print(f"6. 对比解析解与单域MPM结果")
        compare_with_mpm(analytical_results, mpm_results, output_dir, save_image=True, grid_size=grid_size)

        print(f"实验 {i+1} 完成! 结果保存在: {output_dir}")

        # 收集结果用于汇总图
        all_mpm_results.append(mpm_results)
        all_grid_sizes.append(grid_size)
        if analytical_result_for_summary is None:
            analytical_result_for_summary = analytical_results

        # 实验完成，子进程已自动清理资源
        print(f"实验 {i+1} 完成! 结果已保存，准备下一个实验...")


    print(f"\n{'='*60}")
    print(f"批量网格研究完成! 共测试了 {len(grid_sizes)} 个网格大小")
    print("结果目录:")
    for grid_size in grid_sizes:
        print(f"  - experiment_results/analytical_grid{grid_size}/")

    # 生成汇总对比图
    if all_mpm_results and analytical_result_for_summary:
        print(f"\n生成汇总对比图和保存汇总数据...")
        summary_output_dir = "experiment_results/analytical_summary"
        create_grid_study_summary_plot(analytical_result_for_summary, all_mpm_results, all_grid_sizes, summary_output_dir)
        print(f"汇总结果保存在: {summary_output_dir}/")

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
    parser.add_argument('--compare-only', action='store_true', 
                       help='跳过模拟，直接加载现有结果进行比较分析')
    parser.add_argument('--max-y-threshold', type=float, default=0.05,
                       help='边界粒子的最大y坐标阈值，用于过滤上边界粒子 (默认: 0.05 m)')
    parser.add_argument('--use-schwarz', action='store_true',
                       help='使用Schwarz域分解方法运行和比较（默认使用单域MPM）')
    parser.add_argument('--schwarz-config', default="config/schwarz_2d_test3.json",
                       help='Schwarz域分解配置文件路径')
    parser.add_argument('--batch-grid-study', action='store_true',
                       help='批量网格研究模式：自动生成不同grid-size的配置并运行实验')
    parser.add_argument('--grid-range', nargs=2, type=int, default=[64, 160],
                       help='网格大小范围 [开始, 结束] (默认: 64 160)')
    parser.add_argument('--grid-step', type=int, default=16,
                       help='网格大小步长 (默认: 16)')
    parser.add_argument('--summary-only', action='store_true',
                       help='仅生成汇总图和保存汇总数据，从现有结果中加载数据')

    args = parser.parse_args()

    print("实验3: Hertz接触垂直应力分析")
    print("=" * 50)

    # 仅汇总模式
    if args.summary_only:
        print("\n启动汇总模式：从现有结果生成汇总图和保存汇总数据")
        run_summary_only_mode(args)
        return

    # 批量网格研究模式
    if args.batch_grid_study:
        print("\n启动批量网格研究模式")
        run_batch_grid_study(args)
        return

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

    if args.compare_only:
        # 直接比较模式：跳过模拟，直接加载现有结果
        print("\n直接比较模式：跳过模拟，加载现有结果进行分析")

        if args.use_schwarz:
            print(f"\n1. 加载现有Schwarz Domain2边界粒子应力数据")
            config_path = args.schwarz_config
            mpm_results = load_mpm_boundary_particle_stress(use_schwarz=True, config_path=config_path, max_y_threshold=args.max_y_threshold)
        else:
            print(f"\n1. 加载现有单域MPM边界粒子应力数据")
            config_path = args.config
            mpm_results = load_mpm_boundary_particle_stress(use_schwarz=False, config_path=config_path, max_y_threshold=args.max_y_threshold)

        if mpm_results is None:
            print("❌ 无法加载现有模拟结果，请先运行模拟或检查结果目录")
            return

        # 计算解析解（从已保存的文件中读取actual_mass）
        print(f"\n2. 计算对应的Hertz接触解析解")
        results_dir = mpm_results.get('results_dir')
        actual_mass = get_actual_volume_force_mass(config_path, results_dir=results_dir)
        analytical_results = calculate_hertz_stress_field(config_path, actual_mass=actual_mass)
        save_results(analytical_results, args.output_dir)

        print(f"解析解垂直应力统计:")
        print(f"σ_yy: 最小值={analytical_results['stress_yy'].min():.2e} Pa")
        print(f"σ_yy: 最大值={analytical_results['stress_yy'].max():.2e} Pa")

        # 对比分析
        print("\n3. 对比解析解与现有MPM结果")
        compare_with_mpm(analytical_results, mpm_results, args.output_dir, save_image=not args.no_save_image)

        solver_type = "Schwarz域分解" if args.use_schwarz else "单域MPM"
        print(f"\n{solver_type}直接比较完成! 结果保存在: {args.output_dir}")
        return

    # 完整流程：先模拟，再根据模拟结果计算解析解
    if args.use_schwarz:
        # Schwarz域分解模式
        print("\n1. 运行Schwarz域分解模拟")
        results_dir, actual_mass = run_schwarz_simulation(args.schwarz_config)

        if results_dir is None:
            print("Schwarz域分解模拟失败，无法进行分析")
            return

        # 2. 加载Domain2应力数据，获取实际粒子分布
        print(f"\n2. 加载Domain2边界粒子应力数据")
        mpm_results = load_mpm_boundary_particle_stress(use_schwarz=True, config_path=args.schwarz_config, max_y_threshold=args.max_y_threshold)

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
        results_dir, actual_mass = run_mpm_simulation(args.config)

        if results_dir is None:
            print("单域MPM模拟失败，无法进行分析")
            return

        # 2. 加载MPM应力数据，获取实际粒子分布
        print(f"\n2. 加载MPM边界粒子应力数据")
        mpm_results = load_mpm_boundary_particle_stress(use_schwarz=False, config_path=args.config, max_y_threshold=args.max_y_threshold)

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