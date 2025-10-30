#!/usr/bin/env python3
"""
应力数据可视化工具
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os
import glob

def load_stress_data(output_dir, frame_number=None):
    """加载应力数据（单域版本）"""
    import re
    from datetime import datetime

    # 检查是否是统一目录结构或手动指定目录
    base_output_dir = "experiment_results"
    if output_dir in ['stress_output', 'single']:
        # 查找统一输出目录下的最新单域结果
        if not os.path.exists(base_output_dir):
            raise FileNotFoundError(f"统一输出目录不存在: {base_output_dir}")

        single_pattern = os.path.join(base_output_dir, "single_domain_*")
        single_dirs = glob.glob(single_pattern)

        if not single_dirs:
            raise FileNotFoundError("未找到单域MPM结果目录")

        # 从目录名中提取时间戳并排序，选择最新的
        def extract_timestamp(dirname):
            basename = os.path.basename(dirname)
            match = re.search(r'single_domain_(\d{8}_\d{6})', basename)
            if match:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            return datetime.min

        # 按时间戳排序，选择最新的
        latest_dir = max(single_dirs, key=extract_timestamp)
        actual_output_dir = latest_dir
        print(f"找到 {len(single_dirs)} 个单域结果目录，使用最新的: {latest_dir}")
    else:
        # 用户指定了具体目录路径，直接使用
        actual_output_dir = output_dir
        if not os.path.exists(actual_output_dir):
            raise FileNotFoundError(f"指定的目录不存在: {actual_output_dir}")
        print(f"使用指定目录: {actual_output_dir}")

    # 首先搜索时间戳子目录（向下兼容旧格式）
    timestamped_dirs = []
    if os.path.exists(actual_output_dir):
        for item in os.listdir(actual_output_dir):
            item_path = os.path.join(actual_output_dir, item)
            if os.path.isdir(item_path) and item.startswith('frame_'):
                timestamped_dirs.append(item_path)
    
    if frame_number is None:
        if timestamped_dirs:
            # 如果有时间戳目录，找到最新的时间戳目录
            def extract_timestamp(dirname):
                # 从 frame_100_20250902_142624 格式中提取时间戳
                parts = dirname.split('_')
                if len(parts) >= 4:
                    date_part = parts[-2]  # 20250902
                    time_part = parts[-1]  # 142624
                    return date_part + time_part  # 20250902142624
                return '00000000000000'  # 默认值
            
            # 按时间戳排序，取最新的
            latest_dir = max(timestamped_dirs, key=lambda x: extract_timestamp(os.path.basename(x)))
            
        # 查找最新的帧号
        stress_files = glob.glob(f"{actual_output_dir}/stress_frame_*.npy")
        if not stress_files:
            raise FileNotFoundError(f"在目录 {actual_output_dir} 中未找到应力数据文件")

        # 从文件名中提取帧号
        frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in stress_files]
        frame_number = max(frame_numbers)

    # 构建文件路径
    stress_file = f"{actual_output_dir}/stress_frame_{frame_number}.npy"
    positions_file = f"{actual_output_dir}/positions_frame_{frame_number}.npy"

    if not all(os.path.exists(f) for f in [stress_file, positions_file]):
        missing_files = [f for f in [stress_file, positions_file] if not os.path.exists(f)]
        raise FileNotFoundError(f"缺少帧 {frame_number} 的数据文件: {missing_files}")

    stress_data = np.load(stress_file)
    positions = np.load(positions_file)

    return stress_data, positions, frame_number

def load_schwarz_stress_data(output_dir, frame_number=None):
    """加载Schwarz双域应力数据"""
    import re
    from datetime import datetime

    # 检查是否是统一目录结构或手动指定目录
    base_output_dir = "experiment_results"
    if output_dir in ['stress_output_schwarz', 'schwarz']:
        # 查找统一输出目录下的最新Schwarz结果
        if not os.path.exists(base_output_dir):
            raise FileNotFoundError(f"统一输出目录不存在: {base_output_dir}")

        schwarz_pattern = os.path.join(base_output_dir, "schwarz_*")
        schwarz_dirs = glob.glob(schwarz_pattern)

        if not schwarz_dirs:
            raise FileNotFoundError("未找到Schwarz域分解结果目录")

        # 从目录名中提取时间戳并排序，选择最新的
        def extract_timestamp(dirname):
            basename = os.path.basename(dirname)
            match = re.search(r'schwarz_(\d{8}_\d{6})', basename)
            if match:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            return datetime.min

        # 按时间戳排序，选择最新的
        latest_dir = max(schwarz_dirs, key=extract_timestamp)
        actual_output_dir = latest_dir
        print(f"找到 {len(schwarz_dirs)} 个Schwarz结果目录，使用最新的: {latest_dir}")
    else:
        # 用户指定了具体目录路径，直接使用
        actual_output_dir = output_dir
        if not os.path.exists(actual_output_dir):
            raise FileNotFoundError(f"指定的目录不存在: {actual_output_dir}")
        print(f"使用指定目录: {actual_output_dir}")

    # 首先搜索时间戳子目录（向下兼容旧格式）
    timestamped_dirs = []
    if os.path.exists(actual_output_dir):
        for item in os.listdir(actual_output_dir):
            item_path = os.path.join(actual_output_dir, item)
            if os.path.isdir(item_path) and item.startswith('frame_'):
                timestamped_dirs.append(item_path)
    
    if frame_number is None:
        # 查找最新的帧号
        domain1_files = glob.glob(f"{actual_output_dir}/domain1_stress_frame_*.npy")
        if not domain1_files:
            raise FileNotFoundError(f"在目录 {actual_output_dir} 中未找到Domain1应力数据文件")

        # 从文件名中提取帧号
        frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in domain1_files]
        frame_number = max(frame_numbers)

    # 构建文件路径
    d1_stress_file = f"{actual_output_dir}/domain1_stress_frame_{frame_number}.npy"
    d1_positions_file = f"{actual_output_dir}/domain1_positions_frame_{frame_number}.npy"

    d2_stress_file = f"{actual_output_dir}/domain2_stress_frame_{frame_number}.npy"
    d2_positions_file = f"{actual_output_dir}/domain2_positions_frame_{frame_number}.npy"

    all_files = [d1_stress_file, d1_positions_file, d2_stress_file, d2_positions_file]

    if not all(os.path.exists(f) for f in all_files):
        missing_files = [f for f in all_files if not os.path.exists(f)]
        raise FileNotFoundError(f"缺少帧 {frame_number} 的数据文件: {missing_files}")

    # 加载Domain1数据
    d1_stress = np.load(d1_stress_file)
    d1_positions = np.load(d1_positions_file)

    # 加载Domain2数据
    d2_stress = np.load(d2_stress_file)
    d2_positions = np.load(d2_positions_file)

    return {
        'domain1': {'stress': d1_stress, 'positions': d1_positions},
        'domain2': {'stress': d2_stress, 'positions': d2_positions},
        'frame_number': frame_number
    }

def compute_von_mises_stress(stress_data):
    """计算von Mises应力"""
    von_mises = []
    dim = stress_data.shape[1]
    
    for i in range(stress_data.shape[0]):
        s = stress_data[i]
        if dim == 2:
            # 2D von Mises stress
            vm = np.sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2)
        else:
            # 3D von Mises stress
            vm = np.sqrt(0.5*((s[0,0]-s[1,1])**2 + (s[1,1]-s[2,2])**2 + (s[2,2]-s[0,0])**2) + 
                        3*(s[0,1]**2 + s[1,2]**2 + s[2,0]**2))
        von_mises.append(vm)
    
    return np.array(von_mises)

# Hydrostatic pressure functions removed - focusing only on von Mises stress

def calculate_adaptive_radius(positions, coverage_factor=0.8):
    """
    Calculate adaptive radius for particle visualization based on particle spacing
    
    Args:
        positions: Array of particle positions (N, 2) or (N, 3)
        coverage_factor: Factor to control coverage (0-1), higher means larger particles
        
    Returns:
        Optimal radius for scatter plot
    """
    if len(positions) < 2:
        return 20  # Default radius for single particle
    
    try:
        from scipy.spatial.distance import pdist
        # Calculate pairwise distances using scipy
        distances = pdist(positions)
    except ImportError:
        # Fallback: simple pairwise distance calculation
        n = len(positions)
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        distances = np.array(distances)
    
    # Use the 10th percentile of non-zero distances as characteristic spacing
    non_zero_distances = distances[distances > 1e-10]
    if len(non_zero_distances) == 0:
        return 20  # Default if all particles are at same position
    
    characteristic_distance = np.percentile(non_zero_distances, 10)
    
    # Convert distance to matplotlib scatter size (s parameter)
    # s parameter is area in points^2, so radius^2 * pi
    # We want radius to be roughly coverage_factor * characteristic_distance/2
    radius_world = characteristic_distance * coverage_factor * 0.5
    
    # Convert world coordinates to matplotlib points
    # This is approximate and may need adjustment based on axis limits
    radius_points = radius_world * 72  # 72 points per inch, approximate conversion
    
    # Clamp to reasonable range
    radius_points = np.clip(radius_points, 5, 100)
    
    return radius_points ** 2  # Return s parameter (area)

def visualize_stress_2d(positions, von_mises, frame_number, save_path=None, use_log=False, stress_cmap='coolwarm', max_stress=None, particle_size=None):
    """2D应力可视化（仅显示von Mises应力）

    Args:
        max_stress: Manual maximum value for von Mises stress color range. If None, uses data maximum.
        particle_size: Manual particle size (s parameter). If None, uses adaptive calculation.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Calculate adaptive particle size if not specified
    if particle_size is None:
        particle_size = calculate_adaptive_radius(positions)
    print(f"Using particle size (s parameter): {particle_size:.2f}")

    if use_log:
        from matplotlib.colors import LogNorm

        # von Mises stress (using logarithmic scale)
        von_mises_positive = np.maximum(von_mises, 1e-10)  # 避免零值
        vm_vmin = np.min(von_mises_positive)
        vm_vmax = max_stress if max_stress is not None else np.max(von_mises_positive)
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=von_mises_positive,
                           cmap=stress_cmap, s=particle_size, alpha=0.8,
                           norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax))
        ax.set_title(f'von Mises Stress Distribution (Log Scale, Frame {frame_number})')
        cbar_label = 'von Mises Stress (Log)'
    else:
        # von Mises stress (linear scale)
        vm_vmax = max_stress if max_stress is not None else np.max(von_mises)
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=von_mises,
                           cmap=stress_cmap, s=particle_size, alpha=0.8,
                           vmax=vm_vmax)
        ax.set_title(f'von Mises Stress Distribution (Frame {frame_number})')
        cbar_label = 'von Mises Stress'

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label=cbar_label)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")

    plt.show()


def visualize_schwarz_stress_2d(data_dict, save_path=None, use_log=False, stress_cmap='coolwarm', max_stress=None, particle_size=None):
    """2D Schwarz双域应力可视化"""

    frame_number = data_dict['frame_number']
    d1_data = data_dict['domain1']
    d2_data = data_dict['domain2']

    # Calculate von Mises stress for both domains
    d1_von_mises = compute_von_mises_stress(d1_data['stress'])
    d2_von_mises = compute_von_mises_stress(d2_data['stress'])

    # Merge data to unify scale range
    all_von_mises = np.concatenate([d1_von_mises, d2_von_mises])

    # Calculate adaptive radius for each domain
    if particle_size is None:
        d1_particle_size = calculate_adaptive_radius(d1_data['positions'])
        d2_particle_size = calculate_adaptive_radius(d2_data['positions'])
    else:
        d1_particle_size = particle_size
        d2_particle_size = particle_size

    print(f"Domain1 particle size (s parameter): {d1_particle_size:.2f}")
    print(f"Domain2 particle size (s parameter): {d2_particle_size:.2f}")

    # Create 1x2 subplot layout (只显示von Mises应力)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    if use_log:
        from matplotlib.colors import LogNorm

        # Calculate logarithmic scale range
        vm_positive = np.maximum(all_von_mises, 1e-10)
        vm_vmin = np.min(vm_positive)
        vm_vmax = max_stress if max_stress is not None else np.max(vm_positive)
        
        # Domain1 von Mises stress (log scale)
        d1_vm_positive = np.maximum(d1_von_mises, 1e-10)
        scatter1 = ax1.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1],
                              c=d1_vm_positive, cmap=stress_cmap, s=d1_particle_size, alpha=0.8,
                              norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax), edgecolors='none')
        ax1.set_title(f'Domain1 von Mises Stress (Log Scale, Frame {frame_number})')
        cbar1_label = 'von Mises Stress (Log)'

        # Domain2 von Mises stress (log scale)
        d2_vm_positive = np.maximum(d2_von_mises, 1e-10)
        scatter2 = ax2.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1],
                              c=d2_vm_positive, cmap=stress_cmap, s=d2_particle_size, alpha=0.8,
                              norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax), edgecolors='none')
        ax2.set_title(f'Domain2 von Mises Stress (Log Scale, Frame {frame_number})')
        cbar2_label = 'von Mises Stress (Log)'
    else:
        # Linear scale
        vm_vmin = np.min(all_von_mises)
        vm_vmax = max_stress if max_stress is not None else np.max(all_von_mises)

        # Domain1 von Mises stress (linear scale)
        scatter1 = ax1.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1],
                              c=d1_von_mises, cmap=stress_cmap, s=d1_particle_size, alpha=0.8,
                              vmin=vm_vmin, vmax=vm_vmax, edgecolors='none')
        ax1.set_title(f'Domain1 von Mises Stress (Frame {frame_number})')
        cbar1_label = 'von Mises Stress'

        # Domain2 von Mises stress (linear scale)
        scatter2 = ax2.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1],
                              c=d2_von_mises, cmap=stress_cmap, s=d2_particle_size, alpha=0.8,
                              vmin=vm_vmin, vmax=vm_vmax, edgecolors='none')
        ax2.set_title(f'Domain2 von Mises Stress (Frame {frame_number})')
        cbar2_label = 'von Mises Stress'
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label=cbar1_label)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label=cbar2_label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dual domain stress image saved to: {save_path}")
    
    plt.show()

def visualize_schwarz_stress_combined_2d(data_dict, save_path=None, use_log=False, stress_cmap='coolwarm', max_stress=None, particle_size=None):
    """2D Schwarz双域合并应力可视化（仅显示von Mises应力）"""

    frame_number = data_dict['frame_number']
    d1_data = data_dict['domain1']
    d2_data = data_dict['domain2']

    # Calculate von Mises stress for both domains
    d1_von_mises = compute_von_mises_stress(d1_data['stress'])
    d2_von_mises = compute_von_mises_stress(d2_data['stress'])

    # Merge data to unify scale range
    all_von_mises = np.concatenate([d1_von_mises, d2_von_mises])

    # Calculate adaptive radius for each domain
    if particle_size is None:
        d1_particle_size = calculate_adaptive_radius(d1_data['positions'])
        d2_particle_size = calculate_adaptive_radius(d2_data['positions'])
    else:
        d1_particle_size = particle_size
        d2_particle_size = particle_size

    print(f"Domain1 particle size (s parameter): {d1_particle_size:.2f}")
    print(f"Domain2 particle size (s parameter): {d2_particle_size:.2f}")

    # Create single subplot layout (only von Mises stress)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if use_log:
        from matplotlib.colors import LogNorm

        # Calculate logarithmic scale range
        vm_positive = np.maximum(all_von_mises, 1e-10)
        vm_vmin = np.min(vm_positive)
        vm_vmax = max_stress if max_stress is not None else np.max(vm_positive)

        # Combined von Mises stress visualization (log scale)
        d1_vm_positive = np.maximum(d1_von_mises, 1e-10)
        d2_vm_positive = np.maximum(d2_von_mises, 1e-10)

        scatter_d1 = ax.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1],
                               c=d1_vm_positive, cmap=stress_cmap, s=d1_particle_size, alpha=0.8,
                               norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax),
                               edgecolors='none', label='Domain1')
        scatter_d2 = ax.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1],
                               c=d2_vm_positive, cmap=stress_cmap, s=d2_particle_size, alpha=0.8,
                               norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax),
                               edgecolors='none', label='Domain2')
        ax.set_title(f'Dual Domain von Mises Stress Distribution (Log Scale, Frame {frame_number})')
        cbar_label = 'von Mises Stress (Log)'
    else:
        # Linear scale范围
        vm_vmin = np.min(all_von_mises)
        vm_vmax = max_stress if max_stress is not None else np.max(all_von_mises)

        # Combined von Mises stress visualization (linear scale)
        scatter_d1 = ax.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1],
                               c=d1_von_mises, cmap=stress_cmap, s=d1_particle_size, alpha=0.8,
                               vmin=vm_vmin, vmax=vm_vmax,
                               edgecolors='none', label='Domain1')
        scatter_d2 = ax.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1],
                               c=d2_von_mises, cmap=stress_cmap, s=d2_particle_size, alpha=0.8,
                               vmin=vm_vmin, vmax=vm_vmax,
                               edgecolors='none', label='Domain2')
        ax.set_title(f'Dual Domain von Mises Stress Distribution (Frame {frame_number})')
        cbar_label = 'von Mises Stress'

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend()
    plt.colorbar(scatter_d1, ax=ax, label=cbar_label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dual domain combined stress image saved to: {save_path}")
    
    plt.show()


def print_schwarz_stress_statistics(data_dict):
    """打印Schwarz双域应力统计信息（仅包含von Mises应力）"""
    frame_number = data_dict['frame_number']
    d1_data = data_dict['domain1']
    d2_data = data_dict['domain2']

    print(f"Schwarz Dual Domain Stress Statistics (Frame {frame_number}):")
    print("=" * 60)

    for domain_name, domain_data in [("Domain1", d1_data), ("Domain2", d2_data)]:
        print(f"\n{domain_name}:")
        print("-" * 30)

        # Calculate stress scalars
        von_mises = compute_von_mises_stress(domain_data['stress'])

        print(f"Number of particles: {len(domain_data['positions'])}")
        print(f"Dimensions: {domain_data['positions'].shape[1]}D")

        print(f"von Mises Stress:")
        print(f"  Min: {np.min(von_mises):.3e}")
        print(f"  Max: {np.max(von_mises):.3e}")
        print(f"  Mean: {np.mean(von_mises):.3e}")
        print(f"  Std: {np.std(von_mises):.3e}")

        # Stress component statistics
        stress_data = domain_data['stress']
        dim = stress_data.shape[1]
        print(f"Stress Components ({dim}D):")
        for i in range(dim):
            for j in range(dim):
                component = stress_data[:, i, j]
                print(f"  σ_{i+1}{j+1}: {np.mean(component):.3e} ± {np.std(component):.3e}")

def print_stress_statistics(von_mises, stress_data):
    """Print stress statistics information（仅包含von Mises应力）"""
    print("Stress Statistics:")
    print("-" * 40)
    print(f"von Mises Stress:")
    print(f"  Min: {np.min(von_mises):.3e}")
    print(f"  Max: {np.max(von_mises):.3e}")
    print(f"  Mean: {np.mean(von_mises):.3e}")
    print(f"  Std: {np.std(von_mises):.3e}")

    # Stress component statistics
    dim = stress_data.shape[1]
    print(f"\nStress Components ({dim}D):")
    for i in range(dim):
        for j in range(dim):
            component = stress_data[:, i, j]
            print(f"  σ_{i+1}{j+1}: {np.mean(component):.3e} ± {np.std(component):.3e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize stress data')
    parser.add_argument('--dir', '-d', default='stress_output',
                       help='Data directory: "stress_output"/"single" for latest single domain, "schwarz" for latest Schwarz, or specific path like "experiment_results/single_domain_20251029_025635"')
    parser.add_argument('--frame', '-f', type=int, default=None,
                       help='Specify frame number (default: latest frame)')
    parser.add_argument('--save', '-s', default=None,
                       help='Save image path')
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--schwarz', action='store_true',
                       help='Use Schwarz dual domain data (automatically finds latest results from experiment_results/)')
    parser.add_argument('--combined', action='store_true',
                       help='Dual domain combined visualization (only for Schwarz mode)')
    parser.add_argument('--log', action='store_true',
                       help='Use logarithmic scale visualization (default: linear scale)')
    parser.add_argument('--cmap', default='viridis',
                       help='Colormap for stress visualization (default: coolwarm, options: viridis, plasma, inferno, magma, coolwarm, RdYlBu, seismic)')
    parser.add_argument('--max-stress', type=float, default=None,
                       help='Manually specify maximum value for stress color range (applies to von Mises stress only)')
    parser.add_argument('--particle-size', type=float, default=None,
                       help='Manually specify particle size for visualization (s parameter for scatter plot). If not specified, uses adaptive calculation based on particle spacing.')


    args = parser.parse_args()
    
    try:
        if args.schwarz:
            # Schwarz dual domain mode
            if args.dir == 'stress_output':
                args.dir = 'schwarz'  # 使用新的统一目录结构
            
            print(f"Loading Schwarz dual domain data from {args.dir}...")
            data_dict = load_schwarz_stress_data(args.dir, args.frame)
            frame_number = data_dict['frame_number']
            
            print(f"Successfully loaded dual domain data for frame {frame_number}")
            print(f"Domain1 particle count: {len(data_dict['domain1']['positions'])}")
            print(f"Domain2 particle count: {len(data_dict['domain2']['positions'])}")
            print(f"Dimensions: {data_dict['domain1']['positions'].shape[1]}D")
            
            # Print statistics
            if args.stats:
                print_schwarz_stress_statistics(data_dict)
            
            # Visualization
            dim = data_dict['domain1']['positions'].shape[1]
            save_path = args.save
            if save_path and not save_path.endswith(('.png', '.jpg', '.pdf')):
                save_path += '.png'
            
            print("Generating dual domain visualization...")
            if dim != 2:
                print(f"Error: Only 2D data visualization is supported, current data dimension is {dim}D")
                return 1
                
            if args.combined:
                # Combined visualization mode
                visualize_schwarz_stress_combined_2d(data_dict, save_path, args.log, args.cmap, args.max_stress, args.particle_size)
            else:
                # Separate visualization mode
                visualize_schwarz_stress_2d(data_dict, save_path, args.log, args.cmap, args.max_stress, args.particle_size)
                    
        else:
            # Single domain mode (original functionality)
            print(f"Loading data from {args.dir}...")
            stress_data, positions, frame_number = load_stress_data(args.dir, args.frame)
            print(f"Successfully loaded data for frame {frame_number}")
            print(f"Particle count: {positions.shape[0]}")
            print(f"维度: {positions.shape[1]}D")
            
            # Calculate stress scalars
            von_mises = compute_von_mises_stress(stress_data)

            # Print statistics
            if args.stats:
                print_stress_statistics(von_mises, stress_data)

            # Visualization
            dim = positions.shape[1]
            save_path = args.save
            if save_path and not save_path.endswith(('.png', '.jpg', '.pdf')):
                save_path += '.png'

            print("Generating visualization...")
            if dim != 2:
                print(f"Error: Only 2D data visualization is supported, current data dimension is {dim}D")
                return 1

            visualize_stress_2d(positions, von_mises, frame_number, save_path, args.log, args.cmap, args.max_stress, args.particle_size)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())