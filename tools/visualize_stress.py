#!/usr/bin/env python3
"""
应力数据可视化工具
"""
import numpy as np
import sys
import matplotlib
# 只在保存文件模式或all-frames模式下使用Agg后端（内存优化）
# 否则使用默认后端以支持交互式显示
if '--save' in sys.argv or '--all-frames' in sys.argv:
    matplotlib.use('Agg')  # Non-interactive backend for lower memory overhead
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import argparse
import os
import glob
import gc

def create_custom_colormap():
    """创建自定义colormap"""
    # 创建与图中相同的colormap
    # 从图中可以看到颜色从蓝色->青色->绿色->黄色->橙色->红色
    colors_list = [
        (0.0, '#0000FF'),    # 蓝色
        (0.2, '#0080FF'),    # 浅蓝
        (0.4, '#00FFFF'),    # 青色
        (0.6, '#00FF00'),    # 绿色
        (0.8, '#FFFF00'),    # 黄色
        (0.9, '#FF8000'),    # 橙色
        (1.0, '#FF0000'),    # 红色
    ]

    # 创建自定义colormap
    n_bins = 256
    cmap_name = 'custom_stress_colormap'
    cm = LinearSegmentedColormap.from_list(
        cmap_name,
        [color[1] for color in colors_list],
        N=n_bins
    )

    return cm

# 创建全局自定义colormap
CUSTOM_COLORMAP = create_custom_colormap()

def create_discrete_colormap(base_cmap, n_levels=10):
    """创建离散化的colormap"""
    if isinstance(base_cmap, str):
        # 如果是字符串名称，获取matplotlib colormap
        import matplotlib.pyplot as plt
        base_cmap = plt.cm.get_cmap(base_cmap)

    # 创建离散化的colormap
    from matplotlib.colors import ListedColormap
    colors = base_cmap(np.linspace(0, 1, n_levels))
    discrete_cmap = ListedColormap(colors)
    return discrete_cmap

def compute_spatial_averaged_stress(positions, stress_values, averaging_radius):
    """
    计算每个粒子的空间平均应力值 (优化版: 使用KDTree替代距离矩阵)

    Args:
        positions: 粒子位置数组 (N, 2) 或 (N, 3)
        stress_values: 应力值数组 (N,)
        averaging_radius: 平均化半径

    Returns:
        averaged_stress: 平均化后的应力值 (N,)
    """
    import time

    print(f"开始计算空间平均应力，平均化半径: {averaging_radius:.6f}")
    start_time = time.time()

    n_particles = len(positions)
    averaged_stress = np.zeros(n_particles)

    # 使用KDTree进行高效近邻搜索 - O(N log N) 构建, O(k) 查询
    tree = cKDTree(positions)

    # 为每个粒子计算平均应力
    for i in range(n_particles):
        # 找到在平均化半径内的粒子 (包括自身)
        neighbor_indices = tree.query_ball_point(positions[i], averaging_radius)

        # 计算这些粒子的平均应力
        if len(neighbor_indices) > 0:
            averaged_stress[i] = np.mean(stress_values[neighbor_indices])
        else:
            # 如果没有邻居（不应该发生，因为至少包含自身），使用自己的值
            averaged_stress[i] = stress_values[i]

    elapsed_time = time.time() - start_time
    print(f"空间平均计算完成，用时: {elapsed_time:.3f}s (KDTree优化)")

    return averaged_stress

def estimate_optimal_averaging_radius(positions, factor=2.0):
    """
    估算最优的平均化半径 (优化版: 大数据集使用采样)

    Args:
        positions: 粒子位置数组 (N, 2) 或 (N, 3)
        factor: 平均粒子间距的倍数

    Returns:
        optimal_radius: 推荐的平均化半径
    """
    if len(positions) < 2:
        return 0.01  # 默认值

    MAX_SAMPLE_SIZE = 10000  # 最大采样数，避免O(N²)内存消耗

    try:
        from scipy.spatial.distance import pdist

        # 对大数据集使用采样，避免计算所有N(N-1)/2个距离
        if len(positions) > MAX_SAMPLE_SIZE:
            print(f"数据集较大 ({len(positions)} 粒子)，使用采样估计 ({MAX_SAMPLE_SIZE} samples)")
            sample_indices = np.random.choice(len(positions), MAX_SAMPLE_SIZE, replace=False)
            sampled_positions = positions[sample_indices]
            distances = pdist(sampled_positions)
        else:
            # 小数据集直接计算
            distances = pdist(positions)
    except ImportError:
        # 回退方法：随机采样计算
        n_samples = min(1000, len(positions))
        indices = np.random.choice(len(positions), n_samples, replace=False)
        sampled_positions = positions[indices]
        distances = []
        for i in range(len(sampled_positions)):
            for j in range(i+1, len(sampled_positions)):
                dist = np.linalg.norm(sampled_positions[i] - sampled_positions[j])
                distances.append(dist)
        distances = np.array(distances)

    # 使用10%分位数作为特征距离（排除最近邻）
    non_zero_distances = distances[distances > 1e-10]
    if len(non_zero_distances) == 0:
        return 0.01

    characteristic_distance = np.percentile(non_zero_distances, 10)
    optimal_radius = characteristic_distance * factor

    print(f"估算的最优平均化半径: {optimal_radius:.6f} (特征距离: {characteristic_distance:.6f})")
    return optimal_radius

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
        # 首先尝试新的目录结构 (stress_data子目录)
        stress_data_dir = os.path.join(actual_output_dir, "stress_data")
        if os.path.exists(stress_data_dir):
            actual_data_dir = stress_data_dir
            # 查找frame_X子目录
            frame_dirs = glob.glob(f"{stress_data_dir}/frame_*")
            if frame_dirs:
                # 新的frame子目录结构
                frame_numbers = []
                for frame_dir in frame_dirs:
                    frame_name = os.path.basename(frame_dir)
                    if frame_name.startswith('frame_'):
                        try:
                            frame_num = int(frame_name.split('_')[1])
                            frame_numbers.append(frame_num)
                        except (IndexError, ValueError):
                            continue
                if frame_numbers:
                    frame_number = max(frame_numbers)
                else:
                    raise FileNotFoundError(f"在目录 {stress_data_dir} 中未找到有效的frame目录")
            else:
                # 回退到旧的文件命名结构
                stress_files = glob.glob(f"{stress_data_dir}/stress_frame_*.npy")
                if stress_files:
                    frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in stress_files]
                    frame_number = max(frame_numbers)
                else:
                    raise FileNotFoundError(f"在目录 {stress_data_dir} 中未找到应力数据文件")
        else:
            # 回退到旧的目录结构
            stress_files = glob.glob(f"{actual_output_dir}/stress_frame_*.npy")
            actual_data_dir = actual_output_dir
            if not stress_files:
                raise FileNotFoundError(f"在目录 {actual_output_dir} 中未找到应力数据文件")
            frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in stress_files]
            frame_number = max(frame_numbers)

    # 构建文件路径（使用正确的数据目录）
    if 'actual_data_dir' not in locals():
        # 如果是指定帧号的情况，需要重新确定数据目录
        stress_data_dir = os.path.join(actual_output_dir, "stress_data")
        if os.path.exists(stress_data_dir):
            actual_data_dir = stress_data_dir
        else:
            actual_data_dir = actual_output_dir

    # 尝试新的目录结构：frame_X/stress.npy
    frame_dir = f"{actual_data_dir}/frame_{frame_number}"
    if os.path.exists(frame_dir):
        stress_file = f"{frame_dir}/stress.npy"
        positions_file = f"{frame_dir}/positions.npy"
    else:
        # 回退到旧的文件命名结构
        stress_file = f"{actual_data_dir}/stress_frame_{frame_number}.npy"
        positions_file = f"{actual_data_dir}/positions_frame_{frame_number}.npy"

    if not all(os.path.exists(f) for f in [stress_file, positions_file]):
        missing_files = [f for f in [stress_file, positions_file] if not os.path.exists(f)]
        raise FileNotFoundError(f"缺少帧 {frame_number} 的数据文件: {missing_files}")

    stress_data = np.load(stress_file)
    positions = np.load(positions_file)

    return stress_data, positions, frame_number

def get_all_available_frames(output_dir, is_schwarz=False):
    """获取目录中所有可用的应力数据帧"""
    import re
    from datetime import datetime

    # 确定实际输出目录
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
    elif output_dir == 'schwarz':
        # 查找最新的Schwarz结果
        schwarz_pattern = os.path.join(base_output_dir, "schwarz_*")
        schwarz_dirs = glob.glob(schwarz_pattern)

        if not schwarz_dirs:
            raise FileNotFoundError("未找到Schwarz结果目录")

        def extract_timestamp(dirname):
            basename = os.path.basename(dirname)
            match = re.search(r'schwarz_(\d{8}_\d{6})', basename)
            if match:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            return datetime.min

        latest_dir = max(schwarz_dirs, key=extract_timestamp)
        actual_output_dir = latest_dir
    else:
        # 用户指定了具体目录路径
        actual_output_dir = output_dir
        if not os.path.exists(actual_output_dir):
            raise FileNotFoundError(f"指定的目录不存在: {actual_output_dir}")

    # 确定实际数据目录（检查新的目录结构）
    stress_data_dir = os.path.join(actual_output_dir, "stress_data")
    if os.path.exists(stress_data_dir):
        search_dir = stress_data_dir
    else:
        search_dir = actual_output_dir

    # 查找所有应力数据文件
    frame_numbers = set()

    # 首先尝试新的frame子目录结构
    frame_dirs = glob.glob(f"{search_dir}/frame_*")
    if frame_dirs:
        for frame_dir in frame_dirs:
            frame_name = os.path.basename(frame_dir)
            if frame_name.startswith('frame_'):
                try:
                    frame_num = int(frame_name.split('_')[1])
                    # 验证该frame目录包含正确的数据文件
                    if is_schwarz:
                        if (os.path.exists(f"{frame_dir}/domain1_stress.npy") and
                            os.path.exists(f"{frame_dir}/domain2_stress.npy")):
                            frame_numbers.add(frame_num)
                    else:
                        if os.path.exists(f"{frame_dir}/stress.npy"):
                            frame_numbers.add(frame_num)
                except (IndexError, ValueError):
                    continue

    # 如果没有找到frame子目录，回退到旧的文件命名结构
    if not frame_numbers:
        if is_schwarz:
            stress_files = glob.glob(f"{search_dir}/domain*_stress_frame_*.npy")
            for f in stress_files:
                match = re.search(r'stress_frame_(\d+)\.npy$', f)
                if match:
                    frame_numbers.add(int(match.group(1)))
        else:
            stress_files = glob.glob(f"{search_dir}/stress_frame_*.npy")
            for f in stress_files:
                match = re.search(r'stress_frame_(\d+)\.npy$', f)
                if match:
                    frame_numbers.add(int(match.group(1)))

    return sorted(list(frame_numbers)), actual_output_dir

def visualize_single_frame(args, averaging_radius=None):
    """可视化单个帧的应力数据"""
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
            visualize_schwarz_stress_combined_2d(data_dict, save_path, args.log, args.cmap, args.max_stress, args.particle_size, averaging_radius, not args.continuous_colormap, args.colormap_levels)
        else:
            # Separate visualization mode
            visualize_schwarz_stress_2d(data_dict, save_path, args.log, args.cmap, args.max_stress, args.particle_size, averaging_radius, not args.continuous_colormap, args.colormap_levels)

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

        visualize_stress_2d(positions, von_mises, frame_number, save_path, args.log, args.cmap, args.max_stress, args.particle_size, averaging_radius, not args.continuous_colormap, args.colormap_levels)

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
    
    # 确定实际数据目录（检查新的目录结构）
    stress_data_dir = os.path.join(actual_output_dir, "stress_data")
    if os.path.exists(stress_data_dir):
        actual_data_dir = stress_data_dir
    else:
        actual_data_dir = actual_output_dir

    if frame_number is None:
        # 查找最新的帧号
        # 首先尝试新的frame子目录结构
        frame_dirs = glob.glob(f"{actual_data_dir}/frame_*")
        if frame_dirs:
            frame_numbers = []
            for frame_dir in frame_dirs:
                frame_name = os.path.basename(frame_dir)
                if frame_name.startswith('frame_'):
                    try:
                        frame_num = int(frame_name.split('_')[1])
                        # 验证该frame目录确实包含domain数据
                        if (os.path.exists(f"{frame_dir}/domain1_stress.npy") and
                            os.path.exists(f"{frame_dir}/domain2_stress.npy")):
                            frame_numbers.append(frame_num)
                    except (IndexError, ValueError):
                        continue
            if frame_numbers:
                frame_number = max(frame_numbers)
            else:
                raise FileNotFoundError(f"在目录 {actual_data_dir} 中未找到有效的frame目录")
        else:
            # 回退到旧的文件命名结构
            domain1_files = glob.glob(f"{actual_data_dir}/domain1_stress_frame_*.npy")
            if not domain1_files:
                raise FileNotFoundError(f"在目录 {actual_data_dir} 中未找到Domain1应力数据文件")
            frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in domain1_files]
            frame_number = max(frame_numbers)

    # 尝试新的目录结构：frame_X/domain1_stress.npy
    frame_dir = f"{actual_data_dir}/frame_{frame_number}"
    if os.path.exists(frame_dir):
        d1_stress_file = f"{frame_dir}/domain1_stress.npy"
        d1_positions_file = f"{frame_dir}/domain1_positions.npy"
        d2_stress_file = f"{frame_dir}/domain2_stress.npy"
        d2_positions_file = f"{frame_dir}/domain2_positions.npy"
    else:
        # 回退到旧的文件命名结构
        d1_stress_file = f"{actual_data_dir}/domain1_stress_frame_{frame_number}.npy"
        d1_positions_file = f"{actual_data_dir}/domain1_positions_frame_{frame_number}.npy"
        d2_stress_file = f"{actual_data_dir}/domain2_stress_frame_{frame_number}.npy"
        d2_positions_file = f"{actual_data_dir}/domain2_positions_frame_{frame_number}.npy"

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
    """计算von Mises应力 (完全向量化版本)"""
    dim = stress_data.shape[1]

    if dim == 2:
        # 2D von Mises stress - 完全向量化
        s00 = stress_data[:, 0, 0]
        s11 = stress_data[:, 1, 1]
        s01 = stress_data[:, 0, 1]
        von_mises = np.sqrt(s00**2 + s11**2 - s00*s11 + 3*s01**2)
    else:
        # 3D von Mises stress - 完全向量化
        s00 = stress_data[:, 0, 0]
        s11 = stress_data[:, 1, 1]
        s22 = stress_data[:, 2, 2]
        s01 = stress_data[:, 0, 1]
        s12 = stress_data[:, 1, 2]
        s20 = stress_data[:, 2, 0]
        von_mises = np.sqrt(0.5*((s00-s11)**2 + (s11-s22)**2 + (s22-s00)**2) +
                           3*(s01**2 + s12**2 + s20**2))

    return von_mises

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

def visualize_stress_2d(positions, von_mises, frame_number, save_path=None, use_log=False, stress_cmap=None, max_stress=None, particle_size=None, averaging_radius=None, use_discrete=True, n_levels=10):
    """2D应力可视化（仅显示von Mises应力）

    Args:
        max_stress: Manual maximum value for von Mises stress color range. If None, uses data maximum.
        particle_size: Manual particle size (s parameter). If None, uses adaptive calculation.
        averaging_radius: 空间平均化半径。If None, 不进行平均化; if 'auto', 自动计算最优半径
        use_discrete: 是否使用离散colormap
        n_levels: 离散colormap的级数
    """
    # 使用自定义colormap作为默认值
    if stress_cmap is None:
        stress_cmap = CUSTOM_COLORMAP

    # 处理离散colormap
    if not use_discrete:
        final_cmap = stress_cmap
    else:
        final_cmap = create_discrete_colormap(stress_cmap, n_levels)
        print(f"使用离散colormap，级数: {n_levels}")

    # 应力空间平均化
    if averaging_radius is not None:
        if averaging_radius == 'auto':
            averaging_radius = estimate_optimal_averaging_radius(positions)
        von_mises = compute_spatial_averaged_stress(positions, von_mises, averaging_radius)
        print(f"应用空间平均化，半径: {averaging_radius:.6f}")
    else:
        print("跳过空间平均化（averaging_radius为None）")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Calculate adaptive particle size if not specified
    if particle_size is None:
        particle_size = calculate_adaptive_radius(positions)
    print(f"Using particle size (s parameter): {particle_size:.2f}")

    if use_log:
        from matplotlib.colors import LogNorm

        # von Mises stress (using logarithmic scale)
        von_mises_positive = von_mises.copy()
        np.clip(von_mises_positive, 1e-10, None, out=von_mises_positive)  # In-place, 避免零值
        vm_vmin = np.min(von_mises_positive)
        vm_vmax = max_stress if max_stress is not None else np.max(von_mises_positive)
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=von_mises_positive,
                           cmap=final_cmap, s=particle_size, alpha=0.8,
                           norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax), edgecolors='none')
        ax.set_title(f'von Mises Stress Distribution (Log Scale, Frame {frame_number})')
        cbar_label = 'von Mises Stress (Log)'
    else:
        # von Mises stress (linear scale)
        vm_vmax = max_stress if max_stress is not None else np.max(von_mises)
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=von_mises,
                           cmap=final_cmap, s=particle_size, alpha=0.8,
                           vmax=vm_vmax, edgecolors='none')
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
    plt.close()  # 显式关闭figure以释放内存


def visualize_schwarz_stress_2d(data_dict, save_path=None, use_log=False, stress_cmap=None, max_stress=None, particle_size=None, averaging_radius=None, use_discrete=True, n_levels=10):
    """2D Schwarz双域应力可视化"""
    # 使用自定义colormap作为默认值
    if stress_cmap is None:
        stress_cmap = CUSTOM_COLORMAP

    # 处理离散colormap
    if not use_discrete:
        final_cmap = stress_cmap
    else:
        final_cmap = create_discrete_colormap(stress_cmap, n_levels)
        print(f"使用离散colormap，级数: {n_levels}")

    frame_number = data_dict['frame_number']
    d1_data = data_dict['domain1']
    d2_data = data_dict['domain2']

    # Calculate von Mises stress for both domains
    d1_von_mises = compute_von_mises_stress(d1_data['stress'])
    d2_von_mises = compute_von_mises_stress(d2_data['stress'])

    # 应力空间平均化
    if averaging_radius is not None:
        if averaging_radius == 'auto':
            d1_radius = estimate_optimal_averaging_radius(d1_data['positions'])
            d2_radius = estimate_optimal_averaging_radius(d2_data['positions'])
            print(f"自动计算平均化半径 - Domain1: {d1_radius:.6f}, Domain2: {d2_radius:.6f}")
        else:
            d1_radius = d2_radius = averaging_radius

        d1_von_mises = compute_spatial_averaged_stress(d1_data['positions'], d1_von_mises, d1_radius)
        d2_von_mises = compute_spatial_averaged_stress(d2_data['positions'], d2_von_mises, d2_radius)
        print(f"对双域应力应用空间平均化")
    else:
        print("跳过双域空间平均化（averaging_radius为None）")

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
        vm_positive = all_von_mises.copy()
        np.clip(vm_positive, 1e-10, None, out=vm_positive)  # In-place
        vm_vmin = np.min(vm_positive)
        vm_vmax = max_stress if max_stress is not None else np.max(vm_positive)

        # Domain1 von Mises stress (log scale)
        d1_vm_positive = d1_von_mises.copy()
        np.clip(d1_vm_positive, 1e-10, None, out=d1_vm_positive)  # In-place
        scatter1 = ax1.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1],
                              c=d1_vm_positive, cmap=final_cmap, s=d1_particle_size, alpha=0.8,
                              norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax), edgecolors='none')
        ax1.set_title(f'Domain1 von Mises Stress (Log Scale, Frame {frame_number})')
        cbar1_label = 'von Mises Stress (Log)'

        # Domain2 von Mises stress (log scale)
        d2_vm_positive = d2_von_mises.copy()
        np.clip(d2_vm_positive, 1e-10, None, out=d2_vm_positive)  # In-place
        scatter2 = ax2.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1],
                              c=d2_vm_positive, cmap=final_cmap, s=d2_particle_size, alpha=0.8,
                              norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax), edgecolors='none')
        ax2.set_title(f'Domain2 von Mises Stress (Log Scale, Frame {frame_number})')
        cbar2_label = 'von Mises Stress (Log)'
    else:
        # Linear scale
        vm_vmin = np.min(all_von_mises)
        vm_vmax = max_stress if max_stress is not None else np.max(all_von_mises)

        # Domain1 von Mises stress (linear scale)
        scatter1 = ax1.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1],
                              c=d1_von_mises, cmap=final_cmap, s=d1_particle_size, alpha=0.8,
                              vmin=vm_vmin, vmax=vm_vmax, edgecolors='none')
        ax1.set_title(f'Domain1 von Mises Stress (Frame {frame_number})')
        cbar1_label = 'von Mises Stress'

        # Domain2 von Mises stress (linear scale)
        scatter2 = ax2.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1],
                              c=d2_von_mises, cmap=final_cmap, s=d2_particle_size, alpha=0.8,
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
    plt.close()  # 显式关闭figure以释放内存

def visualize_schwarz_stress_combined_2d(data_dict, save_path=None, use_log=False, stress_cmap=None, max_stress=None, particle_size=None, averaging_radius=None, use_discrete=True, n_levels=10):
    """2D Schwarz双域合并应力可视化（仅显示von Mises应力）"""
    # 使用自定义colormap作为默认值
    if stress_cmap is None:
        stress_cmap = CUSTOM_COLORMAP

    # 处理离散colormap
    if not use_discrete:
        final_cmap = stress_cmap
    else:
        final_cmap = create_discrete_colormap(stress_cmap, n_levels)
        print(f"使用离散colormap，级数: {n_levels}")

    frame_number = data_dict['frame_number']
    d1_data = data_dict['domain1']
    d2_data = data_dict['domain2']

    # Calculate von Mises stress for both domains
    d1_von_mises = compute_von_mises_stress(d1_data['stress'])
    d2_von_mises = compute_von_mises_stress(d2_data['stress'])

    # 应力空间平均化
    if averaging_radius is not None:
        if averaging_radius == 'auto':
            d1_radius = estimate_optimal_averaging_radius(d1_data['positions'])
            d2_radius = estimate_optimal_averaging_radius(d2_data['positions'])
            print(f"自动计算平均化半径 - Domain1: {d1_radius:.6f}, Domain2: {d2_radius:.6f}")
        else:
            d1_radius = d2_radius = averaging_radius

        d1_von_mises = compute_spatial_averaged_stress(d1_data['positions'], d1_von_mises, d1_radius)
        d2_von_mises = compute_spatial_averaged_stress(d2_data['positions'], d2_von_mises, d2_radius)
        print(f"对双域应力应用空间平均化")
    else:
        print("跳过双域空间平均化（averaging_radius为None）")

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
        vm_positive = all_von_mises.copy()
        np.clip(vm_positive, 1e-10, None, out=vm_positive)  # In-place
        vm_vmin = np.min(vm_positive)
        vm_vmax = max_stress if max_stress is not None else np.max(vm_positive)

        # Combined von Mises stress visualization (log scale)
        d1_vm_positive = d1_von_mises.copy()
        np.clip(d1_vm_positive, 1e-10, None, out=d1_vm_positive)  # In-place
        d2_vm_positive = d2_von_mises.copy()
        np.clip(d2_vm_positive, 1e-10, None, out=d2_vm_positive)  # In-place

        scatter_d1 = ax.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1],
                               c=d1_vm_positive, cmap=final_cmap, s=d1_particle_size, alpha=0.8,
                               norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax),
                               edgecolors='none', label='Domain1')
        scatter_d2 = ax.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1],
                               c=d2_vm_positive, cmap=final_cmap, s=d2_particle_size, alpha=0.8,
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
                               c=d1_von_mises, cmap=final_cmap, s=d1_particle_size, alpha=0.8,
                               vmin=vm_vmin, vmax=vm_vmax,
                               edgecolors='none', label='Domain1')
        scatter_d2 = ax.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1],
                               c=d2_von_mises, cmap=final_cmap, s=d2_particle_size, alpha=0.8,
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
    plt.close()  # 显式关闭figure以释放内存


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
    parser.add_argument('--cmap', default=None,
                       help='Colormap for stress visualization (default: custom blue-cyan-green-yellow-red colormap, options: viridis, plasma, inferno, magma, coolwarm, RdYlBu, seismic)')
    parser.add_argument('--max-stress', type=float, default=None,
                       help='Manually specify maximum value for stress color range (applies to von Mises stress only)')
    parser.add_argument('--particle-size', type=float, default=5,
                       help='Manually specify particle size for visualization (s parameter for scatter plot). If not specified, uses adaptive calculation based on particle spacing.')
    parser.add_argument('--all-frames', action='store_true',
                       help='Visualize all saved stress frames from the simulation (creates multiple plots)')
    parser.add_argument('--averaging-radius', type=str, default=None,
                       help='Spatial averaging radius for stress smoothing. Use "auto" for automatic estimation, a number for fixed radius, or None to disable averaging (default: None)')
    parser.add_argument('--continuous-colormap', action='store_true', default=False,
                       help='Use continuous colormap instead of discrete levels (default: discrete with 10 levels)')
    parser.add_argument('--colormap-levels', type=int, default=10,
                       help='Number of discrete levels for colormap when using discrete mode (default: 10)')


    args = parser.parse_args()

    # 处理colormap参数
    if args.cmap is None:
        # 使用自定义colormap作为默认值
        args.cmap = CUSTOM_COLORMAP
    elif isinstance(args.cmap, str):
        # 如果是字符串，保持原样让matplotlib处理
        pass

    # 处理平均化半径参数
    averaging_radius = None
    if args.averaging_radius is not None:
        if args.averaging_radius.lower() == 'auto':
            averaging_radius = 'auto'
        else:
            try:
                averaging_radius = float(args.averaging_radius)
                if averaging_radius <= 0:
                    print("错误: 平均化半径必须为正数")
                    return 1
            except ValueError:
                print("错误: 平均化半径必须是数字或'auto'")
                return 1

    try:
        if args.all_frames:
            # 处理所有帧的可视化
            available_frames, actual_dir = get_all_available_frames(args.dir, args.schwarz)

            if not available_frames:
                print("未找到任何应力数据帧")
                return

            print(f"找到 {len(available_frames)} 个保存的应力数据帧: {available_frames}")
            print(f"数据目录: {actual_dir}")

            # 为每个帧创建可视化
            for frame_num in available_frames:
                print(f"\n可视化帧 {frame_num}...")

                # 临时修改参数来可视化特定帧
                temp_args = args
                temp_args.frame = frame_num
                temp_args.all_frames = False  # 避免递归

                # 构造保存文件名（如果用户指定了保存选项）
                if args.save:
                    base_name, ext = os.path.splitext(args.save)
                    temp_args.save = f"{base_name}_frame_{frame_num}{ext}"

                # 可视化当前帧
                visualize_single_frame(temp_args, averaging_radius)

                # 显式清理内存，防止跨帧累积
                plt.close('all')  # 关闭所有matplotlib图形
                gc.collect()      # 强制垃圾回收

            print(f"\n完成！共可视化了 {len(available_frames)} 个帧")
            return

        # 可视化单个帧
        visualize_single_frame(args, averaging_radius)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())