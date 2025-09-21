#!/usr/bin/env python3
"""
应力应变数据可视化工具
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os
import glob

def load_stress_data(output_dir, frame_number=None):
    """加载应力应变数据（单域版本）"""
    # 首先搜索时间戳子目录
    timestamped_dirs = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
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
            
            # 从最新目录中找到帧文件
            stress_files = glob.glob(f"{latest_dir}/stress_frame_*.npy")
            if not stress_files:
                raise FileNotFoundError(f"在最新目录 {latest_dir} 中未找到应力数据文件")
            
            # 从文件名中提取帧号
            frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in stress_files]
            frame_number = max(frame_numbers)
            
            # 直接从最新目录加载
            stress_file = f"{latest_dir}/stress_frame_{frame_number}.npy"
            strain_file = f"{latest_dir}/strain_frame_{frame_number}.npy"
            positions_file = f"{latest_dir}/positions_frame_{frame_number}.npy"
            
            if not all(os.path.exists(f) for f in [stress_file, strain_file, positions_file]):
                raise FileNotFoundError(f"缺少帧 {frame_number} 的数据文件")
                
            found_files = (stress_file, strain_file, positions_file)
        else:
            # 如果没有时间戳目录，回退到旧的搜索方式
            stress_files = glob.glob(f"{output_dir}/stress_frame_*.npy")
            if not stress_files:
                raise FileNotFoundError(f"在 {output_dir} 中未找到应力数据文件")
            
            frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in stress_files]
            frame_number = max(frame_numbers)
            
            stress_file = f"{output_dir}/stress_frame_{frame_number}.npy"
            strain_file = f"{output_dir}/strain_frame_{frame_number}.npy"
            positions_file = f"{output_dir}/positions_frame_{frame_number}.npy"
            
            if not all(os.path.exists(f) for f in [stress_file, strain_file, positions_file]):
                raise FileNotFoundError(f"缺少帧 {frame_number} 的数据文件")
                
            found_files = (stress_file, strain_file, positions_file)
    else:
        # 如果指定了帧号，在所有目录中搜索
        search_dirs = timestamped_dirs if timestamped_dirs else [output_dir]
        found_files = None
        for search_dir in search_dirs:
            stress_file = f"{search_dir}/stress_frame_{frame_number}.npy"
            strain_file = f"{search_dir}/strain_frame_{frame_number}.npy"
            positions_file = f"{search_dir}/positions_frame_{frame_number}.npy"
            
            if all(os.path.exists(f) for f in [stress_file, strain_file, positions_file]):
                found_files = (stress_file, strain_file, positions_file)
                break
        
        if found_files is None:
            raise FileNotFoundError(f"缺少帧 {frame_number} 的数据文件")
    
    stress_data = np.load(found_files[0])
    strain_data = np.load(found_files[1])
    positions = np.load(found_files[2])
    
    return stress_data, strain_data, positions, frame_number

def load_schwarz_stress_data(output_dir, frame_number=None):
    """加载Schwarz双域应力应变数据"""
    # 首先搜索时间戳子目录
    timestamped_dirs = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
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
            
            # 从最新目录中找到domain1帧文件
            domain1_files = glob.glob(f"{latest_dir}/domain1_stress_frame_*.npy")
            if not domain1_files:
                raise FileNotFoundError(f"在最新目录 {latest_dir} 中未找到Domain1应力数据文件")
            
            # 从文件名中提取帧号
            frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in domain1_files]
            frame_number = max(frame_numbers)
            
            # 直接从最新目录加载
            d1_stress_file = f"{latest_dir}/domain1_stress_frame_{frame_number}.npy"
            d1_strain_file = f"{latest_dir}/domain1_strain_frame_{frame_number}.npy"
            d1_positions_file = f"{latest_dir}/domain1_positions_frame_{frame_number}.npy"
            
            d2_stress_file = f"{latest_dir}/domain2_stress_frame_{frame_number}.npy"
            d2_strain_file = f"{latest_dir}/domain2_strain_frame_{frame_number}.npy"
            d2_positions_file = f"{latest_dir}/domain2_positions_frame_{frame_number}.npy"
            
            all_files = [d1_stress_file, d1_strain_file, d1_positions_file,
                         d2_stress_file, d2_strain_file, d2_positions_file]
            
            if not all(os.path.exists(f) for f in all_files):
                raise FileNotFoundError(f"缺少帧 {frame_number} 的数据文件")
                
            found_files = all_files
        else:
            # 如果没有时间戳目录，回退到旧的搜索方式
            domain1_files = glob.glob(f"{output_dir}/domain1_stress_frame_*.npy")
            if not domain1_files:
                raise FileNotFoundError(f"在 {output_dir} 中未找到Domain1应力数据文件")
            
            frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in domain1_files]
            frame_number = max(frame_numbers)
            
            d1_stress_file = f"{output_dir}/domain1_stress_frame_{frame_number}.npy"
            d1_strain_file = f"{output_dir}/domain1_strain_frame_{frame_number}.npy"
            d1_positions_file = f"{output_dir}/domain1_positions_frame_{frame_number}.npy"
            
            d2_stress_file = f"{output_dir}/domain2_stress_frame_{frame_number}.npy"
            d2_strain_file = f"{output_dir}/domain2_strain_frame_{frame_number}.npy"
            d2_positions_file = f"{output_dir}/domain2_positions_frame_{frame_number}.npy"
            
            all_files = [d1_stress_file, d1_strain_file, d1_positions_file,
                         d2_stress_file, d2_strain_file, d2_positions_file]
            
            if not all(os.path.exists(f) for f in all_files):
                raise FileNotFoundError(f"缺少帧 {frame_number} 的数据文件")
                
            found_files = all_files
    else:
        # 如果指定了帧号，在所有目录中搜索
        search_dirs = timestamped_dirs if timestamped_dirs else [output_dir]
        found_files = None
        for search_dir in search_dirs:
            # Domain1文件
            d1_stress_file = f"{search_dir}/domain1_stress_frame_{frame_number}.npy"
            d1_strain_file = f"{search_dir}/domain1_strain_frame_{frame_number}.npy"
            d1_positions_file = f"{search_dir}/domain1_positions_frame_{frame_number}.npy"
            
            # Domain2文件
            d2_stress_file = f"{search_dir}/domain2_stress_frame_{frame_number}.npy"
            d2_strain_file = f"{search_dir}/domain2_strain_frame_{frame_number}.npy"
            d2_positions_file = f"{search_dir}/domain2_positions_frame_{frame_number}.npy"
            
            all_files = [d1_stress_file, d1_strain_file, d1_positions_file, 
                         d2_stress_file, d2_strain_file, d2_positions_file]
            
            if all(os.path.exists(f) for f in all_files):
                found_files = all_files
                break
        
        if found_files is None:
            raise FileNotFoundError(f"缺少帧 {frame_number} 的数据文件")
    
    # 加载Domain1数据
    d1_stress = np.load(found_files[0])
    d1_strain = np.load(found_files[1])
    d1_positions = np.load(found_files[2])
    
    # 加载Domain2数据
    d2_stress = np.load(found_files[3])
    d2_strain = np.load(found_files[4])
    d2_positions = np.load(found_files[5])
    
    return {
        'domain1': {'stress': d1_stress, 'strain': d1_strain, 'positions': d1_positions},
        'domain2': {'stress': d2_stress, 'strain': d2_strain, 'positions': d2_positions},
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

def compute_hydrostatic_pressure(stress_data):
    """计算静水压力"""
    dim = stress_data.shape[1]
    pressure = []
    
    for i in range(stress_data.shape[0]):
        s = stress_data[i]
        if dim == 2:
            p = -(s[0,0] + s[1,1]) / 2
        else:
            p = -(s[0,0] + s[1,1] + s[2,2]) / 3
        pressure.append(p)
    
    return np.array(pressure)

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

def visualize_stress_2d(positions, von_mises, pressure, frame_number, save_path=None, use_log=False, stress_cmap='coolwarm', max_stress=None):
    """2D应力可视化
    
    Args:
        max_stress: Manual maximum value for von Mises stress color range. If None, uses data maximum.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    particle_size = 20
    if use_log:
        from matplotlib.colors import LogNorm, SymLogNorm
        
        # von Mises stress (using logarithmic scale)
        von_mises_positive = np.maximum(von_mises, 1e-10)  # 避免零值
        vm_vmin = np.min(von_mises_positive)
        vm_vmax = max_stress if max_stress is not None else np.max(von_mises_positive)
        scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], c=von_mises_positive, 
                             cmap=stress_cmap, s=particle_size, alpha=0.8, 
                             norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax))
        ax1.set_title(f'von Mises Stress Distribution (Log Scale, Frame {frame_number})')
        cbar1_label = 'von Mises Stress (Log)'
        
        # Hydrostatic pressure (using symmetric logarithmic scale)
        pressure_abs_max = np.max(np.abs(pressure))
        if pressure_abs_max > 0:
            linthresh = pressure_abs_max / 1000
        else:
            linthresh = 1e-10
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], c=pressure, 
                             cmap='RdBu', s=particle_size, alpha=0.8,
                             norm=SymLogNorm(linthresh=linthresh))
        ax2.set_title(f'Hydrostatic Pressure Distribution (SymLog Scale, Frame {frame_number})')
        cbar2_label = 'Hydrostatic Pressure (SymLog)'
    else:
        # von Mises stress (linear scale)
        vm_vmax = max_stress if max_stress is not None else np.max(von_mises)
        scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], c=von_mises, 
                             cmap=stress_cmap, s=particle_size, alpha=0.8,
                             vmax=vm_vmax)
        ax1.set_title(f'von Mises Stress Distribution (Frame {frame_number})')
        cbar1_label = 'von Mises Stress'
        
        # Hydrostatic pressure (linear scale)
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], c=pressure, 
                             cmap='RdBu', s=particle_size, alpha=0.8)
        ax2.set_title(f'Hydrostatic Pressure Distribution (Frame {frame_number})')
        cbar2_label = 'Hydrostatic Pressure'
    
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
        print(f"Image saved to: {save_path}")
    
    plt.show()


def visualize_schwarz_stress_2d(data_dict, save_path=None, use_log=False, stress_cmap='coolwarm', max_stress=None):
    """2D Schwarz双域应力可视化"""
    
    frame_number = data_dict['frame_number']
    d1_data = data_dict['domain1']
    d2_data = data_dict['domain2']
    
    # Calculate von Mises stress and hydrostatic pressure for both domains
    d1_von_mises = compute_von_mises_stress(d1_data['stress'])
    d1_pressure = compute_hydrostatic_pressure(d1_data['stress'])
    d2_von_mises = compute_von_mises_stress(d2_data['stress'])
    d2_pressure = compute_hydrostatic_pressure(d2_data['stress'])
    
    # Merge data to unify scale range
    all_von_mises = np.concatenate([d1_von_mises, d2_von_mises])
    all_pressure = np.concatenate([d1_pressure, d2_pressure])
    
    # Calculate adaptive radius for each domain
    d1_particle_size = 20
    d2_particle_size = 20

    print(f"Domain1 particle size (s parameter): {d1_particle_size:.2f}")
    print(f"Domain2 particle size (s parameter): {d2_particle_size:.2f}")

    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    if use_log:
        from matplotlib.colors import LogNorm, SymLogNorm
        
        # Calculate logarithmic scale range
        vm_positive = np.maximum(all_von_mises, 1e-10)
        vm_vmin = np.min(vm_positive)
        vm_vmax = max_stress if max_stress is not None else np.max(vm_positive)
        
        pressure_abs_max = np.max(np.abs(all_pressure))
        if pressure_abs_max > 0:
            p_linthresh = pressure_abs_max / 1000
        else:
            p_linthresh = 1e-10
        
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
        
        # Domain1 hydrostatic pressure (symmetric log scale)
        scatter3 = ax3.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1], 
                              c=d1_pressure, cmap='RdBu', s=d1_particle_size, alpha=0.8, 
                              norm=SymLogNorm(linthresh=p_linthresh), edgecolors='none')
        ax3.set_title(f'Domain1 Hydrostatic Pressure (SymLog Scale, Frame {frame_number})')
        cbar3_label = 'Hydrostatic Pressure (SymLog)'
        
        # Domain2 hydrostatic pressure (symmetric log scale)
        scatter4 = ax4.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1], 
                              c=d2_pressure, cmap='RdBu', s=d2_particle_size, alpha=0.8, 
                              norm=SymLogNorm(linthresh=p_linthresh), edgecolors='none')
        ax4.set_title(f'Domain2 Hydrostatic Pressure (SymLog Scale, Frame {frame_number})')
        cbar4_label = 'Hydrostatic Pressure (SymLog)'
    else:
        # Linear scale
        vm_vmin = np.min(all_von_mises)
        vm_vmax = max_stress if max_stress is not None else np.max(all_von_mises)
        p_vmin, p_vmax = np.min(all_pressure), np.max(all_pressure)
        
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
        
        # Domain1 hydrostatic pressure (linear scale)
        scatter3 = ax3.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1], 
                              c=d1_pressure, cmap='RdBu', s=d1_particle_size, alpha=0.8, 
                              vmin=p_vmin, vmax=p_vmax, edgecolors='none')
        ax3.set_title(f'Domain1 Hydrostatic Pressure (Frame {frame_number})')
        cbar3_label = 'Hydrostatic Pressure'
        
        # Domain2 hydrostatic pressure (linear scale)
        scatter4 = ax4.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1], 
                              c=d2_pressure, cmap='RdBu', s=d2_particle_size, alpha=0.8, 
                              vmin=p_vmin, vmax=p_vmax, edgecolors='none')
        ax4.set_title(f'Domain2 Hydrostatic Pressure (Frame {frame_number})')
        cbar4_label = 'Hydrostatic Pressure'
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label=cbar1_label)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label=cbar2_label)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3, label=cbar3_label)
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    plt.colorbar(scatter4, ax=ax4, label=cbar4_label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dual domain stress image saved to: {save_path}")
    
    plt.show()

def visualize_schwarz_stress_combined_2d(data_dict, save_path=None, use_log=False, stress_cmap='coolwarm', max_stress=None):
    """2D Schwarz双域合并应力可视化"""
    
    frame_number = data_dict['frame_number']
    d1_data = data_dict['domain1']
    d2_data = data_dict['domain2']
    
    # Calculate von Mises stress and hydrostatic pressure for both domains
    d1_von_mises = compute_von_mises_stress(d1_data['stress'])
    d1_pressure = compute_hydrostatic_pressure(d1_data['stress'])
    d2_von_mises = compute_von_mises_stress(d2_data['stress'])
    d2_pressure = compute_hydrostatic_pressure(d2_data['stress'])
    
    # Merge data to unify scale range
    all_von_mises = np.concatenate([d1_von_mises, d2_von_mises])
    all_pressure = np.concatenate([d1_pressure, d2_pressure])
    
    # Calculate adaptive radius for each domain
    d1_particle_size = calculate_adaptive_radius(d1_data['positions'])
    d2_particle_size = calculate_adaptive_radius(d2_data['positions'])

    print(f"Domain1 particle size (s parameter): {d1_particle_size:.2f}")
    print(f"Domain2 particle size (s parameter): {d2_particle_size:.2f}")

    # Create 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    if use_log:
        from matplotlib.colors import LogNorm, SymLogNorm
        
        # Calculate logarithmic scale range
        vm_positive = np.maximum(all_von_mises, 1e-10)
        vm_vmin = np.min(vm_positive)
        vm_vmax = max_stress if max_stress is not None else np.max(vm_positive)
        
        pressure_abs_max = np.max(np.abs(all_pressure))
        if pressure_abs_max > 0:
            p_linthresh = pressure_abs_max / 1000
        else:
            p_linthresh = 1e-10
        
        # Combined von Mises stress visualization (log scale)
        d1_vm_positive = np.maximum(d1_von_mises, 1e-10)
        d2_vm_positive = np.maximum(d2_von_mises, 1e-10)
        
        scatter1_d1 = ax1.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1], 
                                 c=d1_vm_positive, cmap=stress_cmap, s=d1_particle_size, alpha=0.8, 
                                 norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax), 
                                 edgecolors='black', linewidth=0.3, label='Domain1')
        scatter1_d2 = ax1.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1], 
                                 c=d2_vm_positive, cmap=stress_cmap, s=d2_particle_size, alpha=0.8, 
                                 norm=LogNorm(vmin=vm_vmin, vmax=vm_vmax), 
                                 edgecolors='red', linewidth=0.3, marker='^', label='Domain2')
        ax1.set_title(f'Dual Domain von Mises Stress Distribution (Log Scale, Frame {frame_number})')
        cbar1_label = 'von Mises Stress (Log)'
        
        # Combined hydrostatic pressure visualization (symmetric log scale)
        scatter2_d1 = ax2.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1], 
                                 c=d1_pressure, cmap='RdBu', s=d1_particle_size, alpha=0.8, 
                                 norm=SymLogNorm(linthresh=p_linthresh), 
                                 edgecolors='black', linewidth=0.3, label='Domain1')
        scatter2_d2 = ax2.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1], 
                                 c=d2_pressure, cmap='RdBu', s=d2_particle_size, alpha=0.8, 
                                 norm=SymLogNorm(linthresh=p_linthresh), 
                                 edgecolors='red', linewidth=0.3, marker='^', label='Domain2')
        ax2.set_title(f'Dual Domain Hydrostatic Pressure Distribution (SymLog Scale, Frame {frame_number})')
        cbar2_label = 'Hydrostatic Pressure (SymLog)'
    else:
        # Linear scale范围
        vm_vmin = np.min(all_von_mises)
        vm_vmax = max_stress if max_stress is not None else np.max(all_von_mises)
        p_vmin, p_vmax = np.min(all_pressure), np.max(all_pressure)
        
        # Combined von Mises stress visualization (linear scale)
        scatter1_d1 = ax1.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1], 
                                 c=d1_von_mises, cmap=stress_cmap, s=d1_particle_size, alpha=0.8, 
                                 vmin=vm_vmin, vmax=vm_vmax, 
                                 edgecolors='black', linewidth=0.3, label='Domain1')
        scatter1_d2 = ax1.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1], 
                                 c=d2_von_mises, cmap=stress_cmap, s=d2_particle_size, alpha=0.8, 
                                 vmin=vm_vmin, vmax=vm_vmax, 
                                 edgecolors='red', linewidth=0.3, marker='^', label='Domain2')
        ax1.set_title(f'Dual Domain von Mises Stress Distribution (Frame {frame_number})')
        cbar1_label = 'von Mises Stress'
        
        # Combined hydrostatic pressure visualization (linear scale)
        scatter2_d1 = ax2.scatter(d1_data['positions'][:, 0], d1_data['positions'][:, 1], 
                                 c=d1_pressure, cmap='RdBu', s=d1_particle_size, alpha=0.8, 
                                 vmin=p_vmin, vmax=p_vmax, 
                                 edgecolors='black', linewidth=0.3, label='Domain1')
        scatter2_d2 = ax2.scatter(d2_data['positions'][:, 0], d2_data['positions'][:, 1], 
                                 c=d2_pressure, cmap='RdBu', s=d2_particle_size, alpha=0.8, 
                                 vmin=p_vmin, vmax=p_vmax, 
                                 edgecolors='red', linewidth=0.3, marker='^', label='Domain2')
        ax2.set_title(f'Dual Domain Hydrostatic Pressure Distribution (Frame {frame_number})')
        cbar2_label = 'Hydrostatic Pressure'
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    ax1.legend()
    plt.colorbar(scatter1_d1, ax=ax1, label=cbar1_label)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.legend()
    plt.colorbar(scatter2_d1, ax=ax2, label=cbar2_label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dual domain combined stress image saved to: {save_path}")
    
    plt.show()


def print_schwarz_stress_statistics(data_dict):
    """打印Schwarz双域应力统计信息"""
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
        pressure = compute_hydrostatic_pressure(domain_data['stress'])
        
        print(f"Number of particles: {len(domain_data['positions'])}")
        print(f"Dimensions: {domain_data['positions'].shape[1]}D")
        
        print(f"von Mises Stress:")
        print(f"  Min: {np.min(von_mises):.3e}")
        print(f"  Max: {np.max(von_mises):.3e}")
        print(f"  Mean: {np.mean(von_mises):.3e}")
        print(f"  Std: {np.std(von_mises):.3e}")
        
        print(f"Hydrostatic Pressure:")
        print(f"  Min: {np.min(pressure):.3e}")
        print(f"  Max: {np.max(pressure):.3e}")
        print(f"  Mean: {np.mean(pressure):.3e}")
        print(f"  Std: {np.std(pressure):.3e}")
        
        # Stress component statistics
        stress_data = domain_data['stress']
        dim = stress_data.shape[1]
        print(f"Stress Components ({dim}D):")
        for i in range(dim):
            for j in range(dim):
                component = stress_data[:, i, j]
                print(f"  σ_{i+1}{j+1}: {np.mean(component):.3e} ± {np.std(component):.3e}")

def print_stress_statistics(von_mises, pressure, stress_data):
    """Print stress statistics information"""
    print("Stress Statistics:")
    print("-" * 40)
    print(f"von Mises Stress:")
    print(f"  Min: {np.min(von_mises):.3e}")
    print(f"  Max: {np.max(von_mises):.3e}")
    print(f"  Mean: {np.mean(von_mises):.3e}")
    print(f"  Std: {np.std(von_mises):.3e}")
    
    print(f"\nHydrostatic Pressure:")
    print(f"  Min: {np.min(pressure):.3e}")
    print(f"  Max: {np.max(pressure):.3e}")
    print(f"  Mean: {np.mean(pressure):.3e}")
    print(f"  Std: {np.std(pressure):.3e}")
    
    # Stress component statistics
    dim = stress_data.shape[1]
    print(f"\nStress Components ({dim}D):")
    for i in range(dim):
        for j in range(dim):
            component = stress_data[:, i, j]
            print(f"  σ_{i+1}{j+1}: {np.mean(component):.3e} ± {np.std(component):.3e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize stress and strain data')
    parser.add_argument('--dir', '-d', default='stress_strain_output', 
                       help='Data directory (default: stress_strain_output)')
    parser.add_argument('--frame', '-f', type=int, default=None,
                       help='Specify frame number (default: latest frame)')
    parser.add_argument('--save', '-s', default=None,
                       help='Save image path')
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('--schwarz', action='store_true',
                       help='Use Schwarz dual domain data (from stress_strain_output_schwarz directory)')
    parser.add_argument('--combined', action='store_true',
                       help='Dual domain combined visualization (only for Schwarz mode)')
    parser.add_argument('--log', action='store_true',
                       help='Use logarithmic scale visualization (default: linear scale)')
    parser.add_argument('--cmap', default='viridis',
                       help='Colormap for stress visualization (default: coolwarm, options: viridis, plasma, inferno, magma, coolwarm, RdYlBu, seismic)')
    parser.add_argument('--max-stress', type=float, default=None,
                       help='Manually specify maximum value for stress color range (applies to von Mises stress only)')
    
    
    args = parser.parse_args()
    
    try:
        if args.schwarz:
            # Schwarz dual domain mode
            if args.dir == 'stress_strain_output':
                args.dir = 'stress_strain_output_schwarz'  # Default directory
            
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
                visualize_schwarz_stress_combined_2d(data_dict, save_path, args.log, args.cmap, args.max_stress)
            else:
                # Separate visualization mode
                visualize_schwarz_stress_2d(data_dict, save_path, args.log, args.cmap, args.max_stress)
                    
        else:
            # Single domain mode (original functionality)
            print(f"Loading data from {args.dir}...")
            stress_data, strain_data, positions, frame_number = load_stress_data(args.dir, args.frame)
            print(f"Successfully loaded data for frame {frame_number}")
            print(f"Particle count: {positions.shape[0]}")
            print(f"维度: {positions.shape[1]}D")
            
            # Calculate stress scalars
            von_mises = compute_von_mises_stress(stress_data)
            pressure = compute_hydrostatic_pressure(stress_data)
            
            # Print statistics
            if args.stats:
                print_stress_statistics(von_mises, pressure, stress_data)
            
            # Visualization
            dim = positions.shape[1]
            save_path = args.save
            if save_path and not save_path.endswith(('.png', '.jpg', '.pdf')):
                save_path += '.png'
            
            print("Generating visualization...")
            if dim != 2:
                print(f"Error: Only 2D data visualization is supported, current data dimension is {dim}D")
                return 1
                
            visualize_stress_2d(positions, von_mises, pressure, frame_number, save_path, args.log, args.cmap, args.max_stress)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())