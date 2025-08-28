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
    """加载应力应变数据"""
    if frame_number is None:
        # 自动找到最新的帧
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
    
    stress_data = np.load(stress_file)
    strain_data = np.load(strain_file)
    positions = np.load(positions_file)
    
    return stress_data, strain_data, positions, frame_number

def compute_von_mises_stress(stress_data):
    """计算von Mises应力"""
    von_mises = []
    dim = stress_data.shape[1]
    
    for i in range(stress_data.shape[0]):
        s = stress_data[i]
        if dim == 2:
            # 2D von Mises应力
            vm = np.sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2)
        else:
            # 3D von Mises应力
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

def visualize_stress_2d(positions, von_mises, pressure, frame_number, save_path=None):
    """2D应力可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # von Mises应力
    scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], c=von_mises, 
                         cmap='viridis', s=20, alpha=0.8)
    ax1.set_title(f'von Mises应力分布 (帧 {frame_number})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='von Mises应力')
    
    # 静水压力
    scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], c=pressure, 
                         cmap='RdBu', s=20, alpha=0.8)
    ax2.set_title(f'静水压力分布 (帧 {frame_number})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='静水压力')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()

def visualize_stress_3d(positions, von_mises, pressure, frame_number, save_path=None):
    """3D应力可视化"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(20, 8))
    
    # von Mises应力
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                          c=von_mises, cmap='viridis', s=20, alpha=0.6)
    ax1.set_title(f'von Mises应力分布 (帧 {frame_number})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, label='von Mises应力', shrink=0.8)
    
    # 静水压力
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                          c=pressure, cmap='RdBu', s=20, alpha=0.6)
    ax2.set_title(f'静水压力分布 (帧 {frame_number})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.colorbar(scatter2, ax=ax2, label='静水压力', shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()

def print_stress_statistics(von_mises, pressure, stress_data):
    """打印应力统计信息"""
    print("应力统计信息:")
    print("-" * 40)
    print(f"von Mises应力:")
    print(f"  最小值: {np.min(von_mises):.3e}")
    print(f"  最大值: {np.max(von_mises):.3e}")
    print(f"  平均值: {np.mean(von_mises):.3e}")
    print(f"  标准差: {np.std(von_mises):.3e}")
    
    print(f"\n静水压力:")
    print(f"  最小值: {np.min(pressure):.3e}")
    print(f"  最大值: {np.max(pressure):.3e}")
    print(f"  平均值: {np.mean(pressure):.3e}")
    print(f"  标准差: {np.std(pressure):.3e}")
    
    # 应力分量统计
    dim = stress_data.shape[1]
    print(f"\n应力分量统计 ({dim}D):")
    for i in range(dim):
        for j in range(dim):
            component = stress_data[:, i, j]
            print(f"  σ_{i+1}{j+1}: {np.mean(component):.3e} ± {np.std(component):.3e}")

def main():
    parser = argparse.ArgumentParser(description='可视化应力应变数据')
    parser.add_argument('--dir', '-d', default='stress_strain_output', 
                       help='数据目录 (默认: stress_strain_output)')
    parser.add_argument('--frame', '-f', type=int, default=None,
                       help='指定帧号 (默认: 最新帧)')
    parser.add_argument('--save', '-s', default=None,
                       help='保存图像路径')
    parser.add_argument('--stats', action='store_true',
                       help='显示详细统计信息')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        print(f"从 {args.dir} 加载数据...")
        stress_data, strain_data, positions, frame_number = load_stress_data(args.dir, args.frame)
        print(f"成功加载帧 {frame_number} 的数据")
        print(f"粒子数量: {positions.shape[0]}")
        print(f"维度: {positions.shape[1]}D")
        
        # 计算应力标量
        von_mises = compute_von_mises_stress(stress_data)
        pressure = compute_hydrostatic_pressure(stress_data)
        
        # 打印统计信息
        if args.stats:
            print_stress_statistics(von_mises, pressure, stress_data)
        
        # 可视化
        dim = positions.shape[1]
        save_path = args.save
        if save_path and not save_path.endswith(('.png', '.jpg', '.pdf')):
            save_path += '.png'
        
        print("生成可视化...")
        if dim == 2:
            visualize_stress_2d(positions, von_mises, pressure, frame_number, save_path)
        else:
            visualize_stress_3d(positions, von_mises, pressure, frame_number, save_path)
            
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())