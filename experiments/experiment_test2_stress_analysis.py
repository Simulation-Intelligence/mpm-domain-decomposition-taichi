#!/usr/bin/env python3
"""
实验Test2: Schwarz双域应力分析
运行config/schwarz_2d_test2.json配置的双域模拟，分析Domain2的应力分布特征
重点分析Domain2中心横向和纵向的应力变化曲线
"""

import json
import numpy as np
import sys
import os
import subprocess
import time
import matplotlib.pyplot as plt
import glob
import re
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_schwarz_simulation(config_path="config/schwarz_2d_test2.json", output_dir="experiments/test2_results"):
    """运行Schwarz双域模拟"""
    print("="*60)
    print("开始运行Schwarz双域模拟 (Test2)")
    print(f"配置文件: {config_path}")
    print("="*60)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # 运行模拟
    cmd = [
        sys.executable,
        "simulators/implicit_mpm_schwarz.py",
        "--config", config_path,
        "--no-gui"  # 使用无GUI模式
    ]

    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print("模拟失败!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    print(f"模拟完成，耗时: {elapsed_time:.2f}秒")

    # 查找最新的实验结果目录
    base_output_dir = "experiment_results"
    if not os.path.exists(base_output_dir):
        print(f"输出目录不存在: {base_output_dir}")
        return None

    schwarz_pattern = os.path.join(base_output_dir, "schwarz_*")
    schwarz_dirs = glob.glob(schwarz_pattern)

    if not schwarz_dirs:
        print("未找到Schwarz实验结果")
        return None

    # 按时间戳排序，选择最新的
    def extract_timestamp(dirname):
        basename = os.path.basename(dirname)
        match = re.search(r'schwarz_(\d{8}_\d{6})', basename)
        if match:
            timestamp_str = match.group(1)
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return datetime.min

    latest_dir = max(schwarz_dirs, key=extract_timestamp)
    print(f"找到最新实验结果: {latest_dir}")

    return latest_dir

def load_domain2_data(experiment_dir, frame_number=None):
    """加载Domain2的应力和位置数据"""
    if frame_number is None:
        # 查找最新帧
        stress_files = glob.glob(f"{experiment_dir}/domain2_stress_frame_*.npy")
        if not stress_files:
            raise FileNotFoundError(f"在{experiment_dir}中未找到Domain2应力数据")

        frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in stress_files]
        frame_number = max(frame_numbers)

    # 加载数据文件
    stress_file = f"{experiment_dir}/domain2_stress_frame_{frame_number}.npy"
    positions_file = f"{experiment_dir}/domain2_positions_frame_{frame_number}.npy"

    if not all(os.path.exists(f) for f in [stress_file, positions_file]):
        missing_files = [f for f in [stress_file, positions_file] if not os.path.exists(f)]
        raise FileNotFoundError(f"缺少数据文件: {missing_files}")

    stress_data = np.load(stress_file)
    positions = np.load(positions_file)

    print(f"加载Domain2数据 - 帧{frame_number}")
    print(f"  粒子数量: {len(positions)}")
    print(f"  位置范围: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")

    return stress_data, positions, frame_number

def compute_stress_invariants(stress_data):
    """计算应力不变量"""
    von_mises = []
    hydrostatic_pressure = []

    for i in range(stress_data.shape[0]):
        s = stress_data[i]
        # 2D von Mises应力
        vm = np.sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2)
        von_mises.append(vm)

        # 静水压力 (负的正应力迹的平均值)
        hydrostatic = -(s[0,0] + s[1,1]) / 2
        hydrostatic_pressure.append(hydrostatic)

    return np.array(von_mises), np.array(hydrostatic_pressure)

def extract_horizontal_profile(positions, stress_values, tolerance_factor=0.01, n_bins=100):
    """提取沿横向中心线的应力分布，使用bin分割取平均"""
    # 找到纵坐标中心
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    y_center = (y_min + y_max) / 2

    # 设置容差
    y_tolerance = (y_max - y_min) * tolerance_factor

    # 选择在横向中心线附近的粒子
    horizontal_mask = np.abs(positions[:, 1] - y_center) <= y_tolerance

    if not np.any(horizontal_mask):
        print(f"警告: 在纵坐标中心线附近未找到粒子 (Y = {y_center:.3f} ± {y_tolerance:.3f})")
        return None, None

    horizontal_positions = positions[horizontal_mask]
    horizontal_stress = stress_values[horizontal_mask]

    # 获取X坐标范围
    x_min, x_max = horizontal_positions[:, 0].min(), horizontal_positions[:, 0].max()

    # 创建bin边界
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 为每个bin计算平均应力
    bin_stress = []
    valid_bins = []

    for i in range(n_bins):
        # 找到在当前bin内的粒子
        bin_mask = (horizontal_positions[:, 0] >= bin_edges[i]) & (horizontal_positions[:, 0] < bin_edges[i + 1])

        if i == n_bins - 1:  # 最后一个bin包含右边界
            bin_mask = (horizontal_positions[:, 0] >= bin_edges[i]) & (horizontal_positions[:, 0] <= bin_edges[i + 1])

        if np.any(bin_mask):
            bin_stress_values = horizontal_stress[bin_mask]
            bin_stress.append(np.mean(bin_stress_values))
            valid_bins.append(i)

    if not valid_bins:
        print("警告: 所有bin都为空")
        return None, None

    # 只保留有数据的bin
    x_coords = bin_centers[valid_bins]
    stress_values = np.array(bin_stress)

    print(f"横向剖面: {len(horizontal_positions)}个粒子 → {len(x_coords)}个bin沿Y = {y_center:.3f} ± {y_tolerance:.3f}")

    return x_coords, stress_values

def extract_vertical_profile(positions, stress_values, tolerance_factor=0.01, n_bins=100):
    """提取沿纵向中心线的应力分布"""
    # 找到横坐标中心
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    x_center = (x_min + x_max) / 2

    # 设置容差
    x_tolerance = (x_max - x_min) * tolerance_factor

    # 选择在纵向中心线附近的粒子
    vertical_mask = np.abs(positions[:, 0] - x_center) <= x_tolerance

    if not np.any(vertical_mask):
        print(f"警告: 在横坐标中心线附近未找到粒子 (X = {x_center:.3f} ± {x_tolerance:.3f})")
        return None, None

    vertical_positions = positions[vertical_mask]
    vertical_stress = stress_values[vertical_mask]

    # 获取Y坐标范围
    y_min, y_max = vertical_positions[:, 1].min(), vertical_positions[:, 1].max()

    # 创建bin边界
    bin_edges = np.linspace(y_min, y_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 为每个bin计算平均应力
    bin_stress = []
    valid_bins = []

    for i in range(n_bins):
        # 找到在当前bin内的粒子
        bin_mask = (vertical_positions[:, 1] >= bin_edges[i]) & (vertical_positions[:, 1] < bin_edges[i + 1])

        if i == n_bins - 1:  # 最后一个bin包含右边界
            bin_mask = (vertical_positions[:, 1] >= bin_edges[i]) & (vertical_positions[:, 1] <= bin_edges[i + 1])

        if np.any(bin_mask):
            bin_stress_values = vertical_stress[bin_mask]
            bin_stress.append(np.mean(bin_stress_values))
            valid_bins.append(i)

    if not valid_bins:
        print("警告: 所有bin都为空")
        return None, None

    # 只保留有数据的bin
    valid_bin_centers = bin_centers[valid_bins]
    valid_bin_stress = np.array(bin_stress)

    print(f"纵向剖面: {len(valid_bins)}个bin沿X = {x_center:.3f} ± {x_tolerance:.3f}")

    return valid_bin_centers, valid_bin_stress

def smooth_curve(x, y, window_size=3):
    """使用简单移动平均进行光滑化处理"""
    if len(x) < 3:
        return x, y

    # 确保数据按x坐标排序
    sort_indices = np.argsort(x)
    x_sorted = x[sort_indices]
    y_sorted = y[sort_indices]

    # 自动调整窗口大小
    window_size = max(3, min(len(y_sorted) // 3, 10))
    if window_size >= len(y_sorted):
        return x_sorted, y_sorted

    # 简单移动平均光滑化
    y_smooth = np.convolve(y_sorted, np.ones(window_size)/window_size, mode='same')

    return x_sorted, y_smooth

def plot_stress_profiles(positions, von_mises, hydrostatic_pressure, frame_number, save_dir="experiments/test2_results"):
    """绘制Domain2的应力剖面图"""

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 提取横向和纵向剖面
    x_coords, h_von_mises = extract_horizontal_profile(positions, von_mises)
    x_coords_h, h_pressure = extract_horizontal_profile(positions, hydrostatic_pressure)
    y_coords, v_von_mises = extract_vertical_profile(positions, von_mises)
    y_coords_v, v_pressure = extract_vertical_profile(positions, hydrostatic_pressure)

    # 创建综合图形
    fig = plt.figure(figsize=(16, 12))

    # 2x2布局
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 横向von Mises应力
    ax1 = fig.add_subplot(gs[0, 0])
    if x_coords is not None and h_von_mises is not None and len(x_coords) > 0:
        print(f"横向数据: {len(x_coords)}个点, X范围: [{x_coords.min():.3f}, {x_coords.max():.3f}], 应力范围: [{h_von_mises.min():.2e}, {h_von_mises.max():.2e}]")

        # 光滑化曲线
        if len(x_coords) >= 3:
            try:
                x_smooth, y_smooth = smooth_curve(x_coords, h_von_mises)
                ax1.plot(x_smooth, y_smooth, 'b-', linewidth=2)
            except Exception as e:
                print(f"光滑化失败: {e}")
                # 如果光滑化失败，只绘制原始数据连线
                ax1.plot(x_coords, h_von_mises, 'b-', linewidth=2)
        else:
            ax1.plot(x_coords, h_von_mises, 'b-', linewidth=2)

        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('von Mises Stress')
        ax1.set_title('Horizontal Stress Distribution (Along Y-center)')
        ax1.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f'Min: {h_von_mises.min():.2e}\nMax: {h_von_mises.max():.2e}\nMean: {h_von_mises.mean():.2e}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, 'No horizontal data available', ha='center', va='center', transform=ax1.transAxes)

    # 纵向von Mises应力 (交换坐标)
    ax2 = fig.add_subplot(gs[0, 1])
    if y_coords is not None and v_von_mises is not None and len(y_coords) > 0:
        print(f"纵向数据: {len(y_coords)}个点, Y范围: [{y_coords.min():.3f}, {y_coords.max():.3f}], 应力范围: [{v_von_mises.min():.2e}, {v_von_mises.max():.2e}]")

        # 光滑化曲线 (交换x,y坐标)
        if len(y_coords) >= 3:
            try:
                y_smooth, stress_smooth = smooth_curve(y_coords, v_von_mises)
                ax2.plot(y_smooth, stress_smooth, 'r-', linewidth=2)
            except Exception as e:
                print(f"纵向光滑化失败: {e}")
                ax2.plot(y_coords, v_von_mises, 'r-', linewidth=2)
        else:
            ax2.plot(y_coords, v_von_mises, 'r-', linewidth=2)

        ax2.set_xlabel('Y Coordinate')  # 交换后的标签
        ax2.set_ylabel('von Mises Stress')  # 交换后的标签
        ax2.set_title('Vertical Stress Distribution (Along X-center)')
        ax2.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f'Min: {v_von_mises.min():.2e}\nMax: {v_von_mises.max():.2e}\nMean: {v_von_mises.mean():.2e}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No vertical data available', ha='center', va='center', transform=ax2.transAxes)

    # 横向静水压力
    ax3 = fig.add_subplot(gs[1, 0])
    if x_coords_h is not None and h_pressure is not None and len(x_coords_h) > 0:
        # 光滑化曲线
        if len(x_coords_h) >= 3:
            try:
                x_smooth, pressure_smooth = smooth_curve(x_coords_h, h_pressure)
                ax3.plot(x_smooth, pressure_smooth, 'g-', linewidth=2)
            except Exception as e:
                ax3.plot(x_coords_h, h_pressure, 'g-', linewidth=2)
        else:
            ax3.plot(x_coords_h, h_pressure, 'g-', linewidth=2)
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Hydrostatic Pressure')
        ax3.set_title('Horizontal Pressure Distribution (Along Y-center)')
        ax3.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f'Min: {h_pressure.min():.2e}\nMax: {h_pressure.max():.2e}\nMean: {h_pressure.mean():.2e}'
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No horizontal pressure data', ha='center', va='center', transform=ax3.transAxes)

    # 纵向静水压力 (交换坐标)
    ax4 = fig.add_subplot(gs[1, 1])
    if y_coords_v is not None and v_pressure is not None and len(y_coords_v) > 0:
        # 光滑化曲线 (交换x,y坐标)
        if len(y_coords_v) >= 3:
            try:
                y_smooth, pressure_smooth = smooth_curve(y_coords_v, v_pressure)
                ax4.plot(y_smooth, pressure_smooth, 'm-', linewidth=2)
            except Exception as e:
                ax4.plot(y_coords_v, v_pressure, 'm-', linewidth=2)
        else:
            ax4.plot(y_coords_v, v_pressure, 'm-', linewidth=2)
        ax4.set_xlabel('Y Coordinate')  # 交换后的标签
        ax4.set_ylabel('Hydrostatic Pressure')  # 交换后的标签
        ax4.set_title('Vertical Pressure Distribution (Along X-center)')
        ax4.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f'Min: {v_pressure.min():.2e}\nMax: {v_pressure.max():.2e}\nMean: {v_pressure.mean():.2e}'
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No vertical pressure data', ha='center', va='center', transform=ax4.transAxes)

    plt.suptitle(f'Domain2 Stress Profile Analysis (Frame {frame_number})', fontsize=16)

    # 保存图形
    save_path = os.path.join(save_dir, f'domain2_stress_profiles_frame_{frame_number}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"应力剖面图已保存至: {save_path}")

    plt.show()

    return save_path

def analyze_stress_distribution(positions, stress_data, frame_number, save_dir="experiments/test2_results"):
    """分析Domain2的应力分布特征"""
    print("\n" + "="*50)
    print(f"Domain2应力分析 (帧 {frame_number})")
    print("="*50)

    # 计算应力不变量
    von_mises, hydrostatic_pressure = compute_stress_invariants(stress_data)

    # 整体统计
    print(f"\n整体应力统计:")
    print(f"von Mises应力:")
    print(f"  最小值: {von_mises.min():.3e}")
    print(f"  最大值: {von_mises.max():.3e}")
    print(f"  平均值: {von_mises.mean():.3e}")
    print(f"  标准差: {von_mises.std():.3e}")

    print(f"\n静水压力:")
    print(f"  最小值: {hydrostatic_pressure.min():.3e}")
    print(f"  最大值: {hydrostatic_pressure.max():.3e}")
    print(f"  平均值: {hydrostatic_pressure.mean():.3e}")
    print(f"  标准差: {hydrostatic_pressure.std():.3e}")

    # 应力分量统计
    print(f"\n应力分量统计:")
    stress_components = ['σ_xx', 'σ_xy', 'σ_yx', 'σ_yy']
    for i in range(2):
        for j in range(2):
            component = stress_data[:, i, j]
            comp_name = stress_components[i*2 + j]
            print(f"  {comp_name}: {component.mean():.3e} ± {component.std():.3e}")

    # 绘制应力剖面
    plot_path = plot_stress_profiles(positions, von_mises, hydrostatic_pressure, frame_number, save_dir)

    return {
        'von_mises': von_mises,
        'hydrostatic_pressure': hydrostatic_pressure,
        'plot_path': plot_path
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test2实验: Schwarz双域应力分析')
    parser.add_argument('--skip-sim', action='store_true', help='跳过模拟，使用已有结果')
    parser.add_argument('--experiment-dir', type=str, help='指定实验目录（跳过模拟时使用）')
    parser.add_argument('--frame', type=int, help='指定分析帧号（默认：最新帧）')
    parser.add_argument('--config', type=str, default='config/schwarz_2d_test2.json', help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='experiments/test2_results', help='结果保存目录')

    args = parser.parse_args()

    try:
        # 步骤1: 运行模拟（除非跳过）
        if args.skip_sim:
            if args.experiment_dir:
                experiment_dir = args.experiment_dir
            else:
                # 查找最新的实验结果
                base_output_dir = "experiment_results"
                schwarz_pattern = os.path.join(base_output_dir, "schwarz_*")
                schwarz_dirs = glob.glob(schwarz_pattern)

                if not schwarz_dirs:
                    print("未找到已有的Schwarz实验结果")
                    return 1

                def extract_timestamp(dirname):
                    basename = os.path.basename(dirname)
                    match = re.search(r'schwarz_(\d{8}_\d{6})', basename)
                    if match:
                        timestamp_str = match.group(1)
                        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    return datetime.min

                experiment_dir = max(schwarz_dirs, key=extract_timestamp)

            print(f"使用已有实验结果: {experiment_dir}")
        else:
            experiment_dir = run_schwarz_simulation(args.config, args.output_dir)
            if experiment_dir is None:
                return 1

        # 步骤2: 加载Domain2数据
        print("\n加载Domain2数据...")
        stress_data, positions, frame_number = load_domain2_data(experiment_dir, args.frame)

        # 步骤3: 应力分析
        print("\n开始应力分析...")
        results = analyze_stress_distribution(positions, stress_data, frame_number, args.output_dir)

        print(f"\n实验完成! 结果保存在: {args.output_dir}")
        print(f"应力剖面图: {results['plot_path']}")

    except Exception as e:
        print(f"实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())