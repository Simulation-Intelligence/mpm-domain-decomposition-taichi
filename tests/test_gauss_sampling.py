#!/usr/bin/env python3
"""
测试高斯积分点采样功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from Geometry.GaussQuadrature import GaussQuadrature
from Geometry.ParticleGenerator import ParticleGenerator
from Util.Config import Config

def test_gauss_quadrature_data():
    """测试高斯积分点数据"""
    print("=== 测试高斯积分点数据 ===")
    
    for n in range(1, 6):
        points, weights = GaussQuadrature.get_1d_points_and_weights(n)
        print(f"n={n}: points={points}, weights={weights}, sum_weights={np.sum(weights):.6f}")
        
        # 测试2D网格点
        positions_2d, weights_2d = GaussQuadrature.get_2d_grid_points_and_weights(n, 1.0)
        print(f"  2D: {len(positions_2d)} points, total_weight={np.sum(weights_2d):.6f}")

def test_particle_generation():
    """测试粒子生成"""
    print("\n=== 测试粒子生成 ===")
    
    # 测试不同的particles_per_grid设置
    test_cases = [1, 4, 9, 16, 25]
    
    for ppg in test_cases:
        print(f"\nparticles_per_grid = {ppg}")
        
        # 创建粒子生成器
        generator = ParticleGenerator(dim=2, sampling_method="gauss", particles_per_grid=ppg, grid_size=16)
        
        # 定义矩形区域
        rect_range = [[0.3, 0.7], [0.3, 0.7]]
        
        try:
            # 生成粒子
            particles = generator._generate_gauss_quadrature_particles(rect_range, 1000)
            print(f"  成功生成 {len(particles)} 个粒子")
            
            # 检查粒子位置
            if len(particles) > 0:
                x_coords = [p[0] for p in particles]
                y_coords = [p[1] for p in particles]
                print(f"  X范围: [{min(x_coords):.4f}, {max(x_coords):.4f}]")
                print(f"  Y范围: [{min(y_coords):.4f}, {max(y_coords):.4f}]")
        except Exception as e:
            print(f"  错误: {e}")

def test_with_config():
    """使用配置文件测试"""
    print("\n=== 测试配置文件集成 ===")
    
    # 加载配置
    config_path = "../config/config_gauss_test.json"
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在")
        return
    
    try:
        config = Config(config_path)
        print(f"成功加载配置: {config_path}")
        print(f"sampling_method: {config.get('sampling_method')}")
        print(f"particles_per_grid: {config.get('particles_per_grid')}")
        
        # 验证particles_per_grid是完全平方数
        ppg = config.get('particles_per_grid')
        n_1d = GaussQuadrature.validate_particles_per_grid(ppg)
        print(f"验证通过: {ppg} = {n_1d}²")
        
    except Exception as e:
        print(f"配置文件测试失败: {e}")

def visualize_gauss_particles():
    """可视化高斯积分点分布"""
    print("\n=== 可视化高斯积分点分布 ===")
    
    # 测试不同的particles_per_grid值
    ppg_values = [4, 9, 16]  # 2x2 和 3x3 和 4x4
    grid_size = 8
    rect_range = [[0.1, 0.9], [0.1, 0.9]]
    
    fig, axes = plt.subplots(2, len(ppg_values), figsize=(12, 10))
    
    for j, ppg in enumerate(ppg_values):
        methods = ["regular", "gauss"]
        
        for i, method in enumerate(methods):
            ax = axes[i, j]
            generator = ParticleGenerator(dim=2, sampling_method=method, particles_per_grid=ppg, grid_size=grid_size)

            area_size = (rect_range[0][1] - rect_range[0][0]) * (rect_range[1][1] - rect_range[1][0]) * grid_size * grid_size
            n_particles = int(area_size * ppg)

            if method == "gauss":
                particles = generator._generate_gauss_quadrature_particles(rect_range, n_particles)
            else:
                particles = generator._generate_regular_grid_particles(rect_range, n_particles)
            
            x_coords = [p[0] for p in particles]
            y_coords = [p[1] for p in particles]
            
            ax.scatter(x_coords, y_coords, s=30, alpha=0.7, c='red', label='Particles')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f'{method.capitalize()} Sampling\nPPG={ppg}, {len(particles)} particles')
            
            # 绘制网格线（只绘制在感兴趣区域内的网格线）
            dx = 1.0 / grid_size
            x_min, x_max = rect_range[0]
            y_min, y_max = rect_range[1]
            
            # 找到区域边界对应的网格索引范围
            grid_x_min = max(0, int(x_min / dx))
            grid_x_max = min(grid_size, int(x_max / dx) + 1)
            grid_y_min = max(0, int(y_min / dx))
            grid_y_max = min(grid_size, int(y_max / dx) + 1)
            
            # 只绘制相关区域的网格线
            for k in range(grid_x_min, grid_x_max + 1):
                ax.axvline(k * dx, color='lightgray', alpha=0.6, linewidth=0.8)
            for k in range(grid_y_min, grid_y_max + 1):
                ax.axhline(k * dx, color='lightgray', alpha=0.6, linewidth=0.8)
            
            # 绘制网格中心点（只绘制相关区域内的）
            for gi in range(grid_x_min, grid_x_max):
                for gj in range(grid_y_min, grid_y_max):
                    center_x = gi * dx
                    center_y = gj * dx
                    # 只绘制在形状区域附近的网格中心点
                    if (center_x >= x_min - dx and center_x <= x_max + dx and 
                        center_y >= y_min - dx and center_y <= y_max + dx):
                        ax.plot(center_x, center_y, 'bo', markersize=2, alpha=0.4)
            
            # 绘制区域边界
            rect_x = [rect_range[0][0], rect_range[0][1], rect_range[0][1], rect_range[0][0], rect_range[0][0]]
            rect_y = [rect_range[1][0], rect_range[1][0], rect_range[1][1], rect_range[1][1], rect_range[1][0]]
            ax.plot(rect_x, rect_y, 'b-', linewidth=2, alpha=0.8, label='Shape Boundary')
            
            ax.legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig('./tests/gauss_sampling_comparison.png', dpi=150, bbox_inches='tight')
    print("可视化结果保存为: tests/gauss_sampling_comparison.png")
    plt.close()

if __name__ == "__main__":
    test_gauss_quadrature_data()
    test_particle_generation()
    test_with_config()
    visualize_gauss_particles()
    print("\n=== 测试完成 ===")