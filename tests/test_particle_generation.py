#!/usr/bin/env python3
"""
测试粒子生成器的三种采样方式
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Geometry.ParticleGenerator import ParticleGenerator
import matplotlib.pyplot as plt
import numpy as np

def test_particle_generation():
    """测试三种采样方式"""
    
    # 测试参数
    dim = 2
    rect_range = [[0.2, 0.8], [0.2, 0.8]]  # 矩形区域
    n_particles = 100
    
    # 定义形状
    shape = {
        "type": "rectangle",
        "params": {"range": rect_range},
        "operation": "add"
    }
    
    # 创建三个生成器测试不同采样方式
    sampling_methods = ["uniform", "poisson", "regular"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, method in enumerate(sampling_methods):
        print(f"\n测试 {method} 采样方式:")
        
        # 创建粒子生成器
        generator = ParticleGenerator(dim=dim, sampling_method=method)
        
        # 生成粒子
        particles = generator.generate_particles_for_shape(shape, n_particles)
        
        print(f"目标粒子数: {n_particles}")
        print(f"实际生成粒子数: {len(particles)}")
        
        # 提取位置数据用于可视化
        if particles:
            x_coords = [p[0] for p in particles]
            y_coords = [p[1] for p in particles]
            
            # 绘制粒子分布
            axes[i].scatter(x_coords, y_coords, s=20, alpha=0.7)
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].set_aspect('equal')
            axes[i].set_title(f'{method.capitalize()} Sampling\n({len(particles)} particles)')
            axes[i].grid(True, alpha=0.3)
            
            # 添加矩形边界
            from matplotlib.patches import Rectangle
            rect = Rectangle((rect_range[0][0], rect_range[1][0]), 
                           rect_range[0][1] - rect_range[0][0], 
                           rect_range[1][1] - rect_range[1][0], 
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('particle_generation_test.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到 particle_generation_test.png")
    
    return True

def test_regular_grid_calculation():
    """测试规则分布的计算逻辑"""
    print("\n=== 规则分布计算测试 ===")
    
    # 测试不同的particles_per_grid设置
    grid_size = 16
    particles_per_grid_values = [3, 9, 16]
    
    for particles_per_grid in particles_per_grid_values:
        print(f"\nGrid size: {grid_size}, Particles per grid: {particles_per_grid}")
        
        # 模拟一个单位正方形区域
        rect_range = [[0.0, 1.0], [0.0, 1.0]]
        area = 1.0  # 单位正方形面积
        
        # 计算粒子数量（按照Particles.py中的逻辑）
        n_particles = int(grid_size**2 * area * particles_per_grid)
        
        print(f"目标粒子数: {n_particles}")
        
        # 创建规则分布生成器并生成粒子
        generator = ParticleGenerator(dim=2, sampling_method="regular")
        
        shape = {
            "type": "rectangle", 
            "params": {"range": rect_range}
        }
        
        particles = generator.generate_particles_for_shape(shape, n_particles)
        
        print(f"实际生成粒子数: {len(particles)}")
        
        # 计算理论格点间距
        particle_volume = area / n_particles
        expected_spacing = particle_volume ** 0.5  # 2D情况
        print(f"理论格点间距: {expected_spacing:.4f}")
        
        # 验证生成的粒子是否规则分布
        if len(particles) >= 4:  # 至少需要几个粒子来验证
            x_coords = sorted([p[0] for p in particles])
            y_coords = sorted([p[1] for p in particles])
            
            # 检查x方向的间距
            x_spacings = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
            x_spacings = [s for s in x_spacings if s > 1e-6]  # 过滤掉几乎为0的间距
            
            if x_spacings:
                avg_x_spacing = np.mean(x_spacings)
                print(f"实际x方向平均间距: {avg_x_spacing:.4f}")

if __name__ == "__main__":
    print("开始测试粒子生成器...")
    
    # 测试粒子生成
    try:
        test_particle_generation()
        print("✓ 粒子生成测试完成")
    except Exception as e:
        print(f"✗ 粒子生成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试规则分布计算
    try:
        test_regular_grid_calculation()
        print("✓ 规则分布计算测试完成")
    except Exception as e:
        print(f"✗ 规则分布计算测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试完成！")