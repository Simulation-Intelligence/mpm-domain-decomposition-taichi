#!/usr/bin/env python3
"""
测试基于Poisson采样半径的Alpha值优化效果
"""
import taichi as ti
from Util.Config import Config
from Geometry.Particles import Particles

def compare_boundary_detection():
    """对比使用和不使用Poisson半径的边界检测效果"""
    ti.init(arch=ti.cpu)
    
    print("=== 边界检测方法对比测试 ===\n")
    
    # 加载配置
    config = Config('config/example_shapes.json')
    
    # 测试1：使用Poisson半径优化的方法
    print("1. 使用Poisson半径优化的Alpha Shape边界检测:")
    particles1 = Particles(config)
    
    boundary_count1 = sum(particles1.is_boundary_particle.to_numpy())
    poisson_radius = particles1.particle_generator.get_last_poisson_radius()
    suggested_alpha = particles1.boundary_detector.suggested_alpha
    
    print(f"   - 总粒子数: {particles1.n_particles}")
    print(f"   - Poisson采样半径: {poisson_radius:.6f}")
    print(f"   - 建议Alpha值: {suggested_alpha:.6f}")
    print(f"   - 边界粒子数: {boundary_count1}")
    print(f"   - 边界粒子比例: {boundary_count1/particles1.n_particles:.3f}")
    
    print("\n" + "="*50)
    
    # 测试2：强制使用传统距离方法
    print("\n2. 传统基于距离统计的Alpha Shape边界检测:")
    
    # 重新初始化Taichi以清除之前的状态
    ti.reset()
    ti.init(arch=ti.cpu)
    
    particles2 = Particles(config)
    
    # 手动调用传统方法（不传递Poisson半径）
    positions = particles2.x.to_numpy()
    traditional_boundary_flags = particles2.boundary_detector.detect_boundaries(
        positions, particles2.dim, poisson_radius=None  # 强制不使用Poisson半径
    )
    particles2.is_boundary_particle.from_numpy(traditional_boundary_flags)
    
    boundary_count2 = sum(particles2.is_boundary_particle.to_numpy())
    
    print(f"   - 总粒子数: {particles2.n_particles}")
    print(f"   - 使用方法: 距离中位数估算")
    print(f"   - 边界粒子数: {boundary_count2}")
    print(f"   - 边界粒子比例: {boundary_count2/particles2.n_particles:.3f}")
    
    print("\n" + "="*50)
    
    # 对比分析
    print("\n3. 对比分析:")
    
    efficiency_ratio = boundary_count1 / boundary_count2 if boundary_count2 > 0 else float('inf')
    
    print(f"   - Poisson优化方法边界粒子数: {boundary_count1}")
    print(f"   - 传统方法边界粒子数: {boundary_count2}")
    print(f"   - 效率比 (Poisson/传统): {efficiency_ratio:.3f}")
    
    if efficiency_ratio < 1.0:
        print("   + Poisson优化方法检测到更少但可能更精确的边界粒子")
    elif efficiency_ratio > 1.2:
        print("   ! Poisson优化方法可能过于敏感")
    else:
        print("   + 两种方法结果接近，Poisson方法更稳定")
    
    # Alpha值对比
    if poisson_radius and suggested_alpha:
        ratio_to_radius = suggested_alpha / poisson_radius
        print(f"   - Alpha/Poisson半径比值: {ratio_to_radius:.2f}")
        print(f"   - 这个比值应该在2-4之间比较合理")

if __name__ == "__main__":
    compare_boundary_detection()