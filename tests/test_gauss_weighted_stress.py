#!/usr/bin/env python3
"""
测试高斯积分点权重加权的应力应变保存功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
from Util.Config import Config

# 添加simulators目录到路径
simulators_path = os.path.join(os.path.dirname(__file__), '..', 'simulators')
sys.path.append(simulators_path)

from implicit_mpm import ImplicitMPM

def test_gauss_weighted_stress_saving():
    """测试高斯积分点权重加权的应力应变保存"""
    print("=== 测试高斯积分点权重加权应力应变保存 ===")
    
    # 初始化Taichi
    ti.init(arch=ti.cpu)
    
    # 创建测试配置：高斯采样
    config_gauss = {
        "arch": "cpu",
        "float_type": "f64",
        "dim": 2,
        "grid_size": 32,
        "particles_per_grid": 4,  # 2x2
        "sampling_method": "gauss",
        "dt": 5e-5,
        "max_iter": 2,
        "shapes": [
            {
                "type": "rectangle",
                "params": {
                    "range": [[0.3, 0.7], [0.3, 0.7]]
                },
                "operation": "add",
                "material_id": 0
            }
        ],
        "material_params": [
            {
                "id": 0,
                "name": "test_material",
                "E": 1e5,
                "nu": 0.3,
                "rho": 1000
            }
        ]
    }
    
    # 测试高斯采样的应力应变保存
    print("\n--- 测试高斯采样应力应变保存 ---")
    try:
        config_obj = Config(data=config_gauss)
        mpm = ImplicitMPM(config_obj)
        
        print(f"采样方式: {mpm.particles.sampling_method}")
        print(f"粒子数量: {mpm.particles.n_particles}")
        print(f"网格大小: {mpm.particles.grid_size}")
        print(f"每网格粒子数: {mpm.particles.particles_per_grid}")
        
        # 运行几步模拟
        for step in range(2):
            print(f"运行步骤 {step+1}...")
            mpm.step()
        
        # 测试应力应变数据加权保存
        print("测试应力应变数据处理...")
        stress_data, strain_data, positions = mpm._weight_gauss_data_to_grid()
        
        print(f"加权后数据点数量: {len(positions)}")
        print(f"位置范围: X[{positions[:,0].min():.3f}, {positions[:,0].max():.3f}], Y[{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]")
        
        # 检查数据的形状和类型
        print(f"应力数据形状: {stress_data.shape}")
        print(f"应变数据形状: {strain_data.shape}")
        print(f"位置数据形状: {positions.shape}")
        
        # 可视化结果对比
        visualize_stress_data_comparison(mpm, stress_data, positions)
        
        print("✓ 高斯积分点权重加权测试通过")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def visualize_stress_data_comparison(mpm, weighted_stress, weighted_positions):
    """可视化原始粒子数据和加权后的网格数据对比"""
    print("\n=== 可视化应力数据对比 ===")
    
    # 获取原始粒子数据
    particle_stress = mmp.particles.stress.to_numpy()
    particle_positions = mmp.particles.x.to_numpy()
    
    # 计算von Mises应力
    def compute_von_mises_2d(stress_tensor):
        s = stress_tensor
        return np.sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2)
    
    particle_von_mises = [compute_von_mises_2d(s) for s in particle_stress]
    weighted_von_mises = [compute_von_mises_2d(s) for s in weighted_stress]
    
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始粒子数据
    sc1 = axes[0].scatter(particle_positions[:,0], particle_positions[:,1], 
                         c=particle_von_mises, s=20, alpha=0.7, cmap='viridis')
    axes[0].set_title(f'Original Particles\n{len(particle_positions)} points')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[0], label='von Mises Stress')
    
    # 加权后的网格数据
    sc2 = axes[1].scatter(weighted_positions[:,0], weighted_positions[:,1], 
                         c=weighted_von_mises, s=50, alpha=0.8, cmap='viridis')
    axes[1].set_title(f'Weighted Grid Points\n{len(weighted_positions)} points')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=axes[1], label='von Mises Stress')
    
    # 添加网格线
    grid_size = mmp.particles.grid_size
    dx = 1.0 / grid_size
    for i in range(grid_size + 1):
        for ax in axes:
            ax.axvline(i * dx, color='lightgray', alpha=0.4, linewidth=0.5)
            ax.axhline(i * dx, color='lightgray', alpha=0.4, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('./tests/stress_data_comparison.png', dpi=150, bbox_inches='tight')
    print("对比图保存为: tests/stress_data_comparison.png")
    plt.close()

def test_regular_vs_gauss_comparison():
    """对比regular和gauss采样的应力应变保存结果"""
    print("\n=== 对比Regular和Gauss采样结果 ===")
    
    methods = ["regular", "gauss"]
    results = {}
    
    for method in methods:
        print(f"\n--- 测试 {method.upper()} 采样 ---")
        
        # 重置Taichi
        ti.reset()
        ti.init(arch=ti.cpu)
        
        config = {
            "arch": "cpu",
            "float_type": "f64", 
            "dim": 2,
            "grid_size": 16,
            "particles_per_grid": 9 if method == "gauss" else 9,
            "sampling_method": method,
            "dt": 5e-5,
            "max_iter": 1,
            "shapes": [
                {
                    "type": "rectangle", 
                    "params": {
                        "range": [[0.2, 0.8], [0.2, 0.8]]
                    },
                    "operation": "add",
                    "material_id": 0
                }
            ],
            "material_params": [
                {
                    "id": 0,
                    "name": "test_material",
                    "E": 1e5,
                    "nu": 0.3,
                    "rho": 1000
                }
            ]
        }
        
        try:
            config_obj = Config(data=config)
            mpm = ImplicitMPM(config_obj)
            
            # 运行一步
            mpm.step()
            
            # 获取应力应变数据
            if method == "gauss":
                stress, strain, positions = mpm._weight_gauss_data_to_grid()
            else:
                stress = mpm.particles.stress.to_numpy()
                strain = mpm.particles.strain.to_numpy()
                positions = mpm.particles.x.to_numpy()
            
            results[method] = {
                'stress': stress,
                'positions': positions,
                'count': len(positions)
            }
            
            print(f"{method} 采样结果: {len(positions)} 个数据点")
            
        except Exception as e:
            print(f"{method} 采样测试失败: {e}")
    
    # 输出对比结果
    if len(results) == 2:
        print(f"\n对比结果:")
        print(f"Regular采样: {results['regular']['count']} 个数据点")
        print(f"Gauss采样: {results['gauss']['count']} 个数据点")

if __name__ == "__main__":
    test_gauss_weighted_stress_saving()
    test_regular_vs_gauss_comparison()
    print("\n=== 测试完成 ===")