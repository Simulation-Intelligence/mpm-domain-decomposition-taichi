#!/usr/bin/env python3
"""
测试柏松采样缓存功能
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from Util.poisson_disk_sampling import poisson_disk_sampling_by_count_cached, _load_cache

def test_poisson_cache():
    """测试柏松采样缓存功能"""
    print("=" * 50)
    print("测试柏松采样缓存功能")
    print("=" * 50)
    
    # 测试参数
    region_size = [0.8, 0.8]  # 2D区域
    n_target = 2048  # 目标粒子数
    
    print(f"测试区域: {region_size}")
    print(f"目标粒子数: {n_target}")
    print(f"粒子密度: {n_target / (region_size[0] * region_size[1]):.2f}")
    
    # 第一次运行（应该计算并缓存）
    print("\n第一次运行（计算并缓存）:")
    start_time = time.time()
    points1 = poisson_disk_sampling_by_count_cached(region_size, n_target)
    first_run_time = time.time() - start_time
    print(f"第一次运行耗时: {first_run_time:.2f}秒")
    print(f"得到点数: {len(points1)}")
    
    # 查看缓存内容
    cache = _load_cache()
    print(f"\n缓存条目数: {len(cache)}")
    
    # 第二次运行（应该使用缓存）
    print("\n第二次运行（使用缓存）:")
    start_time = time.time()
    points2 = poisson_disk_sampling_by_count_cached(region_size, n_target)
    second_run_time = time.time() - start_time
    print(f"第二次运行耗时: {second_run_time:.2f}秒")
    print(f"得到点数: {len(points2)}")
    
    # 计算加速比
    if second_run_time > 0:
        speedup = first_run_time / second_run_time
        print(f"\n加速比: {speedup:.1f}x")
    
    # 验证结果一致性
    if len(points1) == len(points2):
        print("✓ 两次运行得到相同数量的点")
    else:
        print("✗ 两次运行点数不同")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_poisson_cache()