import numpy as np 
import matplotlib.pyplot  as plt 
from mpl_toolkits.mplot3d  import Axes3D
import json
import os
import hashlib

# =============== 缓存管理 ===============
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.poisson_cache.json')

def _load_cache():
    """加载柏松采样半径缓存"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def _save_cache(cache):
    """保存柏松采样半径缓存"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError:
        print(f"警告: 无法保存缓存到 {CACHE_FILE}")

def _get_cache_key(region_size, n_target, dim):
    """
    生成统一缓存键（基于密度，4位有效数字）
    """
    volume = np.prod(region_size)
    particle_density = n_target / volume
    
    # 保留4位有效数字
    from math import log10, floor
    if particle_density > 0:
        significant_digits = 4
        power = floor(log10(abs(particle_density)))
        factor = 10 ** (significant_digits - 1 - power)
        rounded_density = round(particle_density * factor) / factor
    else:
        rounded_density = 0
    
    cache_key = f"density_{rounded_density:.3e}_dim_{dim}"
    return cache_key

def _get_ellipse_cache_key(semi_axes, n_target, dim):
    """
    生成统一缓存键（基于密度，4位有效数字）
    """
    if dim == 2:
        area = np.pi * semi_axes[0] * semi_axes[1]
    else:
        area = (4.0/3.0) * np.pi * semi_axes[0] * semi_axes[1] * semi_axes[2]
    
    particle_density = n_target / area
    
    # 保留4位有效数字
    from math import log10, floor
    if particle_density > 0:
        significant_digits = 4
        power = floor(log10(abs(particle_density)))
        factor = 10 ** (significant_digits - 1 - power)
        rounded_density = round(particle_density * factor) / factor
    else:
        rounded_density = 0
    
    cache_key = f"density_{rounded_density:.3e}_dim_{dim}"
    return cache_key

def _get_cached_ellipse_radius(semi_axes, n_target, dim):
    """从缓存中获取椭圆半径"""
    cache = _load_cache()
    cache_key = _get_ellipse_cache_key(semi_axes, n_target, dim)
    return cache.get(cache_key)

def _cache_ellipse_radius(semi_axes, n_target, dim, radius):
    """缓存椭圆半径结果"""
    cache = _load_cache()
    cache_key = _get_ellipse_cache_key(semi_axes, n_target, dim)
    cache[cache_key] = radius
    _save_cache(cache)
    print(f"已缓存椭圆半径 {radius:.6e} 到键 {cache_key}")

def _get_cached_radius(region_size, n_target, dim):
    """从缓存中获取半径"""
    cache = _load_cache()
    cache_key = _get_cache_key(region_size, n_target, dim)
    return cache.get(cache_key)

def _cache_radius(region_size, n_target, dim, radius):
    """缓存半径结果"""
    cache = _load_cache()
    cache_key = _get_cache_key(region_size, n_target, dim)
    cache[cache_key] = radius
    _save_cache(cache)
    print(f"已缓存半径 {radius:.6e} 到键 {cache_key}")

# =============== 主要采样函数 ===============
def poisson_disk_sampling_by_count(region_size, n_target, n_tolerance=0.0001, max_iter=100):
    """
    支持2D/3D的泊松盘采样 
    :param region_size: 区域尺寸 [width, height] 或 [width, height, depth]
    :param n_target: 目标点数 
    :param n_tolerance: 收敛阈值 
    :param max_iter: 最大迭代次数 
    :return: 采样点列表 
    """
    assert all(s > 0 for s in region_size), "区域尺寸必须为正数"
    dim = len(region_size)
    assert dim in [2, 3], "仅支持2D或3D"
 
    # 计算初始半径估计 
    volume = np.prod(region_size) 
    max_possible_r = (volume / n_target) ** (1/dim) * dim 
 
    r_min, r_max = 0.1 * max_possible_r, max_possible_r 
    best_points = []
 
    for _ in range(max_iter):
        r = (r_min + r_max) / 2 
        points = _poisson_disk_sampling_fixed_r(region_size, r)
        n_actual = len(points)

        print(f"半径: {r:.4e}, 采样点数: {n_actual}, 目标点数: {n_target}")
 
        if n_actual >= n_target:
            best_points = points 
            r_min = r 
        else:
            r_max = r 
 
        if ((n_actual - n_target)/n_target < n_tolerance or n_actual - n_target <=1 or r_max - r_min < 1e-6) and n_actual >= n_target:
            break 
    
    print(f"最终半径: {r:.4e}, 采样点数: {len(best_points)}")

    return best_points[:n_target]

def poisson_disk_sampling_by_count_cached(region_size, n_target, n_tolerance=0.0001, max_iter=100):
    """
    支持2D/3D的泊松盘采样，带缓存功能
    """
    assert all(s > 0 for s in region_size), "区域尺寸必须为正数"
    dim = len(region_size)
    assert dim in [2, 3], "仅支持2D或3D"

    # 尝试从缓存中获取半径
    cached_radius = _get_cached_radius(region_size, n_target, dim)
    if cached_radius is not None:
        print(f"使用缓存的半径: {cached_radius:.6e}")
        points = _poisson_disk_sampling_fixed_r(region_size, cached_radius)
        print(f"缓存成功，采样点数: {len(points)} (目标: {n_target})")
        return points

    # 如果缓存未命中或结果不满足要求，进行二分查找
    volume = np.prod(region_size) 
    max_possible_r = (volume / n_target) ** (1/dim) * dim 

    r_min, r_max = 0.1 * max_possible_r, max_possible_r 
    best_points = []
    final_radius = None

    for _ in range(max_iter):
        r = (r_min + r_max) / 2 
        points = _poisson_disk_sampling_fixed_r(region_size, r)
        n_actual = len(points)

        print(f"半径: {r:.4e}, 采样点数: {n_actual}, 目标点数: {n_target}")

        if n_actual >= n_target:
            best_points = points 
            final_radius = r
            r_min = r 
        else:
            r_max = r 

        if ((n_actual - n_target)/n_target < n_tolerance or n_actual - n_target <=1 or r_max - r_min < 1e-6) and n_actual >= n_target:
            break 
    
    print(f"最终半径: {final_radius:.4e}, 采样点数: {len(best_points)}")
    
    # 缓存结果
    if final_radius is not None:
        _cache_radius(region_size, n_target, dim, final_radius)

    return best_points[:n_target]
 
def _poisson_disk_sampling_fixed_r(region_size, r):
    """核心采样算法"""
    dim = len(region_size)
    grid_size = r / np.sqrt(dim)   # 调整网格尺寸计算公式 
    grid_dims = [int(np.ceil(s/grid_size))  for s in region_size]
    
    # 初始化网格数据结构 
    grid = np.empty(grid_dims,  dtype=object)
    points = []
    active = []
 
    # 生成初始点 
    initial_point = np.random.rand(dim)  * region_size 
    points.append(tuple(initial_point)) 
    active.append(initial_point) 
    grid_idx = tuple((initial_point // grid_size).astype(int))
    grid[grid_idx] = initial_point 
 
    while active:
        current_idx = np.random.randint(len(active)) 
        current_point = active[current_idx]
        found = False 
 
        for _ in range(30):
            # 生成候选点 
            direction = np.random.normal(size=dim) 
            direction /= np.linalg.norm(direction) 
            radius = np.random.uniform(r,  2*r)
            new_point = current_point + direction * radius 
 
            if not all(0 <= new_point[i] < region_size[i] for i in range(dim)):
                continue 
 
            # 计算网格索引范围 
            grid_coords = (new_point // grid_size).astype(int)
            search_range = [ 
                (max(0, grid_coords[i]-2), min(grid_dims[i], grid_coords[i]+3))
                for i in range(dim)
            ]
 
            # 检查邻近点 
            valid = True 
            indices = np.array(np.meshgrid(*[np.arange(s,  e) for s, e in search_range])).T.reshape(-1,  dim)
            for idx in indices:
                neighbor = grid[tuple(idx)]
                if neighbor is not None and np.linalg.norm(neighbor  - new_point) < r:
                    valid = False 
                    break 
 
            if valid:
                points.append(tuple(new_point)) 
                active.append(new_point) 
                grid[tuple(grid_coords)] = new_point 
                found = True 
                break 
 
        if not found:
            active.pop(current_idx) 
 
    return points

def poisson_disk_sampling_ellipse(center, semi_axes, n_target, n_tolerance=0.0001, max_iter=100):
    """
    椭圆区域的泊松盘采样（带缓存）
    :param center: 椭圆中心 [x, y] 或 [x, y, z]
    :param semi_axes: 椭圆半轴 [a, b] 或 [a, b, c]
    :param n_target: 目标点数
    :param n_tolerance: 收敛阈值
    :param max_iter: 最大迭代次数
    :return: 采样点列表
    """
    center = np.array(center)
    semi_axes = np.array(semi_axes)
    dim = len(center)
    assert dim in [2, 3], "仅支持2D或3D"
    assert len(semi_axes) == dim, "中心点和半轴维度必须一致"
    
    # 计算椭圆体积/面积
    if dim == 2:
        area = np.pi * semi_axes[0] * semi_axes[1]
    else:
        area = (4.0/3.0) * np.pi * semi_axes[0] * semi_axes[1] * semi_axes[2]
    
    # 尝试从缓存获取半径
    cached_radius = _get_cached_ellipse_radius(semi_axes, n_target, dim)
    if cached_radius is not None:
        print(f"椭圆采样使用缓存半径: {cached_radius:.4e}")
        points = _poisson_disk_sampling_ellipse_fixed_r(center, semi_axes, cached_radius)
        print(f"椭圆采样缓存结果点数: {len(points)}")
        return points[:n_target] if len(points) >= n_target else points
    
    # 缓存未命中，执行二分查找
    print(f"椭圆采样缓存未命中，执行二分查找...")
    max_possible_r = (area / n_target) ** (1/dim) * dim
    r_min, r_max = 0.1 * max_possible_r, max_possible_r
    best_points = []
    final_radius = None
    
    for _ in range(max_iter):
        r = (r_min + r_max) / 2
        points = _poisson_disk_sampling_ellipse_fixed_r(center, semi_axes, r)
        n_actual = len(points)
        
        print(f"椭圆采样 - 半径: {r:.4e}, 采样点数: {n_actual}, 目标点数: {n_target}")
        
        if n_actual >= n_target:
            best_points = points
            final_radius = r
            r_min = r
        else:
            r_max = r
            
        if ((n_actual - n_target)/n_target < n_tolerance or n_actual - n_target <= 1 or r_max - r_min < 1e-6) and n_actual >= n_target:
            break
    
    # 缓存结果
    if final_radius is not None:
        _cache_ellipse_radius(semi_axes, n_target, dim, final_radius)
    
    print(f"椭圆采样最终半径: {final_radius:.4e}, 采样点数: {len(best_points)}")
    return best_points[:n_target]

def _poisson_disk_sampling_ellipse_fixed_r(center, semi_axes, r):
    """椭圆区域的固定半径泊松采样"""
    dim = len(center)
    
    # 创建椭圆边界框
    bbox_min = center - semi_axes
    bbox_max = center + semi_axes
    bbox_size = bbox_max - bbox_min
    
    # 网格设置
    grid_size = r / np.sqrt(dim)
    grid_dims = [int(np.ceil(s/grid_size)) for s in bbox_size]
    
    # 初始化网格
    grid = np.empty(grid_dims, dtype=object)
    points = []
    active = []
    
    # 生成椭圆内的初始点
    initial_point = None
    max_attempts = 1000
    for _ in range(max_attempts):
        candidate = np.random.rand(dim) * bbox_size + bbox_min
        if _is_point_in_ellipse(candidate, center, semi_axes):
            initial_point = candidate
            break
    
    if initial_point is None:
        return []
    
    points.append(tuple(initial_point))
    active.append(initial_point)
    grid_idx = tuple(((initial_point - bbox_min) // grid_size).astype(int))
    # 确保grid_idx在有效范围内
    grid_idx = tuple(min(max(0, idx), grid_dims[i]-1) for i, idx in enumerate(grid_idx))
    grid[grid_idx] = initial_point
    
    while active:
        current_idx = np.random.randint(len(active))
        current_point = active[current_idx]
        found = False
        
        for _ in range(30):
            # 生成候选点
            direction = np.random.normal(size=dim)
            direction /= np.linalg.norm(direction)
            radius = np.random.uniform(r, 2*r)
            new_point = current_point + direction * radius
            
            # 检查是否在椭圆内
            if not _is_point_in_ellipse(new_point, center, semi_axes):
                continue
                
            # 计算网格坐标
            grid_coords = ((new_point - bbox_min) // grid_size).astype(int)
            
            # 确保grid_coords在有效范围内
            if not all(0 <= grid_coords[i] < grid_dims[i] for i in range(dim)):
                continue
            
            # 检查邻近点
            search_range = [
                (max(0, grid_coords[i]-2), min(grid_dims[i], grid_coords[i]+3))
                for i in range(dim)
            ]
            
            valid = True
            indices = np.array(np.meshgrid(*[np.arange(s, e) for s, e in search_range])).T.reshape(-1, dim)
            for idx in indices:
                neighbor = grid[tuple(idx)]
                if neighbor is not None and np.linalg.norm(neighbor - new_point) < r:
                    valid = False
                    break
            
            if valid:
                points.append(tuple(new_point))
                active.append(new_point)
                grid[tuple(grid_coords)] = new_point
                found = True
                break
        
        if not found:
            active.pop(current_idx)
    
    return points

def _is_point_in_ellipse(point, center, semi_axes):
    """检查点是否在椭圆内"""
    normalized_dist = np.sum(((point - center) / semi_axes) ** 2)
    return normalized_dist <= 1.0 
 
def main():
    # 2D示例 
    points_2d = poisson_disk_sampling_by_count([0.2, 0.4], 200)
    plot_2d(points_2d, [0.2, 0.4])
 
    # 3D示例 
    points_3d = poisson_disk_sampling_by_count([0.2, 0.4, 0.3], 5000)
    plot_3d(points_3d, [0.2, 0.4, 0.3])
 
def plot_2d(points, region_size):
    plt.figure(figsize=(8,  8))
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.scatter(x,  y, s=10, c='blue', alpha=0.6)
    plt.xlim(0,  region_size[0])
    plt.ylim(0,  region_size[1])
    plt.gca().set_aspect('equal') 
    plt.show() 
 
def plot_3d(points, region_size):
    fig = plt.figure(figsize=(10,  10))
    ax = fig.add_subplot(111,  projection='3d')
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    ax.scatter(x,  y, z, s=10, alpha=0.6)
    ax.set_xlim(0,  region_size[0])
    ax.set_ylim(0,  region_size[1])
    ax.set_zlim(0,  region_size[2])
    plt.show() 
 
if __name__ == "__main__":
    main()