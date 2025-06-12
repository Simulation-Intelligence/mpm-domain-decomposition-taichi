import numpy as np 
import matplotlib.pyplot  as plt 
from mpl_toolkits.mplot3d  import Axes3D 
 
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