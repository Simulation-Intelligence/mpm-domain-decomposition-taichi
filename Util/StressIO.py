"""
应力数据 I/O 和几何判断工具。
"""

import json
import os
import numpy as np


def is_point_in_single_region(point, region_config, dim):
    """检查点是否在单个区域内。

    Args:
        point: (dim,) array，粒子位置
        region_config: dict，区域配置（type, params）
        dim: int，空间维度（2 或 3）

    Returns:
        bool，点是否在区域内
    """
    region_type = region_config.get("type", "rectangle")
    params = region_config.get("params", {})

    if region_type == "rectangle":
        rect_range = params.get("range", [])
        if len(rect_range) != dim:
            return False

        for d in range(dim):
            if point[d] < rect_range[d][0] or point[d] > rect_range[d][1]:
                return False
        return True

    elif region_type == "ellipse":
        center = params.get("center", [0.5] * dim)
        semi_axes = params.get("semi_axes", [0.1] * dim)

        if len(center) != dim or len(semi_axes) != dim:
            return False

        sum_normalized = 0.0
        for d in range(dim):
            diff = point[d] - center[d]
            sum_normalized += (diff / semi_axes[d]) ** 2
        return sum_normalized <= 1.0

    return False


def is_point_in_regions(point, regions_config, dim):
    """检查点是否在指定区域内（支持多个区域）。

    Args:
        point: (dim,) array，粒子位置
        regions_config: dict 或 list，区域配置
        dim: int，空间维度

    Returns:
        bool，点是否在任何一个区域内
    """
    # 如果是单个区域配置（字典），转换为列表
    if isinstance(regions_config, dict):
        regions_config = [regions_config]

    # 检查点是否在任意一个区域内
    for region in regions_config:
        if is_point_in_single_region(point, region, dim):
            return True

    return False


def compute_von_mises_stress(stress_data, dim):
    """计算 von Mises 应力。

    Args:
        stress_data: (N, dim, dim) array，应力张量数据
        dim: int，空间维度

    Returns:
        (N,) array，von Mises 应力值
    """
    von_mises = []
    for i in range(stress_data.shape[0]):
        s = stress_data[i]
        if dim == 2:
            # 2D von Mises应力
            vm = np.sqrt(s[0, 0] ** 2 + s[1, 1] ** 2 - s[0, 0] * s[1, 1] + 3 * s[0, 1] ** 2)
        else:
            # 3D von Mises应力
            vm = np.sqrt(
                0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
                + 3 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[2, 0] ** 2)
            )
        von_mises.append(vm)
    return np.array(von_mises)


def save_stress_frame(
    frame_dir,
    stress_data,
    positions,
    boundary_flags,
    grid_stress,
    grid_mass,
    grid_meta,
    von_mises_stress,
    actual_masses=None,
):
    """保存单帧的应力数据到 numpy 和 json 文件。

    Args:
        frame_dir: str，帧目录路径
        stress_data: (N, dim, dim) array，粒子应力数据
        positions: (N, dim) array，粒子位置
        boundary_flags: (N,) array，边界标记
        grid_stress: (ngrid, dim, dim) array，网格应力
        grid_mass: (ngrid,) array，网格质量
        grid_meta: dict，网格元数据
        von_mises_stress: (N,) array，von Mises 应力
        actual_masses: optional，实际质量
    """
    os.makedirs(frame_dir, exist_ok=True)

    # 保存粒子数据
    np.save(os.path.join(frame_dir, "stress.npy"), stress_data)
    np.save(os.path.join(frame_dir, "positions.npy"), positions)
    np.save(os.path.join(frame_dir, "boundary_flags.npy"), boundary_flags)

    # 保存网格应力
    np.save(os.path.join(frame_dir, "grid_stress.npy"), grid_stress)
    np.save(os.path.join(frame_dir, "grid_mass.npy"), grid_mass)
    with open(os.path.join(frame_dir, "grid_stress_meta.json"), "w") as f:
        json.dump(grid_meta, f, indent=2)

    # 保存实际质量（如果有）
    if actual_masses is not None:
        np.save(os.path.join(frame_dir, "actual_masses.npy"), np.array(actual_masses))

    # 保存 von Mises 应力到单独的文件
    np.save(os.path.join(frame_dir, "von_mises.npy"), von_mises_stress)


def write_stress_stats_json(
    frame_dir,
    frame_number,
    n_particles,
    n_particles_total,
    n_particles_filtered,
    dim,
    von_mises_stress,
    stress_output_regions=None,
):
    """写入应力统计 JSON 文件。

    Args:
        frame_dir: str，帧目录路径
        frame_number: int，帧号
        n_particles: int，过滤后粒子数
        n_particles_total: int，总粒子数
        n_particles_filtered: int，被过滤掉的粒子数
        dim: int，空间维度
        von_mises_stress: (N,) array，von Mises 应力值
        stress_output_regions: optional，应力输出区域配置
    """
    stats = {
        "frame": frame_number,
        "n_particles": int(n_particles),
        "n_particles_total": int(n_particles_total),
        "n_particles_filtered": int(n_particles_filtered),
        "dimension": int(dim),
        "stress_output_regions": stress_output_regions,
        "von_mises_stress": {
            "min": float(np.min(von_mises_stress)),
            "max": float(np.max(von_mises_stress)),
            "mean": float(np.mean(von_mises_stress)),
            "std": float(np.std(von_mises_stress)),
        },
    }

    with open(os.path.join(frame_dir, "stress_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats
