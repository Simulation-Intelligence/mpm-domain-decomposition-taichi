#!/usr/bin/env python3
"""
实验1 (v3): Eshelby 夹杂/过盈配合测试
严谨版：包含从极坐标到笛卡尔坐标的应力张量变换。
可以直接对比 Sigma_xx 和 Sigma_yy，无需假设取样位置。
"""

import json
import numpy as np
import sys
import os
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid segfault with Taichi
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.plot_style import apply_cmame_style, COLORS as CMAME_COLORS
apply_cmame_style()
from simulators.implicit_mpm import ImplicitMPM
from Util.Config import Config
import taichi as ti

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path):
    """保存配置文件"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def modify_config_grid_size(config, grid_size, use_schwarz=False, modify_domain1=False):
    """
    修改配置文件的网格分辨率

    参数:
        config: 配置字典
        grid_size: 新的网格大小（Schwarz模式下对应 Domain2 的 grid_nx）
        use_schwarz: 是否为Schwarz双域配置
        modify_domain1: Schwarz模式下是否同步缩放 Domain1 的分辨率（默认不修改）

    返回:
        修改后的配置字典
    """
    import copy
    new_config = copy.deepcopy(config)

    def _scale_dt(cfg, old_grid_nx, new_grid_nx, label):
        """按网格分辨率反比例缩放 dt。"""
        if old_grid_nx <= 0 or new_grid_nx <= 0:
            return
        old_dt = cfg.get('dt', None)
        if old_dt is None:
            return
        new_dt = old_dt * (old_grid_nx / new_grid_nx)
        cfg['dt'] = new_dt
        print(f"  {label} dt 调整: {old_dt:g} -> {new_dt:g} (按 {old_grid_nx}/{new_grid_nx})")

    if use_schwarz:
        # Schwarz模式：修改 Domain2；可选同步缩放 Domain1
        if 'Domain2' in new_config:
            orig_d2_nx = new_config['Domain2'].get('grid_nx', 30)
            orig_d2_ny = new_config['Domain2'].get('grid_ny', 30)
            d2_aspect = orig_d2_ny / orig_d2_nx if orig_d2_nx > 0 else 1.0

            new_config['Domain2']['grid_nx'] = grid_size
            new_config['Domain2']['grid_ny'] = int(grid_size * d2_aspect)
            _scale_dt(new_config['Domain2'], orig_d2_nx, grid_size, 'Domain2')

            print(f"修改Schwarz配置: Domain2网格大小设为 {grid_size}x{int(grid_size * d2_aspect)} (比例: {d2_aspect:.3f})")

            if modify_domain1 and 'Domain1' in new_config:
                orig_d1_nx = new_config['Domain1'].get('grid_nx', 30)
                orig_d1_ny = new_config['Domain1'].get('grid_ny', 30)
                d1_aspect = orig_d1_ny / orig_d1_nx if orig_d1_nx > 0 else 1.0
                # 按 Domain2 的缩放比例等比例调整 Domain1
                scale = grid_size / orig_d2_nx if orig_d2_nx > 0 else 1.0
                new_d1_nx = max(1, round(orig_d1_nx * scale))
                new_d1_ny = max(1, round(orig_d1_ny * scale))
                new_config['Domain1']['grid_nx'] = new_d1_nx
                new_config['Domain1']['grid_ny'] = new_d1_ny
                d2_new_dt = new_config['Domain2'].get('dt', None)
                d2_old_dt = config.get('Domain2', {}).get('dt', None)
                d1_old_dt = new_config['Domain1'].get('dt', None)
                if d2_new_dt is not None and d2_old_dt and d1_old_dt is not None:
                    d1_dt_ratio = d1_old_dt / d2_old_dt if d2_old_dt > 0 else 1.0
                    new_config['Domain1']['dt'] = d2_new_dt * d1_dt_ratio
                    print(f"  Domain1 dt 调整: {d1_old_dt:g} -> {new_config['Domain1']['dt']:g} (保持与Domain2的{d1_dt_ratio:g}倍比例)")
                else:
                    _scale_dt(new_config['Domain1'], orig_d1_nx, new_d1_nx, 'Domain1')
                print(f"修改Schwarz配置: Domain1网格大小设为 {new_d1_nx}x{new_d1_ny} (缩放比例: {scale:.3f})")
            elif modify_domain1:
                print("警告: Schwarz配置中未找到Domain1，跳过Domain1分辨率修改")
        else:
            print("警告: Schwarz配置中未找到Domain2")
    else:
        # 原有单域逻辑
        # 获取原始网格大小和宽高比
        original_nx = new_config.get('grid_nx', 100)
        original_ny = new_config.get('grid_ny', 100)
        aspect_ratio = original_ny / original_nx if original_nx > 0 else 1.0

        # 设置新的网格大小
        new_config['grid_nx'] = grid_size
        new_config['grid_ny'] = int(grid_size * aspect_ratio)
        _scale_dt(new_config, original_nx, grid_size, 'Single')

        print(f"修改配置: 网格大小设为 {grid_size}x{int(grid_size * aspect_ratio)} (比例: {aspect_ratio:.3f})")

    return new_config

def get_analytical_stress_cartesian(x_rel, y_rel, R, delta_eff,
                                    E_in, nu_in, E_out, nu_out):
    """
    计算解析解 (支持异质材料, Plane Strain)
    
    参数:
        x_rel, y_rel: 相对于中心的坐标
        R: 夹杂半径
        delta_eff: 过盈量
        E_in, nu_in: 夹杂材料的杨氏模量和泊松比
        E_out, nu_out: 基体材料的杨氏模量和泊松比
    
    返回:
        sigma_xx, sigma_yy: 笛卡尔坐标系下的应力分量
    """
    # 计算剪切模量
    G_in = E_in / (2 * (1 + nu_in))
    G_out = E_out / (2 * (1 + nu_out))

    # 计算接触压力 P (Plane Strain 公式)
    compliance_out = 1.0 / (2 * G_out)
    compliance_in = (1.0 - 2.0 * nu_in) / (2 * G_in)
    P = delta_eff / (compliance_out + compliance_in)

    # 极坐标计算
    r = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(y_rel, x_rel)
    r_safe = np.maximum(r, 1e-10)

    # 初始化应力分量
    sigma_rr = np.zeros_like(r)
    sigma_tt = np.zeros_like(r)

    # 内部 (r < R): 均匀受压状态
    mask_in = r < R
    sigma_rr[mask_in] = -P
    sigma_tt[mask_in] = -P

    # 外部 (r >= R): Lame 解的衰减规律
    mask_out = r >= R
    r_out = r_safe[mask_out]
    sigma_rr[mask_out] = -P * (R / r_out)**2
    sigma_tt[mask_out] = P * (R / r_out)**2

    # 坐标变换到笛卡尔坐标系
    c2 = np.cos(theta)**2
    s2 = np.sin(theta)**2
    sigma_xx = sigma_rr * c2 + sigma_tt * s2
    sigma_yy = sigma_rr * s2 + sigma_tt * c2

    return sigma_xx, sigma_yy

def run_simulation(config_path, use_schwarz=False):
    """
    通过subprocess运行模拟，避免Taichi和matplotlib冲突
    """
    import subprocess
    import glob

    print(f"运行模拟: {config_path}")

    if use_schwarz:
        # 调用 Schwarz 模拟器
        cmd = [
            sys.executable,  # 使用当前Python解释器
            "simulators/implicit_mpm_schwarz.py",
            "--config", config_path
        ]
        print(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        # 退出码-11是segfault，通常发生在模拟完成、数据保存后的绘图阶段
        # 只要数据保存成功就继续
        if result.returncode == -11:
            print("警告: 进程以segfault结束（退出码-11），这通常发生在性能统计绘图时")
            print("检查数据文件是否已保存...")
        elif result.returncode != 0:
            print("模拟失败！")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Schwarz simulation failed with code {result.returncode}")

        print("加载结果数据...")

        # 查找最新的实验结果目录
        experiment_dirs = glob.glob("experiment_results/schwarz_*")
        if not experiment_dirs:
            print("错误: 未找到模拟结果目录")
            if result.returncode == -11:
                print("segfault发生在数据保存之前")
            raise RuntimeError("未找到模拟结果目录")

        latest_dir = max(experiment_dirs, key=os.path.getmtime)
        print(f"从目录加载数据: {latest_dir}")

        # 查找最新的帧目录
        frame_dirs = glob.glob(f"{latest_dir}/stress_data/frame_*")
        if not frame_dirs:
            raise RuntimeError(f"未找到应力数据: {latest_dir}/stress_data/")

        latest_frame_dir = max(frame_dirs, key=lambda x: int(x.split('_')[-1]))
        print(f"加载帧数据: {latest_frame_dir}")

        # 加载Domain1和Domain2数据
        positions1 = np.load(f"{latest_frame_dir}/domain1_positions.npy")
        stresses1 = np.load(f"{latest_frame_dir}/domain1_stress.npy")
        positions2 = np.load(f"{latest_frame_dir}/domain2_positions.npy")
        stresses2 = np.load(f"{latest_frame_dir}/domain2_stress.npy")

        print(f"Domain1: {len(positions1)} particles")
        print(f"Domain2: {len(positions2)} particles")

        # Domain2位置已经是全局坐标（implicit_mpm_schwarz.py中已处理）
        # 但为了确保，我们从配置中读取offset并验证
        cfg = load_config(config_path)
        domain2_offset = np.array(cfg['Domain2']['offset'])

        # positions2 已经在保存时加上了offset，所以这里不需要再加
        # 验证一下positions2的范围是否合理
        print(f"Domain2位置范围: x=[{positions2[:, 0].min():.3f}, {positions2[:, 0].max():.3f}], y=[{positions2[:, 1].min():.3f}, {positions2[:, 1].max():.3f}]")

        timing_dict = None
        try:
            stats_path = os.path.join(latest_dir, 'performance_stats', 'stats_data.json')
            with open(stats_path) as _f:
                _stats = json.load(_f)
            _fd = _stats.get('frame_data', [])
            _big = sum(sum(f['big_domain_solve_time']) for f in _fd)
            _small = sum(sum(f['small_domain_solve_time']) for f in _fd)
            _total = sum(f['total_frame_time'] for f in _fd)
            timing_dict = {
                'mode': 'schwarz',
                'big_domain_solve_time': _big,
                'small_domain_solve_time': _small,
                'other_time': _total - _big - _small,
                'total_time': _total,
            }
        except Exception as _e:
            print(f"警告: 无法加载计时数据: {_e}")
        return (positions1, stresses1, positions2, stresses2, timing_dict)
    else:
        # 单域模式：调用 implicit_mpm.py
        cmd = [
            sys.executable,
            "simulators/implicit_mpm.py",
            "--config", config_path,
            # "--no-gui",
        ]
        print(f"执行命令: {' '.join(cmd)}")

        # 记录子进程启动前已有的目录，避免误 pick 旧结果
        exp_results_dir = "experiment_results"
        existing_dirs = set(
            d for d in os.listdir(exp_results_dir)
            if d.startswith("single_domain_") and
            os.path.isdir(os.path.join(exp_results_dir, d))
        )

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("模拟失败！")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Simulation failed with code {result.returncode}")

        print("模拟完成，加载结果数据...")

        # 找子进程运行后新出现的目录
        all_dirs = set(
            d for d in os.listdir(exp_results_dir)
            if d.startswith("single_domain_") and
            os.path.isdir(os.path.join(exp_results_dir, d))
        )
        new_dirs = sorted(all_dirs - existing_dirs)
        if not new_dirs:
            raise RuntimeError(
                "子进程未创建新的实验结果目录（可能崩溃或未正常保存数据）\n"
                f"STDOUT: {result.stdout[-500:] if result.stdout else ''}\n"
                f"STDERR: {result.stderr[-500:] if result.stderr else ''}"
            )
        latest_exp_dir = os.path.join(exp_results_dir, new_dirs[-1])
        print(f"加载实验结果: {latest_exp_dir}")

        # 查找stress_data目录中的最后一帧
        stress_data_dir = os.path.join(latest_exp_dir, "stress_data")
        if not os.path.exists(stress_data_dir):
            raise RuntimeError(f"未找到应力数据目录: {stress_data_dir}")

        # 获取所有frame目录
        frame_dirs = [d for d in os.listdir(stress_data_dir)
                     if d.startswith("frame_") and
                     os.path.isdir(os.path.join(stress_data_dir, d))]

        if not frame_dirs:
            raise RuntimeError(f"未找到帧数据目录: {stress_data_dir}")

        # 按帧号排序，获取最后一帧
        frame_numbers = [int(d.replace("frame_", "")) for d in frame_dirs]
        latest_frame = max(frame_numbers)
        latest_frame_dir = os.path.join(stress_data_dir, f"frame_{latest_frame}")

        print(f"加载帧数据: {latest_frame_dir}")

        # 加载positions和stress数据
        positions = np.load(os.path.join(latest_frame_dir, "positions.npy"))
        stresses = np.load(os.path.join(latest_frame_dir, "stress.npy"))

        print(f"加载了 {len(positions)} 个粒子的数据")

        timing_dict = None
        try:
            stats_path = os.path.join(latest_exp_dir, 'performance_stats', 'stats_data.json')
            with open(stats_path) as _f:
                _stats = json.load(_f)
            _sum = _stats.get('summary', {})
            _total = _sum.get('total_time', 0.0)
            _solve = _sum.get('solve_time', 0.0)
            timing_dict = {
                'mode': 'single',
                'solve_time': _solve,
                'other_time': _total - _solve,
                'total_time': _total,
            }
        except Exception as _e:
            print(f"警告: 无法加载计时数据: {_e}")
        return (positions, stresses, timing_dict)


def _grid_positions_from_meta(grid_meta):
    """根据网格元数据重建网格节点位置。"""
    nx = int(grid_meta["nx"])
    ny = int(grid_meta["ny"])
    offset = np.array(grid_meta.get("offset", [0.0, 0.0]), dtype=float)
    dx_x = float(grid_meta["dx_x"])
    dx_y = float(grid_meta["dx_y"])

    xs = offset[0] + np.arange(nx) * dx_x
    ys = offset[1] + np.arange(ny) * dx_y
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    return np.column_stack([xx.ravel(), yy.ravel()])


def _load_grid_stress_from_frame(frame_dir, config, use_schwarz=False):
    """从 stress_data/frame_* 目录加载网格应力，并返回 positions/stresses。"""
    def _load_single(prefix=None, domain_cfg=None):
        if prefix is None:
            stress_file = os.path.join(frame_dir, "grid_stress.npy")
            mass_file = os.path.join(frame_dir, "grid_mass.npy")
            meta_file = os.path.join(frame_dir, "grid_stress_meta.json")
        else:
            stress_file = os.path.join(frame_dir, f"{prefix}_grid_stress.npy")
            mass_file = os.path.join(frame_dir, f"{prefix}_grid_mass.npy")
            meta_file = os.path.join(frame_dir, f"{prefix}_grid_stress_meta.json")

        if not os.path.exists(stress_file) or not os.path.exists(meta_file):
            return None

        grid_stress = np.load(stress_file)
        with open(meta_file, "r") as f:
            grid_meta = json.load(f)

        positions = _grid_positions_from_meta(grid_meta)
        stresses = grid_stress.reshape(-1, grid_stress.shape[-2], grid_stress.shape[-1])

        if os.path.exists(mass_file):
            grid_mass = np.load(mass_file)
            valid_mask = grid_mass.reshape(-1) > 1e-10
            positions = positions[valid_mask]
            stresses = stresses[valid_mask]

        return positions, stresses, grid_meta

    if use_schwarz:
        d1 = _load_single("domain1")
        d2 = _load_single("domain2")
        if d1 is None or d2 is None:
            return None
        positions1, stresses1, meta1 = d1
        positions2, stresses2, meta2 = d2
        positions, stresses = merge_schwarz_domains(positions1, stresses1, positions2, stresses2, config)
        return {
            "positions": positions,
            "stresses": stresses,
            "positions1": positions1,
            "stresses1": stresses1,
            "positions2": positions2,
            "stresses2": stresses2,
            "grid_meta1": meta1,
            "grid_meta2": meta2,
        }

    single = _load_single()
    if single is None:
        return None
    positions, stresses, meta = single
    return {
        "positions": positions,
        "stresses": stresses,
        "grid_meta": meta,
    }


def _load_grid_stress_from_result_dir(result_dir, config, use_schwarz=False):
    """从 grid_xxx 结果目录加载已复制出来的网格应力。"""
    def _load_single(prefix=None):
        if prefix is None:
            stress_file = os.path.join(result_dir, "grid_stress.npy")
            mass_file = os.path.join(result_dir, "grid_mass.npy")
            meta_file = os.path.join(result_dir, "grid_stress_meta.json")
        else:
            stress_file = os.path.join(result_dir, f"{prefix}_grid_stress.npy")
            mass_file = os.path.join(result_dir, f"{prefix}_grid_mass.npy")
            meta_file = os.path.join(result_dir, f"{prefix}_grid_stress_meta.json")

        if not os.path.exists(stress_file) or not os.path.exists(meta_file):
            return None

        grid_stress = np.load(stress_file)
        with open(meta_file, "r") as f:
            grid_meta = json.load(f)

        positions = _grid_positions_from_meta(grid_meta)
        stresses = grid_stress.reshape(-1, grid_stress.shape[-2], grid_stress.shape[-1])

        if os.path.exists(mass_file):
            grid_mass = np.load(mass_file)
            valid_mask = grid_mass.reshape(-1) > 1e-10
            positions = positions[valid_mask]
            stresses = stresses[valid_mask]

        return positions, stresses, grid_meta

    if use_schwarz:
        d1 = _load_single("domain1")
        d2 = _load_single("domain2")
        if d1 is None or d2 is None:
            frame_dir = _find_latest_frame_dir(result_dir)
            if frame_dir is None:
                return None
            return _load_grid_stress_from_frame(frame_dir, config, use_schwarz=True)
        positions1, stresses1, meta1 = d1
        positions2, stresses2, meta2 = d2
        positions, stresses = merge_schwarz_domains(positions1, stresses1, positions2, stresses2, config)
        return {
            "positions": positions,
            "stresses": stresses,
            "positions1": positions1,
            "stresses1": stresses1,
            "positions2": positions2,
            "stresses2": stresses2,
            "grid_meta1": meta1,
            "grid_meta2": meta2,
        }

    single = _load_single()
    if single is None:
        frame_dir = _find_latest_frame_dir(result_dir)
        if frame_dir is None:
            return None
        return _load_grid_stress_from_frame(frame_dir, config, use_schwarz=False)
    positions, stresses, meta = single
    return {
        "positions": positions,
        "stresses": stresses,
        "grid_meta": meta,
    }


def _copy_grid_stress_files(latest_frame_dir, output_dir, use_schwarz=False):
    """把最新帧的 grid_stress 文件复制到 grid_xxx 结果目录。"""
    os.makedirs(output_dir, exist_ok=True)

    if use_schwarz:
        filenames = [
            "domain1_grid_stress.npy",
            "domain1_grid_mass.npy",
            "domain1_grid_stress_meta.json",
            "domain2_grid_stress.npy",
            "domain2_grid_mass.npy",
            "domain2_grid_stress_meta.json",
        ]
    else:
        filenames = [
            "grid_stress.npy",
            "grid_mass.npy",
            "grid_stress_meta.json",
        ]

    copied = []
    for name in filenames:
        src = os.path.join(latest_frame_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, name))
            copied.append(name)

    if copied:
        print(f"已复制 grid stress 文件到结果目录: {output_dir} ({', '.join(copied)})")
    else:
        print(f"警告: 未找到可复制的 grid stress 文件: {latest_frame_dir}")


def _find_latest_simulation_result_dir(use_schwarz=False):
    """找到 experiment_results 下最新的单域/双域实验目录。"""
    import glob

    pattern = "experiment_results/schwarz_*" if use_schwarz else "experiment_results/single_domain_*"
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def _find_latest_frame_dir(exp_dir):
    """找到实验目录中最新的 stress_data/frame_* 目录。"""
    stress_data_dir = os.path.join(exp_dir, "stress_data")
    if not os.path.exists(stress_data_dir):
        return None

    frame_dirs = [
        os.path.join(stress_data_dir, d)
        for d in os.listdir(stress_data_dir)
        if d.startswith("frame_") and os.path.isdir(os.path.join(stress_data_dir, d))
    ]
    if not frame_dirs:
        return None
    return max(frame_dirs, key=os.path.getmtime)


def _stress_source_suffix(stress_source):
    """返回文件名后缀。"""
    return '' if stress_source == 'particle' else '_grid'

def calculate_strip_width(config, grid_size=None, use_schwarz=False):
    """
    根据网格尺寸计算截面宽度
    
    参数:
        config: 配置字典
        grid_size: 网格大小（可选，如果提供则覆盖config中的值）
        use_schwarz: 是否为Schwarz模式
    
    返回:
        float: 截面宽度（约为0.5倍网格间距）
    """
    if use_schwarz:
        domain_config = config.get('Domain2', {})
        domain_h = domain_config.get('domain_height', 0.3)
        grid_ny = domain_config.get('grid_ny', 30)
        
        if grid_size is not None:
            # 根据宽高比计算grid_ny
            original_nx = domain_config.get('grid_nx', 30)
            original_ny = domain_config.get('grid_ny', 30)
            aspect_ratio = original_ny / original_nx if original_nx > 0 else 1.0
            grid_ny = int(grid_size * aspect_ratio)
    else:
        domain_h = config.get('domain_height', 1.0)
        grid_ny = config.get('grid_ny', 100)
        
        if grid_size is not None:
            # 根据宽高比计算grid_ny
            original_nx = config.get('grid_nx', 100)
            original_ny = config.get('grid_ny', 100)
            aspect_ratio = original_ny / original_nx if original_nx > 0 else 1.0
            grid_ny = int(grid_size * aspect_ratio)
    
    dy = domain_h / grid_ny
    strip_width = 0.5 * dy  # 0.5倍网格间距
    
    return strip_width

def get_strip_width_from_params(domain_h, grid_size, radius=0.15):
    """
    从参数直接计算截面宽度（用于分析已有结果）
    
    参数:
        domain_h: 域的高度
        grid_size: 网格大小
        radius: 夹杂半径（仅在grid_size<=0时作为fallback）
    
    返回:
        float: 截面宽度
    """
    if grid_size > 0:
        dy_per_grid = domain_h / grid_size
        return 0.5 * dy_per_grid
    else:
        print("警告: grid_size <= 0，使用默认半径的20%作为截面宽度")
        # Fallback
        return radius * 0.2

def extract_domain_config(config, use_schwarz):
    """
    提取域配置参数
    
    参数:
        config: 配置字典
        use_schwarz: 是否为Schwarz模式
    
    返回:
        tuple: (domain_config, domain_w, domain_h, offset)
    """
    if use_schwarz:
        domain_config = config.get('Domain2', {})
        domain_w = domain_config.get('domain_width', 0.3)
        domain_h = domain_config.get('domain_height', 0.3)
        offset = np.array(domain_config.get('offset', [0.35, 0.35]))
    else:
        domain_config = config
        domain_w = config.get('domain_width', 1.0)
        domain_h = config.get('domain_height', 1.0)
        offset = np.array([0.0, 0.0])
    
    return domain_config, domain_w, domain_h, offset

def extract_inclusion_params(domain_config, domain_w, domain_h, offset):
    """
    提取夹杂（inclusion）参数
    
    参数:
        domain_config: 域配置字典
        domain_w, domain_h: 域的宽度和高度
        offset: 偏移量
    
    返回:
        tuple: (center, radius)
    """
    center = None
    radius = 0.15

    for shape in domain_config.get('shapes', []):
        if shape['type'] == 'ellipse' and shape['operation'] == 'change':
            local_center = np.array(shape['params']['center'])
            center = local_center + offset
            radius = shape['params']['semi_axes'][0]
            break

    if center is None:
        center = np.array([domain_w/2.0, domain_h/2.0]) + offset
    
    return center, radius

def extract_material_params(domain_config, config):
    """
    提取材料参数
    
    参数:
        domain_config: 域配置字典
        config: 完整配置字典
    
    返回:
        tuple: (E_in, nu_in, E_out, nu_out, delta)
    """
    mat_params = domain_config.get('material_params', config.get('material_params', []))

    # 基体 (Matrix) - ID=0
    matrix_mat = next(m for m in mat_params if m['id'] == 0)
    E_out = matrix_mat['E']
    nu_out = matrix_mat['nu']

    # 夹杂 (Inclusion) - ID=1
    inclusion_mat = next(m for m in mat_params if m['id'] == 1)
    E_in = inclusion_mat['E']
    nu_in = inclusion_mat['nu']

    # 计算过盈量
    initial_F = inclusion_mat.get('initial_F', [[1.0, 0],[0, 1.0]])
    delta = 1.0 - initial_F[0][0]

    return E_in, nu_in, E_out, nu_out, delta

def merge_schwarz_domains(positions1, stresses1, positions2, stresses2, config):
    """
    合并Schwarz双域数据
    
    参数:
        positions1, stresses1: Domain1的位置和应力
        positions2, stresses2: Domain2的位置和应力
        config: 配置字典
    
    返回:
        tuple: (positions, stresses) 合并后的数据
    """
    # 获取Domain2的范围
    domain2_config = config.get('Domain2', {})
    domain2_offset = np.array(domain2_config.get('offset', [0.35, 0.35]))
    domain2_width = domain2_config.get('domain_width', 0.3)
    domain2_height = domain2_config.get('domain_height', 0.3)

    # Domain2的全局范围
    d2_xmin, d2_xmax = domain2_offset[0], domain2_offset[0] + domain2_width
    d2_ymin, d2_ymax = domain2_offset[1], domain2_offset[1] + domain2_height

    # 从Domain1中排除Domain2范围内的粒子（基于KD-tree邻近过滤，保留D2空洞中的D1粒子）
    mask_inside_d2 = ((positions1[:, 0] >= d2_xmin) & (positions1[:, 0] <= d2_xmax) &
                      (positions1[:, 1] >= d2_ymin) & (positions1[:, 1] <= d2_ymax))

    positions1_inside = positions1[mask_inside_d2]
    stresses1_inside = stresses1[mask_inside_d2]

    if len(positions1_inside) > 0 and len(positions2) > 0:
        # 估计Domain2的格间距
        grid_nx2 = domain2_config.get('grid_nx', 30)
        dx2 = domain2_width / grid_nx2
        # KD-tree：只去掉D2边界附近的D1粒子，保留D2空洞中的D1粒子
        tree2 = KDTree(positions2)
        dist, _ = tree2.query(positions1_inside, k=1)
        keep_in_hole = dist > dx2 * 0.8
        positions1_filtered = np.vstack([positions1[~mask_inside_d2],
                                         positions1_inside[keep_in_hole]])
        stresses1_filtered = np.vstack([stresses1[~mask_inside_d2],
                                        stresses1_inside[keep_in_hole]])
    else:
        positions1_filtered = positions1[~mask_inside_d2]
        stresses1_filtered = stresses1[~mask_inside_d2]

    # 合并Domain1(过滤后) + Domain2
    positions = np.vstack([positions1_filtered, positions2])
    stresses = np.vstack([stresses1_filtered, stresses2])

    return positions, stresses

def analyze_and_plot(positions_or_tuple, stresses_or_none, config, output_dir, grid_size=None, use_schwarz=False):
    """
    分析和绘制结果

    参数:
        positions_or_tuple: 单域模式下为positions数组，Schwarz模式下为(pos1, stress1, pos2, stress2)元组
        stresses_or_none: 单域模式下为stresses数组，Schwarz模式下为None
        config: 配置字典
        output_dir: 输出目录
        grid_size: 网格大小（用于标注），可选
        use_schwarz: 是否为Schwarz模式

    返回:
        dict: 包含plot_path, positions, stresses, analytical_data等信息
    """
    # 处理Schwarz双域数据
    if use_schwarz:
        positions1, stresses1, positions2, stresses2 = positions_or_tuple
        # 保存双域原始数据，供后续 --analyze-only 可视化使用
        np.save(os.path.join(output_dir, 'domain1_positions.npy'), positions1)
        np.save(os.path.join(output_dir, 'domain1_stresses.npy'), stresses1)
        np.save(os.path.join(output_dir, 'domain2_positions.npy'), positions2)
        np.save(os.path.join(output_dir, 'domain2_stresses.npy'), stresses2)
        positions, stresses = merge_schwarz_domains(positions1, stresses1, positions2, stresses2, config)
    else:
        positions = positions_or_tuple
        stresses = stresses_or_none

    # 提取参数
    domain_config, domain_w, domain_h, offset = extract_domain_config(config, use_schwarz)
    center, radius = extract_inclusion_params(domain_config, domain_w, domain_h, offset)
    E_in, nu_in, E_out, nu_out, delta = extract_material_params(domain_config, config)

    print(f"参数: Center={center}, Radius={radius}, Delta={delta}")
    print(f"材料: E_in={E_in}, nu_in={nu_in}, E_out={E_out}, nu_out={nu_out}")

    # 提取截面数据 (沿中心水平线)
    strip_width = calculate_strip_width(config, grid_size, use_schwarz)
    mask_strip = np.abs(positions[:, 1] - center[1]) < strip_width
    
    mpm_x = positions[mask_strip, 0]
    mpm_y = positions[mask_strip, 1]
    mpm_sig_xx = stresses[mask_strip, 0, 0]
    mpm_sig_yy = stresses[mask_strip, 1, 1]

    print(f"提取截面数据: {len(mpm_x)} 个粒子 (总粒子数: {len(positions)})")
    print(f"应力范围: σ_xx [{np.min(mpm_sig_xx):.2e}, {np.max(mpm_sig_xx):.2e}], σ_yy [{np.min(mpm_sig_yy):.2e}, {np.max(mpm_sig_yy):.2e}]")

    # 计算解析解（生成光滑曲线）
    line_x = np.linspace(center[0] - 3*radius, center[0] + 3*radius, 500)
    line_y = np.full_like(line_x, center[1])

    ana_sig_xx, ana_sig_yy = get_analytical_stress_cartesian(
        line_x - center[0], line_y - center[1],
        radius, delta, E_in, nu_in, E_out, nu_out
    )

    _, (ax_xx, ax_yy) = plt.subplots(1, 2, figsize=(10, 4.5))

    sort_idx = np.argsort(mpm_x)
    mpm_x_s = mpm_x[sort_idx] - center[0]
    mpm_sig_xx_s = mpm_sig_xx[sort_idx]
    mpm_sig_yy_s = mpm_sig_yy[sort_idx]

    # Left: Sigma_XX
    ax_xx.plot(mpm_x_s, mpm_sig_xx_s, color='#3182bd', lw=1.0, label='MPM')
    ax_xx.plot(line_x - center[0], ana_sig_xx, color='#b2182b', linestyle='--',
               lw=1.2, label='Analytical')
    ax_xx.axvline(-radius, color='k', linestyle=':', alpha=0.3, label='Inclusion boundary')
    ax_xx.axvline(radius, color='k', linestyle=':', alpha=0.3)
    ax_xx.set_xlabel(r'Radial position (m)')
    ax_xx.set_ylabel(r'$\sigma_{xx}$ (Pa)')
    ax_xx.set_xlim(-3*radius, 3*radius)
    ax_xx.grid(True, alpha=0.3)

    # Right: Sigma_YY
    ax_yy.plot(mpm_x_s, mpm_sig_yy_s, color='#2ca02c', lw=1.0, label='MPM')
    ax_yy.plot(line_x - center[0], ana_sig_yy, color='#b2182b', linestyle='--',
               lw=1.2, label='Analytical')
    ax_yy.axvline(-radius, color='k', linestyle=':', alpha=0.3, label='Inclusion boundary')
    ax_yy.axvline(radius, color='k', linestyle=':', alpha=0.3)
    ax_yy.set_xlabel(r'Radial position (m)')
    ax_yy.set_ylabel(r'$\sigma_{yy}$ (Pa)')
    ax_yy.set_xlim(-3*radius, 3*radius)
    ax_yy.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, 'result_v3_cartesian.pdf')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"图表已保存: {save_path}")

    # 保存数据
    np.save(os.path.join(output_dir, 'positions.npy'), positions)
    np.save(os.path.join(output_dir, 'stresses.npy'), stresses)
    return {
        'plot_path': save_path,
        'positions': positions,
        'stresses': stresses,
        'analytical': {
            'line_x': line_x,
            'line_y': line_y,
            'ana_sig_xx': ana_sig_xx,
            'ana_sig_yy': ana_sig_yy
        },
        'params': {
            'center': center,
            'radius': radius,
            'delta': delta,
            'E_in': E_in,
            'nu_in': nu_in,
            'E_out': E_out,
            'nu_out': nu_out,
            'domain_w': domain_w,
            'domain_h': domain_h
        }
    }

def run_single_experiment(config_path, grid_size, output_dir, use_schwarz=False, stress_source='particle'):
    """
    运行单个实验

    参数:
        config_path: 配置文件路径
        grid_size: 网格大小（可选，None表示使用配置文件中的值）
        output_dir: 输出目录
        use_schwarz: 是否使用Schwarz求解器
        stress_source: 'particle' 或 'grid'，决定比较时使用哪类应力数据

    返回:
        dict: 实验结果
    """
    print(f"\n{'='*60}")
    if grid_size:
        mode = "Schwarz" if use_schwarz else "Single"
        print(f"运行实验 ({mode}模式) - 网格大小: {grid_size}x{grid_size}")
    else:
        print(f"运行实验 - 使用配置文件中的网格大小")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    config = load_config(config_path)

    # 运行模拟
    sim_results = run_simulation(config_path, use_schwarz)
    timing_dict = sim_results[-1]   # 最后一个元素是 timing_dict (or None)
    sim_data = sim_results[:-1]     # 前面是 positions/stresses

    latest_exp_dir = _find_latest_simulation_result_dir(use_schwarz=use_schwarz)
    latest_frame_dir = _find_latest_frame_dir(latest_exp_dir) if latest_exp_dir else None
    if latest_frame_dir is not None:
        _copy_grid_stress_files(latest_frame_dir, output_dir, use_schwarz=use_schwarz)
    else:
        print("警告: 未找到最新的 stress_data/frame_* 目录，无法复制 grid stress 文件")

    if timing_dict is not None:
        with open(os.path.join(output_dir, 'timing.json'), 'w') as _f:
            json.dump(timing_dict, _f, indent=4)

    if stress_source == 'grid':
        if latest_frame_dir is None:
            raise RuntimeError("选择了 grid stress，但未找到可用的 grid_stress 数据")
        grid_payload = _load_grid_stress_from_frame(latest_frame_dir, config, use_schwarz=use_schwarz)
        if grid_payload is None:
            raise RuntimeError("选择了 grid stress，但无法从最新结果帧加载 grid_stress")
        if use_schwarz:
            sim_data = (
                grid_payload['positions1'],
                grid_payload['stresses1'],
                grid_payload['positions2'],
                grid_payload['stresses2'],
            )
        else:
            sim_data = (grid_payload['positions'], grid_payload['stresses'])

    # 分析和绘图
    if use_schwarz:
        results = analyze_and_plot(sim_data, None, config, output_dir, grid_size, use_schwarz=True)
    else:
        positions, stresses = sim_data
        results = analyze_and_plot(positions, stresses, config, output_dir, grid_size, use_schwarz=False)

    # 保存配置备份
    config_backup_path = os.path.join(output_dir, 'config_backup.json')
    save_config(config, config_backup_path)

    print(f"实验完成! 结果保存到: {output_dir}")

    return results

def run_batch_experiments(args):
    """
    运行批量实验

    参数:
        args: 命令行参数对象

    返回:
        dict: 批量实验结果汇总
    """
    print("=" * 80)
    print("开始批量网格分辨率实验")
    print("=" * 80)

    # 生成网格大小列表
    if args.dx_start is not None and args.dx_end is not None and args.dx_step is not None:
        # 从 dx 范围推导 grid_size：需要先读 domain_width
        base_config_for_dx = load_config(args.config)
        if args.schwarz:
            domain_w = base_config_for_dx.get('Domain2', {}).get('domain_width', 1.0)
        else:
            domain_w = base_config_for_dx.get('domain_width', 1.0)
        dx_low = min(args.dx_start, args.dx_end)
        dx_high = max(args.dx_start, args.dx_end)
        dx_step = abs(args.dx_step)
        dx_values = []
        # 按从大到小的顺序执行：先粗网格，后细网格
        dx = dx_high
        while dx >= dx_low - 1e-12:
            dx_values.append(dx)
            dx -= dx_step
        grid_sizes = [max(1, round(domain_w / d)) for d in dx_values]
        print(f"dx 列表:     {[round(d, 6) for d in dx_values]}")
        print(f"网格大小列表: {grid_sizes}")
    else:
        grid_sizes = list(range(args.grid_start, args.grid_end + 1, args.grid_step))
        print(f"网格大小列表: {grid_sizes}")
    print(f"总共 {len(grid_sizes)} 个实验")

    # 加载基础配置
    base_config = load_config(args.config)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 记录实验结果
    results = {
        'base_config': args.config,
        'grid_sizes': grid_sizes,
        'successful_runs': [],
        'failed_runs': [],
        'result_dirs': {},
        'experiment_data': {}
    }

    for i, grid_size in enumerate(grid_sizes):
        print(f"\n{'='*20} 实验 {i+1}/{len(grid_sizes)}: 网格 {grid_size}x{grid_size} {'='*20}")

        try:
            # 1. 修改配置
            modify_d1 = getattr(args, 'modify_domain1', False) and args.schwarz
            modified_config = modify_config_grid_size(base_config, grid_size,
                                                      use_schwarz=args.schwarz,
                                                      modify_domain1=modify_d1)

            # 2. 创建临时配置文件
            temp_config_path = os.path.join(args.output_dir, f"temp_config_grid{grid_size}.json")
            save_config(modified_config, temp_config_path)

            # 3. 创建输出子目录
            grid_output_dir = os.path.join(args.output_dir, f"grid_{grid_size}")

            # 4. 运行单个实验
            exp_results = run_single_experiment(
                temp_config_path,
                grid_size,
                grid_output_dir,
                use_schwarz=args.schwarz,
                stress_source=args.stress_source,
            )

            # 5. 记录结果
            results['successful_runs'].append(grid_size)
            results['result_dirs'][grid_size] = grid_output_dir
            results['experiment_data'][grid_size] = exp_results

            print(f"✓ 实验 {i+1} 成功完成")

            # 清理临时配置文件
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

        except Exception as e:
            print(f"✗ 实验 {i+1} 失败: {e}")
            import traceback
            traceback.print_exc()
            results['failed_runs'].append(grid_size)

    # 保存实验汇总
    from datetime import datetime
    results['completed_time'] = datetime.now().isoformat()
    summary_file = os.path.join(args.output_dir, "experiment_summary.json")

    # 移除不能序列化的numpy数组
    summary_results = {k: v for k, v in results.items() if k != 'experiment_data'}
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=4)

    print(f"\n{'='*80}")
    print("批量实验完成!")
    print(f"成功: {len(results['successful_runs'])}/{len(grid_sizes)} 个实验")
    print(f"成功的网格大小: {results['successful_runs']}")
    if results['failed_runs']:
        print(f"失败的网格大小: {results['failed_runs']}")
    print(f"实验汇总保存到: {summary_file}")
    print("=" * 80)

    return results

def _save_legend_pdf(handles, labels, path):
    """将 legend 单独保存为 PDF。"""
    fig_leg = plt.figure(figsize=(2.5, max(len(labels) * 0.28 + 0.2, 0.6)))
    fig_leg.legend(handles, labels, loc='center', frameon=True, fontsize='x-small')
    fig_leg.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig_leg)


def _draw_summary_axes(ax1, ax2, experiment_data, params, analytical,
                       yy_only, colors, boundary_exclusion=None, xx_only=False):
    """
    在 ax1 (σ_xx) 和/或 ax2 (σ_yy) 上绘制解析解 + 各网格 MPM 结果。
    yy_only=True: 只画 σ_yy (ax2)。xx_only=True: 只画 σ_xx (ax1)。
    返回 (handles, labels) 供外部保存 legend。
    """
    line_x = analytical['line_x']
    ana_sig_xx = analytical['ana_sig_xx']
    ana_sig_yy = analytical['ana_sig_yy']
    center = params['center']
    radius = params['radius']

    draw_xx = not yy_only
    draw_yy = not xx_only

    if draw_xx:
        ax1.plot(line_x - center[0], ana_sig_xx, color='#b2182b', linestyle='--',
                 lw=1.2, label='Analytical', zorder=10, alpha=0.9)
    if draw_yy:
        ax2.plot(line_x - center[0], ana_sig_yy, color='#b2182b', linestyle='--',
                 lw=1.2, label='Analytical', zorder=10, alpha=0.9)

    for i, (grid_size, exp_data) in enumerate(sorted(experiment_data.items())):
        color = colors[i % len(colors)]

        positions = exp_data['positions']
        stresses = exp_data['stresses']

        strip_width = get_strip_width_from_params(params['domain_h'], grid_size, radius)
        mask_strip = np.abs(positions[:, 1] - center[1]) < strip_width

        pos_s = positions[mask_strip]
        str_s = stresses[mask_strip]

        if boundary_exclusion is not None:
            dx = params['domain_w'] / grid_size
            excl = boundary_exclusion * dx
            x_rel = pos_s[:, 0] - center[0]
            y_rel = pos_s[:, 1] - center[1]
            r = np.sqrt(x_rel**2 + y_rel**2)
            mask_excl = np.abs(r - radius) > excl
            pos_s = pos_s[mask_excl]
            str_s = str_s[mask_excl]

        mpm_x = pos_s[:, 0]
        mpm_sig_xx = str_s[:, 0, 0]
        mpm_sig_yy = str_s[:, 1, 1]

        label = f'{params["domain_h"]/grid_size:.2g}'
        idx = np.argsort(mpm_x)

        if draw_xx:
            ax1.plot(mpm_x[idx] - center[0], mpm_sig_xx[idx], '-', color=color,
                     linewidth=1.0, alpha=0.9, label=label)
        if draw_yy:
            ax2.plot(mpm_x[idx] - center[0], mpm_sig_yy[idx], '-', color=color,
                     linewidth=1.0, alpha=0.9, label=label)

    ref_ax = ax1 if draw_xx else ax2
    active_axes = ([ax1] if xx_only else []) + ([ax2] if not xx_only else [])
    for ax in active_axes:
        ax.axvline(-radius, color='k', linestyle=':', alpha=0.3, label='Inclusion boundary')
        ax.axvline(radius, color='k', linestyle=':', alpha=0.3)
        ax.set_xlabel(r'Radial position (m)')
        ax.set_xlim(-3*radius, 3*radius)
        ax.grid(True, alpha=0.3)
    if draw_xx:
        ax1.set_ylabel(r'$\sigma_{xx}$ (Pa)')
    if draw_yy:
        ax2.set_ylabel(r'$\sigma_{yy}$ (Pa)')
    handles, labels = ref_ax.get_legend_handles_labels()
    return handles, labels


def create_summary_plot(results, output_dir, yy_only=False, boundary_exclusion=1.0, file_suffix=''):
    """
    创建汇总图，将所有网格大小的结果绘制在一张图上。
    额外生成一张排除 inclusion 边界附近粒子的版本 (summary_all_grids_excl.pdf)。

    参数:
        results: 批量实验结果字典
        output_dir: 输出目录
        yy_only: 仅绘制yy方向应力
        boundary_exclusion: 排除宽度，单位为 grid size 倍数
    """
    print("\n创建汇总对比图...")

    experiment_data = results['experiment_data']
    if not experiment_data:
        print("警告: 没有可用的实验数据")
        return

    colors = CMAME_COLORS

    first_grid = list(experiment_data.keys())[0]
    first_exp = experiment_data[first_grid]
    analytical = first_exp['analytical']
    params = first_exp['params']

    print(f"网格数据粒子数统计:")
    for grid_size, exp_data in sorted(experiment_data.items()):
        strip_width = get_strip_width_from_params(params['domain_h'], grid_size, params['radius'])
        n = np.sum(np.abs(exp_data['positions'][:, 1] - params['center'][1]) < strip_width)
        print(f"网格 {grid_size}x{grid_size}: 提取截面数据: {n} 个粒子")

    # ── 图1: 全部粒子 ────────────────────────────────────────────────────────
    if yy_only:
        fig, ax2 = plt.subplots(1, 1, figsize=(8.0, 4.5))
        ax1 = None
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    handles, labels = _draw_summary_axes(ax1, ax2, experiment_data, params, analytical,
                                          yy_only, colors, boundary_exclusion=None)
    fig.tight_layout()
    summary_plot_path = os.path.join(output_dir, f"summary_all_grids{file_suffix}.pdf")
    plt.savefig(summary_plot_path)
    plt.close()
    legend_path = os.path.join(output_dir, f"summary_legend{file_suffix}.pdf")
    _save_legend_pdf(handles, labels, legend_path)
    print(f"汇总对比图已保存到: {summary_plot_path}")
    print(f"Legend 已保存到: {legend_path}")

    # ── 单独保存 σ_xx 和 σ_yy（全部粒子版）─────────────────────────────────
    for component, xx_only_flag, yy_only_flag, ylabel, fname in [
        ('xx', True,  False, r'$\sigma_{xx}$ (Pa)', f'summary_xx{file_suffix}.pdf'),
        ('yy', False, True,  r'$\sigma_{yy}$ (Pa)', f'summary_yy{file_suffix}.pdf'),
    ]:
        fig_c, ax_c = plt.subplots(1, 1, figsize=(5.5, 4.5))
        ax1_c = ax_c if not yy_only_flag else None
        ax2_c = ax_c if not xx_only_flag else None
        _draw_summary_axes(ax1_c, ax2_c, experiment_data, params, analytical,
                           yy_only_flag, colors, boundary_exclusion=None,
                           xx_only=xx_only_flag)
        fig_c.tight_layout()
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"单独保存 {component} 图: {fname}")

    # ── 图2: 排除 inclusion 边界附近粒子 ─────────────────────────────────────
    if yy_only:
        fig2, ax2b = plt.subplots(1, 1, figsize=(8.0, 4.5))
        ax1b = None
    else:
        fig2, (ax1b, ax2b) = plt.subplots(1, 2, figsize=(10, 4.5))

    _draw_summary_axes(ax1b, ax2b, experiment_data, params, analytical,
                       yy_only, colors, boundary_exclusion=boundary_exclusion)
    fig2.tight_layout()
    excl_plot_path = os.path.join(output_dir, f"summary_all_grids_excl{file_suffix}.pdf")
    plt.savefig(excl_plot_path)
    plt.close()
    print(f"排除边界版汇总图已保存到: {excl_plot_path} (排除宽度={boundary_exclusion}×dx)")

    # ── 单独保存 σ_xx 和 σ_yy（排除边界版）─────────────────────────────────
    for component, xx_only_flag, yy_only_flag, fname in [
        ('xx', True,  False, f'summary_excl_xx{file_suffix}.pdf'),
        ('yy', False, True,  f'summary_excl_yy{file_suffix}.pdf'),
    ]:
        fig_c, ax_c = plt.subplots(1, 1, figsize=(5.5, 4.5))
        ax1_c = ax_c if not yy_only_flag else None
        ax2_c = ax_c if not xx_only_flag else None
        _draw_summary_axes(ax1_c, ax2_c, experiment_data, params, analytical,
                           yy_only_flag, colors, boundary_exclusion=boundary_exclusion,
                           xx_only=xx_only_flag)
        fig_c.tight_layout()
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"单独保存 {component} 排除边界版: {fname}")

    return summary_plot_path

def create_integrated_error_plot(results, output_dir, boundary_exclusion=1.0, file_suffix=''):
    """
    创建积分误差对比图，显示不同网格大小下的归一化积分误差

    参数:
        results: 批量实验结果字典
        output_dir: 输出目录
    """
    print("\n创建积分误差对比图...")

    experiment_data = results['experiment_data']
    if not experiment_data:
        print("警告: 没有可用的实验数据")
        return

    # 存储每个网格大小的积分误差
    grid_sizes = []
    integrated_errors_xx = []
    integrated_errors_yy = []
    integrated_errors_total = []

    # 获取参数（所有实验应该使用相同的参数）
    first_grid = list(experiment_data.keys())[0]
    first_exp = experiment_data[first_grid]
    params = first_exp['params']

    center = params['center']
    radius = params['radius']
    delta = params['delta']
    E_in = params['E_in']
    nu_in = params['nu_in']
    E_out = params['E_out']
    nu_out = params['nu_out']

    print(f"计算积分误差，参数: center={center}, radius={radius}")

    # 对每个网格大小计算积分误差
    for grid_size, exp_data in sorted(experiment_data.items()):
        print(f"处理网格大小: {grid_size}x{grid_size}")

        # 获取所有粒子的位置和应力
        positions = exp_data['positions']
        stresses = exp_data['stresses']

        # 只提取水平截面上的粒子（与create_summary_plot一致）
        # 使用统一的strip_width计算函数
        strip_width = get_strip_width_from_params(params['domain_h'], grid_size, radius)
        mask_strip = np.abs(positions[:, 1] - center[1]) < strip_width
        
        positions_strip = positions[mask_strip]
        stresses_strip = stresses[mask_strip]
        
        # 进一步限制在x方向±3倍半径范围内（与汇总图范围一致）
        x_rel_all = positions_strip[:, 0] - center[0]
        mask_x_range = np.abs(x_rel_all) <= 1 * radius

        positions_strip = positions_strip[mask_x_range]
        stresses_strip = stresses_strip[mask_x_range]

        # 排除距离 inclusion 边界 ±boundary_exclusion 个 grid size 以内的粒子
        dx = params['domain_w'] / grid_size
        excl_width = boundary_exclusion * dx
        x_rel = positions_strip[:, 0] - center[0]
        y_rel = positions_strip[:, 1] - center[1]
        r = np.sqrt(x_rel**2 + y_rel**2)
        mask_exclude_boundary = np.abs(r - radius) > excl_width
        positions_strip = positions_strip[mask_exclude_boundary]
        stresses_strip  = stresses_strip[mask_exclude_boundary]
        x_rel = x_rel[mask_exclude_boundary]
        y_rel = y_rel[mask_exclude_boundary]

        n_particles = len(positions_strip)
        print(f"  排除 inclusion 边界附近粒子后剩余: {n_particles} (dx={dx:.4f}, 排除带宽=±{excl_width:.4f} [{boundary_exclusion}×dx])")

        ana_sig_xx, ana_sig_yy = get_analytical_stress_cartesian(
            x_rel, y_rel, radius, delta,
            E_in, nu_in, E_out, nu_out
        )

        # 提取MPM结果
        mpm_sig_xx = stresses_strip[:, 0, 0]
        mpm_sig_yy = stresses_strip[:, 1, 1]

        # 计算误差（绝对值）
        error_xx = np.abs(mpm_sig_xx - ana_sig_xx)
        error_yy = np.abs(mpm_sig_yy - ana_sig_yy)

        # 计算积分误差（求和）并归一化（除以粒子数量）
        integrated_error_xx = np.sum(error_xx) / n_particles
        integrated_error_yy = np.sum(error_yy) / n_particles
        integrated_error_total = (integrated_error_xx + integrated_error_yy) / 2.0

        print(f"  粒子数: {n_particles}")
        print(f"  归一化积分误差 σ_xx: {integrated_error_xx:.2e}")
        print(f"  归一化积分误差 σ_yy: {integrated_error_yy:.2e}")
        print(f"  归一化积分误差 总计: {integrated_error_total:.2e}")

        grid_sizes.append(grid_size)
        integrated_errors_xx.append(integrated_error_xx)
        integrated_errors_yy.append(integrated_error_yy)
        integrated_errors_total.append(integrated_error_total)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.plot(grid_sizes, integrated_errors_xx, '-', linewidth=1.0,
             label=r'$\sigma_{xx}$', color='#3182bd')
    ax1.plot(grid_sizes, integrated_errors_yy, '-', linewidth=1.0,
             label=r'$\sigma_{yy}$', color='#2ca02c')
    ax1.set_xlabel(r'Grid size')
    ax1.set_ylabel(r'Normalized integrated error (Pa)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.plot(grid_sizes, integrated_errors_total, '-', linewidth=1.0,
             color='#b2182b', label='Average error')
    ax2.set_xlabel(r'Grid size')
    ax2.set_ylabel(r'Normalized integrated error (Pa)')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    error_plot_path = os.path.join(output_dir, f"integrated_error_comparison{file_suffix}.pdf")
    plt.savefig(error_plot_path)
    plt.close()

    print(f"积分误差对比图已保存到: {error_plot_path}")

    # 创建收敛率图 (log-log plot)
    # 计算网格间距 dx
    dx_values = [params['domain_h'] / gs for gs in grid_sizes]
    
    _, ax_conv = plt.subplots(1, 1, figsize=(5.5, 4.5))

    ax_conv.loglog(dx_values, integrated_errors_total, '-', linewidth=1.0,
                   color='#b2182b', label='Total error')

    log_dx = np.log(dx_values)
    log_error = np.log(integrated_errors_total)
    coeffs = np.polyfit(log_dx, log_error, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    dx_fit = np.array([min(dx_values), max(dx_values)])
    error_fit = np.exp(intercept) * dx_fit**slope
    ax_conv.loglog(dx_fit, error_fit, '--', linewidth=1.5, color='gray',
                   label=rf'Slope $= {slope:.2f}$')

    ax_conv.set_xlabel(r'Grid spacing $dx$ (m)')
    ax_conv.set_ylabel(r'Normalized integrated error (Pa)')
    ax_conv.minorticks_on()
    ax_conv.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    convergence_plot_path = os.path.join(output_dir, f"convergence_rate{file_suffix}.pdf")
    plt.savefig(convergence_plot_path)
    plt.close()
    
    print(f"收敛率图已保存到: {convergence_plot_path}")
    print(f"收敛率 (斜率): {slope:.3f}")

    # 保存数值数据到JSON
    error_data = {
        'grid_sizes': grid_sizes,
        'dx_values': [float(x) for x in dx_values],
        'integrated_errors_xx': [float(x) for x in integrated_errors_xx],
        'integrated_errors_yy': [float(x) for x in integrated_errors_yy],
        'integrated_errors_total': [float(x) for x in integrated_errors_total],
        'convergence_rate': float(slope)
    }

    error_data_path = os.path.join(output_dir, f"integrated_error_data{file_suffix}.json")
    with open(error_data_path, 'w') as f:
        json.dump(error_data, f, indent=4)

    print(f"积分误差数据已保存到: {error_data_path}")

    # 加载各 grid 的计时数据并打印汇总表
    _result_dirs = results.get('result_dirs', {})
    timing_list = []
    for _gs in grid_sizes:
        _t = None
        _gd = _result_dirs.get(_gs) or _result_dirs.get(str(_gs))
        if _gd:
            _tp = os.path.join(_gd, 'timing.json')
            if os.path.exists(_tp):
                with open(_tp) as _f:
                    _t = json.load(_f)
        timing_list.append(_t)

    _use_schwarz = (timing_list[0].get('mode') == 'schwarz'
                    if timing_list and timing_list[0] else False)
    print_results_table(grid_sizes, dx_values,
                        integrated_errors_xx, integrated_errors_yy,
                        integrated_errors_total, slope,
                        timing_list, _use_schwarz)

    return error_plot_path


def create_stress_distribution_plots(results, output_dir, file_suffix=''):
    """
    画最细网格的全域应力分布图（σ_xx, σ_yy, σ_xy），CMAME 风格。
    """
    experiment_data = results['experiment_data']
    if not experiment_data:
        return

    finest_grid = max(experiment_data.keys())
    exp_data = experiment_data[finest_grid]
    params    = exp_data['params']

    # 应力分布图优先使用双域原始叠加数据（避免合并时D2空洞出现空白）
    has_two_domains = (exp_data.get('positions1') is not None and
                       exp_data.get('positions2') is not None)

    domain_w = params['domain_w']

    # 只显示 [0.3, 0.7] × [0.3, 0.7] 区域
    plot_xmin, plot_xmax = 0.3, 0.7
    plot_ymin, plot_ymax = 0.3, 0.7
    plot_span = plot_xmax - plot_xmin
    pts_per_unit_crop = 5.5 * 72.0 / plot_span

    def _crop(pos):
        return ((pos[:, 0] >= plot_xmin) & (pos[:, 0] <= plot_xmax) &
                (pos[:, 1] >= plot_ymin) & (pos[:, 1] <= plot_ymax))

    def _scatter(ax, pos, vals, vmin, vmax, particle_pitch):
        # 用每个粒子的实际覆盖尺度来设定方形 marker，减少双域不同分辨率叠加时的锯齿感。
        side_pts = max(1.0, particle_pitch * pts_per_unit_crop * 1.05)
        s = side_pts ** 2
        return ax.scatter(pos[:, 0], pos[:, 1], c=vals, cmap='turbo',
                          vmin=vmin, vmax=vmax, s=s, linewidths=0,
                          marker='s', rasterized=True)

    components = [
        ((0, 0), r'$\sigma_{xx}$ (Pa)', f'stress_dist_xx{file_suffix}.pdf'),
        ((1, 1), r'$\sigma_{yy}$ (Pa)', f'stress_dist_yy{file_suffix}.pdf'),
        ((0, 1), r'$\sigma_{xy}$ (Pa)', f'stress_dist_xy{file_suffix}.pdf'),
    ]

    if has_two_domains:
        positions1 = exp_data['positions1']
        stresses1  = exp_data['stresses1']
        positions2 = exp_data['positions2']
        stresses2  = exp_data['stresses2']
        d1_pitch = params.get('d1_particle_pitch')
        d2_pitch = params.get('d2_particle_pitch')
        if d1_pitch is None:
            d1_pitch = params['domain_h'] / finest_grid
        if d2_pitch is None:
            d2_pitch = domain_w / finest_grid
        mask1 = _crop(positions1)
        mask2 = _crop(positions2)
    else:
        positions_all = exp_data['positions']
        stresses_all  = exp_data['stresses']
        particle_pitch = params.get('particle_pitch', domain_w / finest_grid)
        mask_all = _crop(positions_all)

    def _draw_boundary_labels(ax):
        """在两层虚线圆附近标注 Domain1 / intersection / Domain2。"""
        if params.get('d1_hole_center') is None or params.get('d2_boundary_center') is None:
            return

        import matplotlib.patheffects as pe

        cx, cy = params['d2_boundary_center']
        r_outer = params['d2_boundary_radius']
        r_inner = params['d1_hole_radius']
        text_kwargs = dict(
            ha='center',
            va='center',
            fontsize=8.5,
            color='black',
            path_effects=[pe.withStroke(linewidth=3.5, foreground='white')],
            zorder=8,
        )
        # 三行都放在图内，沿着中心线向下排列
        ax.text(cx, cy - r_inner + 0.010, 'domain1 boundary', **text_kwargs)
        ax.text(cx, cy - 0.5 * (r_inner + r_outer), 'intersection', **text_kwargs)
        ax.text(cx, cy - r_outer - 0.007, 'domain2 boundary', **text_kwargs)

    print(f"\n创建应力分布图 (finest grid={finest_grid})...")
    for (i, j), clabel, fname in components:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5 + 0.8))
        sc = None

        if has_two_domains:
            vals1_crop = stresses1[mask1, i, j]
            vals2_crop = stresses2[mask2, i, j]
            vmax = float(np.max(np.abs(np.concatenate([vals1_crop, vals2_crop])))) if (mask1.any() or mask2.any()) else 1.0
            # D1 底层，D2 覆盖
            if mask1.any():
                sc = _scatter(ax, positions1[mask1], vals1_crop, -vmax, vmax, d1_pitch)
            if mask2.any():
                sc = _scatter(ax, positions2[mask2], vals2_crop, -vmax, vmax, d2_pitch)
        else:
            vals_crop = stresses_all[mask_all, i, j]
            vmax = float(np.max(np.abs(vals_crop))) if mask_all.any() else 1.0
            sc = _scatter(ax, positions_all[mask_all], vals_crop, -vmax, vmax, particle_pitch)
        if sc is None:
            sc = matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax),
                cmap='turbo'
            )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label(clabel)
        # 画两层虚线圆：Domain1 空洞边界和 Domain2 外边界
        if params.get('d1_hole_center') is not None:
            circle1 = plt.Circle(params['d1_hole_center'], params['d1_hole_radius'],
                                 fill=False, linestyle='--', linewidth=1.0,
                                 edgecolor='black', zorder=6)
            ax.add_patch(circle1)
        if params.get('d2_boundary_center') is not None:
            circle2 = plt.Circle(params['d2_boundary_center'], params['d2_boundary_radius'],
                                 fill=False, linestyle='--', linewidth=1.0,
                                 edgecolor='black', zorder=6)
            ax.add_patch(circle2)
        _draw_boundary_labels(ax)
        ax.set_xlim(plot_xmin, plot_xmax)
        ax.set_ylim(plot_ymin, plot_ymax)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$y$ (m)')
        # 统一 x/y 刻度间距
        import matplotlib.ticker as ticker
        raw = plot_span / 4.0
        mag = 10 ** np.floor(np.log10(raw))
        tick_interval = round(raw / mag) * mag
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"  已保存: {fname}")


def generate_comparison_plots(results, output_dir, stress_source='particle',
                              yy_only=False, boundary_exclusion=1.0):
    """生成某一种应力来源对应的一整套比较图。"""
    file_suffix = _stress_source_suffix(stress_source)
    source_label = '粒子应力' if stress_source == 'particle' else '网格应力'
    print(f"\n生成 {source_label} 对应的比较图...")
    create_summary_plot(results, output_dir, yy_only=yy_only,
                        boundary_exclusion=boundary_exclusion, file_suffix=file_suffix)
    create_integrated_error_plot(results, output_dir,
                                 boundary_exclusion=boundary_exclusion,
                                 file_suffix=file_suffix)
    create_stress_distribution_plots(results, output_dir, file_suffix=file_suffix)


def print_results_table(grid_sizes, dx_values, integrated_errors_xx,
                        integrated_errors_yy, integrated_errors_total,
                        convergence_rate, timing_list=None, use_schwarz=False):
    has_timing = timing_list and any(t is not None for t in timing_list)
    if use_schwarz:
        W = 104
        hdr = (f"  {'Grid':>6}  {'dx':>8}  {'Err_xx(Pa)':>12}  {'Err_yy(Pa)':>12}"
               f"  {'Total(Pa)':>12}  {'BigDom(s)':>10}  {'SmDom(s)':>10}"
               f"  {'Other(s)':>9}  {'Total(s)':>9}")
    else:
        W = 88
        hdr = (f"  {'Grid':>6}  {'dx':>8}  {'Err_xx(Pa)':>12}  {'Err_yy(Pa)':>12}"
               f"  {'Total(Pa)':>12}  {'Solve(s)':>9}  {'Other(s)':>9}  {'Total(s)':>9}")

    print("\n" + "=" * W)
    print(f"  Eshelby 夹杂实验结果汇总 ({'Schwarz' if use_schwarz else 'Single Domain'})")
    print("=" * W)
    print(hdr)
    print("-" * W)

    for i, (gs, dx, exx, eyy, etot) in enumerate(zip(
            grid_sizes, dx_values, integrated_errors_xx,
            integrated_errors_yy, integrated_errors_total)):
        t = timing_list[i] if (has_timing and timing_list) else None
        if use_schwarz:
            ts = (f"  {t['big_domain_solve_time']:>10.2f}  {t['small_domain_solve_time']:>10.2f}"
                  f"  {t['other_time']:>9.2f}  {t['total_time']:>9.2f}") if t else \
                 f"  {'N/A':>10}  {'N/A':>10}  {'N/A':>9}  {'N/A':>9}"
        else:
            ts = (f"  {t['solve_time']:>9.2f}  {t['other_time']:>9.2f}"
                  f"  {t['total_time']:>9.2f}") if t else \
                 f"  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}"
        print(f"  {gs:>6}  {dx:>8.4f}  {exx:>12.3e}  {eyy:>12.3e}  {etot:>12.3e}{ts}")

    print("-" * W)
    print(f"  收敛率 (log-log Total vs dx): {convergence_rate:.3f}")
    print("=" * W + "\n")

def load_experiment_results(output_dir, stress_source='particle'):
    """
    从已有的实验输出目录加载结果数据
    
    参数:
        output_dir: 之前运行实验创建的输出目录
        stress_source: 'particle' 或 'grid'，决定读取哪类 stress 数据
    
    返回:
        dict: 实验结果字典，格式与run_batch_experiments返回的results相同
    """
    print(f"从目录加载实验结果: {output_dir}")
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")
    
    # 查找所有grid_*子目录
    grid_dirs = [d for d in os.listdir(output_dir) 
                 if d.startswith('grid_') and os.path.isdir(os.path.join(output_dir, d))]
    
    if not grid_dirs:
        raise RuntimeError(f"未找到grid_*子目录: {output_dir}")
    
    print(f"找到 {len(grid_dirs)} 个实验结果目录")
    
    results = {
        'successful_runs': [],
        'failed_runs': [],
        'result_dirs': {},
        'experiment_data': {}
    }
    
    for grid_dir in sorted(grid_dirs):
        # 提取网格大小
        try:
            grid_size = int(grid_dir.replace('grid_', ''))
        except ValueError:
            print(f"警告: 无法解析网格大小: {grid_dir}")
            continue
        
        grid_path = os.path.join(output_dir, grid_dir)
        print(f"加载网格大小 {grid_size} 的数据...")
        
        try:
            source_for_grid = stress_source
            grid_meta = grid_meta1 = grid_meta2 = None
            # 加载保存的数据
            config = load_config(os.path.join(grid_path, 'config_backup.json'))

            # 检测是否为Schwarz配置
            use_schwarz = 'Domain2' in config

            # 对于 Schwarz 配置，可优先加载 grid stress；否则回退到粒子 stress
            positions1 = positions2 = stresses1 = stresses2 = None
            if source_for_grid == 'grid':
                grid_payload = _load_grid_stress_from_result_dir(grid_path, config, use_schwarz=use_schwarz)
                if grid_payload is not None:
                    positions = grid_payload['positions']
                    stresses = grid_payload['stresses']
                    positions1 = grid_payload.get('positions1')
                    stresses1 = grid_payload.get('stresses1')
                    positions2 = grid_payload.get('positions2')
                    stresses2 = grid_payload.get('stresses2')
                    grid_meta = grid_payload.get('grid_meta')
                    grid_meta1 = grid_payload.get('grid_meta1')
                    grid_meta2 = grid_payload.get('grid_meta2')
                    if use_schwarz:
                        print(f"  网格应力文件已加载 (D1:{len(positions1)}, D2:{len(positions2)}，用于比较)")
                    else:
                        print(f"  网格应力文件已加载 (N:{len(positions)}，用于比较)")
                else:
                    print(f"  警告: 未找到 grid stress 文件，回退到粒子 stress: {grid_path}")
                    source_for_grid = 'particle'

            if source_for_grid == 'particle':
                # 对于Schwarz配置：若有domain1/domain2分域文件，用KD-tree重新合并（保留D2空洞中的D1粒子）
                d1_path = os.path.join(grid_path, 'domain1_positions.npy')
                d2_path = os.path.join(grid_path, 'domain2_positions.npy')
                positions = np.load(os.path.join(grid_path, 'positions.npy'))
                stresses = np.load(os.path.join(grid_path, 'stresses.npy'))

                # 额外加载分域文件（仅用于应力分布图可视化）
                if use_schwarz and os.path.exists(d1_path) and os.path.exists(d2_path):
                    positions1 = np.load(d1_path)
                    stresses1 = np.load(os.path.join(grid_path, 'domain1_stresses.npy'))
                    positions2 = np.load(d2_path)
                    stresses2 = np.load(os.path.join(grid_path, 'domain2_stresses.npy'))
                    print(f"  分域文件已加载 (D1:{len(positions1)}, D2:{len(positions2)}，仅用于应力分布图)")
            
            if use_schwarz:
                domain_config = config.get('Domain2', {})
                domain1_config = config.get('Domain1', {})
                domain_w = domain_config.get('domain_width', 0.3)
                domain_h = domain_config.get('domain_height', 0.3)
                offset = np.array(domain_config.get('offset', [0.35, 0.35]))
            else:
                domain_config = config
                domain1_config = None
                domain_w = config.get('domain_width', 1.0)
                domain_h = config.get('domain_height', 1.0)
                offset = np.array([0.0, 0.0])
            
            # 查找inclusion
            center = None
            radius = 0.15
            d2_boundary_center = None
            d2_boundary_radius = None
            
            for shape in domain_config.get('shapes', []):
                if shape['type'] == 'ellipse' and shape['operation'] == 'add' and d2_boundary_center is None:
                    local_center = np.array(shape['params']['center'])
                    d2_boundary_center = local_center + offset
                    d2_boundary_radius = shape['params']['semi_axes'][0]
                if shape['type'] == 'ellipse' and shape['operation'] == 'change':
                    local_center = np.array(shape['params']['center'])
                    center = local_center + offset
                    radius = shape['params']['semi_axes'][0]
                    break
            
            if center is None:
                center = np.array([domain_w/2.0, domain_h/2.0]) + offset

            # 查找Domain1中被减去的圆（空洞边界）
            d1_hole_center = None
            d1_hole_radius = None
            if use_schwarz:
                d1_config = config.get('Domain1', {})
                d1_offset = np.array(d1_config.get('offset', [0.0, 0.0]))
                for shape in d1_config.get('shapes', []):
                    if shape['type'] == 'ellipse' and shape['operation'] == 'subtract':
                        d1_hole_center = np.array(shape['params']['center']) + d1_offset
                        d1_hole_radius = shape['params']['semi_axes'][0]
                        break

            # 提取材料参数
            mat_params = domain_config.get('material_params', config.get('material_params', []))
            matrix_mat = next(m for m in mat_params if m['id'] == 0)
            E_out = matrix_mat['E']
            nu_out = matrix_mat['nu']
            
            inclusion_mat = next(m for m in mat_params if m['id'] == 1)
            E_in = inclusion_mat['E']
            nu_in = inclusion_mat['nu']
            
            initial_F = inclusion_mat.get('initial_F', [[1.0, 0],[0, 1.0]])
            delta = 1.0 - initial_F[0][0]

            # 根据数据源设置应力图标记粒度
            if source_for_grid == 'grid':
                if use_schwarz and grid_meta1 is not None and grid_meta2 is not None:
                    d1_particle_pitch = float(grid_meta1['dx_x'])
                    d2_particle_pitch = float(grid_meta2['dx_x'])
                    particle_pitch = None
                elif grid_meta is not None:
                    particle_pitch = float(grid_meta['dx_x'])
                    d1_particle_pitch = None
                    d2_particle_pitch = None
                else:
                    particle_pitch = domain_w / domain_config.get('grid_nx', grid_size) / \
                                     np.sqrt(domain_config.get('particles_per_grid', 1))
                    d1_particle_pitch = None
                    d2_particle_pitch = None
            else:
                particle_pitch = domain_w / domain_config.get('grid_nx', grid_size) / \
                                 np.sqrt(domain_config.get('particles_per_grid', 1))
                d1_particle_pitch = (
                    domain1_config.get('domain_width', 1.0) / domain1_config.get('grid_nx', 1) /
                    np.sqrt(domain1_config.get('particles_per_grid', 1))
                ) if use_schwarz and domain1_config is not None else None
                d2_particle_pitch = (
                    domain_config.get('domain_width', domain_w) / domain_config.get('grid_nx', grid_size) /
                    np.sqrt(domain_config.get('particles_per_grid', 1))
                ) if use_schwarz else None
            
            # 计算解析解（用于生成汇总图）
            line_x = np.linspace(center[0] - 3*radius, center[0] + 3*radius, 500)
            line_y = np.full_like(line_x, center[1])
            
            ana_sig_xx, ana_sig_yy = get_analytical_stress_cartesian(
                line_x - center[0],
                line_y - center[1],
                radius, delta,
                E_in, nu_in, E_out, nu_out
            )
            
            # 构建实验数据
            exp_data = {
                'positions': positions,
                'stresses': stresses,
                'positions1': positions1,
                'stresses1': stresses1,
                'positions2': positions2,
                'stresses2': stresses2,
                'analytical': {
                    'line_x': line_x,
                    'line_y': line_y,
                    'ana_sig_xx': ana_sig_xx,
                    'ana_sig_yy': ana_sig_yy
                },
                'params': {
                    'center': center,
                    'radius': radius,
                    'delta': delta,
                    'E_in': E_in,
                    'nu_in': nu_in,
                    'E_out': E_out,
                    'nu_out': nu_out,
                    'domain_w': domain_w,
                    'domain_h': domain_h,
                    'use_schwarz': use_schwarz,
                    'offset': offset,
                    'd1_hole_center': d1_hole_center,
                    'd1_hole_radius': d1_hole_radius,
                    'd2_boundary_center': d2_boundary_center,
                    'd2_boundary_radius': d2_boundary_radius,
                    'stress_source': source_for_grid,
                    'particle_pitch': particle_pitch,
                    'd1_particle_pitch': d1_particle_pitch,
                    'd2_particle_pitch': d2_particle_pitch,
                }
            }
            
            results['successful_runs'].append(grid_size)
            results['result_dirs'][grid_size] = grid_path
            results['experiment_data'][grid_size] = exp_data
            
            print(f"  ✓ 成功加载 {len(positions)} 个粒子的数据")
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            results['failed_runs'].append(grid_size)
    
    print(f"\n成功加载 {len(results['successful_runs'])} 个实验的数据")
    if results['failed_runs']:
        print(f"加载失败的网格大小: {results['failed_runs']}")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='实验1: Eshelby 夹杂/过盈配合测试 - 支持批量网格分辨率实验')

    # 基础参数
    parser.add_argument('--config', default="config/config_2d_test1.json",
                       help='配置文件路径')
    parser.add_argument('--output-dir', default="experiments/test1_v3",
                       help='输出目录')

    # Schwarz模式
    parser.add_argument('--schwarz', action='store_true',
                       help='使用Schwarz域分解求解器')
    parser.add_argument('--modify-domain1', action='store_true',
                       help='Schwarz批量模式下同步缩放Domain1的分辨率（默认不修改Domain1）')

    # 分析模式参数
    parser.add_argument('--analyze-only', action='store_true',
                       help='仅分析模式：不运行模拟，只从现有结果目录生成汇总图和误差图')
    parser.add_argument('--summary-yy-only', action='store_true',
                       help='汇总对比图仅绘制yy方向应力')
    parser.add_argument('--boundary-exclusion', type=float, default=0.0,
                       help='排除 inclusion 边界附近粒子的宽度，单位为 grid size 倍数 (默认: 0.0)')
    parser.add_argument('--stress-source', choices=['particle', 'grid', 'both'], default='both',
                       help='比较时使用粒子应力、网格应力或两者都生成 (默认: both)')

    # 批量模式参数
    parser.add_argument('--batch-mode', action='store_true',
                       help='启用批量模式：运行多个不同网格分辨率的实验')
    parser.add_argument('--grid-start', type=int, default=30,
                       help='批量模式（grid）：起始网格大小 (默认: 30)')
    parser.add_argument('--grid-end', type=int, default=90,
                       help='批量模式（grid）：结束网格大小 (默认: 90)')
    parser.add_argument('--grid-step', type=int, default=15,
                       help='批量模式（grid）：网格大小步长 (默认: 15)')
    parser.add_argument('--dx-start', type=float, default=None,
                       help='批量模式（dx）：起始网格间距，提供后优先于 --grid-* 参数')
    parser.add_argument('--dx-end', type=float, default=None,
                       help='批量模式（dx）：结束网格间距（含）')
    parser.add_argument('--dx-step', type=float, default=None,
                       help='批量模式（dx）：网格间距步长')

    args = parser.parse_args()

    # 如果使用Schwarz模式但没有指定配置文件，使用Schwarz默认配置
    if args.schwarz and args.config == "config/config_2d_test1.json":
        args.config = "config/schwarz_2d_test1.json"
        print(f"Schwarz模式：使用默认配置 {args.config}")

    # 仅分析模式
    if args.analyze_only:
        print("="*80)
        print("仅分析模式：从现有结果生成汇总图和误差图")
        print(f"结果目录: {args.output_dir}")
        print("="*80)
        
        sources = ['particle', 'grid'] if args.stress_source == 'both' else [args.stress_source]
        for source in sources:
            results = load_experiment_results(args.output_dir, stress_source=source)
            if results['successful_runs']:
                generate_comparison_plots(
                    results,
                    args.output_dir,
                    stress_source=source,
                    yy_only=args.summary_yy_only,
                    boundary_exclusion=args.boundary_exclusion,
                )
            else:
                print(f"错误：{source} 来源没有可用的实验数据")

        print(f"\n分析完成！图表已保存到: {args.output_dir}")
        
        return

    if args.batch_mode:
        # 批量模式
        print("=" * 80)
        mode_str = "Schwarz域分解" if args.schwarz else "单域"
        print(f"批量{mode_str}网格分辨率实验模式")
        print(f"网格范围: {args.grid_start} - {args.grid_end}, 步长: {args.grid_step}")
        print("=" * 80)

        results = run_batch_experiments(args)

        sources = ['particle', 'grid'] if args.stress_source == 'both' else [args.stress_source]
        for source in sources:
            loaded_results = load_experiment_results(args.output_dir, stress_source=source)
            if loaded_results['successful_runs']:
                generate_comparison_plots(
                    loaded_results,
                    args.output_dir,
                    stress_source=source,
                    yy_only=args.summary_yy_only,
                    boundary_exclusion=args.boundary_exclusion,
                )
            else:
                print(f"警告: {source} 来源没有可用数据，跳过对应图表")

        print(f"\n批量实验全部完成!")
        print(f"结果目录: {args.output_dir}")

    else:
        # 单次运行模式（原始功能）
        print("=" * 80)
        mode_str = "Schwarz域分解" if args.schwarz else "单域"
        print(f"{mode_str}单次实验模式")
        print("=" * 80)

        run_single_experiment(
            args.config,
            None,
            args.output_dir,
            use_schwarz=args.schwarz,
            stress_source=args.stress_source,
        )

if __name__ == "__main__":
    main()
