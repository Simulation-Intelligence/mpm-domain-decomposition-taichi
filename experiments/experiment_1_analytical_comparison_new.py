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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid segfault with Taichi
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

def modify_config_grid_size(config, grid_size, use_schwarz=False):
    """
    修改配置文件的网格分辨率

    参数:
        config: 配置字典
        grid_size: 新的网格大小
        use_schwarz: 是否为Schwarz双域配置

    返回:
        修改后的配置字典
    """
    import copy
    new_config = copy.deepcopy(config)

    if use_schwarz:
        # Schwarz模式：只修改Domain2的网格大小
        if 'Domain2' in new_config:
            original_nx = new_config['Domain2'].get('grid_nx', 30)
            original_ny = new_config['Domain2'].get('grid_ny', 30)
            aspect_ratio = original_ny / original_nx if original_nx > 0 else 1.0

            new_config['Domain2']['grid_nx'] = grid_size
            new_config['Domain2']['grid_ny'] = int(grid_size * aspect_ratio)

            print(f"修改Schwarz配置: Domain2网格大小设为 {grid_size}x{int(grid_size * aspect_ratio)} (比例: {aspect_ratio:.3f})")
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
            "--config", config_path,
            "--no-gui"
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

        return (positions1, stresses1, positions2, stresses2)
    else:
        # 单域模式：调用 implicit_mpm.py
        cmd = [
            sys.executable,
            "simulators/implicit_mpm.py",
            "--config", config_path
        ]
        print(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("模拟失败！")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Simulation failed with code {result.returncode}")

        print("模拟完成，加载结果数据...")

        # 单域模式需要从experiment_results/single_domain_<timestamp>加载
        # 找到最新的实验目录
        exp_results_dir = "experiment_results"
        single_domain_dirs = [d for d in os.listdir(exp_results_dir)
                             if d.startswith("single_domain_") and
                             os.path.isdir(os.path.join(exp_results_dir, d))]

        if not single_domain_dirs:
            raise RuntimeError("未找到单域模式的实验结果目录")

        # 按时间戳排序，获取最新的
        single_domain_dirs.sort()
        latest_exp_dir = os.path.join(exp_results_dir, single_domain_dirs[-1])
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

        return (positions, stresses)

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

    # 从Domain1中排除Domain2范围内的粒子
    mask_outside_d2 = ~((positions1[:, 0] >= d2_xmin) & (positions1[:, 0] <= d2_xmax) &
                        (positions1[:, 1] >= d2_ymin) & (positions1[:, 1] <= d2_ymax))
    
    positions1_filtered = positions1[mask_outside_d2]
    stresses1_filtered = stresses1[mask_outside_d2]

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

    # 绘图
    plt.figure(figsize=(12, 5))
    title_suffix = f" (Grid {grid_size}x{grid_size})" if grid_size else ""

    # 左图: Sigma_XX
    plt.subplot(1, 2, 1)
    plt.title(f"Transverse Stress $\sigma_{{xx}}${title_suffix}")
    plt.scatter(mpm_x - center[0], mpm_sig_xx, s=10, alpha=0.6, label='MPM', c='blue', edgecolors='none')
    plt.plot(line_x - center[0], ana_sig_xx, 'r--', lw=2, label='Analytical')
    plt.axvline(-radius, color='k', linestyle=':', alpha=0.3, label='Inclusion boundary')
    plt.axvline(radius, color='k', linestyle=':', alpha=0.3)
    plt.xlabel('X position (relative to center)')
    plt.ylabel('Stress XX (Pa)')
    plt.xlim(-3*radius, 3*radius)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图: Sigma_YY
    plt.subplot(1, 2, 2)
    plt.title(f"Radial Stress $\sigma_{{yy}}${title_suffix}")
    plt.scatter(mpm_x - center[0], mpm_sig_yy, s=10, alpha=0.6, label='MPM', c='green', edgecolors='none')
    plt.plot(line_x - center[0], ana_sig_yy, 'r--', lw=2, label='Analytical')
    plt.axvline(-radius, color='k', linestyle=':', alpha=0.3, label='Inclusion boundary')
    plt.axvline(radius, color='k', linestyle=':', alpha=0.3)
    plt.xlabel('X position (relative to center)')
    plt.ylabel('Stress YY (Pa)')
    plt.xlim(-3*radius, 3*radius)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图形
    save_path = os.path.join(output_dir, 'result_v3_cartesian.png')
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.9, wspace=0.25)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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

def run_single_experiment(config_path, grid_size, output_dir, use_schwarz=False):
    """
    运行单个实验

    参数:
        config_path: 配置文件路径
        grid_size: 网格大小（可选，None表示使用配置文件中的值）
        output_dir: 输出目录
        use_schwarz: 是否使用Schwarz求解器

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

    # 分析和绘图
    if use_schwarz:
        results = analyze_and_plot(sim_results, None, config, output_dir, grid_size, use_schwarz=True)
    else:
        positions, stresses = sim_results
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
            modified_config = modify_config_grid_size(base_config, grid_size, use_schwarz=args.schwarz)

            # 2. 创建临时配置文件
            temp_config_path = os.path.join(args.output_dir, f"temp_config_grid{grid_size}.json")
            save_config(modified_config, temp_config_path)

            # 3. 创建输出子目录
            grid_output_dir = os.path.join(args.output_dir, f"grid_{grid_size}")

            # 4. 运行单个实验
            exp_results = run_single_experiment(temp_config_path, grid_size, grid_output_dir, use_schwarz=args.schwarz)

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

def create_summary_plot(results, output_dir):
    """
    创建汇总图，将所有网格大小的结果绘制在一张图上

    参数:
        results: 批量实验结果字典
        output_dir: 输出目录
    """
    print("\n创建汇总对比图...")

    experiment_data = results['experiment_data']
    if not experiment_data:
        print("警告: 没有可用的实验数据")
        return

    # 定义颜色列表
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

    # 创建图形
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 获取第一个实验的解析解和参数（所有实验的解析解应该相同）
    first_grid = list(experiment_data.keys())[0]
    first_exp = experiment_data[first_grid]
    analytical = first_exp['analytical']
    params = first_exp['params']

    line_x = analytical['line_x']
    ana_sig_xx = analytical['ana_sig_xx']
    ana_sig_yy = analytical['ana_sig_yy']
    center = params['center']
    radius = params['radius']

    # 左图: Sigma_XX
    ax1.set_title(f"Transverse Stress $\sigma_{{xx}}$ - All Grid Sizes", fontsize=14)
    # 坐标平移：使中心点变为0
    ax1.plot(line_x - center[0], ana_sig_xx, 'r--', lw=3, label='Analytical', zorder=10, alpha=0.9)

    # 右图: Sigma_YY
    ax2.set_title(f"Radial Stress $\sigma_{{yy}}$ - All Grid Sizes", fontsize=14)
    # 坐标平移：使中心点变为0
    ax2.plot(line_x - center[0], ana_sig_yy, 'r--', lw=3, label='Analytical', zorder=10, alpha=0.9)

    # 绘制所有网格大小的MPM结果
    for i, (grid_size, exp_data) in enumerate(sorted(experiment_data.items())):
        color = colors[i % len(colors)]

        # 提取截面数据
        positions = exp_data['positions']
        stresses = exp_data['stresses']
        
        # 使用统一的strip_width计算函数
        strip_width = get_strip_width_from_params(params['domain_h'], grid_size, radius)
        mask_strip = np.abs(positions[:, 1] - center[1]) < strip_width

        print(f"网格 {grid_size}x{grid_size}: 提取截面数据: {np.sum(mask_strip)} 个粒子")
        mpm_x = positions[mask_strip, 0]
        mpm_sig_xx = stresses[mask_strip, 0, 0]
        mpm_sig_yy = stresses[mask_strip, 1, 1]

        label = f'Grid {grid_size}x{grid_size}'

        # Sigma_XX (坐标平移：使中心点变为0)
        ax1.scatter(mpm_x - center[0], mpm_sig_xx, s=15, alpha=0.7, c=color, label=label, edgecolors='none')

        # Sigma_YY (坐标平移：使中心点变为0)
        ax2.scatter(mpm_x - center[0], mpm_sig_yy, s=15, alpha=0.7, c=color, label=label, edgecolors='none')

    # 添加辅助线（相对坐标）
    for ax in [ax1, ax2]:
        ax.axvline(-radius, color='k', linestyle=':', alpha=0.3, label='Inclusion boundary' if ax == ax1 else '')
        ax.axvline(radius, color='k', linestyle=':', alpha=0.3)
        ax.set_xlabel('X position (relative to center)', fontsize=12)
        ax.set_ylabel('Stress (Pa)', fontsize=12)
        ax.set_xlim(-3*radius, 3*radius)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存汇总图
    summary_plot_path = os.path.join(output_dir, "summary_all_grids.png")
    plt.savefig(summary_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"汇总对比图已保存到: {summary_plot_path}")

    return summary_plot_path

def create_integrated_error_plot(results, output_dir):
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
        n_particles = len(positions_strip)

        # 计算这些位置对应的解析解
        x_rel = positions_strip[:, 0] - center[0]
        y_rel = positions_strip[:, 1] - center[1]

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

    # 创建图形
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 分别显示σ_xx和σ_yy的误差
    ax1.plot(grid_sizes, integrated_errors_xx, 'o-', linewidth=2, markersize=8,
             label='$\sigma_{xx}$', color='blue')
    ax1.plot(grid_sizes, integrated_errors_yy, 's-', linewidth=2, markersize=8,
             label='$\sigma_{yy}$', color='green')
    ax1.set_xlabel('Grid Size', fontsize=12)
    ax1.set_ylabel('Normalized Integrated Error (Pa)', fontsize=12)
    ax1.set_title('Normalized Integrated Error by Stress Component', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数坐标以便更好地观察趋势

    # 右图: 总体误差
    ax2.plot(grid_sizes, integrated_errors_total, 'o-', linewidth=2, markersize=8,
             color='red', label='Average Error')
    ax2.set_xlabel('Grid Size', fontsize=12)
    ax2.set_ylabel('Normalized Integrated Error (Pa)', fontsize=12)
    ax2.set_title('Total Normalized Integrated Error', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 使用对数坐标

    plt.tight_layout()

    # 保存图形
    error_plot_path = os.path.join(output_dir, "integrated_error_comparison.png")
    plt.savefig(error_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"积分误差对比图已保存到: {error_plot_path}")

    # 创建收敛率图 (log-log plot)
    # 计算网格间距 dx
    dx_values = [params['domain_h'] / gs for gs in grid_sizes]
    
    _, ax_conv = plt.subplots(1, 1, figsize=(8, 6))
    
    # 绘制误差-网格间距关系 (log-log)
    ax_conv.loglog(dx_values, integrated_errors_total, 'o-', linewidth=2, markersize=8,
                   color='red', label='Total Error')
    
    # 拟合收敛率（在log空间中进行线性拟合）
    log_dx = np.log(dx_values)
    log_error = np.log(integrated_errors_total)
    
    # 线性拟合: log(error) = slope * log(dx) + intercept
    coeffs = np.polyfit(log_dx, log_error, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # 绘制拟合直线
    dx_fit = np.array([min(dx_values), max(dx_values)])
    error_fit = np.exp(intercept) * dx_fit**slope
    ax_conv.loglog(dx_fit, error_fit, '--', linewidth=2, color='gray',
                   label=f'Slope = {slope:.2f}')
    
    ax_conv.set_xlabel('Grid Spacing $dx$ (m)', fontsize=12)
    ax_conv.set_ylabel('Normalized Integrated Error (Pa)', fontsize=12)
    ax_conv.set_title('Convergence Rate (log-log scale)', fontsize=14)
    ax_conv.legend(fontsize=11)
    ax_conv.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # 保存收敛率图
    convergence_plot_path = os.path.join(output_dir, "convergence_rate.png")
    plt.savefig(convergence_plot_path, dpi=200, bbox_inches='tight')
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

    error_data_path = os.path.join(output_dir, "integrated_error_data.json")
    with open(error_data_path, 'w') as f:
        json.dump(error_data, f, indent=4)

    print(f"积分误差数据已保存到: {error_data_path}")

    return error_plot_path

def load_experiment_results(output_dir):
    """
    从已有的实验输出目录加载结果数据
    
    参数:
        output_dir: 之前运行实验创建的输出目录
    
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
            # 加载保存的数据
            positions = np.load(os.path.join(grid_path, 'positions.npy'))
            stresses = np.load(os.path.join(grid_path, 'stresses.npy'))
            config = load_config(os.path.join(grid_path, 'config_backup.json'))
            
            # 提取参数（与analyze_and_plot中的逻辑一致）
            # 检测是否为Schwarz配置
            use_schwarz = 'Domain2' in config
            
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
            
            # 查找inclusion
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

    # 分析模式参数
    parser.add_argument('--analyze-only', action='store_true',
                       help='仅分析模式：不运行模拟，只从现有结果目录生成汇总图和误差图')

    # 批量模式参数
    parser.add_argument('--batch-mode', action='store_true',
                       help='启用批量模式：运行多个不同网格分辨率的实验')
    parser.add_argument('--grid-start', type=int, default=20,
                       help='批量模式：起始网格大小 (默认: 40)')
    parser.add_argument('--grid-end', type=int, default=80,
                       help='批量模式：结束网格大小 (默认: 80)')
    parser.add_argument('--grid-step', type=int, default=20,
                       help='批量模式：网格大小步长 (默认: 20)')

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
        
        # 加载已有的实验结果
        results = load_experiment_results(args.output_dir)
        
        if results['successful_runs']:
            # 创建汇总图
            create_summary_plot(results, args.output_dir)
            
            # 创建积分误差对比图
            create_integrated_error_plot(results, args.output_dir)
            
            print(f"\n分析完成！图表已保存到: {args.output_dir}")
        else:
            print("错误：没有可用的实验数据")
        
        return

    if args.batch_mode:
        # 批量模式
        print("=" * 80)
        mode_str = "Schwarz域分解" if args.schwarz else "单域"
        print(f"批量{mode_str}网格分辨率实验模式")
        print(f"网格范围: {args.grid_start} - {args.grid_end}, 步长: {args.grid_step}")
        print("=" * 80)

        results = run_batch_experiments(args)

        # 创建汇总图
        # if results['successful_runs']:
        create_summary_plot(results, args.output_dir)

        # 创建积分误差对比图
        create_integrated_error_plot(results, args.output_dir)

        print(f"\n批量实验全部完成!")
        print(f"结果目录: {args.output_dir}")

    else:
        # 单次运行模式（原始功能）
        print("=" * 80)
        mode_str = "Schwarz域分解" if args.schwarz else "单域"
        print(f"{mode_str}单次实验模式")
        print("=" * 80)

        run_single_experiment(args.config, None, args.output_dir, use_schwarz=args.schwarz)

if __name__ == "__main__":
    main()