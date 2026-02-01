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
    """
    import sys
    # 1. 计算剪切模量 G
    G_in = E_in / (2 * (1 + nu_in))
    G_out = E_out / (2 * (1 + nu_out))

    # 2. 计算接触压力 P (Plane Strain 公式)
    # 分母项：外侧柔度 + 内侧柔度
    compliance_out = 1.0 / (2 * G_out)
    compliance_in = (1.0 - 2.0 * nu_in) / (2 * G_in)

    P = delta_eff / (compliance_out + compliance_in)

    print(f"Calculated Contact Pressure P = {P:.2e} Pa")
    sys.stdout.flush()

    # 3. 极坐标计算
    print(f"DEBUG ANALYTICAL: 输入参数 x_rel shape={np.shape(x_rel)}, y_rel shape={np.shape(y_rel)}, R={R}")
    sys.stdout.flush()

    try:
        r = np.sqrt(x_rel**2 + y_rel**2)
        print(f"DEBUG ANALYTICAL: r 计算完成, shape={r.shape}")
        sys.stdout.flush()

        theta = np.arctan2(y_rel, x_rel)
        print(f"DEBUG ANALYTICAL: theta 计算完成")
        sys.stdout.flush()

        r_safe = np.maximum(r, 1e-10)

        sigma_rr = np.zeros_like(r)
        sigma_tt = np.zeros_like(r)
        print(f"DEBUG ANALYTICAL: sigma数组创建完成")
        sys.stdout.flush()

        # --- 内部 (r < R) ---
        # 内部依然是均匀受压状态
        mask_in = r < R
        sigma_rr[mask_in] = -P
        sigma_tt[mask_in] = -P
        print(f"DEBUG ANALYTICAL: 内部区域应力设置完成")
        sys.stdout.flush()

        # --- 外部 (r >= R) ---
        # 外部依然遵循 Lame 解的衰减规律
        mask_out = r >= R
        r_out = r_safe[mask_out]
        sigma_rr[mask_out] = -P * (R / r_out)**2
        sigma_tt[mask_out] =  P * (R / r_out)**2
        print(f"DEBUG ANALYTICAL: 外部区域应力设置完成")
        sys.stdout.flush()

        # 4. 坐标变换 -> Cartesian
        c2 = np.cos(theta)**2
        s2 = np.sin(theta)**2
        print(f"DEBUG ANALYTICAL: 三角函数计算完成")
        sys.stdout.flush()

        sigma_xx = sigma_rr * c2 + sigma_tt * s2
        sigma_yy = sigma_rr * s2 + sigma_tt * c2
        print(f"DEBUG ANALYTICAL: Cartesian应力转换完成")
        sys.stdout.flush()

        return sigma_xx, sigma_yy
    except Exception as e:
        print(f"DEBUG ANALYTICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

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

        # 单域模式需要从stress_strain_output加载
        # TODO: 实现单域模式的数据加载
        raise NotImplementedError("单域模式subprocess方式尚未实现，请使用--schwarz模式")

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
    # --- 0. 处理Schwarz双域数据 ---
    if use_schwarz:
        import sys
        print("DEBUG: 开始处理Schwarz双域数据...")
        sys.stdout.flush()

        positions1, stresses1, positions2, stresses2 = positions_or_tuple
        print(f"DEBUG: Domain1 数据形状: positions1={positions1.shape}, stresses1={stresses1.shape}")
        print(f"DEBUG: Domain2 数据形状: positions2={positions2.shape}, stresses2={stresses2.shape}")
        sys.stdout.flush()

        # 合并数据：优先使用Domain2
        # 从config获取Domain2的范围
        domain2_config = config.get('Domain2', {})
        domain2_offset = np.array(domain2_config.get('offset', [0.35, 0.35]))
        domain2_width = domain2_config.get('domain_width', 0.3)
        domain2_height = domain2_config.get('domain_height', 0.3)

        # Domain2的全局范围
        d2_xmin, d2_xmax = domain2_offset[0], domain2_offset[0] + domain2_width
        d2_ymin, d2_ymax = domain2_offset[1], domain2_offset[1] + domain2_height

        print(f"DEBUG: Domain2 全局范围: x=[{d2_xmin}, {d2_xmax}], y=[{d2_ymin}, {d2_ymax}]")
        sys.stdout.flush()

        # 从Domain1中排除Domain2范围内的粒子
        print("DEBUG: 正在过滤Domain1数据...")
        sys.stdout.flush()

        try:
            mask_outside_d2 = ~((positions1[:, 0] >= d2_xmin) & (positions1[:, 0] <= d2_xmax) &
                                (positions1[:, 1] >= d2_ymin) & (positions1[:, 1] <= d2_ymax))
            print(f"DEBUG: 过滤mask创建成功, 保留 {np.sum(mask_outside_d2)}/{len(positions1)} 个Domain1粒子")
            sys.stdout.flush()

            positions1_filtered = positions1[mask_outside_d2]
            stresses1_filtered = stresses1[mask_outside_d2]
            print(f"DEBUG: Domain1 过滤完成")
            sys.stdout.flush()
        except Exception as e:
            print(f"DEBUG: 过滤Domain1时出错: {e}")
            sys.stdout.flush()
            raise

        # 合并Domain1(过滤后) + Domain2
        print("DEBUG: 正在合并数据...")
        sys.stdout.flush()

        try:
            positions = np.vstack([positions1_filtered, positions2])
            print(f"DEBUG: positions合并成功, 形状: {positions.shape}")
            sys.stdout.flush()

            stresses = np.vstack([stresses1_filtered, stresses2])
            print(f"DEBUG: stresses合并成功, 形状: {stresses.shape}")
            sys.stdout.flush()
        except Exception as e:
            print(f"DEBUG: 合并数据时出错: {e}")
            sys.stdout.flush()
            raise
    else:
        positions = positions_or_tuple
        stresses = stresses_or_none

    # --- 1. 参数提取 ---
    # 从Domain2配置提取参数（如果是Schwarz模式）
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

    # 查找inclusion (ellipse with "change" operation)
    center = None
    radius = 0.15

    for shape in domain_config.get('shapes', []):
        if shape['type'] == 'ellipse' and shape['operation'] == 'change':
            # Domain2中的局部坐标
            local_center = np.array(shape['params']['center'])
            # 转换为全局坐标
            center = local_center + offset
            radius = shape['params']['semi_axes'][0]
            break

    if center is None:
        # 如果没找到，使用默认值
        center = np.array([domain_w/2.0, domain_h/2.0]) + offset

    # 提取材料参数
    mat_params = domain_config.get('material_params', config.get('material_params', []))

    # 获取基体 (Matrix) - 假设 ID=0
    matrix_mat = next(m for m in mat_params if m['id'] == 0)
    E_out = matrix_mat['E']
    nu_out = matrix_mat['nu']

    # 获取夹杂 (Inclusion) - 假设 ID=1
    inclusion_mat = next(m for m in mat_params if m['id'] == 1)
    E_in = inclusion_mat['E']
    nu_in = inclusion_mat['nu']

    # 计算过盈量
    initial_F = inclusion_mat.get('initial_F', [[1.0, 0],[0, 1.0]])
    delta = 1.0 - initial_F[0][0]


    print(f"Params: Center={center}, Radius={radius}, Delta={delta}, E_in={E_in}, nu_in={nu_in}, E_out={E_out}, nu_out={nu_out}")

    # --- 2. 提取截面数据 (MPM) ---
    # 依然取中心水平线，因为这里应力变化最典型
    import sys
    print(f"提取截面数据，总粒子数: {len(positions)}")
    print(f"DEBUG: positions数据类型: {type(positions)}, 形状: {positions.shape}")
    print(f"DEBUG: stresses数据类型: {type(stresses)}, 形状: {stresses.shape}, dtype: {stresses.dtype}")
    sys.stdout.flush()

    strip_width = radius * 0.2
    print(f"DEBUG: strip_width = {strip_width}, center = {center}")
    sys.stdout.flush()

    try:
        mask_strip = np.abs(positions[:, 1] - center[1]) < strip_width
        print(f"DEBUG: 截面mask创建成功, 选中 {np.sum(mask_strip)}/{len(positions)} 个粒子")
        sys.stdout.flush()

        mpm_x = positions[mask_strip, 0]
        mpm_y = positions[mask_strip, 1]
        print(f"DEBUG: 位置数据提取成功")
        sys.stdout.flush()

        mpm_sig_xx = stresses[mask_strip, 0, 0]
        print(f"DEBUG: sig_xx 提取成功")
        sys.stdout.flush()

        mpm_sig_yy = stresses[mask_strip, 1, 1]
        print(f"DEBUG: sig_yy 提取成功")
        sys.stdout.flush()

        print(f"截面内粒子数: {len(mpm_x)}, 应力范围: sig_xx [{np.min(mpm_sig_xx):.2e}, {np.max(mpm_sig_xx):.2e}], sig_yy [{np.min(mpm_sig_yy):.2e}, {np.max(mpm_sig_yy):.2e}]")
        sys.stdout.flush()
    except Exception as e:
        print(f"DEBUG: 提取截面数据时出错: {e}")
        print(f"DEBUG: 错误类型: {type(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

    # --- 3. 计算对应的解析解 (Analytical) ---
    # 我们直接使用 MPM 粒子的坐标来计算其对应的解析解，这样点对点对比最准确
    # 或者为了画光滑曲线，我们生成一组密集的点

    # 方法 A: 画光滑理论曲线
    # 只画3倍半径范围
    line_x = np.linspace(center[0] - 3*radius, center[0] + 3*radius, 500)
    line_y = np.full_like(line_x, center[1]) # 严格在 y=center 上

    ana_sig_xx, ana_sig_yy = get_analytical_stress_cartesian(
        line_x - center[0],
        line_y - center[1],
        radius, delta,
        E_in, nu_in, E_out, nu_out
    )

    # --- 4. 绘图 ---
    print("DEBUG PLOT: 开始绘图...")
    sys.stdout.flush()

    try:
        plt.figure(figsize=(12, 5))
        print("DEBUG PLOT: Figure创建成功")
        sys.stdout.flush()

        # 标题包含网格大小信息（如果提供）
        title_suffix = f" (Grid {grid_size}x{grid_size})" if grid_size else ""

        # 左图: Sigma_XX
        plt.subplot(1, 2, 1)
        print("DEBUG PLOT: subplot(1,2,1)创建成功")
        sys.stdout.flush()

        plt.title(f"Transverse Stress $\sigma_{{xx}}${title_suffix}")
        print("DEBUG PLOT: title设置成功")
        sys.stdout.flush()

        # 坐标平移：使中心点变为0
        print(f"DEBUG PLOT: 准备scatter, mpm_x shape={mpm_x.shape}, center[0]={center[0]}")
        sys.stdout.flush()

        plt.scatter(mpm_x - center[0], mpm_sig_xx, s=10, alpha=0.6, label='MPM', c='blue', edgecolors='none')
        print("DEBUG PLOT: scatter plot完成")
        sys.stdout.flush()

        plt.plot(line_x - center[0], ana_sig_xx, 'r--', lw=2, label='Analytical')
        print("DEBUG PLOT: line plot完成")
        sys.stdout.flush()

        # 辅助线（相对坐标）
        plt.axvline(-radius, color='k', linestyle=':', alpha=0.3, label='Inclusion boundary')
        plt.axvline(radius, color='k', linestyle=':', alpha=0.3)
        plt.xlabel('X position (relative to center)')
        plt.ylabel('Stress XX (Pa)')
        plt.xlim(-3*radius, 3*radius)
        plt.legend()
        plt.grid(True, alpha=0.3)
        print("DEBUG PLOT: 左图完成")
        sys.stdout.flush()

        # 右图: Sigma_YY
        plt.subplot(1, 2, 2)
        print("DEBUG PLOT: subplot(1,2,2)创建成功")
        sys.stdout.flush()

        plt.title(f"Radial Stress $\sigma_{{yy}}${title_suffix}")
        # 坐标平移：使中心点变为0
        plt.scatter(mpm_x - center[0], mpm_sig_yy, s=10, alpha=0.6, label='MPM', c='green', edgecolors='none')
        print("DEBUG PLOT: 右图scatter完成")
        sys.stdout.flush()

        plt.plot(line_x - center[0], ana_sig_yy, 'r--', lw=2, label='Analytical')
        print("DEBUG PLOT: 右图line plot完成")
        sys.stdout.flush()

        # 辅助线（相对坐标）
        plt.axvline(-radius, color='k', linestyle=':', alpha=0.3, label='Inclusion boundary')
        plt.axvline(radius, color='k', linestyle=':', alpha=0.3)
        plt.xlabel('X position (relative to center)')
        plt.ylabel('Stress YY (Pa)')
        plt.xlim(-3*radius, 3*radius)
        plt.legend()
        plt.grid(True, alpha=0.3)
        print("DEBUG PLOT: 右图完成")
        sys.stdout.flush()

        save_path = os.path.join(output_dir, 'result_v3_cartesian.png')
        print(f"DEBUG PLOT: 准备保存到 {save_path}")
        sys.stdout.flush()

        # Synchronize Taichi to ensure all computations are complete
        try:
            ti.sync()
            print("DEBUG PLOT: Taichi sync完成")
            sys.stdout.flush()
        except:
            print("DEBUG PLOT: Taichi sync跳过 (可能不需要)")
            sys.stdout.flush()

        # Use subplots_adjust instead of tight_layout to avoid re-rendering
        plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.9, wspace=0.25)
        print("DEBUG PLOT: subplots_adjust完成")
        sys.stdout.flush()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("DEBUG PLOT: savefig完成")
        sys.stdout.flush()

        plt.close()
        print("DEBUG PLOT: close完成")
        sys.stdout.flush()

        print(f"\nSaved to: {save_path}")
    except Exception as e:
        print(f"DEBUG PLOT ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

    # --- 5. 保存数据（用于批量汇总） ---
    # 保存原始位置和应力数据
    np.save(os.path.join(output_dir, 'positions.npy'), positions)
    np.save(os.path.join(output_dir, 'stresses.npy'), stresses)

    # 返回结果信息
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

        strip_width = radius * 0.2
        mask_strip = np.abs(positions[:, 1] - center[1]) < strip_width

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

    # 批量模式参数
    parser.add_argument('--batch-mode', action='store_true',
                       help='启用批量模式：运行多个不同网格分辨率的实验')
    parser.add_argument('--grid-start', type=int, default=40,
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