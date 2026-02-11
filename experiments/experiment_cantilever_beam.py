#!/usr/bin/env python3
"""
静态悬臂梁实验
支持单域模式（基于config/config_2d_test4.json）和双域模式（基于config/schwarz_2d_test4.json）
研究gamma=12*rho*g*L^3*(1-nu^2)/(E*h^2)对悬臂梁变形的影响
分析最右下角粒子的位移，绘制log(h/w)随log(gamma)的变化曲线
双域模式：梁的左端（固定端）在Domain2，右端（自由端）在Domain1
"""

import json
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import subprocess
import glob
import shutil
import gc
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path):
    """保存配置文件"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def extract_beam_parameters(config, use_schwarz=False):
    """从配置文件中提取悬臂梁几何和材料参数"""

    if use_schwarz:
        # 双域模式：分析Domain1和Domain2
        domain1 = config['Domain1']
        domain2 = config['Domain2']

        # 材料参数（假设两个域使用相同材料）
        material = domain1['material_params'][0]
        E = material['E']
        nu = material['nu']
        rho = material['rho']

        # 获取Domain1和Domain2的几何形状
        domain1_shape = None
        for shape in domain1['shapes']:
            if shape['type'] == 'rectangle' and shape['operation'] == 'add':
                domain1_shape = shape
                break

        domain2_shape = None
        for shape in domain2['shapes']:
            if shape['type'] == 'rectangle' and shape['operation'] == 'add':
                domain2_shape = shape
                break

        if domain1_shape is None or domain2_shape is None:
            raise ValueError("在双域配置中未找到悬臂梁几何形状")

        # Domain1的范围（包含自由端）
        domain1_range = domain1_shape['params']['range']
        # Domain2的范围（包含固定端）
        domain2_range = domain2_shape['params']['range']

        # 固定端位置（从Domain2的DBC_range获取）
        fixed_end_x = domain2['DBC_range'][0][0][1]  # 固定端右边界

        # 自由端位置（Domain1的右端）
        free_end_x = domain1_range[0][1]

        # 总长度
        L = free_end_x - fixed_end_x

        # 高度（假设两个域的梁高度相同）
        h1 = domain1_range[1][1] - domain1_range[1][0]
        h2 = domain2_range[1][1] - domain2_range[1][0]
        h = max(h1, h2)  # 使用较大的高度

        return {
            'E': E,
            'nu': nu,
            'rho': rho,
            'L': L,
            'h': h,
            'fixed_end_x': fixed_end_x,
            'free_end_x': free_end_x,
            'domain1_range': domain1_range,
            'domain2_range': domain2_range,
            'use_schwarz': True
        }

    else:
        # 单域模式：原有逻辑
        # 材料参数
        material = config['material_params'][0]
        E = material['E']
        nu = material['nu']
        rho = material['rho']

        # 悬臂梁几何参数（从shapes中获取rectangle）
        beam_shape = None
        for shape in config['shapes']:
            if shape['type'] == 'rectangle' and shape['operation'] == 'add':
                beam_shape = shape
                break

        if beam_shape is None:
            raise ValueError("未找到悬臂梁几何形状")

        beam_range = beam_shape['params']['range']
        # 固定端位置（从DBC_range获取）
        fixed_end_x = config['DBC_range'][0][0][1]  # 固定端右边界

        L = beam_range[0][1] - fixed_end_x  # 长度（x方向）
        h = beam_range[1][1] - beam_range[1][0]  # 高度（y方向）

        # 自由端位置
        free_end_x = beam_range[0][1]  # 梁的右端

        return {
            'E': E,
            'nu': nu,
            'rho': rho,
            'L': L,
            'h': h,
            'fixed_end_x': fixed_end_x,
            'free_end_x': free_end_x,
            'beam_range': beam_range,
            'use_schwarz': False
        }

def calculate_gamma(rho, g, L, E, h, nu):
    """计算无量纲参数gamma = 12*rho*g*L^3*(1-nu^2)/(E*h^2)"""
    return 12 * rho * g * L**3 * (1 - nu**2) / (E * h**2)

def extract_gravity_schedule(config):
    """
    从配置中提取重力调度表

    参数:
        config: 配置字典

    返回:
        重力调度列表，每个条目包含 {'frame': int, 'gravity': [x, y]}
    """
    # 检查Schwarz模式 (双域)
    if 'Domain1' in config and 'gravity_schedule' in config['Domain1']:
        schedule = config['Domain1']['gravity_schedule']
    # 检查单域模式
    elif 'gravity_schedule' in config:
        schedule = config['gravity_schedule']
    # 如果没有调度表，创建单个条目
    elif 'Domain1' in config and 'gravity' in config['Domain1']:
        schedule = [{'frame': 0, 'gravity': config['Domain1']['gravity']}]
    elif 'gravity' in config:
        schedule = [{'frame': 0, 'gravity': config['gravity']}]
    else:
        # 默认零重力
        schedule = [{'frame': 0, 'gravity': [0.0, 0.0]}]

    # 按帧号排序
    sorted_schedule = sorted(schedule, key=lambda x: x['frame'])
    return sorted_schedule

def calculate_gamma_from_gravity(gravity_y, beam_params):
    """
    根据重力值计算gamma

    参数:
        gravity_y: 重力的y分量 (通常为负值)
        beam_params: 从extract_beam_parameters()获取的梁参数字典

    返回:
        gamma值
    """
    g = abs(gravity_y)
    return calculate_gamma(
        beam_params['rho'],
        g,
        beam_params['L'],
        beam_params['E'],
        beam_params['h'],
        beam_params['nu']
    )

def calculate_gravity_for_gamma(target_gamma, rho, L, E, h, nu):
    """根据目标gamma值计算所需的重力加速度"""
    # gamma = 12*rho*g*L^3*(1-nu^2)/(E*h^2)
    # g = gamma * E * h^2 / (12 * rho * L^3 * (1-nu^2))
    g = target_gamma * E * h**2 / (12 * rho * L**3 * (1 - nu**2))
    return g

def create_config_for_gamma(base_config_path, target_gamma, output_config_path, use_schwarz=False):
    """为指定的gamma值创建配置文件"""

    # 加载基础配置
    config = load_config(base_config_path)

    # 提取参数
    params = extract_beam_parameters(config, use_schwarz=use_schwarz)

    # 计算所需的重力
    g = calculate_gravity_for_gamma(target_gamma, params['rho'], params['L'],
                                   params['E'], params['h'], params['nu'])

    if use_schwarz:
        # 双域模式：修改两个域的重力
        config['Domain1']['gravity'] = [0.0, -g]
        config['Domain2']['gravity'] = [0.0, -g]
        print(f"创建双域配置文件: {output_config_path}")
        print(f"  Domain1和Domain2重力都设置为: [0.0, {-g:.2e}]")
    else:
        # 单域模式：修改配置中的重力
        config['gravity'] = [0.0, -g]
        print(f"创建单域配置文件: {output_config_path}")

    # 保存新配置
    save_config(config, output_config_path)

    print(f"  gamma = {target_gamma:.2e}")
    print(f"  g = {g:.2e} m/s²")
    print(f"  E = {params['E']:.2e} Pa (保持不变)")

    return config, g

def run_single_simulation(config_path, use_schwarz=False, output_name=None):
    """运行单次MPM模拟"""

    print(f"运行模拟: {config_path}")

    if use_schwarz:
        # 双域模式：使用Schwarz求解器
        cmd = [
            "python", "simulators/implicit_mpm_schwarz.py",
            "--config", config_path,"--no-gui"
        ]
        print(f"  使用双域Schwarz求解器")
    else:
        # 单域模式：使用单域求解器
        cmd = [
            "python", "simulators/implicit_mpm.py",
            "--config", config_path
        ]
        print(f"  使用单域求解器")

    # 准备保存输出的文件路径
    if output_name:
        log_dir = os.path.dirname(config_path)
        stdout_log = os.path.join(log_dir, f"{output_name}_stdout.log")
        stderr_log = os.path.join(log_dir, f"{output_name}_stderr.log")
        cmd_log = os.path.join(log_dir, f"{output_name}_command.log")
    else:
        # 使用配置文件名作为基础
        config_base = os.path.splitext(os.path.basename(config_path))[0]
        log_dir = os.path.dirname(config_path)
        stdout_log = os.path.join(log_dir, f"{config_base}_stdout.log")
        stderr_log = os.path.join(log_dir, f"{config_base}_stderr.log")
        cmd_log = os.path.join(log_dir, f"{config_base}_command.log")

    try:
        # 保存运行的命令
        with open(cmd_log, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Working directory: {os.getcwd()}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")

        print(f"  命令日志保存到: {cmd_log}")

        # 运行模拟
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)

        # 保存标准输出
        with open(stdout_log, 'w') as f:
            f.write(result.stdout)
        print(f"  标准输出保存到: {stdout_log}")

        # 保存错误输出
        with open(stderr_log, 'w') as f:
            f.write(result.stderr)
        print(f"  错误输出保存到: {stderr_log}")

        if result.returncode == 0:
            print(f"  模拟完成")
            return True
        else:
            print(f"  模拟失败，返回码: {result.returncode}")
            if result.stderr:
                print(f"  错误信息: {result.stderr[-500:]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  模拟超时（超过2小时）")
        # 保存超时信息
        with open(stderr_log, 'w') as f:
            f.write("SIMULATION TIMEOUT: Exceeded 2 hours\n")
        return False
    except Exception as e:
        print(f"  模拟过程出错: {e}")
        # 保存异常信息
        with open(stderr_log, 'w') as f:
            f.write(f"EXCEPTION: {str(e)}\n")
        return False

def find_rightmost_bottom_particle(positions, beam_params, domain_id=None):
    """找到最右下角的粒子（y最小，x最大）"""

    if beam_params.get('use_schwarz', False) and domain_id == 1:
        # 双域模式：在Domain1范围内查找最右下角粒子
        domain1_positions = positions

        print(f"  调试信息：输入positions长度: {len(positions)}, domain1_positions长度: {len(domain1_positions)}")

        if len(domain1_positions) == 0:
            raise ValueError("在Domain1范围内未找到粒子")

        # 在Domain1范围内找到y坐标最小的粒子
        min_y = np.min(domain1_positions[:, 1])
        bottom_particles_mask = np.abs(domain1_positions[:, 1] - min_y) < 1e-4
        bottom_particles = domain1_positions[bottom_particles_mask]

        # 在底部粒子中找到x坐标最大的
        max_x_idx = np.argmax(bottom_particles[:, 0])
        rightmost_bottom = bottom_particles[max_x_idx]

        print(f"  Domain1范围内找到最右下角粒子: ({rightmost_bottom[0]:.6f}, {rightmost_bottom[1]:.6f})")
        return rightmost_bottom
    else:
        # 单域模式或Domain2：在全局范围内查找
        # 找到y坐标最小的粒子
        min_y = np.min(positions[:, 1])
        bottom_particles_mask = np.abs(positions[:, 1] - min_y) < 1e-6
        bottom_particles = positions[bottom_particles_mask]

        # 在底部粒子中找到x坐标最大的
        max_x_idx = np.argmax(bottom_particles[:, 0])
        rightmost_bottom = bottom_particles[max_x_idx]

        return rightmost_bottom

def find_closest_frame(target_frame, available_frames, mode='before'):
    """
    找到最接近目标帧的已保存帧

    参数:
        target_frame: 目标帧号
        available_frames: 可用帧号列表
        mode: 'before' (最大的≤目标帧), 'after' (最小的≥目标帧), 'nearest' (最接近的帧)

    返回:
        选中的帧号，如果未找到则返回None
    """
    if not available_frames:
        return None

    sorted_frames = sorted(available_frames)

    if mode == 'before':
        # 找到最大的 <= target_frame 的帧
        candidates = [f for f in sorted_frames if f <= target_frame]
        return max(candidates) if candidates else None
    elif mode == 'after':
        # 找到最小的 >= target_frame 的帧
        candidates = [f for f in sorted_frames if f >= target_frame]
        return min(candidates) if candidates else None
    elif mode == 'nearest':
        # 找到最接近的帧
        return min(sorted_frames, key=lambda f: abs(f - target_frame))
    else:
        raise ValueError(f"Unknown mode: {mode}")

def load_frame_positions(results_dir, frame_num, domain_id=1):
    """
    从特定帧目录加载位置数据

    参数:
        results_dir: 实验结果目录路径
        frame_num: 帧号
        domain_id: 域ID (1 或 2)

    返回:
        位置数组 (n_particles, 2)
    """
    frame_dir = os.path.join(results_dir, 'stress_data', f'frame_{frame_num}')
    positions_file = os.path.join(frame_dir, f'domain{domain_id}_positions.npy')

    if not os.path.exists(positions_file):
        raise FileNotFoundError(f"未找到位置文件: {positions_file}")

    positions = np.load(positions_file)
    return positions

def load_simulation_results(results_dir=None, use_schwarz=False, domain_id=None):
    """加载模拟结果（适配新的stress_data/frame_*目录结构）"""

    if results_dir is None:
        # 查找最新的结果目录
        if use_schwarz:
            result_dirs = glob.glob("experiment_results/schwarz_*")
        else:
            result_dirs = glob.glob("experiment_results/single_domain_*")

        if not result_dirs:
            print("未找到模拟结果目录")
            return None

        # 选择最新的目录（按创建时间）
        results_dir = max(result_dirs, key=os.path.getctime)
        print(f"  从 {len(result_dirs)} 个目录中选择最新的: {os.path.basename(results_dir)}")

    print(f"从目录加载结果: {results_dir}")

    # 检查目录是否存在
    if not os.path.exists(results_dir):
        print(f"错误: 结果目录不存在: {results_dir}")
        return None

    # 新格式：数据保存在 stress_data/frame_* 目录下
    stress_data_dir = os.path.join(results_dir, "stress_data")

    if os.path.exists(stress_data_dir):
        # 新格式：查找 stress_data/frame_* 目录
        frame_dirs = glob.glob(os.path.join(stress_data_dir, "frame_*"))

        if not frame_dirs:
            print(f"  错误: 在 {stress_data_dir} 中未找到frame目录")
            return None

        # 使用最后一帧的数据
        frame_numbers = [int(os.path.basename(d).replace("frame_", "")) for d in frame_dirs]
        last_frame = max(frame_numbers)
        latest_frame_dir = os.path.join(stress_data_dir, f"frame_{last_frame}")

        print(f"  找到 {len(frame_numbers)} 个帧目录, 使用最新帧: {last_frame}")

        # 根据模式加载对应的位置文件
        if use_schwarz and domain_id is not None:
            # 双域模式：加载指定域的结果
            positions_file = os.path.join(latest_frame_dir, f"domain{domain_id}_positions.npy")
            print(f"  加载Domain{domain_id}的结果")
        else:
            # 单域模式：加载普通位置文件
            positions_file = os.path.join(latest_frame_dir, "positions.npy")

        if not os.path.exists(positions_file):
            print(f"  错误: 位置文件不存在: {positions_file}")
            print(f"  frame目录内容: {os.listdir(latest_frame_dir)}")
            return None

        # 加载位置数据
        positions = np.load(positions_file)
        print(f"加载了第 {last_frame} 帧的 {len(positions)} 个粒子位置")

    else:
        # 旧格式：向后兼容，直接从实验结果目录加载
        print("  使用旧格式加载数据（从实验结果目录直接加载）")

        if use_schwarz and domain_id is not None:
            # 双域模式：加载指定域的结果
            position_files = glob.glob(os.path.join(results_dir, f"domain{domain_id}_positions_frame_*.npy"))
            file_prefix = f"domain{domain_id}_positions_frame_"
            print(f"  加载Domain{domain_id}的结果")
        else:
            # 单域模式：加载普通位置文件
            position_files = glob.glob(os.path.join(results_dir, "positions_frame_*.npy"))
            file_prefix = "positions_frame_"

        if not position_files:
            print(f"  错误: 在目录 {results_dir} 中未找到位置数据文件")
            print(f"  目录内容: {os.listdir(results_dir) if os.path.exists(results_dir) else '目录不存在'}")
            return None

        # 使用最后一帧的数据
        frame_numbers = []
        for file in position_files:
            filename = os.path.basename(file)
            frame_num_str = filename.replace(file_prefix, "").replace(".npy", "")
            frame_num = int(frame_num_str)
            frame_numbers.append(frame_num)

        last_frame = max(frame_numbers)
        print(f"  找到帧数: {sorted(frame_numbers)}, 使用最新帧: {last_frame}")

        if use_schwarz and domain_id is not None:
            positions_file = os.path.join(results_dir, f"domain{domain_id}_positions_frame_{last_frame}.npy")
        else:
            positions_file = os.path.join(results_dir, f"positions_frame_{last_frame}.npy")

        # 加载位置数据
        positions = np.load(positions_file)
        print(f"加载了第 {last_frame} 帧的 {len(positions)} 个粒子位置")

    result = {
        'positions': positions,
        'frame': last_frame,
        'results_dir': results_dir
    }

    # 清理临时变量
    del frame_dirs, frame_numbers
    gc.collect()

    return result

def calculate_displacement(initial_pos, final_pos, beam_params):
    """计算位移并返回h/w比值"""

    if beam_params.get('use_schwarz', False):
        # 双域模式：使用domain1_range作为参考
        w = abs(final_pos[0] - beam_params['domain1_range'][0][0])
    else:
        # 单域模式：使用beam_range作为参考
        w = abs(final_pos[0] - beam_params['beam_range'][0][0])

    # h: y方向位移（距离初始y坐标的距离）
    h_displacement = abs(final_pos[1] - initial_pos[1])

    # 计算h/w比值
    if w > 1e-12:  # 避免除零
        h_over_w = h_displacement / w
    else:
        h_over_w = float('inf')

    return {
        'w': w,
        'h_displacement': h_displacement,
        'h_over_w': h_over_w,
        'initial_pos': initial_pos,
        'final_pos': final_pos
    }

def parse_experiment_results_dir(results_dir):
    """
    解析单个实验结果目录，分析重力调度下的梁变形

    参数:
        results_dir: 实验结果目录路径

    返回:
        包含梁参数、重力调度和分析结果的字典
    """
    print(f"\n解析实验结果目录: {results_dir}")

    # 1. 加载配置
    config_path = os.path.join(results_dir, 'config_backup.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # 2. 提取重力调度和梁参数
    gravity_schedule = extract_gravity_schedule(config)
    beam_params = extract_beam_parameters(config, use_schwarz=True)

    print(f"  梁参数: E={beam_params['E']}, L={beam_params['L']}, h={beam_params['h']}")
    print(f"  重力调度: {len(gravity_schedule)} 个重力级别")

    # 3. 发现可用帧
    stress_data_dir = os.path.join(results_dir, 'stress_data')
    if not os.path.exists(stress_data_dir):
        raise FileNotFoundError(f"未找到stress_data目录: {stress_data_dir}")

    frame_dirs = [d for d in os.listdir(stress_data_dir) if d.startswith('frame_')]
    available_frames = sorted([int(d.replace('frame_', '')) for d in frame_dirs])

    print(f"  可用帧: {len(available_frames)} 个 (范围: {min(available_frames)} - {max(available_frames)})")

    # 4. 获取DBC_range的右端x坐标（固定端位置）
    # 从Domain2获取DBC_range
    dbc_right_x = config['Domain2']['DBC_range'][0][0][1]  # DBC x范围的右端
    print(f"  固定端位置 (DBC右端): x = {dbc_right_x}")

    # 5. 在最早的frame找到最右下角粒子并记住其索引
    first_frame = min(available_frames)
    print(f"\n  在第一帧 (frame {first_frame}) 找到跟踪粒子...")
    first_positions = load_frame_positions(results_dir, first_frame, domain_id=1)

    # 找到最右下角粒子
    tracked_particle_pos = find_rightmost_bottom_particle(first_positions, beam_params, domain_id=1)

    # 找到该粒子的索引
    # 通过比较位置找到精确匹配的粒子
    tolerance = 1e-10
    tracked_particle_idx = None
    for idx, pos in enumerate(first_positions):
        if abs(pos[0] - tracked_particle_pos[0]) < tolerance and abs(pos[1] - tracked_particle_pos[1]) < tolerance:
            tracked_particle_idx = idx
            break

    if tracked_particle_idx is None:
        raise RuntimeError("无法找到跟踪粒子的索引")

    # 记录初始位置
    initial_x = tracked_particle_pos[0]
    initial_y = tracked_particle_pos[1]

    print(f"  跟踪粒子索引: {tracked_particle_idx}")
    print(f"  初始位置: x={initial_x:.6f}, y={initial_y:.6f}")

    # 清理
    del first_positions
    gc.collect()

    # 6. 处理每个重力级别
    results = []

    for i, schedule_entry in enumerate(gravity_schedule):
        gravity_frame = schedule_entry['frame']
        gravity_value = schedule_entry['gravity']

        print(f"\n  处理重力级别 {i+1}/{len(gravity_schedule)}:")
        print(f"    帧: {gravity_frame}, 重力: {gravity_value}")

        # 6.1 确定测量帧
        # 如果不是最后一个重力级别，使用下一个重力变化的帧
        # 如果是最后一个，使用最后可用帧
        if i < len(gravity_schedule) - 1:
            measurement_frame_target = gravity_schedule[i + 1]['frame']
        else:
            measurement_frame_target = max(available_frames)

        # 找到最接近的可用帧（≤目标帧）
        measurement_frame = find_closest_frame(measurement_frame_target, available_frames, mode='before')

        if measurement_frame is None:
            print(f"    警告: 未找到合适的测量帧，跳过此重力级别")
            continue

        print(f"    测量帧: {measurement_frame}")

        try:
            # 6.2 加载位置数据
            positions = load_frame_positions(results_dir, measurement_frame, domain_id=1)

            # 6.3 读取跟踪粒子的当前位置
            if tracked_particle_idx >= len(positions):
                raise ValueError(f"粒子索引 {tracked_particle_idx} 超出范围 (共 {len(positions)} 个粒子)")

            current_pos = positions[tracked_particle_idx]
            current_x = current_pos[0]
            current_y = current_pos[1]

            # 6.4 计算h和w
            # h: 初始y - 当前y (向下位移为正)
            h = initial_y - current_y
            # w: 当前x - 固定端x (从固定端到粒子的水平距离)
            w = current_x - dbc_right_x

            # 计算h/w比值
            if w > 1e-12:
                h_over_w = h / w
            else:
                h_over_w = float('inf')

            # 6.5 计算gamma
            gamma = calculate_gamma_from_gravity(gravity_value[1], beam_params)

            # 6.6 存储结果
            result_entry = {
                'frame': measurement_frame,
                'gravity': gravity_value,
                'gamma': gamma,
                'particle_idx': tracked_particle_idx,
                'initial_pos': [initial_x, initial_y],
                'current_pos': [current_x, current_y],
                'h': h,
                'w': w,
                'h_over_w': h_over_w
            }
            results.append(result_entry)

            print(f"    粒子位置: x={current_x:.6f}, y={current_y:.6f}")
            print(f"    gamma={gamma:.6e}, h={h:.6e}, w={w:.6e}, h/w={h_over_w:.6e}")

            # 清理
            del positions
            gc.collect()

        except Exception as e:
            print(f"    错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 6. 编译摘要
    if results:
        gamma_values = [r['gamma'] for r in results]
        h_over_w_values = [r['h_over_w'] for r in results]

        summary = {
            'n_gravity_levels': len(gravity_schedule),
            'n_successful_analyses': len(results),
            'gamma_range': [min(gamma_values), max(gamma_values)],
            'h_over_w_range': [min(h_over_w_values), max(h_over_w_values)]
        }
    else:
        summary = {
            'n_gravity_levels': len(gravity_schedule),
            'n_successful_analyses': 0,
            'gamma_range': [0, 0],
            'h_over_w_range': [0, 0]
        }

    print(f"\n分析完成: {summary['n_successful_analyses']}/{summary['n_gravity_levels']} 个重力级别成功分析")

    return {
        'results_dir': results_dir,
        'beam_params': beam_params,
        'gravity_schedule': gravity_schedule,
        'available_frames': available_frames,
        'results': results,
        'summary': summary
    }

def analyze_single_experiment(results_dir, plot=True, output_file=None):
    """
    分析单个实验结果并生成图表

    参数:
        results_dir: 实验结果目录路径
        plot: 是否生成图表
        output_file: JSON结果输出文件路径（可选）

    返回:
        完整的分析结果字典
    """
    # 解析实验结果
    analysis_results = parse_experiment_results_dir(results_dir)

    # 生成图表
    if plot and analysis_results['results']:
        print(f"\n生成log(h/w) vs log(gamma)图表...")
        plot_results(analysis_results['results'], output_dir=results_dir)

    # 保存JSON结果
    if output_file:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(analysis_results)

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"结果已保存到: {output_file}")

    # 打印摘要统计
    print(f"\n=== 分析摘要 ===")
    print(f"实验目录: {results_dir}")
    print(f"成功分析: {analysis_results['summary']['n_successful_analyses']}/{analysis_results['summary']['n_gravity_levels']} 个重力级别")
    if analysis_results['results']:
        print(f"Gamma范围: {analysis_results['summary']['gamma_range'][0]:.3e} - {analysis_results['summary']['gamma_range'][1]:.3e}")
        print(f"h/w范围: {analysis_results['summary']['h_over_w_range'][0]:.6f} - {analysis_results['summary']['h_over_w_range'][1]:.6f}")

    return analysis_results

def run_single_experiment(base_config_path="config/config_2d_test4.json",
                         target_gamma=None, output_dir="experiment_results/cantilever_single", use_schwarz=False):
    """运行单次悬臂梁实验"""

    print("=" * 60)
    if use_schwarz:
        print("运行单次悬臂梁实验（双域模式）")
    else:
        print("运行单次悬臂梁实验（单域模式）")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    if target_gamma is None:
        # 使用原始配置的参数
        config_path = base_config_path
        config = load_config(config_path)
        params = extract_beam_parameters(config, use_schwarz=use_schwarz)

        if use_schwarz:
            current_g = abs(config['Domain1']['gravity'][1])
        else:
            current_g = abs(config['gravity'][1])

        gamma = calculate_gamma(params['rho'], current_g, params['L'],
                               params['E'], params['h'], params['nu'])
        print(f"使用原始配置，gamma = {gamma:.2e}")
    else:
        # 创建新配置
        if use_schwarz:
            config_path = os.path.join(output_dir, "config_schwarz.json")
        else:
            config_path = os.path.join(output_dir, "config_single.json")

        config, g = create_config_for_gamma(base_config_path, target_gamma, config_path, use_schwarz=use_schwarz)
        params = extract_beam_parameters(config, use_schwarz=use_schwarz)
        gamma = target_gamma

    # 获取初始的最右下角粒子位置（理论位置）
    if use_schwarz:
        # 双域模式：自由端在Domain1的右下角
        domain1_range = params['domain1_range']
        initial_rightmost_bottom = [domain1_range[0][1], domain1_range[1][0]]  # Domain1右下角
    else:
        # 单域模式：梁的右下角
        beam_range = params['beam_range']
        initial_rightmost_bottom = [beam_range[0][1], beam_range[1][0]]  # 右下角

    print(f"悬臂梁参数:")
    print(f"  长度 L = {params['L']:.3f} m")
    print(f"  高度 h = {params['h']:.3f} m")
    print(f"  弹性模量 E = {params['E']:.2e} Pa")
    print(f"  泊松比 nu = {params['nu']:.3f}")
    print(f"  密度 rho = {params['rho']:.2e} kg/m³")
    print(f"  初始最右下角位置: ({initial_rightmost_bottom[0]:.3f}, {initial_rightmost_bottom[1]:.3f})")
    if use_schwarz:
        print(f"  双域模式：自由端在Domain1，固定端在Domain2")

    # 运行模拟
    output_name = f"single_beam_{'schwarz' if use_schwarz else 'single'}"
    success = run_single_simulation(config_path, use_schwarz=use_schwarz, output_name=output_name)
    if not success:
        print("模拟失败")
        return None

    # 加载结果
    if use_schwarz:
        # 双域模式：加载Domain1的结果（包含自由端）
        results = load_simulation_results(use_schwarz=True, domain_id=1)
    else:
        # 单域模式：加载普通结果
        results = load_simulation_results(use_schwarz=False)

    if results is None:
        print("无法加载模拟结果")
        return None

    # 找到最右下角粒子
    if use_schwarz:
        final_rightmost_bottom = find_rightmost_bottom_particle(results['positions'], params, domain_id=1)
    else:
        final_rightmost_bottom = find_rightmost_bottom_particle(results['positions'], params)

    # 计算位移
    displacement_result = calculate_displacement(initial_rightmost_bottom, final_rightmost_bottom, params)

    print(f"\n位移分析结果:")
    print(f"  最终最右下角位置: ({final_rightmost_bottom[0]:.6f}, {final_rightmost_bottom[1]:.6f})")
    print(f"  x方向距离 = {displacement_result['w']:.6f} m")
    print(f"  y方向位移 = {displacement_result['h_displacement']:.6f} m")
    print(f"  h/w = {displacement_result['h_over_w']:.6f}")
    print(f"  gamma = {gamma:.6e}")

    # 保存结果
    result_data = {
        'gamma': gamma,
        'h_over_w': displacement_result['h_over_w'],
        'beam_params': params,
        'displacement': displacement_result,
        'config_path': config_path,
        'use_schwarz': use_schwarz
    }

    result_file = os.path.join(output_dir, "single_experiment_result.json")
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=4, default=str)

    print(f"\n结果保存到: {result_file}")
    return result_data

def run_batch_experiments(base_config_path="config/config_2d_test4.json",
                         gamma_range=(0.01, 1e4), n_points=6,
                         output_dir="experiment_results/cantilever_batch", use_schwarz=False):
    """运行批量悬臂梁实验"""

    print("=" * 60)
    if use_schwarz:
        print("运行批量悬臂梁实验（双域模式）")
    else:
        print("运行批量悬臂梁实验（单域模式）")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成gamma值序列（对数均匀分布）
    gamma_values = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), n_points)

    print(f"gamma范围: {gamma_range[0]:.2e} - {gamma_range[1]:.2e}")
    print(f"测试点数: {n_points}")
    print(f"通过修改重力g来变化gamma参数（保持弹性模量E不变）")
    print(f"gamma值: {gamma_values}")

    gravity_values = []
    for gamma in gamma_values:
        # 计算对应的重力值
        config = load_config(base_config_path)
        params = extract_beam_parameters(config, use_schwarz=use_schwarz)
        g = calculate_gravity_for_gamma(gamma, params['rho'], params['L'], params['E'], params['h'], params['nu'])
        gravity_values.append(g)
        print(f"  gamma = {gamma:.2e} -> g = {g:.2e} m/s²")
    # 存储结果
    results = []

    # 加载基础配置获取参数
    base_config = load_config(base_config_path)
    beam_params = extract_beam_parameters(base_config, use_schwarz=use_schwarz)

    if use_schwarz:
        # 双域模式：自由端在Domain1的右下角
        initial_rightmost_bottom = [beam_params['domain1_range'][0][1], beam_params['domain1_range'][1][0]]
        base_g = abs(base_config['Domain1']['gravity'][1])
    else:
        # 单域模式：梁的右下角
        initial_rightmost_bottom = [beam_params['beam_range'][0][1], beam_params['beam_range'][1][0]]
        base_g = abs(base_config['gravity'][1])
    print(f"\n基础配置参数:")
    print(f"  原始重力 g = {base_g:.2e} m/s²")
    print(f"  弹性模量 E = {beam_params['E']:.2e} Pa (保持不变)")
    print(f"  泊松比 ν = {beam_params['nu']:.3f}")
    print(f"  梁长度 L = {beam_params['L']:.3f} m")
    print(f"  梁高度 h = {beam_params['h']:.3f} m")

    for i, gamma in enumerate(gamma_values):
        print(f"\n{'='*20} 实验 {i+1}/{n_points}: gamma = {gamma:.2e} {'='*20}")

        # 创建配置文件
        if use_schwarz:
            config_path = os.path.join(output_dir, f"config_schwarz_gamma_{gamma:.2e}.json")
        else:
            config_path = os.path.join(output_dir, f"config_gamma_{gamma:.2e}.json")

        config, g = create_config_for_gamma(base_config_path, gamma, config_path, use_schwarz=use_schwarz)

        # 运行模拟
        run_single_simulation(config_path, use_schwarz=use_schwarz, output_name=f"gamma_{gamma:.2e}")


        # 加载结果
        if use_schwarz:
            # 双域模式：加载Domain1的结果（包含自由端）
            sim_results = load_simulation_results(use_schwarz=True, domain_id=1)
        else:
            # 单域模式：加载普通结果
            sim_results = load_simulation_results(use_schwarz=False)

        if sim_results is None:
            print(f"实验 {i+1} 无法加载结果，跳过")
            continue

        # 复制结果目录以便区分（保留原始目录）
        if use_schwarz:
            new_results_dir = os.path.join(output_dir, f"results_schwarz_gamma_{gamma:.2e}")
        else:
            new_results_dir = os.path.join(output_dir, f"results_gamma_{gamma:.2e}")

        if os.path.exists(new_results_dir):
            shutil.rmtree(new_results_dir)
        shutil.copytree(sim_results['results_dir'], new_results_dir)
        print(f"  结果已复制到: {new_results_dir}")
        print(f"  原始结果保留在: {sim_results['results_dir']}")

        # 找到最右下角粒子
        if use_schwarz:
            final_rightmost_bottom = find_rightmost_bottom_particle(sim_results['positions'], beam_params, domain_id=1)
        else:
            final_rightmost_bottom = find_rightmost_bottom_particle(sim_results['positions'], beam_params)

        # 计算位移
        displacement_result = calculate_displacement(initial_rightmost_bottom, final_rightmost_bottom, beam_params)

        # 存储结果（只保留标量值，不保存大型数组）
        result_data = {
            'gamma': gamma,
            'gravity': g,
            'h_over_w': displacement_result['h_over_w'],
            'w': displacement_result['w'],
            'h_displacement': displacement_result['h_displacement'],
            'initial_pos': displacement_result['initial_pos'].tolist() if hasattr(displacement_result['initial_pos'], 'tolist') else displacement_result['initial_pos'],
            'final_pos': displacement_result['final_pos'].tolist() if hasattr(displacement_result['final_pos'], 'tolist') else displacement_result['final_pos'],
            'results_dir': new_results_dir
        }
        results.append(result_data)

        print(f"  h/w = {displacement_result['h_over_w']:.6f}")

        # 清理内存：删除所有大型临时变量
        del sim_results
        del displacement_result
        del final_rightmost_bottom
        del config
        # 强制垃圾回收
        gc.collect()

    print(f"\n批量实验完成! 成功完成 {len(results)}/{n_points} 个实验")

    # 保存批量结果
    batch_result = {
        'beam_params': beam_params,
        'gamma_range': gamma_range,
        'n_points': n_points,
        'successful_experiments': len(results),
        'use_schwarz': use_schwarz,
        'results': results
    }

    batch_file = os.path.join(output_dir, "batch_experiment_results.json")
    with open(batch_file, 'w') as f:
        json.dump(batch_result, f, indent=4, default=str)

    # 绘制结果
    plot_results(results, output_dir)

    print(f"\n批量结果保存到: {batch_file}")
    return batch_result

def plot_results(results, output_dir):
    """
    绘制log(h/w)随log(gamma)的变化曲线

    参数:
        results: 结果列表，或包含'results'键的字典（来自analyze_single_experiment）
        output_dir: 输出目录路径
    """
    # 处理不同的输入格式
    if isinstance(results, dict) and 'results' in results:
        # 来自analyze_single_experiment的格式
        results_list = results['results']
    else:
        # 来自run_batch_experiments的格式（直接是列表）
        results_list = results

    if len(results_list) == 0:
        print("没有有效结果用于绘图")
        return

    # 提取数据
    gamma_values = [r['gamma'] for r in results_list]
    h_over_w_values = [r['h_over_w'] for r in results_list]

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 绘制log-log图
    plt.loglog(gamma_values, h_over_w_values, 'bo-', linewidth=2, markersize=8)

    plt.xlabel('γ = 12ρgL³(1-ν²)/(Eh²)', fontsize=12)
    plt.ylabel('h/w', fontsize=12)
    plt.title('cantilever\nlog(h/w) vs log(γ)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 添加数据点标注
    for i, (gamma, h_w) in enumerate(zip(gamma_values, h_over_w_values)):
        if i % max(1, len(gamma_values)//10) == 0:  # 只标注部分点避免重叠
            plt.annotate(f'({gamma:.1e}, {h_w:.3f})',
                        (gamma, h_w),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

    plt.tight_layout()

    # 保存图片
    plot_file = os.path.join(output_dir, "cantilever_beam_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存

    print(f"分析图保存到: {plot_file}")

    # 打印统计信息
    print(f"\n结果统计:")
    print(f"  gamma范围: {min(gamma_values):.2e} - {max(gamma_values):.2e}")
    print(f"  h/w范围: {min(h_over_w_values):.6f} - {max(h_over_w_values):.6f}")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='静态悬臂梁实验')
    parser.add_argument('--config', default="config/config_2d_test4.json",
                       help='基础配置文件路径')
    parser.add_argument('--mode', choices=['single', 'batch', 'analyze'], default='single',
                       help='实验模式：单次(single)、批量(batch)或分析(analyze)')
    parser.add_argument('--use-schwarz', action='store_true',
                       help='使用双域Schwarz模式（默认为单域模式）')
    parser.add_argument('--gamma', type=float, default=None,
                       help='单次实验的gamma值（默认使用配置文件中的重力）')
    parser.add_argument('--gamma-range', nargs=2, type=float, default=[1e-2, 1e4],
                       help='批量实验的gamma范围 [最小值, 最大值]')
    parser.add_argument('--n-points', type=int, default=12,
                       help='批量实验的测试点数')
    parser.add_argument('--output-dir', default=None,
                       help='输出目录')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='实验结果目录路径（用于analyze模式）')

    args = parser.parse_args()

    print("静态悬臂梁实验")
    print("=" * 50)

    # 根据是否使用schwarz模式设置默认配置
    if args.use_schwarz and args.config == "config/config_2d_test4.json":
        args.config = "config/schwarz_2d_test4.json"
        print(f"双域模式：自动切换到配置文件 {args.config}")

    if args.mode == 'single':
        # 单次实验
        if args.use_schwarz:
            output_dir = args.output_dir or "experiment_results/cantilever_single_schwarz"
        else:
            output_dir = args.output_dir or "experiment_results/cantilever_single"

        result = run_single_experiment(
            base_config_path=args.config,
            target_gamma=args.gamma,
            output_dir=output_dir,
            use_schwarz=args.use_schwarz
        )

    elif args.mode == 'batch':
        # 批量实验
        if args.use_schwarz:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = args.output_dir or f"experiment_results/cantilever_batch_schwarz_{timestamp}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = args.output_dir or f"experiment_results/cantilever_batch_{timestamp}"

        result = run_batch_experiments(
            base_config_path=args.config,
            gamma_range=tuple(args.gamma_range),
            n_points=args.n_points,
            output_dir=output_dir,
            use_schwarz=args.use_schwarz
        )

    elif args.mode == 'analyze':
        # 分析已有实验结果
        if not args.results_dir:
            print("错误: analyze模式需要指定--results-dir参数")
            return

        if not os.path.exists(args.results_dir):
            print(f"错误: 结果目录不存在: {args.results_dir}")
            return

        # 设置输出文件路径
        output_file = os.path.join(args.results_dir, 'analysis_results.json')

        # 执行分析
        result = analyze_single_experiment(
            results_dir=args.results_dir,
            plot=True,
            output_file=output_file
        )

    print("\n实验完成!")

if __name__ == "__main__":
    main()