#!/usr/bin/env python3
"""
生成test3的schwarz域分解配置文件
用于圆柱体的下部分区域和上部分区域，支持可配置的厚度和重叠
"""

import json
import numpy as np
import argparse
import os

def generate_test3_schwarz_config(
    domain2_thickness=0.05,         # Domain2的厚度（底部小部分）
    overlap_thickness=0.01,         # 重叠区域的厚度
    semicircle_center=[0.5, 0.45],  # 下半圆的圆心（y=0.45）
    semicircle_radius=0.4,          # 下半圆的半径
    force_layer_thickness=0.01,    # 施力层厚度（从0.415到0.42）
    domain1_grid_size=32,           # Domain1网格大小
    domain2_grid_size=32,           # Domain2网格大小
    particles_per_grid=4,           # 每网格粒子数
    output_file="config/schwarz_2d_test3.json",
    implicit = True,
    schwarz_max_iter = 10,
    schwarz_eta = 1e-6,
    eta = 1e-3,
    no_gui = False,                 # 是否禁用GUI
    is_schwarz = False,              # 是否为Schwarz模式
):
    """
    生成下半圆域分解配置

    参数:
    - domain2_thickness: Domain2的厚度（底部小部分）
    - overlap_thickness: 重叠区域的厚度
    - semicircle_center: 下半圆的圆心 [x, y]
    - semicircle_radius: 下半圆的半径
    - force_layer_thickness: 施力层厚度
    - domain1_grid_size: Domain1网格大小
    - domain2_grid_size: Domain2网格大小
    - particles_per_grid: 每网格粒子数
    - output_file: 输出文件路径
    """

    # 计算下半圆边界
    center_x, center_y = semicircle_center
    domain1_radius = semicircle_radius

    # 下半圆的边界框
    left = center_x - domain1_radius
    right = center_x + domain1_radius
    bottom = center_y - domain1_radius  # 下半圆的底部
    top = center_y              # 下半圆的顶部（圆心处）

    # 计算域分界线
    # Domain2：底部小部分（从底部向上指定厚度）
    domain2_bottom = bottom
    domain2_top = bottom + domain2_thickness
    domain2_width = 2 * np.sqrt(2*domain1_radius*(domain2_thickness + overlap_thickness) - (domain2_thickness + overlap_thickness)**2)*(1+(10/domain2_grid_size))

    # Domain1 和 Domain2 都使用相同的简化配置
    domain1_offset_x = 0.0
    domain1_offset_y = 0.0

    domain2_offset_x = 0.5 * (1-domain2_width)  # 居中放置
    domain2_offset_y = 0

    domain1_width = 1.0
    # 计算Domain1的grid单元大小
    domain1_grid_cell_size = domain1_width / domain1_grid_size
    domain1_height = top + 5 * domain1_grid_cell_size  # 加上5个grid size大小的高度冗余

    domain1_grid_size_x = domain1_grid_size
    domain1_grid_size_y = int(domain1_grid_size * domain1_height / domain1_width)

    domain1_height = domain1_grid_size_y / domain1_grid_size_x * domain1_width

    domain2_radius = domain1_radius  # 按比例缩放半径
    # 重叠区域：Domain2顶部向上延伸
    overlap_bottom = domain2_top
    overlap_top = overlap_bottom + overlap_thickness

    # 计算Domain2的grid单元大小
    domain2_grid_cell_size = domain2_width / domain2_grid_size
    domain2_height = overlap_top + 5 * domain2_grid_cell_size  # 加上5个grid size大小的高度冗余

    domain2_grid_size_x = domain2_grid_size
    domain2_grid_size_y = int(domain2_grid_size * domain2_height / domain2_width)
    domain2_height = domain2_grid_size_y / domain2_grid_size_x * domain2_width

    # Domain1：从重叠区域开始到圆心处
    domain1_bottom = overlap_bottom
    domain1_top = top

    # 施力区域：在Domain1中，从top-force_layer_thickness到top
    force_bottom = top - force_layer_thickness
    force_top = top

    print(f"下半圆配置:")
    print(f"  中心: ({center_x}, {center_y})")
    print(f"  半径: {domain1_radius}")
    print(f"  范围: x=[{left:.3f}, {right:.3f}], y=[{bottom:.3f}, {top:.3f}]")
    print()

    print(f"域分解配置:")
    print(f"  Domain2厚度: {domain2_thickness}")
    print(f"  重叠厚度: {overlap_thickness}")
    print(f"  Domain2: y=[{domain2_bottom:.3f}, {domain2_top:.3f}]")
    print(f"  重叠区域: y=[{overlap_bottom:.3f}, {overlap_top:.3f}]")
    print(f"  Domain1: y=[{domain1_bottom:.3f}, {domain1_top:.3f}]")
    print(f"  施力区域: y=[{force_bottom:.3f}, {force_top:.3f}] (在Domain1中)")
    print()



    print(f"Domain1 (主体部分) 配置:")
    print(f"  offset: [{domain1_offset_x:.3f}, {domain1_offset_y:.3f}]")
    print(f"  物理范围: x=[{domain1_offset_x:.3f}, {domain1_offset_x + domain1_width:.3f}], y=[{domain1_offset_y:.3f}, {domain1_offset_y + domain1_height:.3f}]")
    print()

    print(f"Domain2 (底部小部分) 配置:")
    print(f"  scale: {domain2_width:.3f}")
    print(f"  offset: [{domain2_offset_x:.3f}, {domain2_offset_y:.3f}]")
    print(f"  物理范围: x=[{domain2_offset_x:.3f}, {domain2_offset_x + domain2_width:.3f}], y=[{domain2_offset_y:.3f}, {domain2_offset_y + domain2_height:.3f}]")
    print()

    # 在归一化坐标系中定义几何形状（scale=1，offset=[0,0]，所以归一化坐标=物理坐标）

    # 每个域中的圆形中心和半径（归一化坐标）
    domain1_center_norm = [center_x, center_y]
    domain1_radius_norm = [domain1_radius, domain1_radius]

    domain2_center_norm = [(center_x-domain2_offset_x), (center_y-domain2_offset_y)]
    domain2_radius_norm = [domain2_radius, domain2_radius]

    # Domain1: 圆减去上半部分（保留下半圆）+ 减去底部一些部分（但要保留重叠区域）
    domain1_cut_top_norm = top  # 切掉上半圆
    domain1_cut_bottom_norm = overlap_bottom   # 切掉到重叠区域开始

    # Domain2: 圆减去重叠区域以上的所有部分（保留重叠区域和底部）
    domain2_cut_top_norm = (overlap_top -domain2_offset_y)  # 切掉重叠区域以上的部分

    add_overlap_top = (overlap_top - domain2_offset_y) 

    # Domain1的力施加区域（在Domain1中，从force_bottom到force_top）
    domain1_force_bottom_norm = force_bottom
    domain1_force_top_norm = force_top

    domain2_cut_width = np.sqrt(2*domain1_radius*domain2_thickness - domain2_thickness**2) 

    domain2_cut_right = (0.5 + domain2_cut_width - domain2_offset_x) 
    domain2_cut_left = (0.5 - domain2_cut_width - domain2_offset_x) 
    

    # 生成配置
    config = {
        "max_schwarz_iter": schwarz_max_iter,
        "schwarz_eta": schwarz_eta,
        "max_frames": 10000 if implicit else 100000,
        "use_record": False,
        "visualize_grid": True,
        "no_gui": no_gui,
        "steps": 1,
        "arch": "cpu",
        "float_type": "f64",
        "use_mass_boundary": True,
        "do_small_advect": False,

        "Domain1": {
            "arch": "cpu",
            "float_type": "f64",
            "dim": 2,
            "grid_nx": domain1_grid_size_x,
            "grid_ny": domain1_grid_size_y,
            "domain_width": domain1_width,
            "domain_height": domain1_height,
            "offset": [domain1_offset_x, domain1_offset_y],
            "particles_per_grid": particles_per_grid,
            "sampling_method": "mesh",
            "elasticity_model": "linear",
            "dt": (5e-4 if implicit else 1e-5),
            "max_iter": 1,
            "solve_max_iter": 1000,
            "solve_init_iter": 100,
            "gravity": [0.0, -0.0],
            "material_params": [
                {
                    "id": 0,
                    "name": "default",
                    "E": 2e4,
                    "nu": 0.3,
                    "rho": 1000
                }
            ],
            "implicit": implicit,
            "use_auto_diff": False,
            "compare_diff": False,
            "bound": 1,
            "initial_velocity_y": -0,
            "damping": 0.0 if implicit else 0.5,
            "particle_damping": 0.99,
            "volume_forces": [
                {
                    "type": "rectangle",
                    "params": {
                        "range": [
                            [0, 1],
                            [domain1_force_bottom_norm, domain1_force_top_norm]
                        ]
                    },
                    "force": [0.0, -10.0]
                }
            ],
            "shapes": [
                {
                    "type": "ellipse",
                    "params": {
                        "center": domain1_center_norm,
                        "semi_axes": domain1_radius_norm
                    },
                    "operation": "add",
                    "material_id": 0
                },
                {
                    "type": "rectangle",
                    "params": {
                        "range": [
                            [0, 1],
                            [domain1_cut_top_norm, 1]
                        ]
                    },
                    "operation": "subtract"
                },
                {
                    "type": "rectangle",
                    "params": {
                        "range": [
                            [0, 1],
                            [0, domain1_cut_bottom_norm]
                        ]
                    },
                    "operation": "subtract"
                }
            ],
            "implicit_solver": "Newton",
            "boundary_size": 0.0001,
            "eta": eta
        },

        "Domain2": {
            "arch": "cpu",
            "float_type": "f64",
            "dim": 2,
            "grid_nx": domain2_grid_size_x,
            "grid_ny": domain2_grid_size_y,
            "domain_width": domain2_width,
            "domain_height": domain2_height,
            "offset": [domain2_offset_x, domain2_offset_y],
            "particles_per_grid": particles_per_grid,
            "sampling_method": "mesh",
            "elasticity_model": "linear",
            "dt": 1e-3 if implicit else 1e-5,
            "max_iter": 1,
            "solve_max_iter": 1000,
            "solve_init_iter": 100,
            "gravity": [0.0, -0.0],
            "material_params": [
                {
                    "id": 0,
                    "name": "default",
                    "E": 2e4,
                    "nu": 0.3,
                    "rho": 1000
                }
            ],
            "implicit": implicit,
            "use_auto_diff": False,
            "compare_diff": False,
            "bound": 3,
            "use_mesh_boundary": True,
            "initial_velocity_y": -0,
            "damping": 0.01 if implicit else 0.5,
            "particle_damping": 1.0,
            "shapes": [
                {
                    "type": "ellipse",
                    "params": {
                        "center": domain2_center_norm,
                        "semi_axes": domain2_radius_norm
                    },
                    "operation": "add",
                    "material_id": 0
                },
                {
                    "type": "rectangle",
                    "params": {
                        "range": [
                            [-10, 10],
                            [domain2_cut_top_norm, 10]
                        ]
                    },
                    "operation": "subtract"
                }
                # ,
                # {
                #     "type": "rectangle",
                #     "params": {
                #         "range": [
                #             [domain2_cut_left, domain2_cut_right],
                #             [domain2_cut_top_norm, add_overlap_top]
                #         ]
                #     },
                #     "operation": "add"
                # }
            ],
            "implicit_solver": "Newton",
            "boundary_size": 0.0001,
            "eta": eta
        }
    }

    # 根据is_schwarz参数决定保存行为
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if is_schwarz:
        # Schwarz模式：保存完整的双域配置
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Schwarz双域配置已保存到: {output_file}")

        # 同时保存Domain1配置到config_2d_test3.json（向后兼容）
        domain1_config = config["Domain1"].copy()
        if len(domain1_config["shapes"]) > 2:
            domain1_config["shapes"] = domain1_config["shapes"][:-1]  # 去掉最后一个subtract
            print("已去掉Domain1配置中的最后一个subtract操作")

        domain1_config["max_frames"] = config["max_frames"]
        domain1_config["use_record"] = config["use_record"]
        domain1_config["no_gui"] = config["no_gui"]

        domain1_output_file = "config/config_2d_test3.json"
        os.makedirs(os.path.dirname(domain1_output_file), exist_ok=True)
        with open(domain1_output_file, 'w') as f:
            json.dump(domain1_config, f, indent=4)
        print(f"Domain1配置已保存到: {domain1_output_file}")
    else:
        # 单域模式：只保存修改后的Domain1配置到指定文件
        domain1_config = config["Domain1"].copy()

        # 去掉最后一个subtract操作（保留椭圆add和第一个subtract）
        if len(domain1_config["shapes"]) > 2:
            domain1_config["shapes"] = domain1_config["shapes"][:-1]  # 去掉最后一个
            print("已去掉Domain1配置中的最后一个subtract操作")

        # 添加max_frames、use_record和no_gui
        domain1_config["max_frames"] = config["max_frames"]
        domain1_config["use_record"] = config["use_record"]
        domain1_config["no_gui"] = config["no_gui"]

        # 保存Domain1配置到指定的输出文件
        with open(output_file, 'w') as f:
            json.dump(domain1_config, f, indent=4)
        print(f"单域配置已保存到: {output_file}")

    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成test3的schwarz域分解配置文件')
    parser.add_argument('--domain2-thickness', type=float, default=0.05,
                       help='Domain2的厚度（底部小部分） (默认: 0.05)')
    parser.add_argument('--overlap-thickness', type=float, default=0.02,
                       help='重叠区域的厚度 (默认: 0.03)')
    parser.add_argument('--semicircle-center', nargs=2, type=float, default=[0.5, 0.45],
                       help='下半圆的圆心坐标 (默认: 0.5 0.42)')
    parser.add_argument('--semicircle-radius', type=float, default=0.4,
                       help='下半圆的半径 (默认: 0.4)')
    parser.add_argument('--force-layer-thickness', type=float, default=0.01,
                       help='施力层厚度 (默认: 0.01)')
    parser.add_argument('--domain1-grid-size', type=int, default=64,
                       help='Domain1网格大小 (默认: 32)')
    parser.add_argument('--domain2-grid-size', type=int, default=64,
                       help='Domain2网格大小 (默认: 32)')
    parser.add_argument('--grid-size', type=int, default=None,
                       help='同时设置两个域的网格大小（向后兼容）')
    parser.add_argument('--particles-per-grid', type=int, default=4,
                       help='每网格粒子数 (默认: 4)')
    parser.add_argument('--implicit', action='store_true',
                       help='是否使用隐式求解器 (默认: False)')
    parser.add_argument('--schwarz-max-iter', type=int, default=20,
                       help='Schwarz迭代的最大次数 (默认: 20)')
    parser.add_argument('--schwarz-eta', type=float, default=1e-3,
                       help='Schwarz迭代的收敛阈值 (默认: 1e-3)')
    parser.add_argument('--eta', type=float, default=1e-4,
                       help='隐式求解器的收敛阈值 (默认: 1e-4)')
    parser.add_argument('--output', default='config/schwarz_2d_test3.json',
                       help='输出文件路径 (默认: config/schwarz_2d_test3.json)')
    parser.add_argument('--no-gui', action='store_true',
                       help='禁用GUI界面 (默认: False)')
    parser.add_argument('--schwarz', action='store_true',
                       help='生成Schwarz双域配置，如果未指定则只生成单域配置 (默认: False)')

    args = parser.parse_args()

    # 向后兼容：如果指定了grid-size，则同时设置两个域的网格大小
    if args.grid_size is not None:
        args.domain1_grid_size = args.grid_size
        args.domain2_grid_size = args.grid_size

    print("生成test3的schwarz域分解配置")
    print("=" * 50)
    print(f"Domain2厚度: {args.domain2_thickness}")
    print(f"重叠厚度: {args.overlap_thickness}")
    print(f"下半圆中心: {args.semicircle_center}")
    print(f"下半圆半径: {args.semicircle_radius}")
    print(f"施力层厚度: {args.force_layer_thickness}")
    print(f"Domain1网格大小: {args.domain1_grid_size}")
    print(f"Domain2网格大小: {args.domain2_grid_size}")
    print(f"每网格粒子数: {args.particles_per_grid}")
    print()

    config = generate_test3_schwarz_config(
        domain2_thickness=args.domain2_thickness,
        overlap_thickness=args.overlap_thickness,
        semicircle_center=args.semicircle_center,
        semicircle_radius=args.semicircle_radius,
        force_layer_thickness=args.force_layer_thickness,
        domain1_grid_size=args.domain1_grid_size,
        domain2_grid_size=args.domain2_grid_size,
        particles_per_grid=args.particles_per_grid,
        output_file=args.output,
        implicit = args.implicit,
        schwarz_max_iter = args.schwarz_max_iter,
        schwarz_eta = args.schwarz_eta,
        eta = args.eta,
        no_gui = args.no_gui,
        is_schwarz = args.schwarz
    )

    print("\n配置生成完成!")

if __name__ == "__main__":
    main()