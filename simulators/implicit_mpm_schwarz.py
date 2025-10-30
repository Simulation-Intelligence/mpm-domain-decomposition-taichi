from implicit_mpm import *
from tools.domain_manager import DomainManager
from tools.boundary_exchanger import BoundaryExchanger
from tools.particle_state_manager import ParticleStateManager

from Util.Recorder import *

import matplotlib.pyplot as plt

from Util.Project import project

@ti.data_oriented
class MPM_Schwarz:
    def __init__(self, main_config: Config, no_gui=False, config_overrides=None):
        """
        基于Schwarz域分解的双域MPM求解器

        参数：
            main_config: 包含Domain1和Domain2配置的全局配置
            no_gui: 是否运行在无GUI模式
            config_overrides: 配置覆盖字典，支持域特定覆盖
                格式: {
                    'global': {'max_schwarz_iter': 5},  # 全局参数覆盖
                    'Domain1': {'dt': 1e-5},           # Domain1特定覆盖
                    'Domain2': {'material_params.0.E': 2e5}  # Domain2特定覆盖
                }
        """
        self.no_gui = no_gui

        # 应用配置覆盖
        self.main_config = self._apply_config_overrides(main_config, config_overrides)
        self.original_config = main_config  # 保存原始配置
        # 初始化域管理器
        self.domain_manager = DomainManager(self.main_config)
        
        # 获取域的引用（保持向后兼容）
        self.Domain1 = self.domain_manager.domain1
        self.Domain2 = self.domain_manager.domain2
        self.BigTimeDomain = self.domain_manager.big_time_domain
        self.SmallTimeDomain = self.domain_manager.small_time_domain
        
        # 初始化边界交换器
        self.boundary_exchanger = BoundaryExchanger(
            self.BigTimeDomain, self.SmallTimeDomain
        )
        
        # 初始化粒子状态管理器
        self.particle_state_manager = ParticleStateManager(
            self.SmallTimeDomain.particles
        )
        
        self.do_small_advect = self.main_config.get("do_small_advect", True)  # 是否执行小时间步长域的粒子自由运动

        # 其他公共参数初始化
        self.max_schwarz_iter = self.main_config.get("max_schwarz_iter", 1)  # Schwarz迭代次数
        self.steps=self.main_config.get("steps", 10)  # 迭代步数
        self.max_frames = self.main_config.get("max_frames", 60)  # 最大帧数
        self.use_record = self.main_config.get("use_record", False)  # 是否使用录制
        self.recorder = None
        self.visualize_grid = self.main_config.get("visualize_grid", False)
        if self.use_record:
            lines_begin = None
            lines_end = None
            lines_color = None
            if self.visualize_grid:
                lines_begin_1 = self.Domain1.get_grid_lines_begin()
                lines_end_1 = self.Domain1.get_grid_lines_end()
                lines_begin_2 = self.Domain2.get_grid_lines_begin()
                lines_end_2 = self.Domain2.get_grid_lines_end()
                lines_begin = np.concatenate([lines_begin_1, lines_begin_2])
                lines_end = np.concatenate([lines_end_1, lines_end_2])
                lines_color_1 = np.full(len(lines_begin_1), 0x66CCFF, dtype=np.uint32)
                lines_color_2 = np.full(len(lines_begin_2), 0xED553B, dtype=np.uint32)
                lines_color = np.concatenate([lines_color_1, lines_color_2])
            self.recorder = ParticleRecorder(
            palette=np.array([
                0x66CCFF,  # 域1普通粒子
                0xED553B,  # 域2普通粒子 
                0xFFFFFF   # 边界粒子
            ], dtype=np.uint32),
            max_frames=self.max_frames,
            lines_begin=lines_begin,
            lines_end=lines_end,
            lines_color=lines_color
        )

        self.use_mass_boundary = self.main_config.get("use_mass_boundary", False)

        self.residuals = []

        # Schwarz收敛参数
        self.schwarz_eta = self.main_config.get("schwarz_eta", 1e-6)  # 收敛容差

        # 存储上一次Schwarz迭代的grid_v，用于收敛判别
        self.Domain1_prev_grid_v = ti.Vector.field(self.Domain1.dim, dtype=ti.f32, shape=self.Domain1.grid.v.shape)
        self.Domain2_prev_grid_v = ti.Vector.field(self.Domain2.dim, dtype=ti.f32, shape=self.Domain2.grid.v.shape)

        # 只在非无GUI模式下初始化GUI
        if not self.no_gui:
            self.gui = ti.ui.Window("Implicit MPM Schwarz", res=(800, 800), vsync=False)
            # 预创建网格线条字段
            self._init_grid_lines()
            # 创建用于渲染的位置字段
            self._init_render_fields()
        else:
            self.gui = None

    def _init_grid_lines(self):
        """初始化网格线条字段，只创建一次"""
        if not self.visualize_grid:
            self.grid_vertices1 = None
            self.grid_vertices2 = None
            return

        # 获取网格线条数据
        lines_begin1 = np.array(self.Domain1.grid.get_lines_begin(), dtype=np.float32)
        lines_end1 = np.array(self.Domain1.grid.get_lines_end(), dtype=np.float32)
        lines_begin2 = np.array(self.Domain2.grid.get_lines_begin(), dtype=np.float32)
        lines_end2 = np.array(self.Domain2.grid.get_lines_end(), dtype=np.float32)

        # 创建Domain1网格线条字段
        if len(lines_begin1) > 0 and len(lines_end1) > 0:
            vertices1 = np.empty((len(lines_begin1) * 2, 2), dtype=np.float32)
            vertices1[0::2] = lines_begin1  # 偶数位置放begin点
            vertices1[1::2] = lines_end1    # 奇数位置放end点

            self.grid_vertices1 = ti.Vector.field(2, dtype=ti.f32, shape=len(vertices1))
            self.grid_vertices1.from_numpy(vertices1)
        else:
            self.grid_vertices1 = None

        # 创建Domain2网格线条字段
        if len(lines_begin2) > 0 and len(lines_end2) > 0:
            vertices2 = np.empty((len(lines_begin2) * 2, 2), dtype=np.float32)
            vertices2[0::2] = lines_begin2  # 偶数位置放begin点
            vertices2[1::2] = lines_end2    # 奇数位置放end点

            self.grid_vertices2 = ti.Vector.field(2, dtype=ti.f32, shape=len(vertices2))
            self.grid_vertices2.from_numpy(vertices2)
        else:
            self.grid_vertices2 = None

    def _init_render_fields(self):
        """初始化用于渲染的位置和颜色字段"""
        # 创建渲染位置字段，用于存储加上offset后的位置
        max_particles1 = self.Domain1.particles.x.shape[0]
        max_particles2 = self.Domain2.particles.x.shape[0]

        self.render_pos1 = ti.Vector.field(2, dtype=ti.f32, shape=max_particles1)
        self.render_pos2 = ti.Vector.field(2, dtype=ti.f32, shape=max_particles2)

        # 创建颜色字段，用于区分普通粒子和边界粒子
        self.render_color1 = ti.Vector.field(3, dtype=ti.f32, shape=max_particles1)
        self.render_color2 = ti.Vector.field(3, dtype=ti.f32, shape=max_particles2)

        # 存储offset作为Taichi字段，方便kernel访问
        self.domain1_offset = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.domain2_offset = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.domain1_scale = ti.field(dtype=ti.f32, shape=())
        self.domain2_scale = ti.field(dtype=ti.f32, shape=())

        # 初始化offset和scale
        self.domain1_offset[None] = ti.Vector([self.Domain1.offset[0], self.Domain1.offset[1]])
        self.domain2_offset[None] = ti.Vector([self.Domain2.offset[0], self.Domain2.offset[1]])
        self.domain1_scale[None] = self.Domain1.scale
        self.domain2_scale[None] = self.Domain2.scale

    @ti.kernel
    def _update_render_positions_domain1(self):
        """更新Domain1的渲染位置和颜色"""
        for i in range(self.Domain1.particles.x.shape[0]):
            # 计算变换后的位置：x * scale + offset
            self.render_pos1[i] = self.Domain1.particles.x[i] * self.domain1_scale[None] + self.domain1_offset[None]

            # 设置颜色：边界粒子为白色，普通粒子为Domain1颜色（青色）
            if self.Domain1.particles.is_boundary_particle[i]:
                self.render_color1[i] = ti.Vector([1.0, 1.0, 1.0])  # 白色
            else:
                self.render_color1[i] = ti.Vector([0.024, 0.522, 0.529])  # 青色

    @ti.kernel
    def _update_render_positions_domain2(self):
        """更新Domain2的渲染位置和颜色"""
        for i in range(self.Domain2.particles.x.shape[0]):
            # 计算变换后的位置：x * scale + offset
            self.render_pos2[i] = self.Domain2.particles.x[i] * self.domain2_scale[None] + self.domain2_offset[None]

            # 设置颜色：边界粒子为白色，普通粒子为Domain2颜色（红色）
            if self.Domain2.particles.is_boundary_particle[i]:
                self.render_color2[i] = ti.Vector([1.0, 1.0, 1.0])  # 白色
            else:
                self.render_color2[i] = ti.Vector([0.929, 0.333, 0.231])  # 红色

    def exchange_boundary_conditions(self):
        """
        设置边界条件
        """
        self.boundary_exchanger.exchange_boundary_conditions()

    def save_small_time_domain_particles(self):
        """
        保存小时间步长域的粒子数据到临时数组
        """
        self.particle_state_manager.save_particle_state()

    def restore_small_time_domain_particles(self):
        """
        恢复小时间步长域的粒子数据从临时数组
        """
        self.particle_state_manager.restore_particle_state()


    @ti.kernel
    def apply_average_grid_v(self):
        """
        将网格速度平均到粒子上
        """
        for I in ti.grouped(self.Domain1.grid.v):
            if self.Domain1.grid.m[I] > 0 and self.Domain2.grid.m[I] > 0:
                self.Domain1.grid.v[I] = (self.Domain1.grid.v[I] + self.Domain2.grid.v[I]) / 2.0
                self.Domain2.grid.v[I] = self.Domain1.grid.v[I]

    @ti.kernel
    def save_schwarz_iteration_grid_v(self):
        """
        保存当前Schwarz迭代的grid_v，用于下次迭代的收敛判别
        """
        for I in ti.grouped(self.Domain1.grid.v):
            self.Domain1_prev_grid_v[I] = self.Domain1.grid.v[I]

        for I in ti.grouped(self.Domain2.grid.v):
            self.Domain2_prev_grid_v[I] = self.Domain2.grid.v[I]

    @ti.kernel
    def check_schwarz_convergence(self) -> ti.f32:
        """
        检查Schwarz迭代收敛性：计算overlap区域中当前迭代和上一次迭代的grid_v差的inf-norm
        overlap区域定义为两个domain都有质量（grid.m > 0）的网格点
        """
        max_residual = 0.0

        # 检查overlap区域的收敛性：遍历Domain1，找到同时在Domain1和Domain2中都有质量的网格点
        for I in ti.grouped(self.Domain1.grid.v):
            if self.Domain1.grid.m[I] > 0:
                # 计算Domain1网格点在全局坐标系中的物理位置
                x = self.Domain1.grid.get_grid_pos(I) + self.Domain1.offset

                # 将全局坐标转换为Domain2的局部坐标
                x_local_in_domain2 = x - self.Domain2.offset

                # 计算在Domain2网格中的索引
                base, fx = self.Domain2.grid.particle_to_grid_base_and_fx(x_local_in_domain2)

                # 检查是否在Domain2的网格范围内且该点有质量
                if base[0] >= 0 and base[0] < self.Domain2.grid.nx and \
                   base[1] >= 0 and base[1] < self.Domain2.grid.ny and \
                   self.Domain2.grid.m[base] > 0:
                    # 这是真正的overlap区域，计算Domain1的速度收敛性
                    velocity_diff_norm = (self.Domain1.grid.v[I] - self.Domain1_prev_grid_v[I]).norm()
                    ti.atomic_max(max_residual, velocity_diff_norm)

        # 同样检查Domain2中的overlap区域
        for I in ti.grouped(self.Domain2.grid.v):
            if self.Domain2.grid.m[I] > 0:
                # 计算Domain2网格点在全局坐标系中的物理位置
                x = self.Domain2.grid.get_grid_pos(I) + self.Domain2.offset

                # 将全局坐标转换为Domain1的局部坐标
                x_local_in_domain1 = x - self.Domain1.offset

                # 计算在Domain1网格中的索引
                base, fx = self.Domain1.grid.particle_to_grid_base_and_fx(x_local_in_domain1)

                # 检查是否在Domain1的网格范围内且该点有质量
                if base[0] >= 0 and base[0] < self.Domain1.grid.nx and \
                   base[1] >= 0 and base[1] < self.Domain1.grid.ny and \
                   self.Domain1.grid.m[base] > 0:
                    # 这是真正的overlap区域，计算Domain2的速度收敛性
                    velocity_diff_norm = (self.Domain2.grid.v[I] - self.Domain2_prev_grid_v[I]).norm()
                    ti.atomic_max(max_residual, velocity_diff_norm)

        return max_residual

    @ti.kernel
    def check_grid_v_residual(self) -> ti.f32:
        """
        计算残差 (保留原有函数用于兼容性)
        """
        residual = 0.0
        cnt = 0
        for I in ti.grouped(self.Domain1.grid.v):
            if self.Domain1.grid.m[I] > 0 and self.Domain2.grid.m[I] > 0:
                ti.atomic_add(residual,(self.Domain1.grid.v[I]-self.Domain2.grid.v[I]).norm())
                ti.atomic_add(cnt, 1)

        return residual / cnt
    
    
    def step(self):
        for _ in range(self.steps):
            """
            执行一步Schwarz迭代
            """

            # 1.P2G: 将粒子的质量和动量传递到网格上
            self.domain_manager.pre_step()

            # 记录上一步网格速度
            self.boundary_exchanger.save_grid_velocities()

            self.save_small_time_domain_particles()

            residuals = []

            self.exchange_boundary_conditions()
            self.boundary_exchanger.save_boundary_velocities()

            # 2.迭代求解两个子域 - 使用收敛判别
            converged = False
            for i in range(self.max_schwarz_iter):
                print(f"Schwarz Iteration {i+1}/{self.max_schwarz_iter}")

                # 在第一次迭代前保存初始grid_v状态
                if i == 0:
                    self.save_schwarz_iteration_grid_v()

                timesteps = self.domain_manager.get_timestep_ratio()
                self.BigTimeDomain.solve()

                for j in range(timesteps):
                    self.boundary_exchanger.interpolate_boundary_velocity((j+1) / timesteps)
                    self.SmallTimeDomain.solve()
                    self.SmallTimeDomain.g2p(self.SmallTimeDomain.dt)
                    if self.do_small_advect:
                        self.SmallTimeDomain.particles.advect(self.SmallTimeDomain.dt)
                        if j < timesteps - 1:
                            self.SmallTimeDomain.pre_p2g_sub()
                            self.SmallTimeDomain.p2g()
                            self.SmallTimeDomain.post_p2g()
                    else:
                        self.SmallTimeDomain.solver.save_previous_velocity()

                # 检查收敛性（从第二次迭代开始）
                if i > 0:
                    convergence_residual = self.check_schwarz_convergence()
                    print(f"  Convergence residual (inf-norm): {convergence_residual:.2e}")
                    residuals.append(convergence_residual)

                    if convergence_residual < self.schwarz_eta:
                        print(f"  Schwarz iterations converged after {i+1} iterations!")
                        converged = True

                # 准备下一次迭代或结束
                if not converged and i < self.max_schwarz_iter - 1:
                    # 保存当前状态用于下次迭代的收敛判别
                    self.save_schwarz_iteration_grid_v()

                    self.exchange_boundary_conditions()
                    self.boundary_exchanger.small_time_domain_boundary_v_next.copy_from(self.SmallTimeDomain.grid.boundary_v)
                    if self.do_small_advect:
                        self.restore_small_time_domain_particles()
                        self.SmallTimeDomain.pre_p2g_sub()
                        self.SmallTimeDomain.p2g()
                        self.SmallTimeDomain.post_p2g()
                    else:
                        self.SmallTimeDomain.grid.v_prev.copy_from(self.boundary_exchanger.small_time_domain_temp_grid_v)
                        self.SmallTimeDomain.grid.v.copy_from(self.boundary_exchanger.small_time_domain_temp_grid_v)

                if not self.do_small_advect:
                    self.restore_small_time_domain_particles()

                # 如果收敛了就提前退出
                if converged:
                    break

            if not converged:
                print(f"Warning: Schwarz iterations did not converge after {self.max_schwarz_iter} iterations")

            # self.apply_average_grid_v()

            self.residuals.append(residuals)

            # 3.G2P: 将网格的速度传递回粒子上
            # 4.粒子自由运动
            self.domain_manager.finalize_step(self.do_small_advect)

    def render(self):
        # 在无GUI模式下跳过渲染
        if self.no_gui or self.gui is None:
            return

        # 使用kernel更新渲染位置
        self._update_render_positions_domain1()
        self._update_render_positions_domain2()

        # 使用ti.ui.Window的canvas API
        canvas = self.gui.get_canvas()
        canvas.set_background_color((0.067, 0.184, 0.255))

        # 使用预计算的渲染位置字段和颜色字段
        if self.Domain1.particles.x.shape[0] > 0:
            canvas.circles(self.render_pos1, radius=0.001, per_vertex_color=self.render_color1)

        if self.Domain2.particles.x.shape[0] > 0:
            canvas.circles(self.render_pos2, radius=0.001, per_vertex_color=self.render_color2)

        #绘制grid网格
        if self.visualize_grid:
            # 使用预创建的网格线条字段
            if self.grid_vertices1 is not None:
                canvas.lines(self.grid_vertices1, 0.001, color=(0.024, 0.522, 0.529))

            if self.grid_vertices2 is not None:
                canvas.lines(self.grid_vertices2, 0.001, color=(0.929, 0.333, 0.231))

        self.gui.show()
        # 合并两域粒子数据
        if self.recorder is None:
            return
        print("Frame", len(self.recorder.frame_data))
        all_pos = np.concatenate([
            self.render_pos1.to_numpy(),
            self.render_pos2.to_numpy()
        ])
        
        # 生成颜色索引 (0:域1普通, 1:域2普通, 2:边界)
        d1_colors = np.where(
            self.Domain1.particles.is_boundary_particle.to_numpy(),
            2, 0
        )
        d2_colors = np.where(
            self.Domain2.particles.is_boundary_particle.to_numpy(),
            2, 1
        )
        all_colors = np.concatenate([d1_colors, d2_colors]).astype(np.uint32)
        
        # 捕获帧
        self.recorder.capture(all_pos, all_colors)
    
    def save_stress_data(self, frame_number):
        """保存两个域的应力数据"""
        import json
        import os

        # 计算两个域的应力
        print("正在计算Domain1的应力...")
        self.Domain1.solver.compute_stress_strain()

        print("正在计算Domain2的应力...")
        self.Domain2.solver.compute_stress_strain()

        # 获取两个域的数据
        stress_data1 = self.Domain1.particles.stress.to_numpy()
        positions1_raw = self.Domain1.particles.x.to_numpy()
        boundary_flags1 = self.Domain1.particles.is_boundary_particle.to_numpy()

        stress_data2 = self.Domain2.particles.stress.to_numpy()
        positions2_raw = self.Domain2.particles.x.to_numpy()
        boundary_flags2 = self.Domain2.particles.is_boundary_particle.to_numpy()

        # 先过滤掉位置为(0,0)的粒子，基于原始坐标 - Domain1
        if self.Domain1.dim == 2:
            # 2D情况：过滤(0,0)位置
            valid_mask1 = ~((positions1_raw[:, 0] == 0.0) & (positions1_raw[:, 1] == 0.0))
        else:
            # 3D情况：过滤(0,0,0)位置
            valid_mask1 = ~((positions1_raw[:, 0] == 0.0) & (positions1_raw[:, 1] == 0.0) & (positions1_raw[:, 2] == 0.0))

        # 先过滤掉位置为(0,0)的粒子，基于原始坐标 - Domain2
        if self.Domain2.dim == 2:
            # 2D情况：过滤(0,0)位置
            valid_mask2 = ~((positions2_raw[:, 0] == 0.0) & (positions2_raw[:, 1] == 0.0))
        else:
            # 3D情况：过滤(0,0,0)位置
            valid_mask2 = ~((positions2_raw[:, 0] == 0.0) & (positions2_raw[:, 1] == 0.0) & (positions2_raw[:, 2] == 0.0))

        # 应用过滤掩码并加上offset得到全局坐标
        import numpy as np
        filtered_stress_data1 = stress_data1[valid_mask1]
        filtered_positions1 = positions1_raw[valid_mask1] + np.array(self.Domain1.offset)
        filtered_boundary_flags1 = boundary_flags1[valid_mask1]

        filtered_stress_data2 = stress_data2[valid_mask2]
        filtered_positions2 = positions2_raw[valid_mask2] + np.array(self.Domain2.offset)
        filtered_boundary_flags2 = boundary_flags2[valid_mask2]
        
        print(f"Domain1: Filtered out {np.sum(~valid_mask1)} particles at origin position")
        print(f"Domain1: Saving data for {len(filtered_positions1)} particles")
        print(f"Domain2: Filtered out {np.sum(~valid_mask2)} particles at origin position") 
        print(f"Domain2: Saving data for {len(filtered_positions2)} particles")
        
        # 创建统一的输出目录结构
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = "experiment_results"
        output_dir = os.path.join(base_output_dir, f"schwarz_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 分别保存两个域的数据
        # Domain1数据
        np.save(f"{output_dir}/domain1_stress_frame_{frame_number}.npy", filtered_stress_data1)
        np.save(f"{output_dir}/domain1_positions_frame_{frame_number}.npy", filtered_positions1)
        np.save(f"{output_dir}/domain1_boundary_flags_frame_{frame_number}.npy", filtered_boundary_flags1)

        # Domain2数据
        np.save(f"{output_dir}/domain2_stress_frame_{frame_number}.npy", filtered_stress_data2)
        np.save(f"{output_dir}/domain2_positions_frame_{frame_number}.npy", filtered_positions2)
        np.save(f"{output_dir}/domain2_boundary_flags_frame_{frame_number}.npy", filtered_boundary_flags2)

        # 保存actual mass信息
        try:
            # Domain1 actual mass
            actual_masses1 = self.Domain1.solver.get_volume_force_masses()
            if actual_masses1:
                np.save(f"{output_dir}/domain1_actual_masses_frame_{frame_number}.npy", np.array(actual_masses1))
                print(f"Saved Domain1 actual masses: {actual_masses1}")
        except Exception as e:
            print(f"Warning: Could not save Domain1 actual masses: {e}")

        try:
            # Domain2 actual mass
            actual_masses2 = self.Domain2.solver.get_volume_force_masses()
            if actual_masses2:
                np.save(f"{output_dir}/domain2_actual_masses_frame_{frame_number}.npy", np.array(actual_masses2))
                print(f"Saved Domain2 actual masses: {actual_masses2}")
        except Exception as e:
            print(f"Warning: Could not save Domain2 actual masses: {e}")
        
        # 计算两个域的von Mises应力
        def compute_von_mises_stress(stress_data, dim):
            von_mises_stress = []
            for i in range(stress_data.shape[0]):
                if dim == 2:
                    s = stress_data[i]
                    # 2D von Mises应力
                    von_mises = np.sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2)
                else:
                    # 3D von Mises应力
                    s = stress_data[i]
                    von_mises = np.sqrt(0.5*((s[0,0]-s[1,1])**2 + (s[1,1]-s[2,2])**2 + (s[2,2]-s[0,0])**2) + 3*(s[0,1]**2 + s[1,2]**2 + s[2,0]**2))
                von_mises_stress.append(von_mises)
            return von_mises_stress
        
        von_mises_stress1 = compute_von_mises_stress(filtered_stress_data1, self.Domain1.dim)
        von_mises_stress2 = compute_von_mises_stress(filtered_stress_data2, self.Domain2.dim)
        
        # 保存统计信息
        stats = {
            "frame": frame_number,
            "domain1": {
                "n_particles": int(filtered_stress_data1.shape[0]),
                "n_particles_total": int(stress_data1.shape[0]),
                "n_particles_filtered": int(np.sum(~valid_mask1)),
                "dimension": int(self.Domain1.dim),
                "von_mises_stress": {
                    "min": float(np.min(von_mises_stress1)),
                    "max": float(np.max(von_mises_stress1)),
                    "mean": float(np.mean(von_mises_stress1)),
                    "std": float(np.std(von_mises_stress1))
                }
            },
            "domain2": {
                "n_particles": int(filtered_stress_data2.shape[0]),
                "n_particles_total": int(stress_data2.shape[0]),
                "n_particles_filtered": int(np.sum(~valid_mask2)),
                "dimension": int(self.Domain2.dim),
                "von_mises_stress": {
                    "min": float(np.min(von_mises_stress2)),
                    "max": float(np.max(von_mises_stress2)),
                    "mean": float(np.mean(von_mises_stress2)),
                    "std": float(np.std(von_mises_stress2))
                }
            }
        }
        
        with open(f"{output_dir}/stress_stats_frame_{frame_number}.json", 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"两域应力数据已保存到 {output_dir}/")
        print(f"Domain1 von Mises应力范围: {stats['domain1']['von_mises_stress']['min']:.3e} - {stats['domain1']['von_mises_stress']['max']:.3e}")
        print(f"Domain1 平均von Mises应力: {stats['domain1']['von_mises_stress']['mean']:.3e}")
        print(f"Domain2 von Mises应力范围: {stats['domain2']['von_mises_stress']['min']:.3e} - {stats['domain2']['von_mises_stress']['max']:.3e}")
        print(f"Domain2 平均von Mises应力: {stats['domain2']['von_mises_stress']['mean']:.3e}")
        
        return output_dir

    def _apply_config_overrides(self, config: Config, overrides: dict = None):
        """
        应用配置覆盖参数，支持域特定的配置覆盖

        参数:
            config: 原始配置对象
            overrides: 要覆盖的配置字典，格式：
                {
                    'global': {'max_schwarz_iter': 5, 'steps': 100},  # 全局参数覆盖
                    'Domain1': {'dt': 1e-5, 'material_params.0.E': 2e5},  # Domain1特定覆盖
                    'Domain2': {'implicit': False}  # Domain2特定覆盖
                }

        返回:
            应用覆盖后的新配置对象
        """
        if overrides is None:
            return config

        # 创建配置的深拷贝
        import copy
        new_config_data = copy.deepcopy(config.data)

        # 应用全局覆盖
        if 'global' in overrides:
            for key, value in overrides['global'].items():
                self._set_nested_value(new_config_data, key, value)
                print(f"全局配置覆盖: {key} = {value}")

        # 应用域特定覆盖
        for domain_name in ['Domain1', 'Domain2']:
            if domain_name in overrides and domain_name in new_config_data:
                for key, value in overrides[domain_name].items():
                    self._set_nested_value(new_config_data[domain_name], key, value)
                    print(f"{domain_name}配置覆盖: {key} = {value}")

        # 创建新的Config对象
        new_config = Config()
        new_config.data = new_config_data
        return new_config

    def _set_nested_value(self, data_dict: dict, key_path: str, value):
        """
        在嵌套字典中设置值，支持点分路径

        参数:
            data_dict: 目标字典
            key_path: 键路径，如 'material_params.0.E' 或 'dt'
            value: 要设置的值
        """
        keys = key_path.split('.')
        current = data_dict

        # 遍历到最后一级的父级
        for key in keys[:-1]:
            # 如果键是数字，表示列表索引
            if key.isdigit():
                key = int(key)
                if not isinstance(current, list):
                    raise ValueError(f"尝试用数字索引访问非列表对象: {key}")
                if key >= len(current):
                    raise ValueError(f"列表索引超出范围: {key}")
                current = current[key]
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]

        # 设置最终值
        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
            if not isinstance(current, list):
                raise ValueError(f"尝试用数字索引访问非列表对象: {final_key}")
            if final_key >= len(current):
                raise ValueError(f"列表索引超出范围: {final_key}")

        current[final_key] = value


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Schwarz Dual Domain MPM Simulator')
    parser.add_argument('--config', type=str, default='config/schwarz_2d_test3.json',
                       help='Configuration file path')
    parser.add_argument('--no-gui', action='store_true',
                       help='Run without GUI (headless mode)')

    args = parser.parse_args()

    # 读取配置文件
    cfg = Config(path=args.config)
    float_type=ti.f32 if cfg.get("float_type", "f32") == "f32" else ti.f64
    arch=cfg.get("arch", "cpu")
    if arch == "cuda":
        arch = ti.cuda
    elif arch == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, default_fp=float_type, device_memory_GB=20, log_level=ti.ERROR)

    # 创建Schwarz域分解MPM实例
    mpm = MPM_Schwarz(cfg, no_gui=args.no_gui)

    frame_count = 0
    target_frames = cfg.get("max_frames", 60)

    if args.no_gui:
        # 无GUI模式：直接运行到目标帧数
        print(f"Running simulation in headless mode for {target_frames} frames...")
        while frame_count < target_frames:
            mpm.step()
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Progress: {frame_count}/{target_frames} frames")
        print("Simulation completed.")
    else:
        # GUI模式：使用传统的GUI循环
        while mpm.gui and mpm.gui.running:
            mpm.step()
            mpm.render()
            frame_count += 1

            # 自动停止条件
            if frame_count >= mpm.max_frames:
                break
    
    
    # #绘制最后1组residuals
    # for i in range(1):
    #     plt.plot(mpm.residuals[-i-1])
    # plt.ylabel('Residual')
    # plt.xlabel('Iteration')
    # plt.xscale('log')  # X轴对数化
    # plt.yscale('log')  # Y轴对数化
    # plt.show()
    
    #
    print("记录最终帧的应力数据...")
    mpm.save_stress_data(frame_count)

    if mpm.recorder is None or args.no_gui:
        print("Simulation finished.")
        exit()

    print("Playback finished.")
    # 播放录制动画
    mpm.recorder.play(loop=True, fps=60)