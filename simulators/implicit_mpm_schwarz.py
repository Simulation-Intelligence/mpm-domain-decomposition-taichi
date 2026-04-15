from implicit_mpm import *
from tools.domain_manager import DomainManager
from tools.boundary_exchanger import BoundaryExchanger
from tools.particle_state_manager import ParticleStateManager
from tools.performance_stats import SchwarzPerformanceStats

from Util.Recorder import *
from Util.CheckpointIO import (
    CONFIG_BACKUP_FILENAME,
    get_checkpoint_dir,
    get_config_backup_path,
    load_manifest,
    load_particle_state,
    load_performance_stats,
    save_manifest,
    save_particle_state,
    save_performance_stats,
)
from Util.Visualization import cv2_draw_particles, cv2_draw_particles_colored, FPSCounter
from Util.ExperimentDir import create_experiment_directory
from Util.StressIO import is_point_in_regions, compute_von_mises_stress, save_stress_frame, write_stress_stats_json

import matplotlib.pyplot as plt
import time
import gc
import math
import numpy as np
import cv2


_PROJECT_PHI = math.radians(28.0)
_PROJECT_THETA = math.radians(32.0)
_PROJECT_CP = math.cos(_PROJECT_PHI)
_PROJECT_SP = math.sin(_PROJECT_PHI)
_PROJECT_CT = math.cos(_PROJECT_THETA)
_PROJECT_ST = math.sin(_PROJECT_THETA)

_COLOR_DOMAIN1_NORMAL_HEX = (int(0.024 * 255) << 16) | (int(0.522 * 255) << 8) | int(0.529 * 255)
_COLOR_DOMAIN2_NORMAL_HEX = (int(0.929 * 255) << 16) | (int(0.333 * 255) << 8) | int(0.231 * 255)
_COLOR_BOUNDARY_HEX = 0xFFFFFF

@ti.data_oriented
class MPM_Schwarz:
    def __init__(self, main_config: Config, no_gui=False, config_overrides=None, config_path=None):
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
        self.config_path = config_path

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
        use_p2g_d2_to_d1 = self.main_config.get("use_p2g_d2_to_d1_boundary", False)
        self.boundary_exchanger = BoundaryExchanger(
            self.BigTimeDomain, self.SmallTimeDomain,
            use_p2g_d2_to_d1=use_p2g_d2_to_d1
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
        self.render_fps = self.main_config.get("render_fps", 60)  # GUI帧率上限
        self.use_record = self.main_config.get("use_record", False)  # 是否使用录制

        # 应力保存配置
        self.stress_save_interval = self.main_config.get("stress_save_interval", 0)  # 0表示仅最后一帧保存
        self.saved_stress_frames = []  # 记录已保存应力数据的帧号
        self.recorder = None
        self.visualize_grid = self.main_config.get("visualize_grid", False)
        self.visualize_boundary_grid = self.main_config.get("visualize_boundary_grid", False)
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
            max_particles=self.Domain1.particles.x.shape[0] + self.Domain2.particles.x.shape[0],
            lines_begin=lines_begin,
            lines_end=lines_end,
            lines_color=lines_color
        )

        self.use_mass_boundary = self.main_config.get("use_mass_boundary", False)

        # 残差列表（限制大小防止内存泄漏）
        self.residuals = []
        self.max_residuals_to_keep = 1000  # 只保留最近1000帧的残差

        # 帧计数器（用于渐进重力）
        self.current_frame = 0
        self.completed_frames = 0

        # 性能统计（限制大小防止内存泄漏）
        self.perf_stats = SchwarzPerformanceStats(max_frames_to_keep=1000000)
        self.enable_perf_stats = self.main_config.get("enable_perf_stats", True)

        # 内存profiling（用于调试内存泄漏）
        self.enable_memory_profile = self.main_config.get("profile_memory", False)
        if self.enable_memory_profile:
            from tools.memory_profiler import MemoryProfiler, FrameStageProfiler
            self.mem_profiler = MemoryProfiler(enabled=True, snapshot_top_n=10)
            self.frame_stage_profiler = FrameStageProfiler(enabled=True)
            print("内存profiling已启用")
        else:
            self.mem_profiler = None
            self.frame_stage_profiler = None

        # Schwarz收敛参数
        self.schwarz_eta = self.main_config.get("schwarz_eta", 1e-6)  # 收敛容差

        # 速度收敛检测参数
        self.velocity_convergence_threshold = self.main_config.get("velocity_convergence_threshold", 0)
        self.velocity_convergence_check_interval = self.main_config.get("velocity_convergence_check_interval", 10)
        self.velocity_convergence_consecutive_required = self.main_config.get("velocity_convergence_consecutive_required", 10)
        self.enable_velocity_convergence = self.velocity_convergence_threshold > 0
        self.velocity_convergence_consecutive_count = 0  # 连续收敛次数计数器
        if self.enable_velocity_convergence:
            print(f"启用双域速度收敛检测: 阈值={self.velocity_convergence_threshold:.2e}, 检测间隔={self.velocity_convergence_check_interval}帧, 需要连续{self.velocity_convergence_consecutive_required}次")

        # 存储上一次Schwarz迭代的grid_v，用于收敛判别
        self.Domain1_prev_grid_v = ti.Vector.field(self.Domain1.dim, dtype=ti.f64, shape=self.Domain1.grid.v.shape)
        self.Domain2_prev_grid_v = ti.Vector.field(self.Domain2.dim, dtype=ti.f64, shape=self.Domain2.grid.v.shape)

        # Schwarz收敛检查的accumulator字段（避免在kernel中使用局部变量+atomic）
        self.convergence_max_residual = ti.field(dtype=ti.f64, shape=())
        self.convergence_max_velocity = ti.field(dtype=ti.f64, shape=())

        # 计算渲染缩放系数（基于Domain1的最大边长，使几何体在屏幕上正确缩放）
        self.dim1 = self.Domain1.dim
        self.dim2 = self.Domain2.dim
        if self.dim1 == 3:
            _d1_max_side = max(self.Domain1.grid.domain_width, self.Domain1.grid.domain_height, self.Domain1.grid.domain_depth)
        else:
            _d1_max_side = max(self.Domain1.grid.domain_width, self.Domain1.grid.domain_height)
        self.render_scale = 1.0 / _d1_max_side

        # 只在非无GUI模式下初始化GUI
        self.gui = None  # ti.GUI 不再使用
        self._win_name = "Implicit MPM Schwarz"
        self._fps_counter = FPSCounter()  # FPS 计数器
        if not self.no_gui:
            cv2.namedWindow(self._win_name, cv2.WINDOW_AUTOSIZE)
            self._cv2_running = True
            self._pos1_np = None
            self._pos2_np = None
            self._color1_np = None
            self._color2_np = None
            self._record_pos_np = None
            self._record_color_np = None
            self._record_color1_np = None
            self._record_color2_np = None
            # 预创建网格线条字段
            self._init_grid_lines()
            # 创建用于渲染的位置字段
            self._init_render_fields()
            self._ensure_render_buffers()
        else:
            self._cv2_running = False
            self._pos1_np = None
            self._pos2_np = None
            self._color1_np = None
            self._color2_np = None
            self._record_pos_np = None
            self._record_color_np = None
            self._record_color1_np = None
            self._record_color2_np = None

        # 初始化完成后的内存检查点
        if self.mem_profiler:
            self.mem_profiler.checkpoint("initialization_complete")

    def _init_grid_lines(self):
        """初始化网格线条字段，只创建一次"""
        if not self.visualize_grid:
            self.grid_vertices1 = None
            self.grid_vertices2 = None
            self._grid_vertices1_np = None
            self._grid_vertices2_np = None
            self.boundary_grid_points1 = None
            self.boundary_grid_points2 = None
            self.move_mass_grid_points1 = None
            self.move_mass_grid_points2 = None
            return

        # 获取网格线条数据
        lines_begin1 = np.array(self.Domain1.grid.get_lines_begin(), dtype=np.float32)
        lines_end1 = np.array(self.Domain1.grid.get_lines_end(), dtype=np.float32)
        lines_begin2 = np.array(self.Domain2.grid.get_lines_begin(), dtype=np.float32)
        lines_end2 = np.array(self.Domain2.grid.get_lines_end(), dtype=np.float32)

        # 创建Domain1网格线条字段
        if len(lines_begin1) > 0 and len(lines_end1) > 0:
            vertices1 = np.empty((len(lines_begin1) * 2, 2), dtype=np.float32)
            vertices1[0::2] = lines_begin1 * self.render_scale  # 偶数位置放begin点
            vertices1[1::2] = lines_end1 * self.render_scale    # 奇数位置放end点

            self.grid_vertices1 = ti.Vector.field(2, dtype=ti.f32, shape=len(vertices1))
            self.grid_vertices1.from_numpy(vertices1)
            self._grid_vertices1_np = vertices1
        else:
            self.grid_vertices1 = None
            self._grid_vertices1_np = None

        # 创建Domain2网格线条字段
        if len(lines_begin2) > 0 and len(lines_end2) > 0:
            vertices2 = np.empty((len(lines_begin2) * 2, 2), dtype=np.float32)
            vertices2[0::2] = lines_begin2 * self.render_scale  # 偶数位置放begin点
            vertices2[1::2] = lines_end2 * self.render_scale    # 奇数位置放end点

            self.grid_vertices2 = ti.Vector.field(2, dtype=ti.f32, shape=len(vertices2))
            self.grid_vertices2.from_numpy(vertices2)
            self._grid_vertices2_np = vertices2
        else:
            self.grid_vertices2 = None
            self._grid_vertices2_np = None

        # 初始化边界网格点的可视化字段
        if self.visualize_boundary_grid:
            self._init_boundary_grid_points()
        else:
            self.boundary_grid_points1 = None
            self.boundary_grid_points2 = None
            self.move_mass_grid_points1 = None
            self.move_mass_grid_points2 = None

    def _init_boundary_grid_points(self):
        """初始化边界网格点的可视化字段"""
        # 为每个域预分配最大可能的边界网格点数
        max_grid_points1 = self.Domain1.grid.get_total_grid_points()
        max_grid_points2 = self.Domain2.grid.get_total_grid_points()

        self.boundary_grid_points1 = ti.Vector.field(2, dtype=ti.f32, shape=max_grid_points1)
        self.boundary_grid_points2 = ti.Vector.field(2, dtype=ti.f32, shape=max_grid_points2)
        self.move_mass_grid_points1 = ti.Vector.field(2, dtype=ti.f32, shape=max_grid_points1)
        self.move_mass_grid_points2 = ti.Vector.field(2, dtype=ti.f32, shape=max_grid_points2)
        self._boundary_grid_points1_np = np.zeros((max_grid_points1, 2), dtype=np.float32)
        self._boundary_grid_points2_np = np.zeros((max_grid_points2, 2), dtype=np.float32)
        self._move_mass_grid_points1_np = np.zeros((max_grid_points1, 2), dtype=np.float32)
        self._move_mass_grid_points2_np = np.zeros((max_grid_points2, 2), dtype=np.float32)

        # 记录实际边界网格点数量的字段
        self.boundary_grid_count1 = ti.field(ti.i32, shape=())
        self.boundary_grid_count2 = ti.field(ti.i32, shape=())
        self.move_mass_grid_count1 = ti.field(ti.i32, shape=())
        self.move_mass_grid_count2 = ti.field(ti.i32, shape=())

    @staticmethod
    def _vec3_to_hex(colors_np):
        """Convert (N,3) float color array to (N,) hex integer array for ti.GUI"""
        r = (colors_np[:, 0] * 255).astype(np.uint32)
        g = (colors_np[:, 1] * 255).astype(np.uint32)
        b = (colors_np[:, 2] * 255).astype(np.uint32)
        return (r << 16) | (g << 8) | b

    def _init_render_fields(self):
        """初始化用于渲染的位置和颜色字段"""
        # 创建渲染位置字段，用于存储加上offset后的位置
        max_particles1 = self.Domain1.particles.x.shape[0]
        max_particles2 = self.Domain2.particles.x.shape[0]

        self.render_pos1 = ti.Vector.field(2, dtype=ti.f32, shape=max_particles1)
        self.render_pos2 = ti.Vector.field(2, dtype=ti.f32, shape=max_particles2)

        # 创建颜色字段（hex），用于区分普通粒子和边界粒子
        self.render_color1_hex = ti.field(dtype=ti.u32, shape=max_particles1)
        self.render_color2_hex = ti.field(dtype=ti.u32, shape=max_particles2)

        # 存储offset作为Taichi字段，方便kernel访问（2D域kernel使用；3D域仅占位不使用）
        self.domain1_offset = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.domain2_offset = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.domain1_scale = ti.field(dtype=ti.f32, shape=())
        self.domain2_scale = ti.field(dtype=ti.f32, shape=())
        self.render_scale_field = ti.field(dtype=ti.f32, shape=())
        if self.dim1 == 2:
            self.domain1_offset[None] = ti.Vector([self.Domain1.offset[0], self.Domain1.offset[1]])
        if self.dim2 == 2:
            self.domain2_offset[None] = ti.Vector([self.Domain2.offset[0], self.Domain2.offset[1]])

        # 初始化scale
        self.domain1_scale[None] = self.Domain1.scale
        self.domain2_scale[None] = self.Domain2.scale
        self.render_scale_field[None] = self.render_scale

    @ti.kernel
    def _update_render_positions_domain1(self):
        """更新Domain1的渲染位置和颜色（仅2D域）"""
        if ti.static(self.Domain1.dim == 2):
            for i in range(self.Domain1.particles.x.shape[0]):
                self.render_pos1[i] = (self.Domain1.particles.x[i] * self.domain1_scale[None] + self.domain1_offset[None]) * self.render_scale_field[None]
                if self.Domain1.particles.is_boundary_particle[i]:
                    self.render_color1_hex[i] = _COLOR_BOUNDARY_HEX
                else:
                    self.render_color1_hex[i] = _COLOR_DOMAIN1_NORMAL_HEX

    @ti.kernel
    def _update_render_positions_domain2(self):
        """更新Domain2的渲染位置和颜色（仅2D域）"""
        if ti.static(self.Domain2.dim == 2):
            for i in range(self.Domain2.particles.x.shape[0]):
                self.render_pos2[i] = (self.Domain2.particles.x[i] * self.domain2_scale[None] + self.domain2_offset[None]) * self.render_scale_field[None]
                if self.Domain2.particles.is_boundary_particle[i]:
                    self.render_color2_hex[i] = _COLOR_BOUNDARY_HEX
                else:
                    self.render_color2_hex[i] = _COLOR_DOMAIN2_NORMAL_HEX

    @ti.kernel
    def _update_boundary_grid_points_domain1(self):
        """更新Domain1的边界网格点"""
        boundary_count = 0
        move_mass_count = 0

        for I in ti.grouped(self.Domain1.grid.is_boundary_grid):
            grid_pos = self.Domain1.grid.get_grid_pos(I)
            render_pos = (grid_pos * self.domain1_scale[None] + self.domain1_offset[None]) * self.render_scale_field[None]

            # 检查is_boundary_grid标记（vector field，检查任意分量是否非零）
            is_bnd1 = False
            for _d in ti.static(range(self.Domain1.grid.dim)):
                if self.Domain1.grid.is_boundary_grid[I][_d] != 0:
                    is_bnd1 = True
            if is_bnd1:
                idx = ti.atomic_add(boundary_count, 1)
                if idx < self.boundary_grid_points1.shape[0]:
                    self.boundary_grid_points1[idx] = render_pos

        self.boundary_grid_count1[None] = boundary_count
        self.move_mass_grid_count1[None] = move_mass_count

    @ti.kernel
    def _update_boundary_grid_points_domain2(self):
        """更新Domain2的边界网格点"""
        boundary_count = 0
        move_mass_count = 0

        for I in ti.grouped(self.Domain2.grid.is_boundary_grid):
            grid_pos = self.Domain2.grid.get_grid_pos(I)
            render_pos = (grid_pos * self.domain2_scale[None] + self.domain2_offset[None]) * self.render_scale_field[None]

            # 检查is_boundary_grid标记（vector field，检查任意分量是否非零）
            is_bnd2 = False
            for _d in ti.static(range(self.Domain2.grid.dim)):
                if self.Domain2.grid.is_boundary_grid[I][_d] != 0:
                    is_bnd2 = True
            if is_bnd2:
                idx = ti.atomic_add(boundary_count, 1)
                if idx < self.boundary_grid_points2.shape[0]:
                    self.boundary_grid_points2[idx] = render_pos

        self.boundary_grid_count2[None] = boundary_count
        self.move_mass_grid_count2[None] = move_mass_count

    @ti.kernel
    def _update_move_mass_grid_points_domain1(self):
        """更新Domain1的move_mass_field > 0的网格点"""
        move_mass_count = 0

        for I in ti.grouped(self.Domain1.grid.move_mass_field):
            if self.Domain1.grid.move_mass_field[I] > 0:
                grid_pos = self.Domain1.grid.get_grid_pos(I)
                render_pos = (grid_pos * self.domain1_scale[None] + self.domain1_offset[None]) * self.render_scale_field[None]
                idx = ti.atomic_add(move_mass_count, 1)
                if idx < self.move_mass_grid_points1.shape[0]:
                    self.move_mass_grid_points1[idx] = render_pos

        self.move_mass_grid_count1[None] = move_mass_count

    @ti.kernel
    def _update_move_mass_grid_points_domain2(self):
        """更新Domain2的move_mass_field > 0的网格点"""
        move_mass_count = 0

        for I in ti.grouped(self.Domain2.grid.move_mass_field):
            if self.Domain2.grid.move_mass_field[I] > 0:
                grid_pos = self.Domain2.grid.get_grid_pos(I)
                render_pos = (grid_pos * self.domain2_scale[None] + self.domain2_offset[None]) * self.render_scale_field[None]
                idx = ti.atomic_add(move_mass_count, 1)
                if idx < self.move_mass_grid_points2.shape[0]:
                    self.move_mass_grid_points2[idx] = render_pos

        self.move_mass_grid_count2[None] = move_mass_count

    def _ensure_render_buffers(self):
        n1 = self.Domain1.particles.x.shape[0]
        n2 = self.Domain2.particles.x.shape[0]
        if self._pos1_np is None or self._pos1_np.shape[0] != n1:
            self._pos1_np = np.zeros((n1, 2), dtype=np.float32)
            self._color1_np = np.zeros((n1,), dtype=np.uint32)
            self._record_color1_np = np.zeros((n1,), dtype=np.uint32)
        if self._pos2_np is None or self._pos2_np.shape[0] != n2:
            self._pos2_np = np.zeros((n2, 2), dtype=np.float32)
            self._color2_np = np.zeros((n2,), dtype=np.uint32)
            self._record_color2_np = np.zeros((n2,), dtype=np.uint32)
        total = n1 + n2
        if self._record_pos_np is None or self._record_pos_np.shape[0] != total:
            self._record_pos_np = np.zeros((total, 2), dtype=np.float32)
            self._record_color_np = np.zeros((total,), dtype=np.uint32)

    @ti.kernel
    def _export_render_domain1(self, pos_out: ti.types.ndarray(), color_out: ti.types.ndarray()):
        for i in range(self.Domain1.particles.x.shape[0]):
            pos_out[i, 0] = self.render_pos1[i][0]
            pos_out[i, 1] = self.render_pos1[i][1]
            color_out[i] = ti.cast(self.render_color1_hex[i], ti.u32)

    @ti.kernel
    def _export_render_domain2(self, pos_out: ti.types.ndarray(), color_out: ti.types.ndarray()):
        for i in range(self.Domain2.particles.x.shape[0]):
            pos_out[i, 0] = self.render_pos2[i][0]
            pos_out[i, 1] = self.render_pos2[i][1]
            color_out[i] = ti.cast(self.render_color2_hex[i], ti.u32)

    @ti.kernel
    def _fill_render_domain1_3d(self, pos_out: ti.types.ndarray(), color_out: ti.types.ndarray()):
        for i in range(self.Domain1.particles.x.shape[0]):
            pos = (self.Domain1.particles.x[i] * self.domain1_scale[None] + ti.Vector([self.Domain1.offset[0], self.Domain1.offset[1], self.Domain1.offset[2]])) * self.render_scale
            x = pos[0] - 0.5
            y = pos[1] - 0.5
            z = pos[2] - 0.5
            x2 = x * _PROJECT_CP + z * _PROJECT_SP
            z2 = z * _PROJECT_CP - x * _PROJECT_SP
            u = x2
            v = y * _PROJECT_CT + z2 * _PROJECT_ST
            pos_out[i, 0] = u + 0.5
            pos_out[i, 1] = v + 0.5
            if self.Domain1.particles.is_boundary_particle[i]:
                color_out[i] = ti.cast(_COLOR_BOUNDARY_HEX, ti.u32)
            else:
                color_out[i] = ti.cast(_COLOR_DOMAIN1_NORMAL_HEX, ti.u32)

    @ti.kernel
    def _fill_render_domain2_3d(self, pos_out: ti.types.ndarray(), color_out: ti.types.ndarray()):
        for i in range(self.Domain2.particles.x.shape[0]):
            pos = (self.Domain2.particles.x[i] * self.domain2_scale[None] + ti.Vector([self.Domain2.offset[0], self.Domain2.offset[1], self.Domain2.offset[2]])) * self.render_scale
            x = pos[0] - 0.5
            y = pos[1] - 0.5
            z = pos[2] - 0.5
            x2 = x * _PROJECT_CP + z * _PROJECT_SP
            z2 = z * _PROJECT_CP - x * _PROJECT_SP
            u = x2
            v = y * _PROJECT_CT + z2 * _PROJECT_ST
            pos_out[i, 0] = u + 0.5
            pos_out[i, 1] = v + 0.5
            if self.Domain2.particles.is_boundary_particle[i]:
                color_out[i] = ti.cast(_COLOR_BOUNDARY_HEX, ti.u32)
            else:
                color_out[i] = ti.cast(_COLOR_DOMAIN2_NORMAL_HEX, ti.u32)

    @ti.kernel
    def _fill_record_color_domain1(self, out: ti.types.ndarray()):
        for i in range(self.Domain1.particles.x.shape[0]):
            if self.Domain1.particles.is_boundary_particle[i]:
                out[i] = ti.cast(2, ti.u32)
            else:
                out[i] = ti.cast(0, ti.u32)

    @ti.kernel
    def _fill_record_color_domain2(self, out: ti.types.ndarray()):
        for i in range(self.Domain2.particles.x.shape[0]):
            if self.Domain2.particles.is_boundary_particle[i]:
                out[i] = ti.cast(2, ti.u32)
            else:
                out[i] = ti.cast(1, ti.u32)

    @ti.kernel
    def _export_boundary_points_domain1(self, pos_out: ti.types.ndarray()):
        for i in range(self.boundary_grid_points1.shape[0]):
            if i < self.boundary_grid_count1[None]:
                pos_out[i, 0] = self.boundary_grid_points1[i][0]
                pos_out[i, 1] = self.boundary_grid_points1[i][1]

    @ti.kernel
    def _export_boundary_points_domain2(self, pos_out: ti.types.ndarray()):
        for i in range(self.boundary_grid_points2.shape[0]):
            if i < self.boundary_grid_count2[None]:
                pos_out[i, 0] = self.boundary_grid_points2[i][0]
                pos_out[i, 1] = self.boundary_grid_points2[i][1]

    @ti.kernel
    def _export_move_mass_points_domain1(self, pos_out: ti.types.ndarray()):
        for i in range(self.move_mass_grid_points1.shape[0]):
            if i < self.move_mass_grid_count1[None]:
                pos_out[i, 0] = self.move_mass_grid_points1[i][0]
                pos_out[i, 1] = self.move_mass_grid_points1[i][1]

    @ti.kernel
    def _export_move_mass_points_domain2(self, pos_out: ti.types.ndarray()):
        for i in range(self.move_mass_grid_points2.shape[0]):
            if i < self.move_mass_grid_count2[None]:
                pos_out[i, 0] = self.move_mass_grid_points2[i][0]
                pos_out[i, 1] = self.move_mass_grid_points2[i][1]

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
    def check_schwarz_convergence_kernel(self):
        """
        检查Schwarz迭代收敛性：计算overlap区域中当前迭代和上一次迭代的grid_v差的相对误差
        overlap区域定义为两个domain都有质量（grid.m > 0）的网格点
        相对误差定义为: max(|v_current - v_previous|) / max(|v_current|)
        使用field accumulator避免内存泄漏
        """
        # 重置accumulator
        self.convergence_max_residual[None] = 0.0
        self.convergence_max_velocity[None] = 0.0

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
                in_d2 = base[0] >= 0 and base[0] < self.Domain2.grid.nx and \
                        base[1] >= 0 and base[1] < self.Domain2.grid.ny
                if ti.static(self.Domain2.grid.dim == 3):
                    in_d2 = in_d2 and base[2] >= 0 and base[2] < self.Domain2.grid.nz
                if in_d2 and self.Domain2.grid.m[base] > 0:
                    # 这是真正的overlap区域，计算Domain1的速度收敛性
                    velocity_diff_norm = (self.Domain1.grid.v[I] - self.Domain1_prev_grid_v[I]).norm()
                    ti.atomic_max(self.convergence_max_residual[None], velocity_diff_norm)

                    # 计算当前速度的模长
                    velocity_magnitude = self.Domain1.grid.v[I].norm()
                    ti.atomic_max(self.convergence_max_velocity[None], velocity_magnitude)

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
                in_d1 = base[0] >= 0 and base[0] < self.Domain1.grid.nx and \
                        base[1] >= 0 and base[1] < self.Domain1.grid.ny
                if ti.static(self.Domain1.grid.dim == 3):
                    in_d1 = in_d1 and base[2] >= 0 and base[2] < self.Domain1.grid.nz
                if in_d1 and self.Domain1.grid.m[base] > 0:
                    # 这是真正的overlap区域，计算Domain2的速度收敛性
                    velocity_diff_norm = (self.Domain2.grid.v[I] - self.Domain2_prev_grid_v[I]).norm()
                    ti.atomic_max(self.convergence_max_residual[None], velocity_diff_norm)

                    # 计算当前速度的模长
                    velocity_magnitude = self.Domain2.grid.v[I].norm()
                    ti.atomic_max(self.convergence_max_velocity[None], velocity_magnitude)

    def check_schwarz_convergence(self) -> float:
        """
        检查Schwarz迭代收敛性的wrapper函数
        返回相对残差: max_velocity_diff / max_velocity
        当max_velocity接近零时，返回绝对残差
        """
        self.check_schwarz_convergence_kernel()

        # 定义小的epsilon用于零速度判定
        velocity_epsilon = 1e-14

        max_velocity = self.convergence_max_velocity[None]
        max_residual = self.convergence_max_residual[None]

        # 计算相对残差，带零速度保护
        if max_velocity < velocity_epsilon:
            # 速度接近零，使用绝对残差
            relative_residual = max_residual
        else:
            # 计算相对残差
            relative_residual = max_residual / max_velocity

        return relative_residual

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

    def check_velocity_convergence(self, frame):
        """检查双域速度是否收敛（需要连续多次小于阈值）"""
        if not self.enable_velocity_convergence:
            return False

        # 只在指定间隔检查
        if frame % self.velocity_convergence_check_interval != 0:
            return False

        # 分别计算两个域的最大网格速度
        max_v_mag_domain1 = self.Domain1.grid.compute_max_velocity_magnitude()
        max_v_mag_domain2 = self.Domain2.grid.compute_max_velocity_magnitude()

        # 取两个域的最大值
        max_v_mag = max(max_v_mag_domain1, max_v_mag_domain2)

        # 检查是否小于阈值
        below_threshold = max_v_mag < self.velocity_convergence_threshold

        if below_threshold:
            self.velocity_convergence_consecutive_count += 1
            print(f"Frame {frame}: 双域速度小于阈值! 最大网格速度: {max_v_mag:.2e} < 阈值: {self.velocity_convergence_threshold:.2e} (连续第{self.velocity_convergence_consecutive_count}次)")
            print(f"  Domain1最大速度: {max_v_mag_domain1:.2e}, Domain2最大速度: {max_v_mag_domain2:.2e}")

            # 检查是否达到连续要求次数
            if self.velocity_convergence_consecutive_count >= self.velocity_convergence_consecutive_required:
                print(f"Frame {frame}: 达到双域速度收敛条件! 连续{self.velocity_convergence_consecutive_count}次小于阈值")
                return True
        else:
            # 重置连续计数器
            if self.velocity_convergence_consecutive_count > 0:
                print(f"Frame {frame}: 双域速度超过阈值，重置连续计数器。最大网格速度: {max_v_mag:.2e} >= 阈值: {self.velocity_convergence_threshold:.2e}")
            else:
                print(f"Frame {frame}: 双域最大网格速度: {max_v_mag:.2e} (Domain1: {max_v_mag_domain1:.2e}, Domain2: {max_v_mag_domain2:.2e})")
            self.velocity_convergence_consecutive_count = 0

        return False

    def step(self):
        for _ in range(self.steps):
            """
            执行一步Schwarz迭代
            """
            # 更新帧计数器
            self.current_frame += 1

            # 更新两个域的当前帧数（用于渐进重力）
            self.Domain1.solver.update_frame(self.current_frame)
            self.Domain2.solver.update_frame(self.current_frame)

            # 帧开始内存检查点
            if self.mem_profiler:
                self.mem_profiler.checkpoint(f"frame_{self.current_frame}_start")

            # 开始帧统计
            if self.enable_perf_stats:
                self.perf_stats.start_frame()

            # 1.P2G: 将粒子的质量和动量传递到网格上
            if self.frame_stage_profiler:
                self.frame_stage_profiler.start_stage("P2G")

            self.domain_manager.pre_step()

            if self.frame_stage_profiler:
                self.frame_stage_profiler.end_stage()
            if self.mem_profiler:
                self.mem_profiler.checkpoint(f"frame_{self.current_frame}_p2g_complete")

            # 记录上一步网格速度
            self.boundary_exchanger.save_grid_velocities()

            self.save_small_time_domain_particles()

            residuals = []

            self.exchange_boundary_conditions()
            self.boundary_exchanger.save_boundary_velocities()

            # 2.迭代求解两个子域 - 使用收敛判别
            if self.frame_stage_profiler:
                self.frame_stage_profiler.start_stage("schwarz_iterations")

            converged = False
            for i in range(self.max_schwarz_iter):
                print(f"Schwarz Iteration {i+1}/{self.max_schwarz_iter}")

                # 只在前10帧进行详细的迭代级 profiling，避免 JSON 文件过大
                enable_detailed_profile = self.mem_profiler and self.current_frame <= 10

                # 迭代开始检查点
                if enable_detailed_profile:
                    self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_start")

                # 在第一次迭代前保存初始grid_v状态
                if i == 0:
                    self.save_schwarz_iteration_grid_v()
                    if enable_detailed_profile:
                        self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_after_save_grid_v")

                timesteps = self.domain_manager.get_timestep_ratio()

                # 大域求解（带统计）
                t_big_start = time.perf_counter()
                big_newton_iters = self.BigTimeDomain.solve()
                if big_newton_iters is None:
                    big_newton_iters = 0
                t_big_end = time.perf_counter()

                # 大域求解后检查点（对所有迭代）
                if enable_detailed_profile:
                    self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_after_big_solve")

                # 小域多步求解（带统计）
                t_small_start = time.perf_counter()
                small_newton_total = 0
                small_substep_iters = []

                for j in range(timesteps):
                    self.boundary_exchanger.interpolate_boundary_velocity((j+1) / timesteps)
                    small_iter = self.SmallTimeDomain.solve()
                    if small_iter is None:
                        small_iter = 0
                    small_newton_total += small_iter
                    small_substep_iters.append(small_iter)

                    if j == timesteps - 1:
                        break
                    self.SmallTimeDomain.g2p(self.SmallTimeDomain.dt)
                    if self.do_small_advect:
                        self.SmallTimeDomain.particles.advect(self.SmallTimeDomain.dt)
                        if j < timesteps - 1:
                            self.SmallTimeDomain.pre_p2g()
                            self.SmallTimeDomain.p2g()
                            self.SmallTimeDomain.post_p2g()
                    else:
                        self.SmallTimeDomain.solver.save_previous_velocity()

                t_small_end = time.perf_counter()

                # 小域求解后检查点
                if enable_detailed_profile:
                    self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_after_small_solve")

                # 检查收敛性（从第二次迭代开始）
                convergence_residual = 1.0  # 默认值
                if i > 0:
                    convergence_residual = self.check_schwarz_convergence()
                    max_v = self.convergence_max_velocity[None]
                    max_diff = self.convergence_max_residual[None]
                    print(f"  Convergence: relative={convergence_residual:.2e}, max_v={max_v:.2e}, max_diff={max_diff:.2e}")
                    residuals.append(convergence_residual)

                    if convergence_residual < self.schwarz_eta:
                        print(f"  Schwarz iterations converged after {i+1} iterations!")
                        converged = True

                    # 收敛性检查后检查点
                    if enable_detailed_profile:
                        self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_after_convergence_check")

                # 记录本次 Schwarz 迭代的统计
                if self.enable_perf_stats:
                    self.perf_stats.record_schwarz_iteration(
                        residual=convergence_residual,
                        big_newton_iters=big_newton_iters,
                        small_newton_iters=small_newton_total,
                        small_substep_iters=small_substep_iters,
                        big_solve_time=t_big_end - t_big_start,
                        small_solve_time=t_small_end - t_small_start,
                    )

                # 准备下一次迭代或结束
                if not converged and i < self.max_schwarz_iter - 1:
                    # 保存当前状态用于下次迭代的收敛判别
                    self.save_schwarz_iteration_grid_v()
                    if enable_detailed_profile:
                        self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_after_save_for_next")

                    self.exchange_boundary_conditions()
                    self.boundary_exchanger.small_time_domain_boundary_v_next.copy_from(self.SmallTimeDomain.grid.boundary_v)
                    if enable_detailed_profile:
                        self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_after_boundary_exchange")

                    if self.do_small_advect:
                        self.restore_small_time_domain_particles()
                        self.SmallTimeDomain.pre_p2g()
                        self.SmallTimeDomain.p2g()
                        self.SmallTimeDomain.post_p2g()
                        if enable_detailed_profile:
                            self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_after_restore_and_p2g")
                    else:
                        self.SmallTimeDomain.grid.v_prev.copy_from(self.boundary_exchanger.small_time_domain_temp_grid_v)
                        self.SmallTimeDomain.grid.v.copy_from(self.boundary_exchanger.small_time_domain_temp_grid_v)
                        if enable_detailed_profile:
                            self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_after_grid_copy")

                if not self.do_small_advect:
                    self.restore_small_time_domain_particles()
                    if enable_detailed_profile:
                        self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_after_final_restore")

                # 迭代结束检查点
                if enable_detailed_profile:
                    self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_iter_{i}_end")

                # 如果收敛了就提前退出
                if converged:
                    break

            if not converged:
                print(f"Warning: Schwarz iterations did not converge after {self.max_schwarz_iter} iterations")

            if self.frame_stage_profiler:
                self.frame_stage_profiler.end_stage()
            if self.mem_profiler:
                self.mem_profiler.checkpoint(f"frame_{self.current_frame}_schwarz_complete")

            # self.apply_average_grid_v()

            self.residuals.append(residuals)

            # 限制 residuals 大小，防止内存泄漏
            if len(self.residuals) > self.max_residuals_to_keep:
                self.residuals = self.residuals[-self.max_residuals_to_keep:]

            # 结束帧统计
            if self.enable_perf_stats:
                self.perf_stats.end_frame(converged=converged)

            # 3.G2P: 将网格的速度传递回粒子上
            # 4.粒子自由运动
            if self.frame_stage_profiler:
                self.frame_stage_profiler.start_stage("G2P_and_advect")

            self.domain_manager.finalize_step(self.do_small_advect)

            if self.frame_stage_profiler:
                self.frame_stage_profiler.end_stage()

            # 帧结束检查点和记录
            if self.mem_profiler:
                stage_data = self.frame_stage_profiler.get_stages() if self.frame_stage_profiler else None
                self.mem_profiler.record_frame(self.current_frame, stage_data)
                self.mem_profiler.checkpoint(f"frame_{self.current_frame}_complete")
                if self.frame_stage_profiler:
                    self.frame_stage_profiler.reset()

            if self.current_frame % 10000 == 0:
                if self.mem_profiler:
                    self.mem_profiler.checkpoint(f"before_full_gc_frame_{self.current_frame}")
                gc.collect()  # 完全GC，清理所有代
                if self.mem_profiler:
                    self.mem_profiler.checkpoint(f"after_full_gc_frame_{self.current_frame}")

    def render(self):
        # 在无GUI模式下跳过渲染
        if self.no_gui or not self._cv2_running:
            return

        if self.mem_profiler:
            self.mem_profiler.checkpoint(f"render_frame_{self.current_frame}_start")

        self._ensure_render_buffers()

        # 更新渲染位置：2D域用kernel导出，3D域用kernel投影
        if self.dim1 == 2:
            self._update_render_positions_domain1()
            self._export_render_domain1(self._pos1_np, self._color1_np)
        else:
            self._fill_render_domain1_3d(self._pos1_np, self._color1_np)
        if self.dim2 == 2:
            self._update_render_positions_domain2()
            self._export_render_domain2(self._pos2_np, self._color2_np)
        else:
            self._fill_render_domain2_3d(self._pos2_np, self._color2_np)

        # 更新边界网格点（如果启用了边界网格可视化）
        if self.visualize_boundary_grid:
            self._update_boundary_grid_points_domain1()
            self._update_boundary_grid_points_domain2()

            # 只有当域有move_boundary配置时才更新move_mass网格点
            if hasattr(self.Domain1.grid, 'has_move_boundary') and self.Domain1.grid.has_move_boundary:
                self._update_move_mass_grid_points_domain1()
            else:
                self.move_mass_grid_count1[None] = 0

            if hasattr(self.Domain2.grid, 'has_move_boundary') and self.Domain2.grid.has_move_boundary:
                self._update_move_mass_grid_points_domain2()
            else:
                self.move_mass_grid_count2[None] = 0

        # 用 cv2 渲染（非阻塞，避免 WSL2 XSync 卡死）
        img = np.full((800, 800, 3), (65, 47, 17), dtype=np.uint8)  # 背景色 0x112F41

        # 渲染Domain1粒子（逐粒子颜色）
        if self.Domain1.particles.x.shape[0] > 0:
            cv2_draw_particles_colored(img, self._pos1_np, self._color1_np, radius=2)

        # 渲染Domain2粒子（逐粒子颜色）
        if self.Domain2.particles.x.shape[0] > 0:
            cv2_draw_particles_colored(img, self._pos2_np, self._color2_np, radius=2)

        # 绘制grid网格线
        if self.visualize_grid:
            if self._grid_vertices1_np is not None:
                verts = self._grid_vertices1_np
                p1s = (verts[::2] * 800).astype(np.int32)
                p2s = (verts[1::2] * 800).astype(np.int32)
                p1s[:, 1] = 800 - p1s[:, 1]
                p2s[:, 1] = 800 - p2s[:, 1]
                for p1, p2 in zip(p1s, p2s):
                    cv2.line(img, tuple(p1), tuple(p2), (135, 133, 6), 1)  # 0x068587

            if self._grid_vertices2_np is not None:
                verts = self._grid_vertices2_np
                p1s = (verts[::2] * 800).astype(np.int32)
                p2s = (verts[1::2] * 800).astype(np.int32)
                p1s[:, 1] = 800 - p1s[:, 1]
                p2s[:, 1] = 800 - p2s[:, 1]
                for p1, p2 in zip(p1s, p2s):
                    cv2.line(img, tuple(p1), tuple(p2), (59, 85, 237), 1)  # 0xED553B

        # 绘制边界网格点
        if self.visualize_boundary_grid and self.visualize_grid:
            if self.boundary_grid_points1 is not None and self.boundary_grid_count1[None] > 0:
                self._export_boundary_points_domain1(self._boundary_grid_points1_np)
                pts = self._boundary_grid_points1_np[:self.boundary_grid_count1[None]]
                cv2_draw_particles(img, pts, radius=3, bgr_color=(153, 153, 255))  # 0xFF9999

            if self.boundary_grid_points2 is not None and self.boundary_grid_count2[None] > 0:
                self._export_boundary_points_domain2(self._boundary_grid_points2_np)
                pts = self._boundary_grid_points2_np[:self.boundary_grid_count2[None]]
                cv2_draw_particles(img, pts, radius=3, bgr_color=(255, 153, 153))  # 0x9999FF

            if self.move_mass_grid_points1 is not None and self.move_mass_grid_count1[None] > 0:
                self._export_move_mass_points_domain1(self._move_mass_grid_points1_np)
                pts = self._move_mass_grid_points1_np[:self.move_mass_grid_count1[None]]
                cv2_draw_particles(img, pts, radius=4, bgr_color=(255, 0, 255))  # 0xFF00FF

            if self.move_mass_grid_points2 is not None and self.move_mass_grid_count2[None] > 0:
                self._export_move_mass_points_domain2(self._move_mass_grid_points2_np)
                pts = self._move_mass_grid_points2_np[:self.move_mass_grid_count2[None]]
                cv2_draw_particles(img, pts, radius=4, bgr_color=(255, 0, 255))  # 0xFF00FF

        self._fps_counter.draw_on_image(img)  # 显示 FPS
        cv2.imshow(self._win_name, img)
        self._fps_counter.update()  # 更新 FPS 计数器
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            self._cv2_running = False
        elif cv2.getWindowProperty(self._win_name, cv2.WND_PROP_VISIBLE) < 1:
            self._cv2_running = False
        # 合并两域粒子数据
        if self.recorder is None:
            if self.mem_profiler:
                self.mem_profiler.checkpoint(f"render_frame_{self.current_frame}_end")
            return
        print("Frame", len(self.recorder.frame_data))
        n1 = self.Domain1.particles.x.shape[0]
        n2 = self.Domain2.particles.x.shape[0]
        total = n1 + n2

        if n1 > 0:
            np.copyto(self._record_pos_np[:n1], self._pos1_np)
            self._fill_record_color_domain1(self._record_color1_np)
            np.copyto(self._record_color_np[:n1], self._record_color1_np)
        if n2 > 0:
            np.copyto(self._record_pos_np[n1:total], self._pos2_np)
            self._fill_record_color_domain2(self._record_color2_np)
            np.copyto(self._record_color_np[n1:total], self._record_color2_np)

        # 捕获帧
        self.recorder.capture(self._record_pos_np[:total], self._record_color_np[:total])

        if self.mem_profiler:
            self.mem_profiler.checkpoint(f"render_frame_{self.current_frame}_end")
    
    def _create_experiment_directory(self, config_path=None):
        """创建实验目录并备份配置文件（委托给 Util.ExperimentDir）"""
        return create_experiment_directory("schwarz", config_path)

    def attach_experiment_dir(self, experiment_dir):
        """绑定已有实验目录，恢复运行时继续写回该目录。"""
        self._experiment_dir = os.fspath(experiment_dir)
        self._stress_data_dir = os.path.join(self._experiment_dir, "stress_data")
        os.makedirs(self._experiment_dir, exist_ok=True)
        os.makedirs(self._stress_data_dir, exist_ok=True)

    def save_checkpoint(self):
        """保存最新 Schwarz 帧边界断点。"""
        if not hasattr(self, '_experiment_dir') or not hasattr(self, '_stress_data_dir'):
            self._experiment_dir, self._stress_data_dir = self._create_experiment_directory(self.config_path)

        checkpoint_dir = get_checkpoint_dir(self._experiment_dir)
        save_particle_state(self.Domain1.particles, checkpoint_dir / "domain1")
        save_particle_state(self.Domain2.particles, checkpoint_dir / "domain2")

        manifest = {
            "simulator_type": "schwarz",
            "config_path": CONFIG_BACKUP_FILENAME,
            "checkpoint_frame": int(self.completed_frames),
            "completed_frames": int(self.completed_frames),
            "current_frame": int(self.current_frame),
            "saved_stress_frames": sorted({int(frame) for frame in self.saved_stress_frames}),
            "velocity_convergence_consecutive_count": int(self.velocity_convergence_consecutive_count),
        }
        save_manifest(self._experiment_dir, manifest)
        if self.enable_perf_stats:
            save_performance_stats(self.perf_stats, self._experiment_dir)
        print(f"Schwarz断点已保存到: {checkpoint_dir}")
        return checkpoint_dir

    def load_checkpoint(self, experiment_dir):
        """从已有实验目录恢复 Schwarz 运行态。"""
        self.attach_experiment_dir(experiment_dir)
        manifest = load_manifest(self._experiment_dir, expected_simulator_type="schwarz")
        checkpoint_dir = get_checkpoint_dir(self._experiment_dir)

        load_particle_state(self.Domain1.particles, checkpoint_dir / "domain1")
        load_particle_state(self.Domain2.particles, checkpoint_dir / "domain2")

        self.completed_frames = int(manifest.get("completed_frames", 0))
        self.current_frame = int(manifest.get("current_frame", 0))
        self.saved_stress_frames = sorted({int(frame) for frame in manifest.get("saved_stress_frames", [])})
        self.velocity_convergence_consecutive_count = int(
            manifest.get("velocity_convergence_consecutive_count", 0)
        )

        if self.enable_perf_stats:
            load_performance_stats(self.perf_stats, self._experiment_dir)
        self.Domain1.solver.update_frame(self.current_frame)
        self.Domain2.solver.update_frame(self.current_frame)

        print(
            f"已恢复Schwarz断点: completed_frames={self.completed_frames}, "
            f"current_frame={self.current_frame}, 已保存stress帧={len(self.saved_stress_frames)}"
        )
        return manifest

    def save_stress_data(self, frame_number):
        """保存两个域的应力数据"""
        import json
        import os

        # 计算两个域的应力
        print("正在计算Domain1的应力...")
        self.Domain1.solver.compute_stress_strain_with_averaging()

        print("正在计算Domain2的应力...")
        self.Domain2.solver.compute_stress_strain_with_averaging()

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

        # 创建或获取实验目录
        if not hasattr(self, '_experiment_dir') or not hasattr(self, '_stress_data_dir'):
            self._experiment_dir, self._stress_data_dir = self._create_experiment_directory(self.config_path)

        # 计算两个域的von Mises应力（使用Util函数）
        von_mises_stress1 = compute_von_mises_stress(filtered_stress_data1, self.Domain1.dim)
        von_mises_stress2 = compute_von_mises_stress(filtered_stress_data2, self.Domain2.dim)

        # 为当前帧创建单独的子目录
        frame_dir = f"{self._stress_data_dir}/frame_{frame_number}"
        os.makedirs(frame_dir, exist_ok=True)

        # 分别保存两个域的数据到该帧的子目录
        # Domain1数据
        np.save(f"{frame_dir}/domain1_stress.npy", filtered_stress_data1)
        np.save(f"{frame_dir}/domain1_positions.npy", filtered_positions1)
        np.save(f"{frame_dir}/domain1_boundary_flags.npy", filtered_boundary_flags1)
        np.save(f"{frame_dir}/domain1_von_mises.npy", von_mises_stress1)

        # Domain2数据
        np.save(f"{frame_dir}/domain2_stress.npy", filtered_stress_data2)
        np.save(f"{frame_dir}/domain2_positions.npy", filtered_positions2)
        np.save(f"{frame_dir}/domain2_boundary_flags.npy", filtered_boundary_flags2)
        np.save(f"{frame_dir}/domain2_von_mises.npy", von_mises_stress2)

        # 保存两个域的网格应力（基于各自F_grid）和网格质量
        d1_grid_stress, d1_grid_mass, d1_grid_meta = self.Domain1.solver.get_grid_stress_from_F_grid_numpy()
        np.save(f"{frame_dir}/domain1_grid_stress.npy", d1_grid_stress)
        np.save(f"{frame_dir}/domain1_grid_mass.npy", d1_grid_mass)
        with open(f"{frame_dir}/domain1_grid_stress_meta.json", 'w') as f:
            json.dump(d1_grid_meta, f, indent=2)

        d2_grid_stress, d2_grid_mass, d2_grid_meta = self.Domain2.solver.get_grid_stress_from_F_grid_numpy()
        np.save(f"{frame_dir}/domain2_grid_stress.npy", d2_grid_stress)
        np.save(f"{frame_dir}/domain2_grid_mass.npy", d2_grid_mass)
        with open(f"{frame_dir}/domain2_grid_stress_meta.json", 'w') as f:
            json.dump(d2_grid_meta, f, indent=2)

        print(
            f"Domain1网格应力已保存: shape={d1_grid_stress.shape}, "
            f"有效网格数={d1_grid_meta['valid_grid_count']}"
        )
        print(
            f"Domain2网格应力已保存: shape={d2_grid_stress.shape}, "
            f"有效网格数={d2_grid_meta['valid_grid_count']}"
        )

        # 保存actual mass信息到该帧的子目录
        try:
            # Domain1 actual mass
            actual_masses1 = self.Domain1.solver.get_volume_force_masses()
            if actual_masses1:
                np.save(f"{frame_dir}/domain1_actual_masses.npy", np.array(actual_masses1))
                print(f"Saved Domain1 actual masses: {actual_masses1}")
        except Exception as e:
            print(f"Warning: Could not save Domain1 actual masses: {e}")

        try:
            # Domain2 actual mass
            actual_masses2 = self.Domain2.solver.get_volume_force_masses()
            if actual_masses2:
                np.save(f"{frame_dir}/domain2_actual_masses.npy", np.array(actual_masses2))
                print(f"Saved Domain2 actual masses: {actual_masses2}")
        except Exception as e:
            print(f"Warning: Could not save Domain2 actual masses: {e}")
        
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
        
        with open(f"{frame_dir}/stress_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"两域应力数据已保存到 {self._experiment_dir}/")
        print(f"stress数据保存在: {self._stress_data_dir}/")
        print(f"Domain1 von Mises应力范围: {stats['domain1']['von_mises_stress']['min']:.3e} - {stats['domain1']['von_mises_stress']['max']:.3e}")
        print(f"Domain1 平均von Mises应力: {stats['domain1']['von_mises_stress']['mean']:.3e}")
        print(f"Domain2 von Mises应力范围: {stats['domain2']['von_mises_stress']['min']:.3e} - {stats['domain2']['von_mises_stress']['max']:.3e}")
        print(f"Domain2 平均von Mises应力: {stats['domain2']['von_mises_stress']['mean']:.3e}")

        # 清理大型numpy数组，释放内存
        del stress_data1, positions1_raw, boundary_flags1
        del stress_data2, positions2_raw, boundary_flags2
        del filtered_stress_data1, filtered_positions1, filtered_boundary_flags1
        del filtered_stress_data2, filtered_positions2, filtered_boundary_flags2
        del valid_mask1, valid_mask2
        del von_mises_stress1, von_mises_stress2
        del d1_grid_stress, d1_grid_mass, d1_grid_meta
        del d2_grid_stress, d2_grid_mass, d2_grid_meta

        return self._experiment_dir

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
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--config', type=str, default=None,
                             help='Configuration file path')
    input_group.add_argument('--dir', type=str, default=None,
                             help='Resume from an existing experiment directory')
    parser.add_argument('--no-gui', action='store_true',
                       help='Run without GUI (headless mode)')

    args = parser.parse_args()

    resume_dir = args.dir
    config_path = args.config or 'config/schwarz_2d_test3.json'
    if resume_dir:
        config_path = os.fspath(get_config_backup_path(resume_dir))

    # 读取配置文件
    cfg = Config(path=config_path)
    float_type=ti.f32 if cfg.get("float_type", "f32") == "f32" else ti.f64
    arch=cfg.get("arch", "cpu")
    print(f"使用浮点类型: {float_type}, 计算架构: {arch}")
    if arch == "cuda":
        arch = ti.cuda
    elif arch == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, default_fp=float_type, log_level=ti.ERROR,enable_fallback=False  )

    # 创建Schwarz域分解MPM实例
    mpm = MPM_Schwarz(cfg, no_gui=args.no_gui, config_path=config_path)

    if resume_dir:
        mpm.load_checkpoint(resume_dir)

    if not resume_dir:
        print("保存初始状态的应力数据...")
        mpm.save_stress_data(frame_number=0)
        mpm.saved_stress_frames.append(0)

    target_frames = mpm.max_frames

    try:
        if args.no_gui:
            # 无GUI模式：直接运行到目标帧数
            print(f"Running simulation in headless mode for {target_frames} frames...")
            while mpm.completed_frames < target_frames:
                mpm.step()
                mpm.completed_frames += 1
                frame_count = mpm.completed_frames

                # 间隔保存应力数据
                if mpm.stress_save_interval > 0 and frame_count % mpm.stress_save_interval == 0:
                    print(f"保存第 {frame_count} 帧的应力数据...")
                    mpm.save_stress_data(frame_count)
                    mpm.saved_stress_frames.append(frame_count)

                # 检查速度收敛
                if mpm.check_velocity_convergence(frame_count):
                    print(f"在第 {frame_count} 帧达到速度收敛，提前终止模拟")
                    break

                if frame_count % 100 == 0:
                    print(f"Progress: {frame_count}/{target_frames} frames")
            print("Simulation completed.")
        else:
            # GUI模式：使用传统的GUI循环
            _render_interval = 1.0 / mpm.render_fps
            _last_render_t = 0.0
            while mpm._cv2_running and mpm.completed_frames < mpm.max_frames:
                mpm.step()
                _now = time.perf_counter()
                if _now - _last_render_t >= _render_interval:
                    mpm.render()
                    _last_render_t = _now
                mpm.completed_frames += 1
                frame_count = mpm.completed_frames

                # 间隔保存应力数据
                if mpm.stress_save_interval > 0 and frame_count % mpm.stress_save_interval == 0:
                    print(f"保存第 {frame_count} 帧的应力数据...")
                    mpm.save_stress_data(frame_count)
                    mpm.saved_stress_frames.append(frame_count)

                # 检查速度收敛
                if mpm.check_velocity_convergence(frame_count):
                    print(f"在第 {frame_count} 帧达到速度收敛，提前终止模拟")
                    break
    finally:
        gc.enable()
        gc.collect()

    frame_count = mpm.completed_frames
    if frame_count not in mpm.saved_stress_frames:
        print("记录最终帧的应力数据...")
        mpm.save_stress_data(frame_count)
        mpm.saved_stress_frames.append(frame_count)

    mpm.saved_stress_frames = sorted({int(frame) for frame in mpm.saved_stress_frames})
    mpm.save_checkpoint()

    if mpm.mem_profiler:
        mpm.mem_profiler.checkpoint("simulation_complete")
        report_path = os.path.join(mpm._experiment_dir, "memory_profile.json")
        mpm.mem_profiler.save_report(report_path)
        mpm.mem_profiler.stop()

    # 打印保存的应力数据帧信息
    if mpm.saved_stress_frames:
        print(f"共保存了 {len(mpm.saved_stress_frames)} 个帧的应力数据: {mpm.saved_stress_frames}")

    # 性能统计报告和可视化
    if mpm.enable_perf_stats and mpm.perf_stats.frame_data:
        print("\n" + "=" * 70)
        print("生成性能统计报告...")

        # 生成汇总报告
        mpm.perf_stats.generate_summary_report()

        # 创建输出目录
        stats_output_dir = os.path.join(mpm._experiment_dir, "performance_stats")
        os.makedirs(stats_output_dir, exist_ok=True)

        # 保存所有图表
        print(f"\n保存性能统计图表到 {stats_output_dir}...")
        mpm.perf_stats.save_all_plots(stats_output_dir, show=False)

        # 保存统计数据到 JSON
        stats_json_path = os.path.join(stats_output_dir, "stats_data.json")
        mpm.perf_stats.save_to_file(stats_json_path)

        print(f"性能统计完成！结果保存在: {stats_output_dir}")

    # 确保关闭所有 matplotlib 图形
    plt.close('all')

    if mpm.recorder is None or args.no_gui:
        print("Simulation finished.")
        exit()

    print("Playback finished.")
    # 播放录制动画
    mpm.recorder.play(loop=True, fps=60)
