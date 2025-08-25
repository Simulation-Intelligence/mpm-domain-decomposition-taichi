import taichi as ti
import numpy as np

from Geometry.BoundaryDetector import BoundaryDetector, NeighborDensityBoundaryDetector
from Geometry.ParticleGenerator import ShapeConfig, ParticleGenerator, ParticleInitializer
from Geometry.ShapeUtils import ManualBoundaryDetector, ParticleNeighborBuilder, ParticleAdvector, ParticleMerger


# ------------------ 粒子模块 ------------------
@ti.data_oriented
class Particles:
    def __init__(self, config, common_particles:'Particles'=None):
        self.dim = config.get("dim", 2)
        self.float_type = ti.f32 if config.get("float_type", "f32") == "f32" else ti.f64
        
        # 解析新的几何形状配置
        self.shapes = ShapeConfig.parse_shapes_config(config, self.dim)
        boundary_range = config.get("boundary_range", None)
        
        self.num_areas = len(self.shapes)
        self.boundary_range = ti.Vector.field(2, self.float_type, shape=(self.dim))
        self.neighbor = (3,) * self.dim

        if boundary_range is not None:
            for d in ti.static(range(self.dim)):
                self.boundary_range[d] = ti.Vector(boundary_range[d])

        # 计算每个形状的面积
        self.areas = ti.field(self.float_type, self.num_areas)
        for i in range(self.num_areas):
            self.areas[i] = ShapeConfig.calculate_shape_area(self.shapes[i], self.dim)

        max_n_per_area = 0
            
        self.particles_per_grid = config.get("particles_per_grid", 8)
        self.grid_size = config.get("grid_size", 16)
        self.n_per_area = ti.field(ti.i32, shape=self.num_areas)
        self.n_particles = 0
        for i in range(self.num_areas):
            n = int(self.grid_size**self.dim * self.areas[i] * self.particles_per_grid)
            self.n_per_area[i] = n
            if n > max_n_per_area:
                max_n_per_area = n
            self.n_particles += n

        self.common_particles = None

        if common_particles is not None:
            self.n_particles += common_particles.n_particles
            self.common_particles = common_particles

        self.use_possion_sampling = config.get("use_possion_sampling", True)
        self.pos_possion = ti.Vector.field(self.dim, self.float_type, shape=max_n_per_area)
        self.p_rho = config.get("p_rho", 1)
        self.p_vol = (1.0/self.grid_size)**self.dim / self.particles_per_grid
        self.p_mass = self.p_vol * self.p_rho
        self.boundary_size = 0.01

        self.float_type = self.float_type
        
        # 粒子字段
        self.x = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.v = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)

        shape = (self.n_particles, 3,3) if self.dim == 2 else (self.n_particles, 3,3,3)
        self.wip=ti.field(self.float_type, shape)
        self.dwip=ti.Vector.field(self.dim, self.float_type, shape)
        
        self.init_vel_y = config.get("initial_velocity_y", -1)
        self.is_boundary_particle = ti.field(ti.i32, self.n_particles)

        # 初始化组件
        self._init_components(config)
        
        # 初始化粒子
        self.initialize()

        # 设置边界
        if boundary_range is not None:
            self.set_boundary(method="manual")
        else:
            self.set_boundary(method="automatic")

    def _init_components(self, config):
        """初始化各种组件"""
        # 粒子生成器
        self.particle_generator = ParticleGenerator(
            dim=self.dim, 
            use_poisson_sampling=self.use_possion_sampling
        )
        
        # 粒子初始化器
        self.particle_initializer = ParticleInitializer(
            dim=self.dim,
            float_type=self.float_type,
            init_vel_y=self.init_vel_y
        )
        
        # 边界检测器
        self.boundary_detector = BoundaryDetector(boundary_size=self.boundary_size)
        self.neighbor_boundary_detector = NeighborDensityBoundaryDetector(boundary_size=self.boundary_size)
        
        # 手动边界检测器
        if hasattr(self, 'boundary_range'):
            self.manual_boundary_detector = ManualBoundaryDetector(
                boundary_range=self.boundary_range,
                boundary_size=self.boundary_size,
                dim=self.dim
            )
        
        # 邻居构建器
        self.neighbor_builder = ParticleNeighborBuilder(
            grid_size=self.grid_size,
            dim=self.dim
        )
        
        # 运动积分器
        self.advector = ParticleAdvector()
        
        # 粒子合并器
        self.merger = ParticleMerger()
    
    def initialize(self):
        """初始化所有粒子，支持多种几何形状"""
        # 使用粒子生成器生成粒子
        all_particles = self.particle_generator.generate_particles_for_shapes(
            self.shapes, 
            [self.n_per_area[i] for i in range(self.num_areas)]
        )
        
        # 将最终粒子数量写入Taichi字段
        self.n_particles = min(len(all_particles), self.n_particles)
        
        # 使用粒子初始化器初始化粒子属性
        self.particle_initializer.initialize_particle_fields(
            all_particles[:self.n_particles],
            self.x, self.v, self.F, self.C
        )
        
        # 处理公共粒子
        if self.common_particles is not None:
            start_num = self.n_particles - self.common_particles.n_particles
            self.merge_common_particles(start_num)

    def merge_common_particles(self, start_num):
        """合并公共粒子"""
        self.merger.merge_common_particles(
            self.x, self.v, self.F, self.C, self.is_boundary_particle,
            self.common_particles.x, self.common_particles.v,
            self.common_particles.F, self.common_particles.C,
            self.common_particles.is_boundary_particle,
            start_num, self.common_particles.n_particles
        )

    def set_boundary_automatic(self):
        """自动检测边界粒子：基于Alpha Shape"""
        self.set_boundary_alpha_shape()

    def set_boundary_alpha_shape(self):
        """基于Alpha Shape的边界检测"""
        # 获取所有粒子位置
        positions = self.x.to_numpy()
        
        # 获取Poisson采样半径（如果可用）
        poisson_radius = self.particle_generator.get_last_poisson_radius()
        
        # 使用边界检测器，传递Poisson半径信息
        boundary_flags = self.boundary_detector.detect_boundaries(
            positions, self.dim, poisson_radius
        )
        
        # 将结果写回Taichi字段
        self.is_boundary_particle.from_numpy(boundary_flags)

    def set_boundary_neighbor_density(self):
        """基于邻居数量统计的边界检测（备选方法）"""
        # 获取所有粒子位置
        positions = self.x.to_numpy()
        
        # 使用邻居密度边界检测器
        boundary_flags = self.neighbor_boundary_detector.detect_boundaries(positions)
        
        # 将结果写回Taichi字段
        self.is_boundary_particle.from_numpy(boundary_flags)

    def set_boundary_manual(self):
        """手动指定边界：基于矩形区域"""
        if hasattr(self, 'manual_boundary_detector'):
            self.manual_boundary_detector.detect_boundaries(
                self.x, self.is_boundary_particle, self.n_particles
            )

    def set_boundary(self, method="automatic"):
        """设置边界粒子
        Args:
            method: "automatic" 自动检测，"manual" 手动指定
        """
        if method == "automatic":
            self.set_boundary_automatic()
        elif method == "manual" and hasattr(self, 'boundary_range') and self.boundary_range is not None:
            self.set_boundary_manual()
        else:
            print(f"Warning: Using automatic boundary detection")
            self.set_boundary_automatic()

    def build_neighbor_list(self):
        """构建邻居列表"""
        self.neighbor_builder.build_neighbor_list(
            self.x, self.wip, self.dwip, self.n_particles
        )

    def advect(self, dt: ti.f32):
        """粒子运动积分"""
        self.advector.advect(self.x, self.v, dt)