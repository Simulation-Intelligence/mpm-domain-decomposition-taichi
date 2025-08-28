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

        # 支持新旧配置格式
        sampling_method = config.get("sampling_method", None)
        if sampling_method is not None:
            # 新格式：sampling_method 可以是 "uniform", "poisson", "regular"
            self.sampling_method = sampling_method
        else:
            # 兼容旧格式：use_possion_sampling 布尔值
            use_poisson = config.get("use_possion_sampling", True)
            self.sampling_method = "poisson" if use_poisson else "uniform"
        self.pos_possion = ti.Vector.field(self.dim, self.float_type, shape=max_n_per_area)
        self.p_rho = config.get("p_rho", 1)
        self.p_vol = (1.0/self.grid_size)**self.dim / self.particles_per_grid
        self.p_mass = self.p_vol * self.p_rho
        self.boundary_size = config.get("boundary_size", None)

        self.float_type = self.float_type
        
        # 粒子字段
        self.x = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.v = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)
        
        # 应力和应变字段
        self.stress = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)
        self.strain = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)

        shape = (self.n_particles, 3,3) if self.dim == 2 else (self.n_particles, 3,3,3)
        self.wip=ti.field(self.float_type, shape)
        self.dwip=ti.Vector.field(self.dim, self.float_type, shape)
        
        self.init_vel_y = config.get("initial_velocity_y", -1)
        self.is_boundary_particle = ti.field(ti.i32, self.n_particles)
        
        # 材料参数表和粒子材料ID
        self.material_params = self._parse_material_params(config)
        self.particle_material_id = ti.field(ti.i32, self.n_particles)
        
        # 创建Taichi字段存储材料参数，以便在kernel中访问
        self.max_materials = 16  # 最大支持16种材料
        self.material_E = ti.field(self.float_type, self.max_materials)
        self.material_nu = ti.field(self.float_type, self.max_materials)
        self.material_rho = ti.field(self.float_type, self.max_materials)
        self.material_mu = ti.field(self.float_type, self.max_materials)
        self.material_lambda = ti.field(self.float_type, self.max_materials)
        self.material_p_mass = ti.field(self.float_type, self.max_materials)
        
        # 将材料参数复制到Taichi字段
        self._copy_material_params_to_fields()

        #检查taichi字段的材料参数
        for mat_id in range(self.max_materials):
            if self.material_E[mat_id] != 0:
                print(f"Material {mat_id}: E={self.material_E[mat_id]}, nu={self.material_nu[mat_id]}, rho={self.material_rho[mat_id]}")

        # 初始化组件
        self._init_components(config)
        
        # 初始化粒子
        self.initialize()

        # 设置边界
        if self.boundary_size is not None:
            boundary_range = config.get("boundary_range", None)
            if boundary_range is not None:
                self.set_boundary(method="manual")
            else:
                self.set_boundary(method="automatic")

    def _parse_material_params(self, config):
        """解析材料参数表"""
        material_params = config.get("material_params", [])
        if not material_params:
            # 如果没有材料参数表，使用默认参数
            E = config.get("E", 4)
            nu = config.get("nu", 0.4)
            rho = config.get("p_rho", 1)
            material_params = [{
                "id": 0,
                "name": "default",
                "E": E,
                "nu": nu,
                "rho": rho
            }]
        
        # 构建材料参数字典，以id为键
        params_dict = {}
        for param in material_params:
            mat_id = param["id"]
            E = param["E"]
            nu = param["nu"]
            rho = param.get("rho", config.get("p_rho", 1))
            
            # 计算拉梅参数
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            
            # 计算粒子质量：p_mass = p_vol * rho
            p_mass = self.p_vol * rho
            
            params_dict[mat_id] = {
                "E": E,
                "nu": nu,
                "rho": rho,
                "mu": mu,
                "lambda": lam,
                "p_mass": p_mass
            }

        return params_dict
    
    def _copy_material_params_to_fields(self):
        """将材料参数复制到Taichi字段"""
        for mat_id, params in self.material_params.items():
            if mat_id < self.max_materials:
                self.material_E[mat_id] = params["E"]
                self.material_nu[mat_id] = params["nu"]
                self.material_rho[mat_id] = params["rho"]
                self.material_mu[mat_id] = params["mu"]
                self.material_lambda[mat_id] = params["lambda"]
                self.material_p_mass[mat_id] = params["p_mass"]

    @ti.func
    def get_material_params(self, particle_id):
        """获取指定粒子的材料参数(Taichi函数)"""
        material_id = self.particle_material_id[particle_id]
        mu = self.material_mu[material_id]
        lam = self.material_lambda[material_id]
        return mu, lam
    
    @ti.func
    def get_particle_mass(self, particle_id):
        """获取指定粒子的质量"""
        material_id = self.particle_material_id[particle_id]
        return self.material_p_mass[material_id]
        
    def get_material_param(self, material_id, param_name):
        """获取指定材料ID的参数值(Python函数)"""
        if material_id in self.material_params:
            return self.material_params[material_id].get(param_name, 0.0)
        return 0.0

    def _init_components(self, config):
        """初始化各种组件"""
        # 粒子生成器
        self.particle_generator = ParticleGenerator(
            dim=self.dim, 
            sampling_method=self.sampling_method
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
            self.x, self.v, self.F, self.C,
            self.particle_material_id
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