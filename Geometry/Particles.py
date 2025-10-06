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

        # 存储配置参数，延后计算粒子数量
        self.particles_per_grid = config.get("particles_per_grid", 8)

        # 支持新的网格配置，区分2D和3D
        if self.dim == 2:
            if "grid_nx" in config.data and "grid_ny" in config.data:
                self.grid_nx = config.get("grid_nx", 16)
                self.grid_ny = config.get("grid_ny", 16)
                self.grid_nz = 1  # 兼容性
                self.grid_size = max(self.grid_nx, self.grid_ny)  # 兼容性
            else:
                self.grid_size = config.get("grid_size", 16)
                self.grid_nx = self.grid_size
                self.grid_ny = self.grid_size
                self.grid_nz = 1  # 兼容性

            # 支持自定义域尺寸
            self.domain_width = config.get("domain_width", 1.0)
            self.domain_height = config.get("domain_height", 1.0)
            self.domain_depth = 1.0

        else:  # 3D情况
            if "grid_nx" in config.data and "grid_ny" in config.data and "grid_nz" in config.data:
                self.grid_nx = config.get("grid_nx", 16)
                self.grid_ny = config.get("grid_ny", 16)
                self.grid_nz = config.get("grid_nz", 16)
                self.grid_size = max(self.grid_nx, self.grid_ny, self.grid_nz)  # 兼容性
            else:
                self.grid_size = config.get("grid_size", 16)
                self.grid_nx = self.grid_size
                self.grid_ny = self.grid_size
                self.grid_nz = self.grid_size

            # 支持自定义域尺寸
            self.domain_width = config.get("domain_width", 1.0)
            self.domain_height = config.get("domain_height", 1.0)
            self.domain_depth = config.get("domain_depth", 1.0)

        self.common_particles = common_particles
        
        # 计算每个形状的面积（Python数组）
        self.areas = []
        for i in range(self.num_areas):
            area = ShapeConfig.calculate_shape_area(self.shapes[i], self.dim)
            self.areas.append(area)
            
        # 粒子数量和相关字段将在generate_and_create_fields()中创建
        self.n_particles = 0
        self.particle_data_generated = False

        # 支持新旧配置格式
        sampling_method = config.get("sampling_method", None)
        if sampling_method is not None:
            # 新格式：sampling_method 可以是 "uniform", "poisson", "regular", "gauss"
            self.sampling_method = sampling_method
        else:
            # 兼容旧格式：use_possion_sampling 布尔值
            use_poisson = config.get("use_possion_sampling", True)
            self.sampling_method = "poisson" if use_poisson else "uniform"
        # 存储基础参数
        self.p_rho = config.get("p_rho", 1)

        # 计算网格单元体积（使用新的矩形网格系统）
        self.grid_cell_volume = (self.domain_width / self.grid_nx) * (self.domain_height / self.grid_ny)
        if self.dim == 3:
            self.grid_cell_volume *= (self.domain_depth / self.grid_nz)
        self.p_vol = self.grid_cell_volume / self.particles_per_grid
        self.p_mass = self.p_vol * self.p_rho

        if self.sampling_method == "gauss":
            self.p_vol = self.grid_cell_volume / 4 / self.particles_per_grid  # gauss采样使用更小的体积
        self.boundary_size = config.get("boundary_size", None)
        self.init_vel_y = config.get("initial_velocity_y", -1)
        
        # 材料参数表
        self.material_params = self._parse_material_params(config)
        
        # 两种材料的参数作为类成员变量（Python变量）
        if len(self.material_params) >= 1:
            mat0 = self.material_params[0]
            self.mu_1 = mat0["mu"]
            self.lam_1 = mat0["lambda"]
            self.p_mass_1 = mat0["p_mass"]
        else:
            # 默认材料1参数
            self.mu_1 = self.mu
            self.lam_1 = self.lam
            self.p_mass_1 = self.p_mass
            
        if len(self.material_params) >= 2:
            mat1 = self.material_params[1]
            self.mu_2 = mat1["mu"] 
            self.lam_2 = mat1["lambda"]
            self.p_mass_2 = mat1["p_mass"]
        else:
            # 默认材料2参数（与材料1相同）
            self.mu_2 = self.mu_1
            self.lam_2 = self.lam_1
            self.p_mass_2 = self.p_mass_1

        #检查材料参数
        print(f"Material 1: mu={self.mu_1}, lambda={self.lam_1}, p_mass={self.p_mass_1}")
        print(f"Material 2: mu={self.mu_2}, lambda={self.lam_2}, p_mass={self.p_mass_2}")

        # 初始化组件
        self._init_components(config)
        
        # 生成粒子并创建字段
        self.generate_and_create_fields()

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
            E = config.get("E", 4.0)
            nu = config.get("nu", 0.4)
            rho = config.get("p_rho", 1)
            material_params = [{
                "id": 0,
                "name": "default",
                "E": E,
                "nu": nu,
                "rho": rho
            }]
            # 保留兼容性的全局参数，但现在主要使用particles.material_params
            self.mu = E / (2*(1+nu))
            self.lam = E*nu/((1+nu)*(1-2*nu))

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

            # 初始变形梯度F，默认为二维单位矩阵
            initial_F = param.get("initial_F", None)
            if initial_F is None:
                # 默认二维单位矩阵
                initial_F = [[1.0, 0.0], [0.0, 1.0]]

            params_dict[mat_id] = {
                "E": E,
                "nu": nu,
                "rho": rho,
                "mu": mu,
                "lambda": lam,
                "p_mass": p_mass,
                "initial_F": initial_F
            }

        return params_dict
    
    @ti.func
    def get_material_params(self, particle_id):
        """获取指定粒子的材料参数(Taichi函数)"""
        material_id = self.particle_material_id[particle_id]
        mu, lam = 0.0, 0.0
        if material_id == 0:
            mu, lam = self.mu_1, self.lam_1
        else:
            mu, lam = self.mu_2, self.lam_2
        return mu, lam

    @ti.func
    def get_particle_mass(self, particle_id):
        """获取指定粒子的质量"""
        material_id = self.particle_material_id[particle_id]
        p_mass = 0.0
        if material_id == 0:
            p_mass = self.p_mass_1
        else:
            p_mass = self.p_mass_2
        return p_mass

    @ti.func  
    def get_particle_weight(self, particle_id):
        """获取指定粒子的高斯积分权重"""
        return self.particle_weight[particle_id]
        
    def get_material_param(self, material_id, param_name):
        """获取指定材料ID的参数值(Python函数)"""
        if material_id in self.material_params:
            return self.material_params[material_id].get(param_name, 0.0)
        return 0.0

    def _init_components(self, config):
        """初始化各种组件"""
        # 粒子生成器
        # 构建参数字典
        generator_params = {
            'dim': self.dim,
            'sampling_method': self.sampling_method,
            'particles_per_grid': self.particles_per_grid,
            'grid_size': self.grid_size,
            'grid_nx': self.grid_nx,
            'grid_ny': self.grid_ny,
            'domain_width': self.domain_width,
            'domain_height': self.domain_height
        }

        # 3D情况下添加额外参数
        if self.dim == 3:
            generator_params['grid_nz'] = self.grid_nz
            generator_params['domain_depth'] = self.domain_depth

        self.particle_generator = ParticleGenerator(**generator_params)
        
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
            inv_dx_x=self.grid_nx / self.domain_width,
            inv_dx_y=self.grid_ny / self.domain_height,
            inv_dx_z=self.grid_nz / self.domain_depth if self.dim == 3 else 1.0,
            float_type=self.float_type,
            dim=self.dim
        )
        
        # 运动积分器
        self.advector = ParticleAdvector()
        
        # 粒子合并器
        self.merger = ParticleMerger()
    
    def generate_and_create_fields(self):
        """生成粒子数据，然后创建对应的Taichi字段"""
        if self.particle_data_generated:
            return
        
        # 第一步：计算每个shape需要的粒子数量
        particles_per_area = []
        for i in range(self.num_areas):
            n = int(self.grid_size**self.dim * self.areas[i] * self.particles_per_grid)
            particles_per_area.append(n)
        
        # 第二步：使用粒子生成器生成所有粒子
        all_particles = self.particle_generator.generate_particles_for_shapes(
            self.shapes, 
            particles_per_area
        )
        
        # 第三步：计算总粒子数量（包括common_particles）
        generated_particle_count = len(all_particles)
        common_particle_count = self.common_particles.n_particles if self.common_particles else 0
        total_particle_count = generated_particle_count + common_particle_count
        
        print(f"Generated particles: {generated_particle_count}")
        print(f"Common particles: {common_particle_count}")
        print(f"Total particles: {total_particle_count}")
        
        # 第四步：设置最终粒子数量并创建Taichi字段
        self.n_particles = total_particle_count
        self._create_taichi_fields()
        
        # 第五步：初始化粒子属性
        self.particle_initializer.initialize_particle_fields(
            all_particles,
            self.x, self.v, self.F, self.C,
            self.particle_material_id,
            self.particle_weight,
            self.material_params
        )
        
        # 第六步：处理公共粒子
        if self.common_particles is not None:
            self.merge_common_particles(generated_particle_count)
            
        self.particle_data_generated = True
    
    def _create_taichi_fields(self):
        """创建所有Taichi字段"""
        # 粒子字段
        self.x = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.v = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)
        
        # 应力和应变字段
        self.stress = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)
        self.strain = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)

        # 权重字段 - 用于存储高斯积分点权重
        self.particle_weight = ti.field(self.float_type, self.n_particles)

        # 权重和梯度字段
        shape = (self.n_particles, 3, 3) if self.dim == 2 else (self.n_particles, 3, 3, 3)
        self.wip = ti.field(self.float_type, shape)
        self.dwip = ti.Vector.field(self.dim, self.float_type, shape)
        
        # 边界和材料ID字段
        self.is_boundary_particle = ti.field(ti.i32, self.n_particles)
        self.particle_material_id = ti.field(ti.i32, self.n_particles)

        # 粒子体积力字段
        self.volume_force = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        
        # Taichi字段用于计算每个area的粒子数（向后兼容）
        self.n_per_area = ti.field(ti.i32, shape=self.num_areas)
        for i in range(self.num_areas):
            area_particles = int(self.grid_size**self.dim * self.areas[i] * self.particles_per_grid)
            self.n_per_area[i] = area_particles
            
        # Poisson采样相关字段（如果需要）
        max_n_per_area = max(int(self.grid_size**self.dim * area * self.particles_per_grid) for area in self.areas) if self.areas else 1
        self.pos_possion = ti.Vector.field(self.dim, self.float_type, shape=max_n_per_area)

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