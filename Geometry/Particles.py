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
        self.init_vel_y = config.get("initial_velocity_y", 0)
        
        # 材料参数表
        self.material_params = self._parse_material_params(config)

        # 边界粒子范围限制
        self.boundary_particle_ranges = self._parse_boundary_particle_ranges(config.get("boundary_particle_range", None))
        
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
            use_mesh_boundary = config.get("use_mesh_boundary", True)  # 默认使用mesh边界

            if self.sampling_method == "mesh" and use_mesh_boundary:
                pass  # mesh采样的边界信息已经在粒子生成时设置，无需额外检测
            elif boundary_range is not None:
                self.set_boundary(method="manual")
            else:
                self.set_boundary(method="automatic")
            
            # 所有边界标记完成后，应用boundary_particle_range过滤
            self.apply_boundary_particle_range_filter()

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

    def _parse_boundary_particle_ranges(self, boundary_particle_range):
        """解析边界粒子范围配置，支持单个区域和多个矩形区域"""
        if boundary_particle_range is None:
            return None

        if isinstance(boundary_particle_range, list) and len(boundary_particle_range) > 0:
            first_elem = boundary_particle_range[0]

            # 检查是否是旧格式：单个区域 [[x_min, x_max], [y_min, y_max]]
            if isinstance(first_elem, list) and len(first_elem) == 2 and \
               all(isinstance(x, (int, float)) for x in first_elem):
                print("检测到单个区域格式")
                return [boundary_particle_range]

            # 检查是否是新格式：多个区域 [[[x1_min, x1_max], [y1_min, y1_max]], [[x2_min, x2_max], [y2_min, y2_max]], ...]
            elif isinstance(first_elem, list) and len(first_elem) > 0 and \
                 isinstance(first_elem[0], list) and len(first_elem[0]) == 2:
                print(f"检测到多区域格式，共 {len(boundary_particle_range)} 个矩形区域")
                return boundary_particle_range

        print(f"Warning: 无法识别的boundary_particle_range格式: {boundary_particle_range}")
        return None

    @ti.func
    def get_material_params(self, particle_id):
        """获取指定粒子的材料参数(Taichi函数)"""
        return self.particle_mu[particle_id], self.particle_lam[particle_id]

    @ti.kernel
    def _init_material_fields_from_id(self):
        """从 material_id 系统初始化 per-particle mu/lam 字段（向后兼容）"""
        for p in range(self.n_particles):
            if self.particle_material_id[p] == 0:
                self.particle_mu[p]  = self.mu_1
                self.particle_lam[p] = self.lam_1
            else:
                self.particle_mu[p]  = self.mu_2
                self.particle_lam[p] = self.lam_2

    def setup_spatial_material(self, config):
        """若 config 包含 material_distribution，按空间坐标覆盖 per-particle mu/lam"""
        dist = config.get("material_distribution", None)
        if dist is None:
            return
        if dist["type"] == "smoothstep_x":
            self._apply_smoothstep_x(dist)

    def _apply_smoothstep_x(self, dist):
        """Smoothstep 插值：x 轴方向软区→硬区，更新 particle_mu / particle_lam"""
        x_pos  = self.x.to_numpy()[:, 0]
        mu_arr  = self.particle_mu.to_numpy().copy()
        lam_arr = self.particle_lam.to_numpy().copy()
        nu       = dist["nu"]
        E_soft   = dist["E_soft"]
        E_hard   = dist["E_hard"]
        x_lo     = dist["x_soft_lo"]
        x_hi     = dist["x_soft_hi"]
        x_trans  = dist["x_trans"]
        cx       = 0.5 * (x_lo + x_hi)
        half_soft = 0.5 * (x_hi - x_lo)
        for i in range(self.n_particles):
            abs_dx = abs(x_pos[i] - cx)
            if abs_dx <= half_soft:
                E = E_soft
            elif abs_dx >= half_soft + x_trans:
                E = E_hard
            else:
                t = (abs_dx - half_soft) / x_trans
                smooth = 3*t*t - 2*t*t*t
                E = E_soft + (E_hard - E_soft) * smooth
            mu_arr[i]  = E / (2.0 * (1.0 + nu))
            lam_arr[i] = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        dtype = np.float32 if self.float_type == ti.f32 else np.float64
        self.particle_mu.from_numpy(mu_arr.astype(dtype))
        self.particle_lam.from_numpy(lam_arr.astype(dtype))

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
            'domain_height': self.domain_height,
            'use_mesh_boundary': config.get("use_mesh_boundary", True)
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

        # 计算总域面积/体积
        if self.dim == 2:
            total_domain_area = self.domain_width * self.domain_height
            total_grid_cells = self.grid_nx * self.grid_ny
        else:  # 3D
            total_domain_area = self.domain_width * self.domain_height * self.domain_depth
            total_grid_cells = self.grid_nx * self.grid_ny * self.grid_nz

        # 计算每单位面积/体积的粒子密度
        particles_per_unit_area = (total_grid_cells * self.particles_per_grid) / total_domain_area

        for i in range(self.num_areas):
            # 根据shape的实际面积/体积计算粒子数量
            n = int(self.areas[i] * particles_per_unit_area)
            particles_per_area.append(n)
            print(f"Shape {i}: 面积={self.areas[i]:.6f}, 粒子数={n}")
        
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
            self.material_params,
            self.is_boundary_particle
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

        # per-particle 材料参数字段（支持空间变化材料）
        self.particle_mu  = ti.field(self.float_type, self.n_particles)
        self.particle_lam = ti.field(self.float_type, self.n_particles)

        # move_boundary相关字段
        self.is_move_boundary_particle = ti.field(ti.i32, self.n_particles)
        self.target_position = ti.Vector.field(self.dim, self.float_type, self.n_particles)

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

        print("使用Alpha Shape进行自动边界检测")
        print(f"粒子总数: {self.x.shape[0]}")
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
            method: "automatic" 自动检测，"manual" 手动指定，"mesh" 使用mesh采样的边界
        """
        if method == "mesh" and self.sampling_method == "mesh":
            # 如果使用mesh采样，边界信息已经在粒子生成时设置，无需额外检测
            print("使用mesh采样的边界信息，跳过边界检测")
        elif method == "automatic":
            self.set_boundary_automatic()
        elif method == "manual" and hasattr(self, 'boundary_range') and self.boundary_range is not None:
            self.set_boundary_manual()
        else:
            print(f"Warning: Using automatic boundary detection")
            self.set_boundary_automatic()



    def apply_boundary_particle_range_filter(self):
        """应用boundary_particle_range过滤器，只保留在指定区域内的边界粒子"""
        if self.boundary_particle_ranges is None:
            return  # 如果没有配置范围限制，直接返回

        print(f"应用boundary_particle_range过滤器: {len(self.boundary_particle_ranges)} 个矩形区域")

        # 统计过滤前的边界粒子数量
        before_count = self.count_boundary_particles()

        # 转换所有区域为numpy数组
        import numpy as np
        regions_array = np.array(self.boundary_particle_ranges, dtype=np.float32 if self.float_type == ti.f32 else np.float64)

        # 应用多区域过滤
        self.filter_boundary_particles_by_multiple_rectangles(regions_array, len(self.boundary_particle_ranges))

        # 统计过滤后的边界粒子数量
        after_count = self.count_boundary_particles()

        print(f"边界粒子范围过滤: {before_count} -> {after_count} (移除了 {before_count - after_count} 个)")

    @ti.kernel
    def filter_boundary_particles_by_rectangle(self, range_array: ti.types.ndarray()):
        """在指定范围外的边界粒子取消边界标记"""
        for i in range(self.n_particles):
            if self.is_boundary_particle[i]:
                pos = self.x[i]
                in_range = True

                # 检查是否在指定范围内 - 避免在静态循环中使用break
                for d in ti.static(range(self.dim)):
                    if in_range and not (range_array[d, 0] <= pos[d] <= range_array[d, 1]):
                        in_range = False

                # 如果不在范围内，取消边界标记
                if not in_range:
                    self.is_boundary_particle[i] = 0

    @ti.kernel
    def filter_boundary_particles_by_multiple_rectangles(self, regions_array: ti.types.ndarray(), num_regions: ti.i32):
        """根据多个矩形区域过滤边界粒子，只保留在任何一个区域内的边界粒子"""
        for i in range(self.n_particles):
            if self.is_boundary_particle[i]:
                pos = self.x[i]
                in_any_region = False

                # 检查是否在任何一个区域内
                for region_idx in range(num_regions):
                    in_this_region = True
                    # 避免使用break，改用逻辑控制
                    for d in ti.static(range(self.dim)):
                        if in_this_region and not (regions_array[region_idx, d, 0] <= pos[d] <= regions_array[region_idx, d, 1]):
                            in_this_region = False

                    # 如果这个区域匹配，记录结果（不能用break，所以继续检查所有区域）
                    if in_this_region:
                        in_any_region = True

                # 如果不在任何区域内，取消边界标记
                if not in_any_region:
                    self.is_boundary_particle[i] = 0

    @ti.kernel
    def count_boundary_particles(self) -> ti.i32:
        """统计边界粒子数量"""
        count = 0
        for i in range(self.n_particles):
            if self.is_boundary_particle[i]:
                count += 1
        return count

    def build_neighbor_list(self):
        """构建邻居列表"""
        self.neighbor_builder.build_neighbor_list(
            self.x, self.wip, self.dwip, self.n_particles
        )


    def advect(self, dt: ti.f64):
        """粒子运动积分"""
        self.advector.advect(self.x, self.v, dt)

    def mark_move_boundary_particles(self, start_region, displacement):
        """标记移动边界区域内的边界粒子"""
        self.mark_particles_in_region(start_region, displacement)

    @ti.kernel
    def mark_particles_in_region(self, start_region: ti.types.ndarray(), displacement: ti.types.ndarray()):
        """标记指定区域内的边界粒子用于移动"""
        for i in range(self.n_particles):
            pos = self.x[i]
            in_region = True

            # 检查是否在起始区域内
            for d in ti.static(range(self.dim)):
                if not (start_region[d, 0] <= pos[d] <= start_region[d, 1]):
                    in_region = False

            if in_region and self.is_boundary_particle[i]:
                self.is_move_boundary_particle[i] = 1
                # 计算目标位置
                target_pos = pos
                for d in ti.static(range(self.dim)):
                    if d < displacement.shape[0]:
                        target_pos[d] += displacement[d]
                self.target_position[i] = target_pos
            else:
                self.is_move_boundary_particle[i] = 0


    def setup_move_boundary(self, grid):
        """设置移动边界，需要grid对象来获取移动配置"""
        if not hasattr(grid, 'has_move_boundary') or not grid.has_move_boundary:
            return

        dtype = np.float32 if grid.float_type == ti.f32 else np.float64
        start_region = list(grid.move_start_region)  # [[x0,x1],[y0,y1],...]

        # select_x_max：找到区域内边界粒子中 x 最大值，收窄到最右一列（宽度 = 0.1 dx）
        if getattr(grid, 'move_select_x_max', False):
            positions = self.x.to_numpy()
            is_boundary = self.is_boundary_particle.to_numpy()
            sr = start_region
            mask = np.ones(len(positions), dtype=bool)
            for d in range(self.dim):
                mask &= (positions[:, d] >= sr[d][0]) & (positions[:, d] <= sr[d][1])
            mask &= is_boundary.astype(bool)
            if mask.any():
                x_max = positions[mask, 0].max()
                half_width = 0.05 * grid.dx_x  # ±0.05 dx，总宽约 0.1 dx
                start_region = list(start_region)
                start_region[0] = [x_max - half_width, x_max + half_width]
                print(f"[move_boundary] select_x_max: x 范围收窄到 [{start_region[0][0]:.6f}, {start_region[0][1]:.6f}]")
            else:
                print("[move_boundary] select_x_max: 区域内未找到边界粒子，跳过收窄")

        start_region_np = np.array(start_region, dtype=dtype)
        displacement_np = np.array(grid.move_displacement, dtype=dtype)

        # 标记移动边界粒子
        self.mark_move_boundary_particles(start_region_np, displacement_np)

    @ti.kernel
    def mark_particles_arc_2d(self, start_region: ti.types.ndarray(),
                               center: ti.types.ndarray(),
                               trig: ti.types.ndarray()):
        """标记2D弧形边界粒子，计算旋转后的目标位置。trig = [cos(angle), sin(angle)]"""
        for i in range(self.n_particles):
            if self.is_boundary_particle[i]:
                pos = self.x[i]
                in_region = True
                for d in ti.static(range(self.dim)):
                    if not (start_region[d, 0] <= pos[d] <= start_region[d, 1]):
                        in_region = False
                if in_region:
                    self.is_move_boundary_particle[i] = 1
                    dx = pos[0] - center[0]
                    dy = pos[1] - center[1]
                    self.target_position[i] = ti.Vector([
                        trig[0] * dx - trig[1] * dy + center[0],
                        trig[1] * dx + trig[0] * dy + center[1]
                    ])
            else:
                self.is_move_boundary_particle[i] = 0

    @ti.kernel
    def mark_particles_arc_3d(self, start_region: ti.types.ndarray(),
                               center: ti.types.ndarray(),
                               axis: ti.types.ndarray(),
                               trig: ti.types.ndarray()):
        """标记3D弧形边界粒子，使用Rodrigues旋转公式。trig = [cos(angle), sin(angle)]"""
        for i in range(self.n_particles):
            if self.is_boundary_particle[i]:
                pos = self.x[i]
                in_region = True
                for d in ti.static(range(self.dim)):
                    if not (start_region[d, 0] <= pos[d] <= start_region[d, 1]):
                        in_region = False
                if in_region:
                    self.is_move_boundary_particle[i] = 1
                    vx = pos[0] - center[0]
                    vy = pos[1] - center[1]
                    vz = pos[2] - center[2]
                    kx, ky, kz = axis[0], axis[1], axis[2]
                    kdotv = kx * vx + ky * vy + kz * vz
                    cos_a, sin_a = trig[0], trig[1]
                    self.target_position[i] = ti.Vector([
                        vx * cos_a + (ky * vz - kz * vy) * sin_a + kx * kdotv * (1.0 - cos_a) + center[0],
                        vy * cos_a + (kz * vx - kx * vz) * sin_a + ky * kdotv * (1.0 - cos_a) + center[1],
                        vz * cos_a + (kx * vy - ky * vx) * sin_a + kz * kdotv * (1.0 - cos_a) + center[2]
                    ])
            else:
                self.is_move_boundary_particle[i] = 0

    def setup_arc_boundary(self, grid):
        """设置弧形边界（支持多个），将各start_region内的边界粒子目标位置设为旋转后的位置"""
        import math
        if not hasattr(grid, 'has_arc_boundary') or not grid.has_arc_boundary:
            return

        dtype = np.float32 if grid.float_type == ti.f32 else np.float64

        for cfg in grid.arc_boundary_configs:
            start_region = cfg.get("start_region", [[0.0, 1.0]] * grid.dim)
            center = cfg.get("center", [0.5] * grid.dim)
            angle = cfg.get("angle", 0.1)
            sr_np = np.array(start_region, dtype=dtype)       # shape (dim, 2)
            center_np = np.array(center, dtype=dtype)
            trig_np = np.array([math.cos(angle), math.sin(angle)], dtype=dtype)

            if grid.dim == 2:
                self.mark_particles_arc_2d(sr_np, center_np, trig_np)
            else:
                axis = cfg.get("axis", [0.0, 0.0, 1.0])
                norm = math.sqrt(sum(a * a for a in axis))
                axis_np = np.array([a / norm for a in axis], dtype=dtype)
                self.mark_particles_arc_3d(sr_np, center_np, axis_np, trig_np)
