"""
粒子生成模块 - 处理各种几何形状的粒子生成
"""
import numpy as np
import math
from .GaussQuadrature import GaussQuadrature


class ShapeConfig:
    """形状配置解析和面积计算"""
    
    @staticmethod
    def parse_shapes_config(config, dim=2):
        """解析形状配置，支持矩形、椭圆和挖空"""
        shapes = []
        
        # 处理新格式的shapes配置
        shapes_config = config.get("shapes", None)
        if shapes_config is not None:
            for shape_config in shapes_config:
                shapes.append({
                    "type": shape_config.get("type", "rectangle"),
                    "params": shape_config.get("params", {}),
                    "operation": shape_config.get("operation", "add"),  # add/subtract
                    "material_id": shape_config.get("material_id", 0)
                })
        
        # 兼容旧格式的initial_position_range
        else:
            default_init_pos_range = [[0.3, 0.6], [0.3, 0.6]] if dim == 2 else [[0.3, 0.6], [0.3, 0.6], [0.3, 0.6]]
            init_pos_range = config.get("initial_position_range", [default_init_pos_range])
            
            for rect_range in init_pos_range:
                shapes.append({
                    "type": "rectangle",
                    "params": {"range": rect_range},
                    "operation": "add"
                })
        
        return shapes
    
    @staticmethod
    def calculate_shape_area(shape, dim=2):
        """计算形状的面积"""
        shape_type = shape["type"]
        params = shape["params"]
        
        if shape_type == "rectangle":
            rect_range = params["range"]
            area = 1.0
            for d in range(dim):
                area *= (rect_range[d][1] - rect_range[d][0])
            return area
            
        elif shape_type == "ellipse":
            if dim == 2:
                # 2D椭圆面积 = π * a * b
                a = params["semi_axes"][0]  # 半长轴
                b = params["semi_axes"][1]  # 半短轴
                return math.pi * a * b
            else:
                # 3D椭球体积 = (4/3) * π * a * b * c
                a, b, c = params["semi_axes"]
                return (4.0/3.0) * math.pi * a * b * c
                
        return 0.0


class ParticleGenerator:
    """粒子生成器"""

    def __init__(self, dim=2, sampling_method="poisson", particles_per_grid=8, grid_size=16):
        """
        初始化粒子生成器
        Args:
            dim: 维度 (2 or 3)
            sampling_method: 采样方式，可选 "uniform", "poisson", "regular", "gauss"
            particles_per_grid: 每个网格的粒子数
            grid_size: 网格大小
        """
        self.dim = dim
        self.sampling_method = sampling_method
        self.particles_per_grid = particles_per_grid
        self.grid_size = grid_size
        self.last_poisson_radius = None  # 存储最后使用的Poisson采样半径
    
    def generate_particles_for_shapes(self, shapes, particles_per_area):
        """为多个形状生成粒子，按照config中的顺序依次执行添加和挖空操作"""
        all_particles = []
        
        # 按照shapes的顺序依次执行操作
        for i, shape in enumerate(shapes):
            if shape["operation"] == "add":
                # 添加操作：生成新粒子并加入总列表
                particles = self.generate_particles_for_shape(shape, particles_per_area[i])
                material_id = shape.get("material_id", 0)  # 获取材料ID，默认为0
                
                # 为每个粒子添加材料ID信息
                particles_with_material = []
                for idx, pos in enumerate(particles):
                    particle_data = {
                        "position": pos,
                        "material_id": material_id
                    }
                    # 如果使用高斯积分点采样并且有权重数据，则添加权重
                    if hasattr(self, 'gauss_weights') and self.gauss_weights is not None and idx < len(self.gauss_weights):
                        particle_data["weight"] = self.gauss_weights[idx]
                    particles_with_material.append(particle_data)
                all_particles.extend(particles_with_material)
                
            elif shape["operation"] == "subtract":
                # 挖空操作：从现有粒子中移除在该形状内的粒子
                all_particles = self._remove_particles_in_shape_with_material(all_particles, shape)
                
            elif shape["operation"] == "change":
                # 改变操作：改变指定形状内粒子的材料ID
                material_id = shape.get("material_id", 0)
                all_particles = self._change_particles_material_in_shape(all_particles, shape, material_id)
        
        return all_particles
    
    def generate_particles_for_shape(self, shape, n_particles):
        """为指定形状生成粒子"""
        shape_type = shape["type"]
        params = shape["params"]
        
        if shape_type == "rectangle":
            return self._generate_rectangle_particles(params, n_particles)
        elif shape_type == "ellipse":
            return self._generate_ellipse_particles(params, n_particles)
        
        return []
    
    def _generate_rectangle_particles(self, params, n_particles):
        """生成矩形区域内的粒子"""
        particles = []
        rect_range = params["range"]
        
        # 计算边界框大小
        region_size = []
        for d in range(self.dim):
            region_size.append(rect_range[d][1] - rect_range[d][0])
        
        if self.sampling_method == "poisson":
            # 使用泊松采样
            from Util.poisson_disk_sampling import poisson_disk_sampling_by_count_cached as poisson_disk_sampling_by_count
            points_np, radius_info = self._poisson_sampling_with_radius(region_size, n_particles)
            
            # 存储半径信息用于边界检测
            if radius_info is not None:
                self.last_poisson_radius = radius_info
            
            for point in points_np:
                pos = []
                for d in range(self.dim):
                    pos.append(point[d] + rect_range[d][0])
                particles.append(pos)
        elif self.sampling_method == "regular":
            # 规则分布采样
            particles = self._generate_regular_grid_particles(rect_range, n_particles)
        elif self.sampling_method == "gauss":
            # 高斯积分点采样
            particles, weights = self._generate_gauss_quadrature_particles(rect_range, n_particles)
            # 将权重信息附加到粒子数据中，以便后续使用
            self.gauss_weights = weights
        else:
            # 均匀随机采样 (uniform)
            for _ in range(n_particles):
                pos = []
                for d in range(self.dim):
                    min_val = rect_range[d][0]
                    max_val = rect_range[d][1]
                    pos.append(np.random.uniform(min_val, max_val))
                particles.append(pos)
        
        return particles
    
    def _generate_regular_grid_particles(self, rect_range, n_particles):
        """生成规则格点分布的粒子"""
        particles = []
        
        # 计算区域总面积/体积
        total_volume = 1.0
        for d in range(self.dim):
            total_volume *= (rect_range[d][1] - rect_range[d][0])
        
        # 计算每个粒子占据的体积
        particle_volume = total_volume / n_particles
        
        # 计算格点间距（假设是正方形/立方体格点）
        spacing = particle_volume ** (1.0 / self.dim)

        #存储
        self.last_poisson_radius = spacing

        # 计算每个维度的格点数量
        grid_counts = []
        for d in range(self.dim):
            dimension_length = rect_range[d][1] - rect_range[d][0]
            count = max(1, int(round(dimension_length / spacing)))
            grid_counts.append(count)
        
        # 重新调整spacing以均匀分布
        adjusted_spacing = []
        for d in range(self.dim):
            dimension_length = rect_range[d][1] - rect_range[d][0]
            adjusted_spacing.append(dimension_length / (grid_counts[d] + 1))
        
        # 生成格点粒子
        if self.dim == 2:
            for i in range(grid_counts[0]):
                for j in range(grid_counts[1]):
                    x = rect_range[0][0] + (i + 1) * adjusted_spacing[0]
                    y = rect_range[1][0] + (j + 1) * adjusted_spacing[1]
                    particles.append([x, y])
        elif self.dim == 3:
            for i in range(grid_counts[0]):
                for j in range(grid_counts[1]):
                    for k in range(grid_counts[2]):
                        x = rect_range[0][0] + (i + 1) * adjusted_spacing[0]
                        y = rect_range[1][0] + (j + 1) * adjusted_spacing[1]
                        z = rect_range[2][0] + (k + 1) * adjusted_spacing[2]
                        particles.append([x, y, z])
        
        # 如果生成的粒子数量超过需要的数量，随机选择一部分
        # if len(particles) > n_particles:
        #     import random
        #     particles = random.sample(particles, n_particles)
        
        return particles
    
    def _generate_gauss_quadrature_particles(self, rect_range, n_particles):
        """生成基于高斯积分点的粒子分布，基于网格结构
        
        Returns:
            tuple: (particles, weights) 其中particles是位置列表，weights是对应的权重列表
        """
        if self.dim != 2:
            raise ValueError("高斯积分点采样目前只支持2D")
        
        # 验证particles_per_grid是完全平方数
        try:
            n_1d = GaussQuadrature.validate_particles_per_grid(self.particles_per_grid)
        except ValueError as e:
            # 如果不是完全平方数，计算最接近的完全平方数
            sqrt_n = int(self.particles_per_grid ** 0.5)
            if sqrt_n * sqrt_n < self.particles_per_grid:
                sqrt_n += 1
            n_1d = min(sqrt_n, 10)  # 最大支持10个点
            print(f"警告: {e}")
            print(f"将使用{n_1d}x{n_1d}={n_1d*n_1d}个高斯积分点")
        
        particles = []
        particle_weights = []
        
        # 计算网格间距
        dx = 1.0 / self.grid_size
        
        # 获取高斯积分点的相对位置和权重（在±0.5dx范围内）
        gauss_positions, gauss_weights = GaussQuadrature.get_2d_grid_points_and_weights(n_1d, dx)
        
        # 遍历所有网格点
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # 网格中心位置：I * dx，其中I是网格索引
                grid_center_x = (i+0.5) * dx
                grid_center_y = (j+0.5) * dx

                # 在每个网格中心周围放置高斯积分点
                for idx, pos in enumerate(gauss_positions):
                    particle_x = grid_center_x + pos[0]
                    particle_y = grid_center_y + pos[1]
                    
                    # 检查粒子是否在指定的形状区域内
                    if (rect_range[0][0] <= particle_x <= rect_range[0][1] and 
                        rect_range[1][0] <= particle_y <= rect_range[1][1]):
                        particles.append([particle_x, particle_y])
                        particle_weights.append(gauss_weights[idx])
        
        
        return particles, particle_weights
    
    def _generate_ellipse_particles(self, params, n_particles):
        """生成椭圆区域内的粒子"""
        center = params["center"]
        semi_axes = params["semi_axes"]
        
        if self.sampling_method == "poisson":
            # 使用椭圆的泊松采样
            try:
                from Util.poisson_disk_sampling import poisson_disk_sampling_ellipse
                points_list, radius_info = self._ellipse_poisson_sampling_with_radius(center, semi_axes, n_particles)
                
                # 存储半径信息
                if radius_info is not None:
                    self.last_poisson_radius = radius_info
                    
                particles = [list(point) for point in points_list]
            except Exception as e:
                print(f"椭圆Poisson采样失败，使用传统方法: {e}")
                # 回退到传统方法
                particles = self._traditional_ellipse_sampling(center, semi_axes, n_particles)
        elif self.sampling_method == "regular":
            # 椭圆的规则采样
            particles = self._generate_regular_ellipse_particles(center, semi_axes, n_particles)
        else:
            # 传统随机采样方法 (uniform)
            particles = self._traditional_ellipse_sampling(center, semi_axes, n_particles)
        
        return particles
    
    def _generate_regular_ellipse_particles(self, center, semi_axes, n_particles):
        """生成椭圆区域内的规则格点分布粒子"""
        particles = []
        
        # 计算椭圆的包围盒
        bbox_min = []
        bbox_max = []
        for d in range(self.dim):
            bbox_min.append(center[d] - semi_axes[d])
            bbox_max.append(center[d] + semi_axes[d])
        
        # 创建包围盒的rect_range格式
        rect_range = []
        for d in range(self.dim):
            rect_range.append([bbox_min[d], bbox_max[d]])
        
        # 计算椭圆面积/体积
        if self.dim == 2:
            ellipse_area = math.pi * semi_axes[0] * semi_axes[1]
        else:
            ellipse_area = (4.0/3.0) * math.pi * semi_axes[0] * semi_axes[1] * semi_axes[2]
        
        # 基于椭圆面积计算理论粒子数量，用于估算格点密度
        bbox_area = 1.0
        for d in range(self.dim):
            bbox_area *= (bbox_max[d] - bbox_min[d])
        
        # 估算需要在包围盒中生成多少格点，使得椭圆内大约有n_particles个粒子
        fill_ratio = ellipse_area / bbox_area
        estimated_bbox_particles = int(0.8*n_particles / fill_ratio)  # 1.2是安全系数
        
        # 在包围盒中生成规则格点
        bbox_particles = self._generate_regular_grid_particles(rect_range, estimated_bbox_particles)
        
        # 筛选出在椭圆内的粒子
        for particle_pos in bbox_particles:
            if self._is_point_in_ellipse(particle_pos, center, semi_axes):
                particles.append(particle_pos)
        
        # # 如果粒子数量过多，随机选择一部分
        # if len(particles) > n_particles:
        #     import random
        #     particles = random.sample(particles, n_particles)
        
        return particles
    
    def _is_point_in_ellipse(self, point, center, semi_axes):
        """检查点是否在椭圆内"""
        sum_normalized = 0.0
        for d in range(self.dim):
            diff = point[d] - center[d]
            sum_normalized += (diff / semi_axes[d])**2
        return sum_normalized <= 1.0
    
    def _remove_particles_in_shape(self, particles, shape):
        """从粒子列表中移除在指定形状内的粒子"""
        remaining_particles = []
        
        for particle_pos in particles:
            if not self._is_particle_in_shape(particle_pos, shape):
                remaining_particles.append(particle_pos)
        
        return remaining_particles
    
    def _remove_particles_in_shape_with_material(self, particles, shape):
        """从包含材料ID的粒子列表中移除在指定形状内的粒子"""
        remaining_particles = []
        
        for particle in particles:
            particle_pos = particle["position"]
            if not self._is_particle_in_shape(particle_pos, shape):
                remaining_particles.append(particle)
        
        return remaining_particles
    
    def _change_particles_material_in_shape(self, particles, shape, new_material_id):
        """改变指定形状内粒子的材料ID"""
        modified_particles = []
        
        for particle in particles:
            particle_pos = particle["position"]
            if self._is_particle_in_shape(particle_pos, shape):
                # 粒子在形状内，改变材料ID
                modified_particle = particle.copy()
                modified_particle["material_id"] = new_material_id
                modified_particles.append(modified_particle)
            else:
                # 粒子不在形状内，保持不变
                modified_particles.append(particle)
        
        return modified_particles
    
    def _poisson_sampling_with_radius(self, region_size, n_particles):
        """执行Poisson采样并返回半径信息"""
        import io
        from contextlib import redirect_stdout
        
        # 捕获输出以提取半径信息
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            from Util.poisson_disk_sampling import poisson_disk_sampling_by_count_cached as poisson_disk_sampling_by_count
            points_np = poisson_disk_sampling_by_count(region_size, n_particles)
        
        # 解析输出获取最终半径
        output_text = captured_output.getvalue()
        radius_info = self._extract_radius_from_output(output_text)
        
        return points_np, radius_info
    
    def _extract_radius_from_output(self, output_text):
        """从输出文本中提取最终半径"""
        lines = output_text.strip().split('\n')
        for line in reversed(lines):  # 从后往前找最终半径
            if '最终半径:' in line or '最终半径' in line:
                try:
                    # 提取半径数值
                    parts = line.split('半径:')[1].split(',')[0].strip()
                    radius = float(parts)
                    return radius
                except (IndexError, ValueError):
                    continue
        return None
    
    def get_last_poisson_radius(self):
        """获取最后使用的Poisson采样半径"""
        return self.last_poisson_radius
    
    def _ellipse_poisson_sampling_with_radius(self, center, semi_axes, n_particles):
        """椭圆Poisson采样并返回半径信息"""
        import io
        from contextlib import redirect_stdout
        
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            from Util.poisson_disk_sampling import poisson_disk_sampling_ellipse
            points_list = poisson_disk_sampling_ellipse(center, semi_axes, n_particles)
        
        output_text = captured_output.getvalue()
        radius_info = self._extract_radius_from_output(output_text)
        
        return points_list, radius_info
    
    def _traditional_ellipse_sampling(self, center, semi_axes, n_particles):
        """传统椭圆采样方法"""
        particles = []
        generated = 0
        max_attempts = n_particles * 10
        attempts = 0
        
        while generated < n_particles and attempts < max_attempts:
            attempts += 1
            
            if self.dim == 2:
                x = np.random.uniform(-semi_axes[0], semi_axes[0])
                y = np.random.uniform(-semi_axes[1], semi_axes[1])
                
                if (x/semi_axes[0])**2 + (y/semi_axes[1])**2 <= 1.0:
                    particles.append([center[0] + x, center[1] + y])
                    generated += 1
            else:
                x = np.random.uniform(-semi_axes[0], semi_axes[0])
                y = np.random.uniform(-semi_axes[1], semi_axes[1])
                z = np.random.uniform(-semi_axes[2], semi_axes[2])
                
                if (x/semi_axes[0])**2 + (y/semi_axes[1])**2 + (z/semi_axes[2])**2 <= 1.0:
                    particles.append([center[0] + x, center[1] + y, center[2] + z])
                    generated += 1
        
        return particles
    
    def _is_particle_in_shape(self, particle_pos, shape):
        """检查粒子是否在指定形状内"""
        shape_type = shape["type"]
        params = shape["params"]
        
        if shape_type == "rectangle":
            rect_range = params["range"]
            for d in range(self.dim):
                if particle_pos[d] < rect_range[d][0] or particle_pos[d] > rect_range[d][1]:
                    return False
            return True
            
        elif shape_type == "ellipse":
            center = params["center"]
            semi_axes = params["semi_axes"]
            
            return self._is_point_in_ellipse(particle_pos, center, semi_axes)
        
        return False


class ParticleInitializer:
    """粒子属性初始化器"""
    
    def __init__(self, dim=2, float_type=np.float32, init_vel_y=-1):
        self.dim = dim
        self.float_type = float_type
        self.init_vel_y = init_vel_y
    
    def initialize_particle_fields(self, particle_data, x_field, v_field, F_field, C_field, material_id_field=None, weight_field=None):
        """初始化粒子的各种属性字段"""
        import taichi as ti
        n_particles = len(particle_data)
        
        # 处理两种格式：旧格式(只有位置)和新格式(包含材料ID和权重)
        positions = []
        material_ids = []
        weights = []
        
        for data in particle_data:
            if isinstance(data, dict) and "position" in data:
                # 新格式：包含位置、材料ID和可选的权重
                positions.append(data["position"])
                material_ids.append(data.get("material_id", 0))
                weights.append(data.get("weight", 1.0))  # 默认权重为1.0
            else:
                # 旧格式：只有位置信息
                positions.append(data)
                material_ids.append(0)
                weights.append(1.0)
        
        # 创建临时numpy数组用于传递粒子属性
        numpy_dtype = np.float32 if self.float_type == ti.f32 else np.float64
        positions_np = np.array(positions, dtype=numpy_dtype)
        material_ids_np = np.array(material_ids, dtype=np.int32)
        weights_np = np.array(weights, dtype=numpy_dtype)
        
        temp_positions = ti.Vector.field(self.dim, self.float_type, shape=n_particles)
        temp_material_ids = ti.field(ti.i32, shape=n_particles)
        temp_weights = ti.field(self.float_type, shape=n_particles)
        
        temp_positions.from_numpy(positions_np)
        temp_material_ids.from_numpy(material_ids_np)
        temp_weights.from_numpy(weights_np)
        
        # 初始化粒子属性
        @ti.kernel
        def init_kernel():
            for i in range(n_particles):
                x_field[i] = temp_positions[i]
                v_field[i] = ti.Vector.zero(self.float_type, self.dim)
                if self.dim >= 2:
                    v_field[i][1] = self.init_vel_y
                F_field[i] = ti.Matrix.identity(self.float_type, self.dim)
                C_field[i] = ti.Matrix.zero(self.float_type, self.dim, self.dim)
        
        @ti.kernel
        def init_material_ids():
            for i in range(n_particles):
                material_id_field[i] = temp_material_ids[i]
        
        @ti.kernel
        def init_weights():
            for i in range(n_particles):
                weight_field[i] = temp_weights[i]
        
        init_kernel()
        
        # 设置材料ID（如果提供了material_id_field）
        if material_id_field is not None:
            init_material_ids()
        
        # 设置权重（如果提供了weight_field）
        if weight_field is not None:
            init_weights()