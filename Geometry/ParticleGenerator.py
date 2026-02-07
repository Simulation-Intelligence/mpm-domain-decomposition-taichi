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

    def __init__(self, dim=2, sampling_method="poisson", particles_per_grid=8, grid_size=16,
                 grid_nx=None, grid_ny=None, grid_nz=None, domain_width=1.0, domain_height=1.0, domain_depth=1.0, use_mesh_boundary=True):
        """
        初始化粒子生成器
        Args:
            dim: 维度 (2 or 3)
            sampling_method: 采样方式，可选 "uniform", "poisson", "regular", "gauss", "mesh"
            particles_per_grid: 每个网格的粒子数
            grid_size: 网格大小（兼容旧版）
            grid_nx: 网格x方向大小
            grid_ny: 网格y方向大小
            domain_width: 域宽度
            domain_height: 域高度
            use_mesh_boundary: 是否在mesh采样时使用边界信息
        """
        self.dim = dim
        self.sampling_method = sampling_method
        self.use_mesh_boundary = use_mesh_boundary
        self.particles_per_grid = particles_per_grid

        # 支持新的网格配置，区分2D和3D
        if dim == 2:
            if grid_nx is not None and grid_ny is not None:
                self.grid_nx = grid_nx
                self.grid_ny = grid_ny
                self.grid_size = max(grid_nx, grid_ny)  # 兼容性
            else:
                self.grid_size = grid_size
                self.grid_nx = grid_size
                self.grid_ny = grid_size

            # 域尺寸
            self.domain_width = domain_width
            self.domain_height = domain_height
            self.domain_depth = 1.0

        else:  # 3D情况
            if grid_nx is not None and grid_ny is not None and grid_nz is not None:
                self.grid_nx = grid_nx
                self.grid_ny = grid_ny
                self.grid_nz = grid_nz
                self.grid_size = max(grid_nx, grid_ny, grid_nz)  # 兼容性
            else:
                self.grid_size = grid_size
                self.grid_nx = grid_size
                self.grid_ny = grid_size
                self.grid_nz = grid_size

            # 域尺寸
            self.domain_width = domain_width
            self.domain_height = domain_height
            self.domain_depth = domain_depth

        self.last_poisson_radius = None  # 存储最后使用的Poisson采样半径
    
    def generate_particles_for_shapes(self, shapes, particles_per_area):
        """为多个形状生成粒子，按照config中的顺序依次执行添加和挖空操作"""

        # 如果使用mesh采样方法，使用特殊的处理流程
        if self.sampling_method == "mesh":
            return self._generate_mesh_particles_for_shapes(shapes, particles_per_area)

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

        # 计算网格间距（物理坐标，使用实际的 domain 尺寸）
        # 对于非正方形域，x和y方向的间距可能不同
        dx = self.domain_width / self.grid_nx
        dy = self.domain_height / self.grid_ny

        # 使用平均间距来获取高斯积分点（假设网格接近正方形）
        avg_spacing = (dx + dy) / 2.0
        gauss_positions, gauss_weights = GaussQuadrature.get_2d_grid_points_and_weights(n_1d, avg_spacing)

        # 遍历所有网格点
        for i in range(self.grid_nx):
            for j in range(self.grid_ny):
                # 网格中心位置（物理坐标）
                grid_center_x = (i + 0.5) * dx
                grid_center_y = (j + 0.5) * dy

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


    def _create_rectangle_poly(self, rect_range):
        """创建矩形的.poly格式数据"""
        x_min, x_max = rect_range[0][0], rect_range[0][1]
        y_min, y_max = rect_range[1][0], rect_range[1][1]

        # 定义矩形的四个顶点 (逆时针顺序)
        vertices = [
            (x_min, y_min),  # 左下
            (x_max, y_min),  # 右下
            (x_max, y_max),  # 右上
            (x_min, y_max)   # 左上
        ]

        # 构建.poly格式的字符串
        poly_content = f"4 2 0 0\n"  # 4个顶点，2维，0个属性，0个边界标记
        for i, (x, y) in enumerate(vertices, 1):
            poly_content += f"{i} {x:.6f} {y:.6f}\n"

        poly_content += "4 0\n"  # 4条边，0个边界标记
        for i in range(4):
            next_i = (i + 1) % 4 + 1
            poly_content += f"{i+1} {i+1} {next_i}\n"

        poly_content += "0\n"  # 0个hole

        return poly_content

    def _create_ellipse_poly(self, center, semi_axes, n_particles):
        """创建椭圆的.poly格式数据（通过多边形近似）"""
        import math

        # 根据椭圆几何特性和目标粒子密度计算边界分段数
        # 计算椭圆面积和周长
        ellipse_area = math.pi * semi_axes[0] * semi_axes[1]

        # 椭圆周长的近似公式（Ramanujan第二近似）
        a, b = semi_axes[0], semi_axes[1]
        h = ((a - b) / (a + b)) ** 2
        ellipse_perimeter = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

        # 计算理想的粒子间距：假设粒子均匀分布在椭圆面积内
        particle_density = n_particles / ellipse_area
        ideal_particle_spacing = 1.0 / math.sqrt(particle_density)

        # 基于理想间距计算边界分段数
        n_segments = max(16, int(ellipse_perimeter / ideal_particle_spacing))

        # 为避免边界过密，限制最大分段数
        n_segments = min(n_segments, n_particles // 2)

        vertices = []
        for i in range(n_segments):
            angle = 2 * math.pi * i / n_segments
            x = center[0] + semi_axes[0] * math.cos(angle)
            y = center[1] + semi_axes[1] * math.sin(angle)
            vertices.append((x, y))

        # 构建.poly格式的字符串
        poly_content = f"{n_segments} 2 0 0\n"
        for i, (x, y) in enumerate(vertices, 1):
            poly_content += f"{i} {x:.6f} {y:.6f}\n"

        poly_content += f"{n_segments} 0\n"
        for i in range(n_segments):
            next_i = (i + 1) % n_segments + 1
            poly_content += f"{i+1} {i+1} {next_i}\n"

        poly_content += "0\n"  # 0个hole

        return poly_content

    def _call_triangle_and_extract_vertices(self, poly_data, target_particle_spacing, save_mesh=True):
        """调用Triangle工具并提取顶点，可选择保存mesh数据"""
        import tempfile
        import os
        import subprocess

        vertices = []

        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 写入.poly文件
                poly_file = os.path.join(temp_dir, "input.poly")
                with open(poly_file, 'w') as f:
                    f.write(poly_data)

                # 基于目标粒子间距计算面积约束
                # 目标三角形边长应该与粒子间距相当
                target_triangle_area = (target_particle_spacing ** 2) * 0.5  # 等边三角形面积
                area_constraint = f"{target_triangle_area:.8f}"

                # 调用Triangle
                triangle_path = "/Users/zhaofen2/Desktop/work/SIG/triangle/triangle"
                # 对于复杂边界，移除YY参数，让Triangle有更多自由度处理边界
                cmd = [triangle_path, f"-pYYqa{area_constraint}", "input"]

                print(f"调用Triangle命令: {triangle_path} -pYYqa{area_constraint} input")
                result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"Triangle执行失败，返回码: {result.returncode}")
                    return self._fallback_sampling(poly_data, target_particle_spacing)

                # 读取生成的.node文件
                node_file = os.path.join(temp_dir, "input.1.node")
                if os.path.exists(node_file):
                    vertices = self._parse_node_file(node_file)

                    # 如果需要保存mesh数据，则保存所有相关文件
                    if save_mesh:
                        self._save_mesh_data(temp_dir, target_particle_spacing)
                else:
                    return self._fallback_sampling(poly_data, target_particle_spacing)

        except Exception as e:
            return self._fallback_sampling(poly_data, target_particle_spacing)

        return vertices

    def _calculate_area_constraint(self, poly_data, n_particles):
        """根据目标粒子数计算面积约束"""
        # 估算形状总面积
        total_area = self._estimate_poly_area(poly_data)
        # 计算每个三角形的平均面积
        avg_triangle_area = total_area / n_particles
        return f"{avg_triangle_area:.8f}"

    def _estimate_poly_area(self, poly_data):
        """估算多边形面积"""
        lines = poly_data.strip().split('\n')
        n_vertices = int(lines[0].split()[0])

        vertices = []
        for i in range(1, n_vertices + 1):
            parts = lines[i].split()
            x, y = float(parts[1]), float(parts[2])
            vertices.append((x, y))

        # 使用鞋带公式计算面积
        area = 0.0
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        return abs(area) / 2.0

    def _parse_node_file(self, node_file):
        """解析Triangle输出的.node文件"""
        vertices = []

        with open(node_file, 'r') as f:
            lines = f.readlines()

        # 跳过注释和空行
        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

        if not data_lines:
            return vertices

        # 第一行：顶点数 维度 属性数 边界标记数
        header = data_lines[0].split()
        n_vertices = int(header[0])

        # 读取顶点数据
        for i in range(1, min(len(data_lines), n_vertices + 1)):
            parts = data_lines[i].split()
            if len(parts) >= 3:  # 顶点索引 x y [其他数据]
                x, y = float(parts[1]), float(parts[2])
                vertices.append([x, y])

        return vertices

    def _save_mesh_data(self, temp_dir, target_particle_spacing):
        """保存Triangle生成的mesh数据到本地目录"""
        import os
        import shutil
        from datetime import datetime

        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mesh_dir = f"mesh_data/mesh_{timestamp}"
        os.makedirs(mesh_dir, exist_ok=True)

        print(f"Saving mesh data to directory: {mesh_dir}")

        # 需要保存的文件列表
        files_to_save = [
            "input.poly",      # 输入边界文件
            "input.1.node",    # 顶点文件
            "input.1.ele",     # 三角形单元文件
            "input.1.edge",    # 边文件
        ]

        saved_files = []
        for file_name in files_to_save:
            source_path = os.path.join(temp_dir, file_name)
            if os.path.exists(source_path):
                dest_path = os.path.join(mesh_dir, file_name)
                shutil.copy2(source_path, dest_path)
                saved_files.append(file_name)
                print(f"  Saved: {file_name}")

        # 保存mesh信息文件
        info_file = os.path.join(mesh_dir, "mesh_info.txt")
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Mesh Generation Time: {timestamp}\n")
            f.write(f"Target Particle Spacing: {target_particle_spacing:.6f}\n")
            f.write(f"Grid Configuration: {self.grid_nx}×{self.grid_ny}\n")
            f.write(f"Domain Size: {self.domain_width}×{self.domain_height}\n")
            f.write(f"Sampling Method: {self.sampling_method}\n")
            f.write(f"Particles Per Grid: {self.particles_per_grid}\n")
            f.write(f"Saved Files: {', '.join(saved_files)}\n")

        # 解析并保存可读的mesh数据摘要
        self._save_mesh_summary(mesh_dir, temp_dir)

        print(f"Mesh data saved successfully, {len(saved_files)} files saved")
        return mesh_dir

    def _save_mesh_summary(self, mesh_dir, temp_dir):
        """保存mesh数据的可读摘要"""
        import os

        summary_file = os.path.join(mesh_dir, "mesh_summary.txt")

        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== Mesh Data Summary ===\n\n")

                # 读取顶点信息
                node_file = os.path.join(temp_dir, "input.1.node")
                if os.path.exists(node_file):
                    vertices = self._parse_node_file(node_file)
                    f.write(f"Number of Vertices: {len(vertices)}\n")
                    if vertices:
                        f.write(f"Vertex Range: X[{min(v[0] for v in vertices):.4f}, {max(v[0] for v in vertices):.4f}], ")
                        f.write(f"Y[{min(v[1] for v in vertices):.4f}, {max(v[1] for v in vertices):.4f}]\n")

                # 读取三角形信息
                ele_file = os.path.join(temp_dir, "input.1.ele")
                if os.path.exists(ele_file):
                    triangles = self._parse_ele_file(ele_file)
                    f.write(f"Number of Triangles: {len(triangles)}\n")

                # 读取边界信息
                edge_file = os.path.join(temp_dir, "input.1.edge")
                if os.path.exists(edge_file):
                    edges = self._parse_edge_file(edge_file)
                    boundary_edges = [e for e in edges if len(e) > 2 and e[2] != 0]
                    f.write(f"Number of Edges: {len(edges)}\n")
                    f.write(f"Number of Boundary Edges: {len(boundary_edges)}\n")

        except Exception as e:
            print(f"Error saving mesh summary: {e}")

    def _parse_ele_file(self, ele_file):
        """解析Triangle输出的.ele文件"""
        triangles = []

        try:
            with open(ele_file, 'r') as f:
                lines = f.readlines()

            data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

            if not data_lines:
                return triangles

            # 第一行：三角形数量 每个三角形的顶点数 属性数
            header = data_lines[0].split()
            n_triangles = int(header[0])

            # 读取三角形数据
            for i in range(1, min(len(data_lines), n_triangles + 1)):
                parts = data_lines[i].split()
                if len(parts) >= 4:  # 三角形索引 顶点1 顶点2 顶点3
                    triangle = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]  # 转换为0索引
                    triangles.append(triangle)

        except Exception as e:
            print(f"Error parsing .ele file: {e}")

        return triangles

    def _parse_edge_file(self, edge_file):
        """解析Triangle输出的.edge文件"""
        edges = []

        try:
            with open(edge_file, 'r') as f:
                lines = f.readlines()

            data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

            if not data_lines:
                return edges

            # 第一行：边数量 边界标记数
            header = data_lines[0].split()
            n_edges = int(header[0])

            # 读取边数据
            for i in range(1, min(len(data_lines), n_edges + 1)):
                parts = data_lines[i].split()
                if len(parts) >= 3:  # 边索引 顶点1 顶点2 [边界标记]
                    edge = [int(parts[1]) - 1, int(parts[2]) - 1]  # 转换为0索引
                    if len(parts) > 3:
                        edge.append(int(parts[3]))  # 边界标记
                    edges.append(edge)

        except Exception as e:
            print(f"Error parsing .edge file: {e}")

        return edges

    def _fallback_sampling(self, poly_data, target_particle_spacing):
        """Triangle失败时的后备采样方法"""

        # 从poly_data中提取边界框
        lines = poly_data.strip().split('\n')
        n_vertices = int(lines[0].split()[0])

        x_coords, y_coords = [], []
        for i in range(1, n_vertices + 1):
            parts = lines[i].split()
            x_coords.append(float(parts[1]))
            y_coords.append(float(parts[2]))

        # 检查是否有有效的坐标数据
        if not x_coords or not y_coords:
            print("警告: poly数据中没有有效的顶点坐标")
            return []

        # 基于目标粒子间距计算采样数量
        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        bbox_area = bbox_width * bbox_height
        n_particles = int(bbox_area / (target_particle_spacing ** 2))

        # 在边界框内随机采样
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        vertices = []
        for _ in range(n_particles):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            vertices.append([x, y])

        return vertices

    def _generate_mesh_particles_for_shapes(self, shapes, particles_per_area):
        """为mesh采样方法专门设计的粒子生成流程
        先构建最终的几何边界，然后用triangle生成网格粒子
        """
        if self.dim != 2:
            raise ValueError("Mesh采样目前只支持2D")

        print("使用mesh采样方法，先构建最终边界...")

        # 基于particles_per_grid计算粒子间距
        # particles_per_grid表示每个网格单元的粒子数
        # 计算网格单元面积
        grid_cell_area = (self.domain_width / self.grid_nx) * (self.domain_height / self.grid_ny)
        particles_per_unit_area = self.particles_per_grid / grid_cell_area
        target_particle_spacing = 1.0 / (particles_per_unit_area ** 0.5)

        # 基于粒子间距确定统一的边界段长度
        target_segment_length = target_particle_spacing * 0.5  # 边界段长度为粒子间距的一半

        print(f"网格配置: {self.grid_nx}×{self.grid_ny}, 域尺寸: {self.domain_width}×{self.domain_height}")
        print(f"每网格粒子数: {self.particles_per_grid}")
        print(f"目标粒子间距: {target_particle_spacing:.6f}")
        print(f"统一边界段长度: {target_segment_length:.6f}")

        # 4. 构建综合的.poly文件，使用统一的段长度，并返回边界点
        poly_content, boundary_points = self._build_combined_poly_file(shapes, target_segment_length)

        # 5. 调用Triangle生成网格
        vertices = self._call_triangle_and_extract_vertices(poly_content, target_particle_spacing)

        # 6. 为粒子分配材料ID和边界标记
        particles_with_material = []
        for pos in vertices:
            # 确定粒子所属的材料
            material_id = self._determine_particle_material(pos, shapes)
            # 检查是否为边界粒子（根据use_mesh_boundary配置决定）
            is_boundary = False
            if self.use_mesh_boundary:
                is_boundary = self._is_boundary_particle_from_input(pos, boundary_points, target_particle_spacing *1e-2)

            particle_data = {
                "position": pos,
                "material_id": material_id,
                "is_boundary": is_boundary
            }
            particles_with_material.append(particle_data)

        boundary_count = sum(1 for p in particles_with_material if p.get("is_boundary", False))
        print(f"Mesh采样生成了 {len(particles_with_material)} 个粒子，其中 {boundary_count} 个边界粒子")
        return particles_with_material

    def _is_boundary_particle_from_input(self, particle_pos, boundary_points, tolerance):
        """检查粒子是否为边界粒子（基于输入边界点）"""
        # 检查粒子位置是否与任何边界点接近
        for boundary_point in boundary_points:
            distance = ((particle_pos[0] - boundary_point[0])**2 + (particle_pos[1] - boundary_point[1])**2)**0.5
            if distance < tolerance:
                return True
        return False

    def _build_combined_poly_file(self, shapes, target_segment_length):
        """构建包含所有几何操作的综合.poly文件
        按照shapes顺序统一处理，每个边界点都进行有效性检查
        """
        import math

        print("构建综合几何边界，按顺序统一处理...")

        all_segments = []
        all_holes = []
        vertex_id_counter = 1

        # 按照shapes顺序统一处理
        for i, shape in enumerate(shapes):
            print(f"处理第 {i+1} 个形状: {shape['type']} ({shape['operation']})")

            if shape["operation"] == "add":
                # 生成add边界，但排除后续subtract区域内的点
                later_subtract_shapes = [s for s in shapes[i+1:] if s["operation"] == "subtract"]
                segments, vertex_id_counter = self._add_shape_boundary_with_exclusion(
                    shape, later_subtract_shapes, vertex_id_counter, boundary_marker=1, target_segment_length=target_segment_length)
                all_segments.extend(segments)
                print(f"  添加了 {len(segments)} 个边界段")

            elif shape["operation"] == "change":
                # change操作：改变材料属性，但不影响几何边界
                print(f"  change操作：不影响边界生成")

            elif shape["operation"] == "subtract":
                # 检查hole点是否在之前的add区域内
                previous_add_shapes = [s for s in shapes[:i] if s["operation"] == "add"]
                hole_point = self._get_shape_center(shape)

                # 首先从all_segments中移除所有在当前subtract范围内的segments
                original_count = len(all_segments)
                all_segments = self._remove_segments_in_shape(all_segments, shape)
                removed_count = original_count - len(all_segments)
                print(f"  移除了 {removed_count} 个在subtract范围内的已有边界段")

                # 检查subtract是否与add区域相交
                intersects_add = self._subtract_intersects_add(shape, previous_add_shapes)

                if self._point_in_add_regions(hole_point, previous_add_shapes):
                    # hole点在add内部：正常的subtract操作
                    segments, vertex_id_counter = self._add_shape_boundary_with_inclusion(
                        shape, previous_add_shapes, vertex_id_counter, boundary_marker=1, target_segment_length=target_segment_length)
                    all_segments.extend(segments)
                    print(f"  添加了 {len(segments)} 个边界段")

                    # 只有当subtract完全在add内部时才添加hole点
                    if self._subtract_completely_inside_add(shape, previous_add_shapes):
                        all_holes.append(hole_point)
                        print(f"  添加hole点")

                elif intersects_add:
                    # hole点不在add内部但与add相交：添加相交部分的边界以闭合边界
                    # 但需要排除在之前subtract内的边界段
                    previous_subtract_shapes = [s for s in shapes[:i] if s["operation"] == "subtract"]
                    segments, vertex_id_counter = self._add_shape_boundary_with_inclusion_excluding_subtract(
                        shape, previous_add_shapes, previous_subtract_shapes, vertex_id_counter, boundary_marker=1, target_segment_length=target_segment_length)
                    all_segments.extend(segments)
                    print(f"  添加了 {len(segments)} 个边界段（相交部分，用于闭合边界）")
                else:
                    print(f"  跳过subtract形状（与add区域无交集）")

        # 构建.poly文件内容，并获取边界点
        poly_content, boundary_points = self._assemble_poly_content(all_segments, all_holes)

        return poly_content, boundary_points

    def _add_rectangle_to_poly(self, params, start_vertex_id, boundary_marker=1, target_segment_length=None):
        """将矩形添加到poly文件中，根据target_segment_length细分边界"""
        rect_range = params["range"]
        x_min, x_max = rect_range[0][0], rect_range[0][1]
        y_min, y_max = rect_range[1][0], rect_range[1][1]

        if target_segment_length is None:
            target_segment_length = 0.01  # 默认段长度

        # 计算矩形的周长和各边长度
        width = x_max - x_min
        height = y_max - y_min

        # 计算每条边需要的分段数
        n_segments_horizontal = max(1, int(width / target_segment_length))
        n_segments_vertical = max(1, int(height / target_segment_length))

        # 生成所有边界点 (逆时针顺序)
        vertices = []

        # 底边 (从左到右)
        for i in range(n_segments_horizontal):
            x = x_min + (i * width) / n_segments_horizontal
            vertices.append((x, y_min))

        # 右边 (从下到上)
        for i in range(n_segments_vertical):
            y = y_min + ((i + 1) * height) / n_segments_vertical
            vertices.append((x_max, y))

        # 顶边 (从右到左)
        for i in range(n_segments_horizontal):
            x = x_max - ((i + 1) * width) / n_segments_horizontal
            vertices.append((x, y_max))

        # 左边 (从上到下)
        for i in range(n_segments_vertical - 1):
            y = y_max - ((i + 1) * height) / n_segments_vertical
            vertices.append((x_min, y))

        # 创建边界段
        segments = []
        total_vertices = len(vertices)
        for i in range(total_vertices):
            vertex1_id = start_vertex_id + i
            vertex2_id = start_vertex_id + (i + 1) % total_vertices
            segments.append({
                "id": len(segments) + 1,
                "vertex1": vertex1_id,
                "vertex2": vertex2_id,
                "vertex1_pos": vertices[i],
                "vertex2_pos": vertices[(i + 1) % total_vertices],
                "boundary_marker": boundary_marker
            })

        return segments, start_vertex_id + total_vertices

    def _add_ellipse_to_poly(self, params, start_vertex_id, boundary_marker=1, target_segment_length=None):
        """将椭圆添加到poly文件中"""
        import math

        center = params["center"]
        semi_axes = params["semi_axes"]

        # 计算椭圆周长
        a, b = semi_axes[0], semi_axes[1]
        h = ((a - b) / (a + b)) ** 2
        ellipse_perimeter = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

        # 使用统一的目标段长度确定分段数
        if target_segment_length is None:
            target_segment_length = 0.01  # 默认段长度

        n_segments = max(16, int(ellipse_perimeter / target_segment_length))

        vertices = []
        for i in range(n_segments):
            angle = 2 * math.pi * i / n_segments
            x = center[0] + semi_axes[0] * math.cos(angle)
            y = center[1] + semi_axes[1] * math.sin(angle)
            vertices.append((x, y))

        segments = []
        for i in range(n_segments):
            vertex1_id = start_vertex_id + i
            vertex2_id = start_vertex_id + (i + 1) % n_segments
            segments.append({
                "id": len(segments) + 1,
                "vertex1": vertex1_id,
                "vertex2": vertex2_id,
                "vertex1_pos": vertices[i],
                "vertex2_pos": vertices[(i + 1) % n_segments],
                "boundary_marker": boundary_marker
            })

        return segments, start_vertex_id + n_segments

    def _get_shape_center(self, shape):
        """获取形状的中心点，用作hole点"""
        if shape["type"] == "rectangle":
            rect_range = shape["params"]["range"]
            center_x = (rect_range[0][0] + rect_range[0][1]) / 2
            center_y = (rect_range[1][0] + rect_range[1][1]) / 2
            return (center_x, center_y)
        elif shape["type"] == "ellipse":
            center = shape["params"]["center"]
            return (center[0], center[1])
        return (0, 0)

    def _assemble_poly_content(self, all_segments, all_holes):
        """组装.poly文件内容，并修复边界连接缺口"""
        # 收集所有唯一顶点
        vertices = {}
        vertex_positions = []

        for segment in all_segments:
            pos1 = segment["vertex1_pos"]
            pos2 = segment["vertex2_pos"]

            if pos1 not in vertices:
                vertices[pos1] = len(vertex_positions) + 1
                vertex_positions.append(pos1)

            if pos2 not in vertices:
                vertices[pos2] = len(vertex_positions) + 1
                vertex_positions.append(pos2)

        # 检测并修复边界连接缺口
        print("检测边界连接缺口...")
        fixed_segments = self._fix_boundary_gaps(all_segments, vertices)

        # 构建.poly内容
        poly_content = f"{len(vertex_positions)} 2 0 0\n"

        # 顶点
        for i, (x, y) in enumerate(vertex_positions, 1):
            poly_content += f"{i} {x:.6f} {y:.6f}\n"

        # 边
        poly_content += f"{len(fixed_segments)} 1\n"  # 包含边界标记
        for i, segment in enumerate(fixed_segments, 1):
            v1_id = vertices[segment["vertex1_pos"]]
            v2_id = vertices[segment["vertex2_pos"]]
            marker = segment["boundary_marker"]
            poly_content += f"{i} {v1_id} {v2_id} {marker}\n"

        # holes
        poly_content += f"{len(all_holes)}\n"
        for i, (x, y) in enumerate(all_holes, 1):
            poly_content += f"{i} {x:.6f} {y:.6f}\n"

        # 过滤出真正的外部边界点（排除多个add形状之间的内部连接点）
        # external_boundary_points = self._filter_external_boundary_points(vertex_positions)
        external_boundary_points = vertex_positions  # 暂时不进行过滤

        # 返回poly文件内容和过滤后的外部边界点位置
        return poly_content, external_boundary_points

    def _filter_external_boundary_points(self, vertex_positions):
        """
        过滤出真正的外部边界点，移除那些有太多近邻的边界点
        这些通常是多个add形状之间的内部连接点
        """
        import numpy as np

        print("过滤内部边界点...")

        if len(vertex_positions) < 3:
            return vertex_positions

        # 计算平均边界点间距作为邻居检测的参考
        all_distances = []
        for i, pos1 in enumerate(vertex_positions):
            for j, pos2 in enumerate(vertex_positions):
                if i != j:
                    dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                    all_distances.append(dist)

        if not all_distances:
            return vertex_positions

        # 使用中位距离的一定倍数作为邻居检测阈值
        all_distances.sort()
        median_distance = all_distances[len(all_distances) // 2]
        neighbor_threshold = median_distance * 0.001  # 1.5倍中位距离内算作邻居

        print(f"中位距离: {median_distance:.6f}, 邻居检测阈值: {neighbor_threshold:.6f}")

        # 对每个边界点，统计其邻居数量
        external_boundary_points = []

        for i, pos in enumerate(vertex_positions):
            neighbor_count = 0

            # 统计邻居数量
            for j, other_pos in enumerate(vertex_positions):
                if i != j:
                    distance = ((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)**0.5
                    if distance < neighbor_threshold:
                        neighbor_count += 1

            # 外部边界点通常邻居较少（2个左右），内部连接点邻居较多
            # 这里使用3作为阈值：邻居数<=3的认为是外部边界点
            if neighbor_count == 0:
                external_boundary_points.append(pos)

        print(f"过滤前边界点数量: {len(vertex_positions)}")
        print(f"过滤后边界点数量: {len(external_boundary_points)}")

        # 如果过滤后点太少，降低要求重新过滤
        if len(external_boundary_points) < len(vertex_positions) * 0.5:
            print("过滤后点数较少，使用更宽松的标准...")
            external_boundary_points = []
            for i, pos in enumerate(vertex_positions):
                neighbor_count = 0
                for j, other_pos in enumerate(vertex_positions):
                    if i != j:
                        distance = ((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)**0.5
                        if distance < neighbor_threshold:
                            neighbor_count += 1
                # 更宽松的标准：邻居数<=5
                if neighbor_count <= 5:
                    external_boundary_points.append(pos)
            print(f"宽松标准过滤后边界点数量: {len(external_boundary_points)}")

        return external_boundary_points

    def _fix_boundary_gaps(self, segments, vertices):
        """检测并修复边界段之间的缺口"""
        if not segments:
            return segments

        # 构建边界图：顶点 -> 连接的顶点
        graph = {}
        for segment in segments:
            v1_pos = segment["vertex1_pos"]
            v2_pos = segment["vertex2_pos"]

            if v1_pos not in graph:
                graph[v1_pos] = []
            if v2_pos not in graph:
                graph[v2_pos] = []

            graph[v1_pos].append(v2_pos)
            graph[v2_pos].append(v1_pos)

        # 找到度数为1的顶点（边界端点）
        endpoints = []
        for vertex, neighbors in graph.items():
            if len(neighbors) == 1:
                endpoints.append(vertex)

        print(f"发现 {len(endpoints)} 个边界端点")

        # 如果有2个端点，连接它们
        if len(endpoints) == 2:
            v1_pos, v2_pos = endpoints
            gap_segment = {
                "id": len(segments) + 1,
                "vertex1": None,  # 稍后设置
                "vertex2": None,  # 稍后设置
                "vertex1_pos": v1_pos,
                "vertex2_pos": v2_pos,
                "boundary_marker": 1
            }
            fixed_segments = segments + [gap_segment]
            print(f"添加连接段，距离: {((v1_pos[0] - v2_pos[0])**2 + (v1_pos[1] - v2_pos[1])**2)**0.5:.6f}")
            return fixed_segments

        elif len(endpoints) > 2:
            # 多个端点，需要连接成对来形成闭合边界
            print(f"需要配对连接多个端点")

            # 贪心算法：重复寻找最近的端点对并连接
            remaining_endpoints = endpoints.copy()
            gap_segments = []

            while len(remaining_endpoints) >= 2:
                # 寻找当前剩余端点中最近的一对
                min_distance = float('inf')
                best_pair = None
                best_indices = None

                for i in range(len(remaining_endpoints)):
                    for j in range(i + 1, len(remaining_endpoints)):
                        v1, v2 = remaining_endpoints[i], remaining_endpoints[j]
                        # 计算欧氏距离
                        distance = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**0.5
                        if distance < min_distance:
                            min_distance = distance
                            best_pair = (v1, v2)
                            best_indices = (i, j)

                if best_pair:
                    v1_pos, v2_pos = best_pair
                    gap_segment = {
                        "id": len(segments) + len(gap_segments) + 1,
                        "vertex1": None,
                        "vertex2": None,
                        "vertex1_pos": v1_pos,
                        "vertex2_pos": v2_pos,
                        "boundary_marker": 1
                    }
                    gap_segments.append(gap_segment)
                    print(f"添加连接段 {len(gap_segments)}, 距离: {min_distance:.6f}")

                    # 从剩余端点中移除已配对的点
                    remaining_endpoints.pop(best_indices[1])  # 先移除较大索引
                    remaining_endpoints.pop(best_indices[0])
                else:
                    break

            if gap_segments:
                fixed_segments = segments + gap_segments
                return fixed_segments

        return segments

    def _determine_particle_material(self, position, shapes):
        """根据粒子位置和形状层次确定材料ID"""
        material_id = 0  # 默认材料

        # 按shapes顺序处理，后面的操作会覆盖前面的
        for shape in shapes:
            if self._is_particle_in_shape(position, shape):
                if shape["operation"] == "add":
                    material_id = shape.get("material_id", 0)
                elif shape["operation"] == "change":
                    material_id = shape.get("material_id", 0)
                # subtract操作不改变材料ID，因为粒子不应该在subtract区域内

        return material_id

    def _shape_intersects_with_add_regions(self, subtract_shape, add_shapes):
        """检查subtract形状是否与任何add形状有交集"""
        for add_shape in add_shapes:
            if self._shapes_intersect(subtract_shape, add_shape):
                return True
        return False

    def _point_in_add_regions(self, point, add_shapes):
        """检查点是否在任何add区域内"""
        for add_shape in add_shapes:
            if self._is_particle_in_shape(point, add_shape):
                return True
        return False

    def _shapes_intersect(self, shape1, shape2):
        """检查两个形状是否相交（简化版本）"""
        # 使用边界框快速检测
        bbox1 = self._get_shape_bbox(shape1)
        bbox2 = self._get_shape_bbox(shape2)

        # 检查边界框是否相交
        return not (bbox1[1][0] < bbox2[0][0] or  # shape1右边 < shape2左边
                   bbox1[0][0] > bbox2[1][0] or   # shape1左边 > shape2右边
                   bbox1[1][1] < bbox2[0][1] or   # shape1上边 < shape2下边
                   bbox1[0][1] > bbox2[1][1])     # shape1下边 > shape2上边

    def _get_shape_bbox(self, shape):
        """获取形状的边界框 [(x_min, y_min), (x_max, y_max)]"""
        if shape["type"] == "rectangle":
            rect_range = shape["params"]["range"]
            return [(rect_range[0][0], rect_range[1][0]),
                   (rect_range[0][1], rect_range[1][1])]
        elif shape["type"] == "ellipse":
            center = shape["params"]["center"]
            semi_axes = shape["params"]["semi_axes"]
            return [(center[0] - semi_axes[0], center[1] - semi_axes[1]),
                   (center[0] + semi_axes[0], center[1] + semi_axes[1])]
        return [(0, 0), (0, 0)]

    def _add_clipped_subtract_boundary(self, subtract_shape, add_shapes, start_vertex_id, target_segment_length):
        """添加裁剪后的subtract边界，只保留在add区域内的部分"""
        # 先生成完整的subtract边界
        if subtract_shape["type"] == "rectangle":
            segments, _ = self._add_rectangle_to_poly(
                subtract_shape["params"], start_vertex_id, boundary_marker=2, target_segment_length=target_segment_length)
        elif subtract_shape["type"] == "ellipse":
            segments, _ = self._add_ellipse_to_poly(
                subtract_shape["params"], start_vertex_id, boundary_marker=2, target_segment_length=target_segment_length)
        else:
            return [], start_vertex_id

        # 过滤掉不在add区域内的边界段
        clipped_segments = []
        vertex_mapping = {}
        new_vertex_id = start_vertex_id

        for segment in segments:
            # 检查边界段的中点是否在add区域内
            v1_pos = segment["vertex1_pos"]
            v2_pos = segment["vertex2_pos"]
            midpoint = ((v1_pos[0] + v2_pos[0]) / 2, (v1_pos[1] + v2_pos[1]) / 2)

            if self._point_in_add_regions(midpoint, add_shapes):
                # 重新映射顶点ID
                if v1_pos not in vertex_mapping:
                    vertex_mapping[v1_pos] = new_vertex_id
                    new_vertex_id += 1
                if v2_pos not in vertex_mapping:
                    vertex_mapping[v2_pos] = new_vertex_id
                    new_vertex_id += 1

                clipped_segment = {
                    "id": len(clipped_segments) + 1,
                    "vertex1": vertex_mapping[v1_pos],
                    "vertex2": vertex_mapping[v2_pos],
                    "vertex1_pos": v1_pos,
                    "vertex2_pos": v2_pos,
                    "boundary_marker": 2
                }
                clipped_segments.append(clipped_segment)

        return clipped_segments, new_vertex_id

    def _add_shape_boundary_with_exclusion(self, shape, exclude_shapes, start_vertex_id, boundary_marker=1, target_segment_length=None):
        """添加形状边界，但排除在exclude_shapes内的边界点"""
        # 先生成完整的边界
        if shape["type"] == "rectangle":
            segments, _ = self._add_rectangle_to_poly(
                shape["params"], start_vertex_id, boundary_marker, target_segment_length)
        elif shape["type"] == "ellipse":
            segments, _ = self._add_ellipse_to_poly(
                shape["params"], start_vertex_id, boundary_marker, target_segment_length)
        else:
            return [], start_vertex_id

        # 过滤掉在exclude_shapes内的边界段
        filtered_segments = []
        vertex_mapping = {}
        new_vertex_id = start_vertex_id

        for segment in segments:
            # 检查边界段的中点是否在任何exclude_shape内
            v1_pos = segment["vertex1_pos"]
            v2_pos = segment["vertex2_pos"]
            midpoint = ((v1_pos[0] + v2_pos[0]) / 2, (v1_pos[1] + v2_pos[1]) / 2)

            # 如果中点不在任何exclude区域内，则保留该边界段
            if not self._point_in_shapes(midpoint, exclude_shapes):
                # 重新映射顶点ID
                if v1_pos not in vertex_mapping:
                    vertex_mapping[v1_pos] = new_vertex_id
                    new_vertex_id += 1
                if v2_pos not in vertex_mapping:
                    vertex_mapping[v2_pos] = new_vertex_id
                    new_vertex_id += 1

                filtered_segment = {
                    "id": len(filtered_segments) + 1,
                    "vertex1": vertex_mapping[v1_pos],
                    "vertex2": vertex_mapping[v2_pos],
                    "vertex1_pos": v1_pos,
                    "vertex2_pos": v2_pos,
                    "boundary_marker": boundary_marker
                }
                filtered_segments.append(filtered_segment)

        return filtered_segments, new_vertex_id

    def _add_shape_boundary_with_inclusion(self, shape, include_shapes, start_vertex_id, boundary_marker=2, target_segment_length=None):
        """添加形状边界，但只保留在include_shapes内的边界点"""
        # 先生成完整的边界
        if shape["type"] == "rectangle":
            segments, _ = self._add_rectangle_to_poly(
                shape["params"], start_vertex_id, boundary_marker, target_segment_length)
        elif shape["type"] == "ellipse":
            segments, _ = self._add_ellipse_to_poly(
                shape["params"], start_vertex_id, boundary_marker, target_segment_length)
        else:
            return [], start_vertex_id

        # 只保留在include_shapes内的边界段
        filtered_segments = []
        vertex_mapping = {}
        new_vertex_id = start_vertex_id

        for segment in segments:
            # 检查边界段的中点是否在任何include_shape内
            v1_pos = segment["vertex1_pos"]
            v2_pos = segment["vertex2_pos"]
            midpoint = ((v1_pos[0] + v2_pos[0]) / 2, (v1_pos[1] + v2_pos[1]) / 2)

            # 如果中点在任何include区域内，则保留该边界段
            if self._point_in_shapes(midpoint, include_shapes):
                # 重新映射顶点ID
                if v1_pos not in vertex_mapping:
                    vertex_mapping[v1_pos] = new_vertex_id
                    new_vertex_id += 1
                if v2_pos not in vertex_mapping:
                    vertex_mapping[v2_pos] = new_vertex_id
                    new_vertex_id += 1

                filtered_segment = {
                    "id": len(filtered_segments) + 1,
                    "vertex1": vertex_mapping[v1_pos],
                    "vertex2": vertex_mapping[v2_pos],
                    "vertex1_pos": v1_pos,
                    "vertex2_pos": v2_pos,
                    "boundary_marker": boundary_marker
                }
                filtered_segments.append(filtered_segment)

        return filtered_segments, new_vertex_id

    def _add_shape_boundary_with_inclusion_excluding_subtract(self, shape, include_shapes, exclude_shapes, start_vertex_id, boundary_marker=2, target_segment_length=None):
        """添加形状边界，保留在include_shapes内但排除在exclude_shapes内的边界点"""
        # 先生成完整的边界
        if shape["type"] == "rectangle":
            segments, _ = self._add_rectangle_to_poly(
                shape["params"], start_vertex_id, boundary_marker, target_segment_length)
        elif shape["type"] == "ellipse":
            segments, _ = self._add_ellipse_to_poly(
                shape["params"], start_vertex_id, boundary_marker, target_segment_length)
        else:
            return [], start_vertex_id

        # 只保留在include_shapes内且不在exclude_shapes内的边界段
        filtered_segments = []
        vertex_mapping = {}
        new_vertex_id = start_vertex_id

        for segment in segments:
            # 检查边界段的中点是否在任何include_shape内
            v1_pos = segment["vertex1_pos"]
            v2_pos = segment["vertex2_pos"]
            midpoint = ((v1_pos[0] + v2_pos[0]) / 2, (v1_pos[1] + v2_pos[1]) / 2)

            # 如果中点在任何include区域内且不在任何exclude区域内，则保留该边界段
            if (self._point_in_shapes(midpoint, include_shapes) and
                not self._point_in_shapes(midpoint, exclude_shapes)):
                # 重新映射顶点ID
                if v1_pos not in vertex_mapping:
                    vertex_mapping[v1_pos] = new_vertex_id
                    new_vertex_id += 1
                if v2_pos not in vertex_mapping:
                    vertex_mapping[v2_pos] = new_vertex_id
                    new_vertex_id += 1

                filtered_segment = {
                    "id": len(filtered_segments) + 1,
                    "vertex1": vertex_mapping[v1_pos],
                    "vertex2": vertex_mapping[v2_pos],
                    "vertex1_pos": v1_pos,
                    "vertex2_pos": v2_pos,
                    "boundary_marker": boundary_marker
                }
                filtered_segments.append(filtered_segment)

        return filtered_segments, new_vertex_id

    def _point_in_shapes(self, point, shapes):
        """检查点是否在任何给定形状内"""
        for shape in shapes:
            if self._is_particle_in_shape(point, shape):
                return True
        return False

    def _subtract_completely_inside_add(self, subtract_shape, add_shapes):
        """检查subtract形状是否完全在add区域内部（不与边界相交）"""
        # 获取subtract形状的边界框
        subtract_bbox = self._get_shape_bbox(subtract_shape)

        # 检查边界框的所有角点是否都在add区域内
        corner_points = [
            subtract_bbox[0],  # (x_min, y_min)
            (subtract_bbox[1][0], subtract_bbox[0][1]),  # (x_max, y_min)
            subtract_bbox[1],  # (x_max, y_max)
            (subtract_bbox[0][0], subtract_bbox[1][1])   # (x_min, y_max)
        ]

        for point in corner_points:
            if not self._point_in_add_regions(point, add_shapes):
                return False

        # 进一步检查：采样subtract边界上的点
        if subtract_shape["type"] == "rectangle":
            boundary_points = self._sample_rectangle_boundary(subtract_shape["params"], 8)
        elif subtract_shape["type"] == "ellipse":
            boundary_points = self._sample_ellipse_boundary(subtract_shape["params"], 16)
        else:
            return True

        for point in boundary_points:
            if not self._point_in_add_regions(point, add_shapes):
                return False

        return True

    def _subtract_intersects_add(self, subtract_shape, add_shapes):
        """检查subtract形状是否与add形状相交"""
        if not add_shapes:
            return False

        # 采样subtract形状的边界点
        if subtract_shape["type"] == "rectangle":
            boundary_points = self._sample_rectangle_boundary(subtract_shape["params"], 20)
        elif subtract_shape["type"] == "ellipse":
            boundary_points = self._sample_ellipse_boundary(subtract_shape["params"], 20)
        else:
            return False

        # 检查是否有任何边界点在add区域内
        for point in boundary_points:
            if self._point_in_add_regions(point, add_shapes):
                return True

        # 反向检查：采样add形状的边界点，看是否有在subtract内
        for add_shape in add_shapes:
            if add_shape["type"] == "rectangle":
                add_boundary_points = self._sample_rectangle_boundary(add_shape["params"], 20)
            elif add_shape["type"] == "ellipse":
                add_boundary_points = self._sample_ellipse_boundary(add_shape["params"], 20)
            else:
                continue

            for point in add_boundary_points:
                if self._point_in_shapes(point, [subtract_shape]):
                    return True

        return False

    def _sample_rectangle_boundary(self, params, n_samples):
        """在矩形边界上采样点"""
        rect_range = params["range"]
        x_min, x_max = rect_range[0][0], rect_range[0][1]
        y_min, y_max = rect_range[1][0], rect_range[1][1]

        points = []
        # 底边
        for i in range(n_samples // 4):
            x = x_min + (i / (n_samples // 4 - 1)) * (x_max - x_min) if n_samples > 4 else (x_min + x_max) / 2
            points.append((x, y_min))
        # 右边
        for i in range(n_samples // 4):
            y = y_min + (i / (n_samples // 4 - 1)) * (y_max - y_min) if n_samples > 4 else (y_min + y_max) / 2
            points.append((x_max, y))
        # 顶边
        for i in range(n_samples // 4):
            x = x_max - (i / (n_samples // 4 - 1)) * (x_max - x_min) if n_samples > 4 else (x_min + x_max) / 2
            points.append((x, y_max))
        # 左边
        for i in range(n_samples // 4):
            y = y_max - (i / (n_samples // 4 - 1)) * (y_max - y_min) if n_samples > 4 else (y_min + y_max) / 2
            points.append((x_min, y))

        return points

    def _sample_ellipse_boundary(self, params, n_samples):
        """在椭圆边界上采样点"""
        import math
        center = params["center"]
        semi_axes = params["semi_axes"]

        points = []
        for i in range(n_samples):
            angle = 2 * math.pi * i / n_samples
            x = center[0] + semi_axes[0] * math.cos(angle)
            y = center[1] + semi_axes[1] * math.sin(angle)
            points.append((x, y))

        return points

    def _remove_segments_in_shape(self, segments, shape):
        """从segments列表中移除所有在指定形状内的segments"""
        filtered_segments = []

        for segment in segments:
            # 检查边界段的中点是否在shape内
            v1_pos = segment["vertex1_pos"]
            v2_pos = segment["vertex2_pos"]
            midpoint = ((v1_pos[0] + v2_pos[0]) / 2, (v1_pos[1] + v2_pos[1]) / 2)

            # 如果中点不在shape内，则保留该边界段
            if not self._is_particle_in_shape(midpoint, shape):
                filtered_segments.append(segment)

        return filtered_segments


class ParticleInitializer:
    """粒子属性初始化器"""
    
    def __init__(self, dim=2, float_type=np.float32, init_vel_y=-1):
        self.dim = dim
        self.float_type = float_type
        self.init_vel_y = init_vel_y
    
    def initialize_particle_fields(self, particle_data, x_field, v_field, F_field, C_field, material_id_field=None, weight_field=None, material_params=None, boundary_field=None):
        """初始化粒子的各种属性字段"""
        import taichi as ti
        n_particles = len(particle_data)
        
        # 处理两种格式：旧格式(只有位置)和新格式(包含材料ID、权重和边界信息)
        positions = []
        material_ids = []
        weights = []
        boundary_flags = []

        for data in particle_data:
            if isinstance(data, dict) and "position" in data:
                # 新格式：包含位置、材料ID、权重和边界信息
                positions.append(data["position"])
                material_ids.append(data.get("material_id", 0))
                weights.append(data.get("weight", 1.0))  # 默认权重为1.0
                boundary_flags.append(1 if data.get("is_boundary", False) else 0)
            else:
                # 旧格式：只有位置信息
                positions.append(data)
                material_ids.append(0)
                weights.append(1.0)
                boundary_flags.append(0)
        
        # 创建临时numpy数组用于传递粒子属性
        numpy_dtype = np.float32 if self.float_type == ti.f32 else np.float64
        positions_np = np.array(positions, dtype=numpy_dtype)
        material_ids_np = np.array(material_ids, dtype=np.int32)
        weights_np = np.array(weights, dtype=numpy_dtype)
        boundary_flags_np = np.array(boundary_flags, dtype=np.int32)

        temp_positions = ti.Vector.field(self.dim, self.float_type, shape=n_particles)
        temp_material_ids = ti.field(ti.i32, shape=n_particles)
        temp_weights = ti.field(self.float_type, shape=n_particles)
        temp_boundary_flags = ti.field(ti.i32, shape=n_particles)

        temp_positions.from_numpy(positions_np)
        temp_material_ids.from_numpy(material_ids_np)
        temp_weights.from_numpy(weights_np)
        temp_boundary_flags.from_numpy(boundary_flags_np)

        # 准备初始F矩阵数据
        use_custom_F = material_params is not None
        temp_initial_F = None

        if use_custom_F:
            # 为每个粒子准备初始F矩阵
            initial_F_list = []
            for i in range(n_particles):
                material_id = material_ids[i]
                if material_id in material_params:
                    initial_F = material_params[material_id].get("initial_F", [[1.0, 0.0], [0.0, 1.0]])
                    # 确保矩阵大小与维度匹配
                    if self.dim == 2:
                        if len(initial_F) != 2 or len(initial_F[0]) != 2:
                            initial_F = [[1.0, 0.0], [0.0, 1.0]]  # 默认二维单位矩阵
                    elif self.dim == 3:
                        if len(initial_F) != 3 or len(initial_F[0]) != 3:
                            initial_F = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # 默认三维单位矩阵
                else:
                    # 默认单位矩阵
                    if self.dim == 2:
                        initial_F = [[1.0, 0.0], [0.0, 1.0]]
                    else:
                        initial_F = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                initial_F_list.append(initial_F)

            # 转换为numpy数组
            initial_F_np = np.array(initial_F_list, dtype=numpy_dtype)
            temp_initial_F = ti.Matrix.field(self.dim, self.dim, self.float_type, shape=n_particles)
            temp_initial_F.from_numpy(initial_F_np)

        # 初始化粒子属性
        @ti.kernel
        def init_kernel():
            for i in range(n_particles):
                x_field[i] = temp_positions[i]
                v_field[i] = ti.Vector.zero(self.float_type, self.dim)
                if self.dim >= 2:
                    v_field[i][1] = self.init_vel_y

                # 根据材料参数设置初始F矩阵
                if use_custom_F:
                    F_field[i] = temp_initial_F[i]
                else:
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

        @ti.kernel
        def init_boundary_flags():
            for i in range(n_particles):
                boundary_field[i] = temp_boundary_flags[i]

        init_kernel()

        # 设置材料ID（如果提供了material_id_field）
        if material_id_field is not None:
            init_material_ids()

        # 设置权重（如果提供了weight_field）
        if weight_field is not None:
            init_weights()

        # 设置边界标记（如果提供了boundary_field）
        if boundary_field is not None:
            init_boundary_flags()