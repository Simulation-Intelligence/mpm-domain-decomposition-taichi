"""
粒子生成模块 - 处理各种几何形状的粒子生成
"""
import numpy as np
import math


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
                    "operation": shape_config.get("operation", "add")  # add/subtract
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
    
    def __init__(self, dim=2, use_poisson_sampling=True):
        self.dim = dim
        self.use_poisson_sampling = use_poisson_sampling
        self.last_poisson_radius = None  # 存储最后使用的Poisson采样半径
    
    def generate_particles_for_shapes(self, shapes, particles_per_area):
        """为多个形状生成粒子"""
        all_particles = []
        
        # 生成所有添加操作的粒子
        for i, shape in enumerate(shapes):
            if shape["operation"] == "add":
                particles = self.generate_particles_for_shape(shape, particles_per_area[i])
                all_particles.extend(particles)
        
        # 应用所有挖空操作
        for shape in shapes:
            if shape["operation"] == "subtract":
                all_particles = self._remove_particles_in_shape(all_particles, shape)
        
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
        
        if self.use_poisson_sampling:
            # 使用泊松采样
            from Util.poisson_disk_sampling import poisson_disk_sampling_by_count
            points_np, radius_info = self._poisson_sampling_with_radius(region_size, n_particles)
            
            # 存储半径信息用于边界检测
            if radius_info is not None:
                self.last_poisson_radius = radius_info
            
            for point in points_np:
                pos = []
                for d in range(self.dim):
                    pos.append(point[d] + rect_range[d][0])
                particles.append(pos)
        else:
            # 均匀随机采样
            for _ in range(n_particles):
                pos = []
                for d in range(self.dim):
                    min_val = rect_range[d][0]
                    max_val = rect_range[d][1]
                    pos.append(np.random.uniform(min_val, max_val))
                particles.append(pos)
        
        return particles
    
    def _generate_ellipse_particles(self, params, n_particles):
        """生成椭圆区域内的粒子"""
        center = params["center"]
        semi_axes = params["semi_axes"]
        
        if self.use_poisson_sampling:
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
        else:
            # 传统随机采样方法
            particles = self._traditional_ellipse_sampling(center, semi_axes, n_particles)
        
        return particles
    
    def _remove_particles_in_shape(self, particles, shape):
        """从粒子列表中移除在指定形状内的粒子"""
        remaining_particles = []
        
        for particle_pos in particles:
            if not self._is_particle_in_shape(particle_pos, shape):
                remaining_particles.append(particle_pos)
        
        return remaining_particles
    
    def _poisson_sampling_with_radius(self, region_size, n_particles):
        """执行Poisson采样并返回半径信息"""
        import io
        import sys
        from contextlib import redirect_stdout
        
        # 捕获输出以提取半径信息
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            from Util.poisson_disk_sampling import poisson_disk_sampling_by_count
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
            
            # 计算椭圆方程
            sum_normalized = 0.0
            for d in range(self.dim):
                diff = particle_pos[d] - center[d]
                sum_normalized += (diff / semi_axes[d])**2
            
            return sum_normalized <= 1.0
        
        return False


class ParticleInitializer:
    """粒子属性初始化器"""
    
    def __init__(self, dim=2, float_type=np.float32, init_vel_y=-1):
        self.dim = dim
        self.float_type = float_type
        self.init_vel_y = init_vel_y
    
    def initialize_particle_fields(self, positions, x_field, v_field, F_field, C_field):
        """初始化粒子的各种属性字段"""
        import taichi as ti
        n_particles = len(positions)
        
        # 创建临时numpy数组用于传递粒子位置
        numpy_dtype = np.float32 if self.float_type == ti.f32 else np.float64
        positions_np = np.array(positions, dtype=numpy_dtype)
        temp_positions = ti.Vector.field(self.dim, self.float_type, shape=n_particles)
        temp_positions.from_numpy(positions_np)
        
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
        
        init_kernel()