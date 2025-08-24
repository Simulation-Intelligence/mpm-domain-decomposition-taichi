import taichi as ti
from Util.poisson_disk_sampling import poisson_disk_sampling_by_count
import numpy as np


# ------------------ 粒子模块 ------------------
@ti.data_oriented
class Particles:
    def __init__(self, config,common_particles:'Particles'=None):
        self.dim = config.get("dim", 2)
        self.float_type = ti.f32 if config.get("float_type", "f32") == "f32" else ti.f64
        default_init_pos_range = [[0.3, 0.6], [0.3, 0.6]] if self.dim == 2 else [[0.3, 0.6], [0.3, 0.6], [0.3, 0.6]]
        init_pos_range = config.get("initial_position_range", [default_init_pos_range])
        boundary_range = config.get("boundary_range", None)
        self.num_areas = len(init_pos_range)
        self.init_pos_range = ti.Vector.field(2, self.float_type, shape=(self.num_areas, self.dim))
        self.boundary_range = ti.Vector.field(2, self.float_type, shape=(self.dim))
        self.neighbor = (3,) * self.dim

        for i in ti.static(range(self.num_areas)):
            for d in ti.static(range(self.dim)):
                self.init_pos_range[i, d] = ti.Vector(init_pos_range[i][d])

        if boundary_range is not None:
            for d in ti.static(range(self.dim)):
                self.boundary_range[d] = ti.Vector(boundary_range[d])

        self.areas = ti.field(self.float_type, self.num_areas)
        for i in ti.static(range(self.num_areas)):
            area = 1.0
            for d in ti.static(range(self.dim)):
                min_val = self.init_pos_range[i, d][0]
                max_val = self.init_pos_range[i, d][1]
                area *= (max_val - min_val)
            self.areas[i] = area

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
        # self.boundary_size= 1.0 / self.grid_size /2
        self.boundary_size = 0.01

        float_type = self.float_type if config.get("float_type", "f32") == "f32" else ti.f64

        self.float_type = float_type
        
        self.x = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.v = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, self.float_type, self.n_particles)

        shape = (self.n_particles, 3,3) if self.dim == 2 else (self.n_particles, 3,3,3)
        self.wip=ti.field(self.float_type, shape)
        self.dwip=ti.Vector.field(self.dim, self.float_type, shape)
        
        
        self.init_vel_y = config.get("initial_velocity_y", -1)

        self.is_boundary_particle = ti.field(ti.i32, self.n_particles)

        self.initialize()

        if boundary_range is not None:
            self.set_boundary(method="manual")
        else:
            self.set_boundary(method="automatic")
    
    def initialize(self):
        start_num=0
        for i in range(self.num_areas):
            region_size=[]
            for d in ti.static(range(self.dim)):
                region_size.append(self.init_pos_range[i, d][1] - self.init_pos_range[i, d][0])
            if self.use_possion_sampling:
                points_np=poisson_disk_sampling_by_count(region_size, self.n_per_area[i])
                self.pos_possion.from_numpy(np.array(points_np))
            self.initialize_area(i, start_num)
            start_num += self.n_per_area[i]
        
        if self.common_particles is not None:
            self.merge_common_particles(start_num)
            start_num += self.common_particles.n_particles
        


    @ti.kernel
    def initialize_area(self,i:ti.i32,start_num:ti.i32):

        for p in range(self.n_per_area[i]):
            pos = ti.Vector.zero(self.float_type, self.dim)
            for d in ti.static(range(self.dim)):
                min_val = self.init_pos_range[i, d][0]
                max_val = self.init_pos_range[i, d][1]
                if self.use_possion_sampling:
                #从泊松盘采样中获取位置
                    pos[d]=self.pos_possion[p][d] + min_val
                else:
                #均匀采样
                    pos[d] = ti.random(self.float_type) * (max_val -min_val) + min_val
                # 边界判断
                # if pos[d] < min_val + self.boundary_size or pos[d] > max_val - self.boundary_size:
                #     self.is_boundary_particle[start_num + p] = 1
            self.x[start_num+p] = pos
            self.v[start_num+p] = ti.Vector.zero(self.float_type, self.dim)
            self.v[start_num+p][1] = self.init_vel_y
            self.F[start_num+p] = ti.Matrix.identity(self.float_type, self.dim)
            self.C[start_num+p] = ti.Matrix.zero(self.float_type, self.dim, self.dim)

    @ti.kernel
    def merge_common_particles(self,start_num:ti.i32):
        for p in range(self.common_particles.n_particles):
            self.x[start_num + p] = self.common_particles.x[p]
            self.v[start_num + p] = self.common_particles.v[p]
            self.F[start_num + p] = self.common_particles.F[p]
            self.C[start_num + p] = self.common_particles.C[p]
            self.is_boundary_particle[start_num + p] = 1 if self.common_particles.is_boundary_particle[p] else 0

    @ti.kernel
    def count_neighbors(self, neighbor_counts: ti.template()):
        """统计每个粒子的邻居数量"""
        search_radius = self.boundary_size
        
        for p in range(self.n_particles):
            count = 0
            for q in range(self.n_particles):
                if p != q:
                    dist_sq = 0.0
                    for d in ti.static(range(self.dim)):
                        diff = self.x[p][d] - self.x[q][d]
                        dist_sq += diff * diff
                    
                    if dist_sq < search_radius * search_radius:
                        count += 1
            neighbor_counts[p] = count

    def set_boundary_automatic(self):
        """自动检测边界粒子：基于Alpha Shape"""
        self.set_boundary_alpha_shape()

    def set_boundary_alpha_shape(self):
        """基于Alpha Shape的边界检测"""
        from scipy.spatial import Delaunay
        
        # 获取所有粒子位置
        positions = self.x.to_numpy()
        
        # 计算自适应alpha值
        alpha = self._calculate_adaptive_alpha(positions)
        print(f"Using alpha = {alpha:.4f}")
        
        # 计算Delaunay三角剖分
        tri = Delaunay(positions)
        
        # 获取alpha shape的边界边
        boundary_edges = self._extract_alpha_shape_edges(positions, tri, alpha)
        print(f"Alpha shape has {len(boundary_edges)} boundary edges")
        
        # 计算每个粒子到边界的距离并标记边界粒子
        self._mark_particles_near_alpha_boundary(positions, boundary_edges)

    def _calculate_adaptive_alpha(self, positions):
        """计算自适应的alpha值"""
        from scipy.spatial.distance import pdist
        
        # 方法1：基于最近邻距离
        n_sample = min(1000, len(positions))  # 采样以提高速度
        sample_indices = np.random.choice(len(positions), n_sample, replace=False)
        sample_positions = positions[sample_indices]
        
        # 计算采样点之间的距离
        distances = pdist(sample_positions)
        distances = distances[distances > 1e-10]  # 排除零距离
        
        if len(distances) == 0:
            return self.boundary_size * 2
        
        # 使用距离的中位数作为基准
        median_dist = np.median(distances)
        alpha = median_dist * 1.5  # 经验系数
        
        return max(alpha, self.boundary_size)

    def _extract_alpha_shape_edges(self, positions, tri, alpha):
        """从Delaunay三角剖分中提取alpha shape的边界边"""
        def circumradius(triangle_points):
            """计算三角形外接圆半径"""
            if self.dim == 2:
                a, b, c = triangle_points
                # 2D三角形外接圆半径公式
                ab = np.linalg.norm(b - a)
                bc = np.linalg.norm(c - b)
                ca = np.linalg.norm(a - c)
                
                # 海伦公式计算面积
                s = (ab + bc + ca) / 2
                area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
                
                if area < 1e-10:
                    return float('inf')
                
                return (ab * bc * ca) / (4 * area)
            else:
                # 3D情况暂时简化处理
                return np.linalg.norm(triangle_points[1] - triangle_points[0])
        
        # 筛选符合alpha条件的三角形
        valid_triangles = []
        for simplex in tri.simplices:
            triangle_points = positions[simplex]
            if circumradius(triangle_points) <= alpha:
                valid_triangles.append(simplex)
        
        # 统计每条边被多少个三角形共享
        edge_count = {}
        for triangle in valid_triangles:
            if self.dim == 2:
                # 2D情况：三角形有3条边
                edges = [(triangle[0], triangle[1]), 
                        (triangle[1], triangle[2]), 
                        (triangle[2], triangle[0])]
            else:
                # 3D情况：四面体的边
                edges = [(triangle[i], triangle[j]) 
                        for i in range(len(triangle)) 
                        for j in range(i+1, len(triangle))]
            
            for edge in edges:
                # 标准化边的表示（小索引在前）
                edge = tuple(sorted(edge))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # 边界边是只被一个三角形共享的边
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        return boundary_edges

    def _mark_particles_near_alpha_boundary(self, positions, boundary_edges):
        """标记距离alpha shape边界小于boundary_size的粒子"""
        # 先全部设为非边界
        for p in range(self.n_particles):
            self.is_boundary_particle[p] = 0
        
        boundary_count = 0
        
        # 对每个粒子计算到边界的最小距离
        for i in range(self.n_particles):
            particle_pos = positions[i]
            min_dist_to_boundary = float('inf')
            
            # 计算到所有边界边的最小距离
            for edge in boundary_edges:
                p1, p2 = positions[edge[0]], positions[edge[1]]
                dist = self._point_to_line_segment_distance(particle_pos, p1, p2)
                min_dist_to_boundary = min(min_dist_to_boundary, dist)
            
            # 如果距离小于threshold，标记为边界粒子
            if min_dist_to_boundary <= self.boundary_size:
                self.is_boundary_particle[i] = 1
                boundary_count += 1
        
        print(f"Marked {boundary_count} particles as boundary particles")

    def _point_to_line_segment_distance(self, point, line_start, line_end):
        """计算点到线段的最短距离"""
        # 向量计算
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-10:  # 线段长度为0
            return np.linalg.norm(point_vec)
        
        # 计算投影参数t
        t = np.dot(point_vec, line_vec) / line_len_sq
        t = max(0, min(1, t))  # 限制在[0,1]范围内
        
        # 计算最近点
        closest_point = line_start + t * line_vec
        return np.linalg.norm(point - closest_point)

    def set_boundary_neighbor_density(self):
        """基于邻居数量统计的边界检测（备选方法）"""
        # 创建临时数组存储每个粒子的邻居数量
        neighbor_counts = ti.field(ti.i32, shape=self.n_particles)
        
        # 统计所有粒子的邻居数量
        self.count_neighbors(neighbor_counts)
        
        # 计算邻居数量的统计特征
        counts_np = neighbor_counts.to_numpy()
        mean_neighbors = np.mean(counts_np)
        max_neighbors = np.max(counts_np)
        std_neighbors = np.std(counts_np)
        
        # 使用最大值的比例作为阈值
        threshold = max(1, int(max_neighbors*3/5))
        print(f"Neighbor density: mean={mean_neighbors:.2f}, std={std_neighbors:.2f}, threshold={threshold}")
        
        # 标记边界粒子
        self.mark_boundary_particles(neighbor_counts, threshold)

    @ti.kernel  
    def mark_boundary_particles(self, neighbor_counts: ti.template(), threshold: ti.i32):
        """根据邻居数量阈值标记边界粒子"""
        for p in range(self.n_particles):
            if neighbor_counts[p] < threshold:
                self.is_boundary_particle[p] = 1
            else:
                self.is_boundary_particle[p] = 0

    @ti.kernel
    def set_boundary_manual(self):
        """手动指定边界：基于矩形区域"""
        for p in range(self.n_particles):
            for d in ti.static(range(self.dim)):
                min_val = self.boundary_range[d][0]
                max_val = self.boundary_range[d][1]
                if self.x[p][d] < min_val + self.boundary_size or self.x[p][d] > max_val - self.boundary_size:
                    self.is_boundary_particle[p] = 1

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

    @ti.kernel
    def build_neighbor_list(self):
        for p in range(self.n_particles):
            base = (self.x[p] * self.grid_size - 0.5).cast(int)
            fx = self.x[p] * self.grid_size - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                dpos = (offset - fx) * self.grid_size
                self.wip[p, offset] = weight
                self.dwip[p, offset] = weight * dpos


    @ti.kernel
    def advect(self, dt: ti.f32):
        for p in self.x:
            self.x[p] += dt * self.v[p]