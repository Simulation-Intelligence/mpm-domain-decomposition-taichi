"""
边界检测模块 - Alpha Shape边界检测实现
"""
import numpy as np
import taichi as ti


class BoundaryDetector:
    """基于Alpha Shape的边界检测器"""
    
    def __init__(self, boundary_size=0.01):
        self.boundary_size = boundary_size
        self.suggested_alpha = None  # 存储建议的alpha值
    
    def detect_boundaries(self, positions, dim=2, poisson_radius=None):
        """检测粒子的边界
        
        Args:
            positions: 粒子位置数组
            dim: 维度 (2D 或 3D)
            poisson_radius: Poisson采样半径（如果可用）
            
        Returns:
            boundary_flags: 边界标记数组 (1为边界粒子，0为内部粒子)
        """
        from scipy.spatial import Delaunay
        
        # 计算自适应alpha值，优先使用Poisson半径信息
        alpha = self._calculate_adaptive_alpha(positions, poisson_radius)
        print(f"Using alpha = {alpha:.4f}")
        
        # 计算Delaunay三角剖分
        tri = Delaunay(positions)
        
        # 获取alpha shape的边界边
        boundary_edges = self._extract_alpha_shape_edges(positions, tri, alpha, dim)
        print(f"Alpha shape has {len(boundary_edges)} boundary edges")
        
        # 标记边界粒子
        boundary_flags = self._mark_particles_near_boundary(positions, boundary_edges)
        
        return boundary_flags
    
    def _calculate_adaptive_alpha(self, positions, poisson_radius=None):
        """计算自适应的alpha值，优先使用Poisson采样半径"""
        from scipy.spatial.distance import pdist
        
        # 方法1：如果有Poisson采样半径，直接使用它来估算alpha
        if poisson_radius is not None:
            print(f"Using Poisson radius for alpha estimation: {poisson_radius:.6f}")
            
            # Poisson采样半径通常是两个点之间的最小距离
            # Alpha值应该略大于这个距离以保证连通性
            alpha = poisson_radius * 2.5  # 经验系数，使得相邻点能形成三角形
            
            # 确保alpha在合理范围内
            min_alpha = self.boundary_size * 0.5
            max_alpha = poisson_radius * 5.0
            alpha = max(min_alpha, min(alpha, max_alpha))
            
            self.suggested_alpha = alpha
            return alpha
        
        # 方法2：回退到基于最近邻距离的方法
        print("Falling back to distance-based alpha estimation")
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
    
    def _extract_alpha_shape_edges(self, positions, tri, alpha, dim):
        """从Delaunay三角剖分中提取alpha shape的边界边"""
        def circumradius(triangle_points):
            """计算三角形外接圆半径"""
            if dim == 2:
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
            if dim == 2:
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
    
    def _mark_particles_near_boundary(self, positions, boundary_edges):
        """标记距离alpha shape边界小于boundary_size的粒子"""
        n_particles = len(positions)
        boundary_flags = np.zeros(n_particles, dtype=np.int32)
        
        boundary_count = 0
        
        # 对每个粒子计算到边界的最小距离
        for i in range(n_particles):
            particle_pos = positions[i]
            min_dist_to_boundary = float('inf')
            
            # 计算到所有边界边的最小距离
            for edge in boundary_edges:
                p1, p2 = positions[edge[0]], positions[edge[1]]
                dist = self._point_to_line_segment_distance(particle_pos, p1, p2)
                min_dist_to_boundary = min(min_dist_to_boundary, dist)
            
            # 如果距离小于threshold，标记为边界粒子
            if min_dist_to_boundary <= self.boundary_size:
                boundary_flags[i] = 1
                boundary_count += 1
        
        print(f"Marked {boundary_count} particles as boundary particles")
        return boundary_flags
    
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


class NeighborDensityBoundaryDetector:
    """基于邻居密度的边界检测器（备选方法）"""
    
    def __init__(self, boundary_size=0.01):
        self.boundary_size = boundary_size
    
    def detect_boundaries(self, positions):
        """基于邻居密度检测边界"""
        n_particles = len(positions)
        neighbor_counts = np.zeros(n_particles, dtype=np.int32)
        
        # 统计邻居数量
        self._count_neighbors(positions, neighbor_counts)
        
        # 计算阈值
        counts_mean = np.mean(neighbor_counts)
        counts_max = np.max(neighbor_counts)
        threshold = max(1, int(counts_max * 3 / 5))
        
        print(f"Neighbor density: mean={counts_mean:.2f}, max={counts_max}, threshold={threshold}")
        
        # 标记边界粒子
        boundary_flags = (neighbor_counts < threshold).astype(np.int32)
        
        return boundary_flags
    
    def _count_neighbors(self, positions, neighbor_counts):
        """统计每个粒子的邻居数量"""
        n_particles = len(positions)
        search_radius = self.boundary_size
        
        for p in range(n_particles):
            count = 0
            for q in range(n_particles):
                if p != q:
                    dist_sq = np.sum((positions[p] - positions[q])**2)
                    if dist_sq < search_radius * search_radius:
                        count += 1
            neighbor_counts[p] = count