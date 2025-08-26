"""
形状工具模块 - 提供几何形状相关的工具函数
"""
import numpy as np
import taichi as ti


@ti.data_oriented
class ManualBoundaryDetector:
    """手动边界检测器 - 基于矩形区域"""
    
    def __init__(self, boundary_range, boundary_size=0.01, dim=2):
        self.boundary_range = boundary_range
        self.boundary_size = boundary_size
        self.dim = dim
    
    @ti.kernel
    def detect_boundaries(self, x_field: ti.template(), boundary_field: ti.template(), n_particles: ti.i32):
        """手动指定边界：基于矩形区域"""
        for p in range(n_particles):
            for d in ti.static(range(self.dim)):
                min_val = self.boundary_range[d][0]
                max_val = self.boundary_range[d][1]
                if x_field[p][d] < min_val + self.boundary_size or x_field[p][d] > max_val - self.boundary_size:
                    boundary_field[p] = 1


@ti.data_oriented
class ParticleNeighborBuilder:
    """粒子邻居列表构建器"""
    
    def __init__(self, grid_size, dim=2):
        self.grid_size = grid_size
        self.dim = dim
        self.neighbor = (3,) * dim
    
    @ti.kernel
    def build_neighbor_list(self, x_field: ti.template(), wip_field: ti.template(), 
                           dwip_field: ti.template(), n_particles: ti.i32):
        """构建粒子的邻居列表和权重"""
        for p in range(n_particles):
            base = (x_field[p] * self.grid_size - 0.5).cast(int)
            fx = x_field[p] * self.grid_size - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                dpos = (offset - fx) * self.grid_size
                wip_field[p, offset] = weight
                dwip_field[p, offset] = weight * dpos


@ti.data_oriented
class ParticleAdvector:
    """粒子运动积分器"""
    
    @ti.kernel 
    def advect(self, x_field: ti.template(), v_field: ti.template(), dt: ti.f32):
        """粒子位置积分"""
        for p in x_field:
            x_field[p] += dt * v_field[p]

@ti.data_oriented
class ParticleMerger:
    """粒子合并工具 - 用于合并公共粒子"""
    
    @ti.kernel
    def merge_common_particles(self, target_x: ti.template(), target_v: ti.template(),
                              target_F: ti.template(), target_C: ti.template(),
                              target_boundary: ti.template(),
                              source_x: ti.template(), source_v: ti.template(),
                              source_F: ti.template(), source_C: ti.template(),
                              source_boundary: ti.template(),
                              start_num: ti.i32, n_common: ti.i32):
        """合并公共粒子到目标粒子系统"""
        for p in range(n_common):
            idx = start_num + p
            target_x[idx] = source_x[p]
            target_v[idx] = source_v[p]
            target_F[idx] = source_F[p]
            target_C[idx] = source_C[p]
            target_boundary[idx] = 1 if source_boundary[p] else 0