import taichi as ti

@ti.data_oriented
class BoundaryExchanger:
    """处理域间边界条件交换的类"""
    
    def __init__(self, big_time_domain, small_time_domain):
        self.big_time_domain = big_time_domain
        self.small_time_domain = small_time_domain
        
        # 分配临时数组用于保存网格速度
        self.big_time_domain_temp_grid_v = ti.Vector.field(
            big_time_domain.grid.dim, big_time_domain.float_type, 
            (big_time_domain.grid.size,) * big_time_domain.grid.dim
        )
        self.small_time_domain_temp_grid_v = ti.Vector.field(
            small_time_domain.grid.dim, small_time_domain.float_type, 
            (small_time_domain.grid.size,) * small_time_domain.grid.dim
        )
        self.small_time_domain_boundary_v_last = ti.Vector.field(
            small_time_domain.grid.dim, small_time_domain.float_type, 
            (small_time_domain.grid.size,) * small_time_domain.grid.dim
        )
        self.small_time_domain_boundary_v_next = ti.Vector.field(
            small_time_domain.grid.dim, small_time_domain.float_type, 
            (small_time_domain.grid.size,) * small_time_domain.grid.dim
        )
    
    def save_grid_velocities(self):
        """保存网格速度"""
        self.big_time_domain_temp_grid_v.copy_from(self.big_time_domain.grid.v)
        self.small_time_domain_temp_grid_v.copy_from(self.small_time_domain.grid.v)
    
    def exchange_boundary_conditions(self):
        """设置边界条件"""
        self.project_to_big_time_domain_boundary(
            self.small_time_domain.grid.v, self.big_time_domain.grid.boundary_v
        )
        self.project_to_small_time_domain_boundary(
            self.big_time_domain.grid.v, self.small_time_domain.grid.boundary_v
        )
    
    def save_boundary_velocities(self):
        """保存边界速度状态"""
        self.small_time_domain_boundary_v_last.copy_from(self.small_time_domain.grid.boundary_v)
        self.small_time_domain_boundary_v_next.copy_from(self.small_time_domain.grid.boundary_v)
    
    @ti.kernel
    def linp(self, dest: ti.template(), a: ti.template(), b: ti.template(), ratio: ti.f32):
        """线性插值"""
        for I in ti.grouped(dest):
            dest[I] = a[I] * (1 - ratio) + b[I] * ratio
    
    def interpolate_boundary_velocity(self, ratio: float):
        """在边界速度之间插值"""
        self.linp(self.small_time_domain.grid.boundary_v, 
                 self.small_time_domain_boundary_v_last, 
                 self.small_time_domain_boundary_v_next, ratio)
    
    @ti.kernel
    def project_to_big_time_domain_boundary(self, from_boundary_v: ti.template(), to_boundary_v: ti.template()):
        """将小时间步长域的速度投影到大时间步长域的边界"""
        for I in ti.grouped(to_boundary_v):
            if self.big_time_domain.grid.is_particle_boundary_grid[I]:
                m = 0.0
                is_small_time_domain_boundary = False
                x = ((I * self.big_time_domain.grid.dx) * self.big_time_domain.scale + self.big_time_domain.offset) - self.small_time_domain.offset
                x = x / self.small_time_domain.scale
                base = (x * self.small_time_domain.grid.inv_dx - 0.5).cast(int)
                fx = x * self.small_time_domain.grid.inv_dx - base.cast(float)
                w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
                to_boundary_v[I] = ti.Vector.zero(self.big_time_domain.grid.float_type, self.big_time_domain.grid.dim)

                for offset in ti.static(ti.grouped(ti.ndrange(*self.big_time_domain.particles.neighbor))):
                    weight = 1.0
                    for d in ti.static(range(self.big_time_domain.grid.dim)):
                        weight *= w[offset[d]][d]
                    if self.small_time_domain.grid.is_particle_boundary_grid[base + offset]:
                        is_small_time_domain_boundary = True
                    m += weight * self.small_time_domain.grid.m[base + offset]
                    to_boundary_v[I] += weight * from_boundary_v[base + offset] * self.small_time_domain.grid.m[base + offset]

                big_time_domain_set_boundary = self.big_time_domain.grid.is_particle_boundary_grid[I] and m > 1e-10
                big_time_domain_set_boundary = big_time_domain_set_boundary and (not is_small_time_domain_boundary or self.big_time_domain.grid.m[I] < m)

                if big_time_domain_set_boundary:
                    self.big_time_domain.grid.is_boundary_grid[I] = [1] * self.big_time_domain.grid.dim
                    to_boundary_v[I] = to_boundary_v[I] / m

    @ti.kernel
    def project_to_small_time_domain_boundary(self, from_boundary_v: ti.template(), to_boundary_v: ti.template()):
        """将大时间步长域的速度投影到小时间步长域的边界"""
        for I in ti.grouped(to_boundary_v):
            if self.small_time_domain.grid.is_particle_boundary_grid[I]:
                m = 0.0
                is_big_time_domain_boundary = False
                x = ((I * self.small_time_domain.grid.dx) * self.small_time_domain.scale + self.small_time_domain.offset) - self.big_time_domain.offset
                x = x / self.big_time_domain.scale
                base = (x * self.big_time_domain.grid.inv_dx - 0.5).cast(int)
                fx = x * self.big_time_domain.grid.inv_dx - base.cast(float)
                w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
                to_boundary_v[I] = ti.Vector.zero(self.small_time_domain.grid.float_type, self.small_time_domain.grid.dim)

                for offset in ti.static(ti.grouped(ti.ndrange(*self.small_time_domain.particles.neighbor))):
                    weight = 1.0
                    for d in ti.static(range(self.small_time_domain.grid.dim)):
                        weight *= w[offset[d]][d]
                    if self.big_time_domain.grid.is_particle_boundary_grid[base + offset]:
                        is_big_time_domain_boundary = True
                    m += weight * self.big_time_domain.grid.m[base + offset]
                    to_boundary_v[I] += weight * from_boundary_v[base + offset] * self.big_time_domain.grid.m[base + offset]

                small_time_domain_set_boundary = self.small_time_domain.grid.is_particle_boundary_grid[I] and m > 1e-10
                small_time_domain_set_boundary = small_time_domain_set_boundary and (not is_big_time_domain_boundary or self.small_time_domain.grid.m[I] < m)

                if small_time_domain_set_boundary:
                    self.small_time_domain.grid.is_boundary_grid[I] = [1] * self.small_time_domain.grid.dim
                    to_boundary_v[I] = to_boundary_v[I] / m