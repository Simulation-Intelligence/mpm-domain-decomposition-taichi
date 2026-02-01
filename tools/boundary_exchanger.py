import taichi as ti
from simulators.implicit_mpm import ImplicitMPM

@ti.data_oriented
class BoundaryExchanger:
    """处理域间边界条件交换的类"""

    def __init__(self, big_time_domain: ImplicitMPM, small_time_domain: ImplicitMPM):
        self.big_time_domain = big_time_domain
        self.small_time_domain = small_time_domain
        
        # 分配临时数组用于保存网格速度，支持矩形网格
        big_grid_shape = self._get_grid_shape(big_time_domain)
        small_grid_shape = self._get_grid_shape(small_time_domain)

        self.big_time_domain_temp_grid_v = ti.Vector.field(
            big_time_domain.grid.dim, big_time_domain.float_type,
            big_grid_shape
        )
        self.small_time_domain_temp_grid_v = ti.Vector.field(
            small_time_domain.grid.dim, small_time_domain.float_type,
            small_grid_shape
        )
        self.small_time_domain_boundary_v_last = ti.Vector.field(
            small_time_domain.grid.dim, small_time_domain.float_type,
            small_grid_shape
        )
        self.small_time_domain_boundary_v_next = ti.Vector.field(
            small_time_domain.grid.dim, small_time_domain.float_type,
            small_grid_shape
        )

    def _get_grid_shape(self, domain):
        """获取域的网格形状"""
        if domain.grid.dim == 2:
            return (domain.grid.nx, domain.grid.ny)
        elif domain.grid.dim == 3:
            return (domain.grid.nx, domain.grid.ny, domain.grid.nz)
        else:
            # 兼容旧格式
            return (domain.grid.size,) * domain.grid.dim

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
    def linp(self, dest: ti.template(), a: ti.template(), b: ti.template(), ratio: ti.f64):
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
                # 计算物理坐标，去掉scale
                x = self.big_time_domain.grid.get_grid_pos(I) + self.big_time_domain.offset - self.small_time_domain.offset

                # 使用已有的函数计算网格索引
                base, fx = self.small_time_domain.grid.particle_to_grid_base_and_fx(x)
                w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]

                # 保存原始的boundary_v值（可能包含move boundary设置的值）
                original_boundary_v = to_boundary_v[I]
                to_boundary_v[I] = ti.Vector.zero(self.big_time_domain.grid.float_type, self.big_time_domain.grid.dim)

                for offset in ti.static(ti.grouped(ti.ndrange(*self.small_time_domain.particles.neighbor))):
                    grid_idx = base + offset
                    # 检查边界，忽略越界情况
                    in_bounds = True
                    if self.small_time_domain.grid.dim == 2:
                        if grid_idx[0] < 0 or grid_idx[0] >= self.small_time_domain.grid.nx or \
                           grid_idx[1] < 0 or grid_idx[1] >= self.small_time_domain.grid.ny:
                            in_bounds = False
                    else:
                        if grid_idx[0] < 0 or grid_idx[0] >= self.small_time_domain.grid.nx or \
                           grid_idx[1] < 0 or grid_idx[1] >= self.small_time_domain.grid.ny or \
                           grid_idx[2] < 0 or grid_idx[2] >= self.small_time_domain.grid.nz:
                            in_bounds = False

                    if in_bounds:
                        weight = 1.0
                        for d in ti.static(range(self.small_time_domain.grid.dim)):
                            weight *= w[offset[d]][d]
                        if self.small_time_domain.grid.is_particle_boundary_grid[grid_idx]:
                            is_small_time_domain_boundary = True
                        m += weight * self.small_time_domain.grid.m[grid_idx]
                        to_boundary_v[I] += weight * from_boundary_v[grid_idx] * self.small_time_domain.grid.m[grid_idx]

                big_time_domain_set_boundary = self.big_time_domain.grid.is_particle_boundary_grid[I] and m > 1e-10
                big_time_domain_set_boundary = big_time_domain_set_boundary and (not is_small_time_domain_boundary or self.big_time_domain.grid.m[I] < m)

                if big_time_domain_set_boundary:
                    self.big_time_domain.grid.is_boundary_grid[I] = [1] * self.big_time_domain.grid.dim
                    to_boundary_v[I] = to_boundary_v[I] / m
                else:
                    # 如果没有边界交换覆盖，恢复原始的boundary_v（保留move boundary设置）
                    to_boundary_v[I] = original_boundary_v

    @ti.kernel
    def project_to_small_time_domain_boundary(self, from_boundary_v: ti.template(), to_boundary_v: ti.template()):
        """将大时间步长域的速度投影到小时间步长域的边界"""
        for I in ti.grouped(to_boundary_v):
            if self.small_time_domain.grid.is_particle_boundary_grid[I]:
                m = 0.0
                is_big_time_domain_boundary = False
                # 计算物理坐标，去掉scale
                x = self.small_time_domain.grid.get_grid_pos(I) + self.small_time_domain.offset - self.big_time_domain.offset

                # 使用已有的函数计算网格索引
                base, fx = self.big_time_domain.grid.particle_to_grid_base_and_fx(x)
                w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]

                # 保存原始的boundary_v值（可能包含move boundary设置的值）
                original_boundary_v = to_boundary_v[I]
                to_boundary_v[I] = ti.Vector.zero(self.small_time_domain.grid.float_type, self.small_time_domain.grid.dim)

                for offset in ti.static(ti.grouped(ti.ndrange(*self.big_time_domain.particles.neighbor))):
                    grid_idx = base + offset
                    # 检查边界，忽略越界情况
                    in_bounds = True
                    if self.big_time_domain.grid.dim == 2:
                        if grid_idx[0] < 0 or grid_idx[0] >= self.big_time_domain.grid.nx or \
                           grid_idx[1] < 0 or grid_idx[1] >= self.big_time_domain.grid.ny:
                            in_bounds = False
                    else:
                        if grid_idx[0] < 0 or grid_idx[0] >= self.big_time_domain.grid.nx or \
                           grid_idx[1] < 0 or grid_idx[1] >= self.big_time_domain.grid.ny or \
                           grid_idx[2] < 0 or grid_idx[2] >= self.big_time_domain.grid.nz:
                            in_bounds = False

                    if in_bounds:
                        weight = 1.0
                        for d in ti.static(range(self.big_time_domain.grid.dim)):
                            weight *= w[offset[d]][d]
                        if self.big_time_domain.grid.is_particle_boundary_grid[grid_idx]:
                            is_big_time_domain_boundary = True
                        m += weight * self.big_time_domain.grid.m[grid_idx]
                        to_boundary_v[I] += weight * from_boundary_v[grid_idx] * self.big_time_domain.grid.m[grid_idx]

                small_time_domain_set_boundary = self.small_time_domain.grid.is_particle_boundary_grid[I] and m > 1e-10
                small_time_domain_set_boundary = small_time_domain_set_boundary and (not is_big_time_domain_boundary or self.small_time_domain.grid.m[I] < m)

                if small_time_domain_set_boundary:
                    self.small_time_domain.grid.is_boundary_grid[I] = [1] * self.small_time_domain.grid.dim
                    to_boundary_v[I] = to_boundary_v[I] / m
                else:
                    # 如果没有边界交换覆盖，恢复原始的boundary_v（保留move boundary设置）
                    to_boundary_v[I] = original_boundary_v