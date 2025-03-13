import taichi as ti



# ------------------ 网格模块 ------------------
@ti.data_oriented
class Grid:
    def __init__(self, size, dim, bound):
        self.size = size
        self.dim = dim
        self.bound = bound
        self.dx = 1.0 / size
        self.inv_dx = size
        
        self.v = ti.Vector.field(dim, ti.f32, (size,)*dim)
        self.m = ti.field(ti.f32, (size,)*dim)
        self.v_prev = ti.Vector.field(dim, ti.f32, (size,)*dim)
        
        # 隐式求解相关
        self.particles = ti.field(ti.i32, (size, size, 32))
        self.wip = ti.field(ti.f32, (size, size, 32))
        self.dwip = ti.Vector.field(dim, ti.f32, (size, size, 32))
        self.particle_count = ti.field(ti.i32, (size, size))

    @ti.kernel
    def apply_boundary_conditions(self):
        for I in ti.grouped(self.v):
            cond = (I < self.bound) & (self.v[I] < 0) | (I > self.size - self.bound) & (self.v[I] > 0)
            self.v[I] = ti.select(cond, 0, self.v[I])

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.v):
            self.m[I] = 0.0
            self.v[I] = [0.0, 0.0]