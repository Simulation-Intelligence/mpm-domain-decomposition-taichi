import taichi as ti

# ------------------ 粒子模块 ------------------
@ti.data_oriented
class Particles:
    def __init__(self, grid_size,init_pos_range,init_vel_y):
        self.dim = 2
        self.n_particles = grid_size**self.dim // 2**(self.dim - 1)
        
        self.x = ti.Vector.field(self.dim, ti.f32, self.n_particles)
        self.v = ti.Vector.field(self.dim, ti.f32, self.n_particles)
        self.F = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        self.C = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        self.temp_P = ti.Matrix.field(2, 2, ti.f32, self.n_particles)

        self.wip=ti.field(ti.f32, (self.n_particles, 3,3))
        self.dwip=ti.Vector.field(self.dim, ti.f32, (self.n_particles, 3,3))
        
        self.init_pos_range = init_pos_range
        self.init_vel_y = init_vel_y

    @ti.kernel
    def initialize(self):
        for p in range(self.n_particles):
            pos = ti.Vector.zero(ti.f32, self.dim)
            for d in ti.static(range(self.dim)):
                pos[d] = ti.random() * (self.init_pos_range[1]-self.init_pos_range[0]) + self.init_pos_range[0]
            self.x[p] = pos
            self.v[p] = ti.Vector.zero(ti.f32, self.dim)
            self.v[p][1] = self.init_vel_y
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)

    @ti.kernel
    def advect(self, dt: ti.f32):
        for p in self.x:
            self.x[p] += dt * self.v[p]