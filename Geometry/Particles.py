import taichi as ti
import json



# ------------------ 粒子模块 ------------------
@ti.data_oriented
class Particles:
    def __init__(self, config, grid_size):
        self.dim = 2
        self.init_pos_range = config.get("initial_position_range", [[0.3, 0.6], [0.3, 0.6]])
        self.area = (self.init_pos_range[0][1]-self.init_pos_range[0][0]) * (self.init_pos_range[1][1]-self.init_pos_range[1][0])
        self.particles_per_grid = config.get("particles_per_grid", 8)
        self.n_particles = (int)(grid_size**self.dim * self.area * self.particles_per_grid)
        self.p_rho = config.get("p_rho", 1)
        self.p_vol = (1.0/grid_size)**self.dim / self.particles_per_grid
        self.p_mass = self.p_vol * self.p_rho
        self.boundary_size= config.get("boundary_size", 0.05)
        
        self.x = ti.Vector.field(self.dim, ti.f32, self.n_particles)
        self.v = ti.Vector.field(self.dim, ti.f32, self.n_particles)
        self.F = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        self.C = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        self.temp_P = ti.Matrix.field(2, 2, ti.f32, self.n_particles)

        self.wip=ti.field(ti.f32, (self.n_particles, 3,3))
        self.dwip=ti.Vector.field(self.dim, ti.f32, (self.n_particles, 3,3))
        
        
        self.init_vel_y = config.get("initial_velocity_y", -1)

        self.is_boundary_particle = ti.field(ti.i32, self.n_particles)

    @ti.kernel
    def initialize(self):
        for p in range(self.n_particles):
            pos = ti.Vector.zero(ti.f32, self.dim)
            for d in ti.static(range(self.dim)):
                pos[d] = ti.random() * (self.init_pos_range[d][1]-self.init_pos_range[d][0]) + self.init_pos_range[d][0]
                # test if the particle is in the boundary
                # if d==1:
                if pos[d] < self.init_pos_range[d][0]+self.boundary_size or pos[d] > self.init_pos_range[d][1] - self.boundary_size:
                    self.is_boundary_particle[p] = 1
            self.x[p] = pos
            self.v[p] = ti.Vector.zero(ti.f32, self.dim)
            self.v[p][1] = self.init_vel_y
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)

    @ti.kernel
    def advect(self, dt: ti.f32):
        for p in self.x:
            self.x[p] += dt * self.v[p]