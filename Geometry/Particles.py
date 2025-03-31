import taichi as ti



# ------------------ 粒子模块 ------------------
@ti.data_oriented
class Particles:
    def __init__(self, config):
        self.dim = config.get("dim", 2)
        self.init_pos_range = config.get("initial_position_range", [[0.3, 0.6], [0.3, 0.6]])
        self.area = (self.init_pos_range[0][1]-self.init_pos_range[0][0]) * (self.init_pos_range[1][1]-self.init_pos_range[1][0])
        self.particles_per_grid = config.get("particles_per_grid", 8)
        self.grid_size = config.get("grid_size", 16)
        self.n_particles = (int)(self.grid_size**self.dim * self.area * self.particles_per_grid)
        self.p_rho = config.get("p_rho", 1)
        self.p_vol = (1.0/self.grid_size)**self.dim / self.particles_per_grid
        self.p_mass = self.p_vol * self.p_rho
        self.boundary_size= config.get("boundary_size", 0.05)

        float_type = ti.f32 if config.get("float_type", "f32") == "f32" else ti.f64

        self.float_type = float_type
        
        self.x = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.v = ti.Vector.field(self.dim, self.float_type, self.n_particles)
        self.F = ti.Matrix.field(2, 2, self.float_type, self.n_particles)
        self.C = ti.Matrix.field(2, 2, self.float_type, self.n_particles)
        self.temp_P = ti.Matrix.field(2, 2, self.float_type, self.n_particles)

        self.wip=ti.field(self.float_type, (self.n_particles, 3,3))
        self.dwip=ti.Vector.field(self.dim, self.float_type, (self.n_particles, 3,3))
        
        
        self.init_vel_y = config.get("initial_velocity_y", -1)

        self.is_boundary_particle = ti.field(ti.i32, self.n_particles)

    @ti.kernel
    def initialize(self):
        for p in range(self.n_particles):
            pos = ti.Vector.zero(self.float_type, self.dim)
            for d in ti.static(range(self.dim)):
                pos[d] = ti.random() * (self.init_pos_range[d][1]-self.init_pos_range[d][0]) + self.init_pos_range[d][0]
                # test if the particle is in the boundary
                # if d==1:
                if pos[d] < self.init_pos_range[d][0]+self.boundary_size or pos[d] > self.init_pos_range[d][1] - self.boundary_size:
                    self.is_boundary_particle[p] = 1
            self.x[p] = pos
            self.v[p] = ti.Vector.zero(self.float_type, self.dim)
            self.v[p][1] = self.init_vel_y
            self.F[p] = ti.Matrix.identity(self.float_type, self.dim)
            self.C[p] = ti.Matrix.zero(self.float_type, self.dim, self.dim)

    @ti.kernel
    def build_neighbor_list(self):
        for p in range(self.n_particles):
            base = (self.x[p] * self.grid_size - 0.5).cast(int)
            fx = self.x[p] * self.grid_size - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
            
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
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