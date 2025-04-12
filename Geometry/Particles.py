import taichi as ti
from Util.poisson_disk_sampling import poisson_disk_sampling_by_count
import numpy as np


# ------------------ 粒子模块 ------------------
@ti.data_oriented
class Particles:
    def __init__(self, config):
        self.dim = config.get("dim", 2)
        init_pos_range = config.get("initial_position_range", [[[0.3, 0.6], [0.3, 0.6]]])
        self.num_areas = len(init_pos_range)
        self.init_pos_range = ti.Vector.field(2, ti.f32, shape=(self.num_areas, self.dim))

        for i in ti.static(range(self.num_areas)):
            for d in ti.static(range(self.dim)):
                self.init_pos_range[i, d] = ti.Vector(init_pos_range[i][d])
        
        self.areas = ti.field(ti.f32, self.num_areas)
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
        self.pos_possion = ti.Vector.field(self.dim, ti.f32, shape=max_n_per_area)
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
    
    def initialize(self):
        start_num=0
        for i in range(self.num_areas):

            points_np=poisson_disk_sampling_by_count(self.init_pos_range[i,0][1]-self.init_pos_range[i,0][0],self.init_pos_range[i,1][1]-self.init_pos_range[i,1][0], self.n_per_area[i])
            self.pos_possion.from_numpy(np.array(points_np))
            self.initialize_area(i, start_num)
            start_num += self.n_per_area[i]

    @ti.kernel
    def initialize_area(self,i:ti.i32,start_num:ti.i32):
        for p in range(self.n_per_area[i]):
            pos = ti.Vector.zero(self.float_type, self.dim)
            for d in ti.static(range(self.dim)):
                min_val = self.init_pos_range[i, d][0]
                max_val = self.init_pos_range[i, d][1]
                #从泊松盘采样中获取位置
                pos[d]=self.pos_possion[p][d] + min_val
                # 边界判断
                if pos[d] < min_val + self.boundary_size or pos[d] > max_val - self.boundary_size:
                    self.is_boundary_particle[start_num + p] = 1
            self.x[start_num+p] = pos
            self.v[start_num+p] = ti.Vector.zero(self.float_type, self.dim)
            self.v[start_num+p][1] = self.init_vel_y
            self.F[start_num+p] = ti.Matrix.identity(self.float_type, self.dim)
            self.C[start_num+p] = ti.Matrix.zero(self.float_type, self.dim, self.dim)

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