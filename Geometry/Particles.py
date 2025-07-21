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
        self.boundary_size= config.get("boundary_size", 0.05)

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
            self.set_boundary()
    
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
    def set_boundary(self):
        for p in range(self.n_particles):
            for d in ti.static(range(self.dim)):
                min_val = self.boundary_range[d][0]
                max_val = self.boundary_range[d][1]
                if self.x[p][d] < min_val + self.boundary_size or self.x[p][d] > max_val - self.boundary_size:
                    self.is_boundary_particle[p] = 1

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