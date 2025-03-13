import taichi as ti
import numpy as np

@ti.data_oriented
class MPM3D:
    def __init__(self, config):
        ti.init(arch=ti.vulkan, debug=config.get('debug', False))
        
        # 基础配置
        self.dim = config.get('dim', 3)
        self.n_grid = config.get('n_grid', 64)
        self.particles_per_grid = config.get('particles_per_grid', 4)
        self.dx = 1 / self.n_grid
        self.dt = config.get('dt', 1e-4)
        self.max_steps = config.get('max_steps', 25)
        self.p_rho = config.get('density', 1)
        self.p_vol = (self.dx)**self.dim
        self.p_mass = self.p_vol * self.p_rho
        
        #set gravity based on dims
        if self.dim == 2:
            self.gravity = config.get('gravity',ti.Vector([0, -9.8]))
        else:
            self.gravity = config.get('gravity', ti.Vector([0, -9.8, 0]))
        self.bound = 3
        self.E = config.get('youngs_modulus', 40)
        self.nu = config.get('poisson_ratio', 0.4)
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.neighbor = (3,) * self.dim
        
        # 初始化场
        grid_shape = tuple([self.n_grid]*self.dim)
        self.grid_momt = ti.Vector.field(self.dim, dtype=ti.f32, shape=grid_shape)
        self.grid_m = ti.field(dtype=ti.f32, shape=grid_shape)
        
        # 粒子属性
        self.n_particles = self.n_grid**self.dim // 2**(self.dim - 1)
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        
        self.init_particles()

    @ti.kernel
    def init_particles(self):
        for i in range(self.n_particles):
            pos = ti.Vector.zero(ti.f32, self.dim)
            for d in ti.static(range(self.dim)):
                pos[d] = ti.random() * 0.5 + 0.25
            self.x[i] = pos
            self.v[i]= ti.Vector.zero(ti.f32, self.dim)
            self.v[i][1] = -1
            self.F[i] = ti.Matrix.identity(ti.f32, self.dim)
            self.C[i] = ti.Matrix.zero(ti.f32, self.dim, self.dim)

    @ti.kernel
    def reset_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_momt[I] = [0] * self.dim
            self.grid_m[I] = 0

    @ti.kernel
    def p2g(self):
        for p in self.x:
            Xp = self.x[p] / self.dx
            base = (Xp - 0.5).cast(int)
            fx = Xp - base.cast(float)
            
            # 三次B样条权重
            w =[0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            
            U , sig, V = ti.svd(self.F[p])
            J=1.0
            for d in ti.static(range(self.dim)):
                J *= sig[d, d]
            stress = 2 * self.mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose() + \
                             ti.Matrix.identity(float, self.dim) * self.la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4) * stress / self.dx**2
            affine = stress + self.p_mass * self.C[p]
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.grid_momt[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def update_grid_v(self):
        for I in ti.grouped(self.grid_m):
            # if self.grid_m[I] > 0:
            #     self.grid_momt[I] /= self.grid_m[I]
            
            #self.grid_momt[I] += self.dt * self.gravity * self.grid_m[I]
            
            # 边界处理
            cond = (I < self.bound) & (self.grid_momt[I] < 0) | (I > self.n_grid - self.bound) & (self.grid_momt[I] > 0)
            self.grid_momt[I] = ti.select(cond, 0, self.grid_momt[I])

    @ti.kernel
    def g2p(self):
        for p in self.x:
            Xp = self.x[p] / self.dx
            base = (Xp - 0.5).cast(int)
            fx = Xp - base.cast(float)
            
            w =[0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            new_v = ti.zero(self.v[p])
            new_C = ti.zero(self.C[p])
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                g_v = self.grid_momt[base + offset] / (self.grid_m[base + offset] + 1e-10)
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx**2
                
            self.v[p] = new_v
            self.x[p] += self.dt * self.v[p]
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.dt * self.C[p]) @ self.F[p]
            self.C[p] = new_C

    def step(self):
        self.reset_grid()
        self.p2g()
        self.update_grid_v()
        self.g2p()

    def project(self, particles_np):
        phi, theta = np.radians(28), np.radians(32)
        particles_np = particles_np - 0.5
        x, y, z = particles_np[:,0], particles_np[:,1], particles_np[:,2]
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        x, z = x * cp + z * sp, z * cp - x * sp
        u, v = x, y * ct + z * st
        projected = np.stack([u, v], axis=1) + 0.5
        return projected

    def render(self):
        gui = ti.GUI("MPM3D", (800, 800))
        
        while gui.running and not gui.get_event(gui.ESCAPE):
            for _ in range(self.max_steps):
                self.step()
            
            projected = None
            # 坐标变换投影
            if self.dim ==3:
                particles_np = self.x.to_numpy()
                projected = self.project(particles_np)
            else:
                projected = self.x.to_numpy()
            
            gui.circles(projected, radius=1.5, color=0x66CCFF)
            gui.show()

# 使用示例
if __name__ == "__main__":
    config = {
        'debug': False,
        'dim': 2,
        'n_grid': 16,
        'dt': 1e-4,
        'max_steps': 50,
        'youngs_modulus':4,
        'density': 1
    }
    
    mpm = MPM3D(config)
    mpm.render()
