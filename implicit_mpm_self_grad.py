import taichi as ti
import json
from Optimizer.BFGS import BFGS
from Optimizer.LBFGS import LBFGS

ti.init(arch=ti.vulkan, random_seed=114514,log_level=ti.ERROR)

# ------------------ 配置模块 ------------------
class Config:
    def __init__(self, path):
        with open(path) as f:
            self.data = json.load(f)
    
    def get(self, key, default=None):
        return self.data.get(key, default)

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

# ------------------ 粒子模块 ------------------
@ti.data_oriented
class Particles:
    def __init__(self, config, grid_size):
        self.dim = 2
        self.n_particles = grid_size**self.dim // 2**(self.dim - 1)
        self.p_rho = config.get("p_rho", 1)
        self.p_vol = (1.0/grid_size)**self.dim
        self.p_mass = self.p_vol * self.p_rho
        
        self.x = ti.Vector.field(self.dim, ti.f32, self.n_particles)
        self.v = ti.Vector.field(self.dim, ti.f32, self.n_particles)
        self.F = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        self.C = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        self.temp_P = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        
        self.init_pos_range = config.get("initial_position_range", [0.3, 0.6])
        self.init_vel_y = config.get("initial_velocity_y", -1)

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

# ------------------ 隐式求解器模块 ------------------
@ti.data_oriented
class ImplicitSolver:
    def __init__(self, grid, particles:Particles, config):
        self.grid = grid
        self.particles = particles
        self.dt = config.get("dt", 2e-3)
        self.max_iter = config.get("max_iter", 1)
        
        E = config.get("E", 4)
        nu = config.get("nu", 0.4)
        self.mu = E / (2*(1+nu))
        self.lam = E*nu/((1+nu)*(1-2*nu))
        
        solver_type = config.get("implicit_solver", "BFGS")
        if solver_type == "BFGS":
            self.optimizer = BFGS(self.compute_energy_grad, grid.size**grid.dim * grid.dim)
        else:
            self.optimizer = LBFGS(self.compute_energy_grad, grid.size**grid.dim * grid.dim)

    def solve(self):
        self.build_neighbor_list()
        self.save_previous_velocity()
        self.set_initial_guess()
        self.optimizer.minimize()
        self.update_velocity()

    @ti.kernel
    def build_neighbor_list(self):
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            self.grid.particle_count[i,j] = 0
            
        for p in range(self.particles.n_particles):
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            fx = self.particles.x[p] * self.grid.inv_dx - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
            
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                cnt = ti.atomic_add(self.grid.particle_count[grid_idx], 1)
                if cnt < 32:
                    self.grid.particles[grid_idx.x, grid_idx.y, cnt] = p
                    weight = 1.0
                    for d in ti.static(range(self.grid.dim)):
                        weight *= w[offset[d]][d]
                    dpos = (offset - fx) * self.grid.inv_dx
                    self.grid.dwip[grid_idx.x, grid_idx.y, cnt] = dpos * weight

    @ti.kernel
    def compute_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()) -> ti.f32:
        total_energy = 0.0
        grad_flat.fill(0.0)
        
        # 粒子阶段
        for p in range(self.particles.n_particles):
            vel_grad = ti.Matrix.zero(ti.f32, 2, 2)
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            fx = self.particles.x[p] * self.grid.inv_dx - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]

            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                weight = 1.0
                for d in ti.static(range(self.grid.dim)):
                    weight *= w[offset[d]][d]
                vidx = grid_idx.x * self.grid.size * 2 + grid_idx.y * 2
                vel = ti.Vector([v_flat[vidx], v_flat[vidx+1]])
                vel_grad += 4 * self.dt * vel.outer_product((offset - fx) * self.grid.inv_dx) * weight
            
            new_F = (ti.Matrix.identity(ti.f32, 2) + vel_grad) @ self.particles.F[p]
            J = new_F.determinant()
            logJ = ti.log(J)
            F_inv_T = new_F.inverse().transpose()
            cauchy = self.mu * (new_F - F_inv_T)*self.particles.F[p].transpose() + self.lam * ti.log(J) * F_inv_T * self.particles.F[p].transpose()
            self.particles.temp_P[p] = (self.particles.p_vol ) * cauchy
            
            energy = 0.5*self.mu*(new_F.norm_sqr() - 2) - self.mu*logJ + 0.5*self.lam*logJ**2
            total_energy += energy * self.particles.p_vol
        
        # 网格阶段
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            if self.grid.m[i,j] == 0:
                continue
                
            vidx = i * self.grid.size * 2 + j * 2
            momentum_grad = self.grid.m[i,j] * ti.Vector([v_flat[vidx]-self.grid.v_prev[i,j][0], 
                                                        v_flat[vidx+1]-self.grid.v_prev[i,j][1]])
            
            energy_grad = ti.Vector.zero(ti.f32, 2)
            for k in range(self.grid.particle_count[i,j]):
                p = self.grid.particles[i,j,k]
                energy_grad += 4*self.particles.temp_P[p] @ self.grid.dwip[i,j,k] * self.dt
            
            grad_flat[vidx] = momentum_grad[0] + energy_grad[0]
            grad_flat[vidx+1] = momentum_grad[1] + energy_grad[1]
            total_energy += 0.5 * self.grid.m[i,j] * (ti.Vector([v_flat[vidx], v_flat[vidx+1]]) - 
                             self.grid.v_prev[i,j]).norm_sqr()
        
        return total_energy

    def save_previous_velocity(self):
        self.grid.v_prev.copy_from(self.grid.v)

    @ti.kernel
    def set_initial_guess(self):
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            idx = i * self.grid.size * 2 + j * 2
            self.optimizer.x[idx] = self.grid.v[i,j][0]
            self.optimizer.x[idx+1] = self.grid.v[i,j][1]

    @ti.kernel
    def update_velocity(self):
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            idx = i * self.grid.size * 2 + j * 2
            self.grid.v[i,j][0] = self.optimizer.x[idx]
            self.grid.v[i,j][1] = self.optimizer.x[idx+1]

# ------------------ 主模拟器 ------------------
@ti.data_oriented
class ImplicitMPM:
    def __init__(self, config_path):
        self.cfg = Config(config_path)
        self.grid = Grid(
            size=self.cfg.get("grid_size", 16),
            dim=2,
            bound=self.cfg.get("bound", 2)
        )
        self.particles = Particles(self.cfg, self.grid.size)
        self.implicit = self.cfg.get("implicit", True)
        self.max_iter = self.cfg.get("max_iter", 1)
        self.dt = self.cfg.get("dt", 2e-3)

        E = self.cfg.get("E", 4)
        nu = self.cfg.get("nu", 0.4)
        self.mu = E / (2 * (1 + nu))  # 现在作为实例属性
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        if self.implicit:
            self.solver = ImplicitSolver(self.grid, self.particles, self.cfg)
        
        self.particles.initialize()

    def step(self):
        for _ in range(self.max_iter):
            self.grid.clear()
            self.p2g()
            if self.implicit:
                self.solver.solve()
            else:
                self.solve_explicit()
            self.grid.apply_boundary_conditions()
            self.g2p()
            self.particles.advect(self.dt)

    @ti.kernel
    def p2g(self):
        for p in range(self.particles.n_particles):
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            fx = self.particles.x[p] * self.grid.inv_dx - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
            
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                weight = 1.0
                for d in ti.static(range(self.grid.dim)):
                    weight *= w[offset[d]][d]
                dpos = (offset - fx) * self.grid.dx
                ti.atomic_add(self.grid.m[grid_idx], weight * self.particles.p_mass)
                ti.atomic_add(self.grid.v[grid_idx], weight * self.particles.p_mass * 
                             (self.particles.v[p] + self.particles.C[p] @ dpos))
        
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            if self.grid.m[i,j] > 1e-10:
                self.grid.v[i,j] /= self.grid.m[i,j]

    @ti.kernel
    def g2p(self):
        for p in self.particles.x:
            Xp = self.particles.x[p] / self.grid.dx
            base = (Xp - 0.5).cast(int)
            fx = Xp - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
            
            new_v = ti.Vector.zero(ti.f32, 2)
            new_C = ti.Matrix.zero(ti.f32, 2, 2)
            
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = base + offset
                weight = 1.0
                for d in ti.static(range(self.grid.dim)):
                    weight *= w[offset[d]][d]
                dpos = (offset - fx) * self.grid.dx
                g_v = self.grid.v[grid_idx]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.grid.dx**2
                
            self.particles.v[p] = new_v
            self.particles.F[p] = (ti.Matrix.identity(ti.f32, 2) + self.dt * self.particles.C[p]) @ self.particles.F[p]
            self.particles.C[p] = new_C

    @ti.kernel
    def solve_explicit(self):
        for p in range(self.particles.n_particles):
            Xp = self.particles.x[p] / self.grid.dx
            base = (Xp - 0.5).cast(int)
            fx = Xp - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]

            U, sig, V = ti.svd(self.particles.F[p])
            if sig.determinant() < 0:
                sig[1,1] = -sig[1,1]
            self.particles.F[p] = U @ sig @ V.transpose()
            
            J = self.particles.F[p].determinant()
            logJ = ti.log(J)
            cauchy = self.mu * (self.particles.F[p] @ self.particles.F[p].transpose()) + ti.Matrix.identity(ti.f32, 2) * (self.lam * logJ - self.mu)
            stress = -(self.dt * self.particles.p_vol * 4 / self.grid.dx**2) * cauchy

            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = base + offset
                weight = 1.0
                for d in ti.static(range(self.grid.dim)):
                    weight *= w[offset[d]][d]
                dpos = (offset - fx) * self.grid.dx
                self.grid.v[grid_idx] += weight * stress @ dpos / self.grid.m[grid_idx]

# 使用示例
if __name__ == "__main__":
    mpm = ImplicitMPM("config.json")
    gui = ti.GUI("Implicit MPM", res=800)
    
    while gui.running:
        mpm.step()
        gui.circles(mpm.particles.x.to_numpy(), radius=1.5, color=0x66CCFF)
        gui.show()