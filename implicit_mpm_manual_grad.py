import taichi as ti
import json

from Geometry.Grid import Grid
from Geometry.Particles import Particles

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


# ------------------ 隐式求解器模块 ------------------
@ti.data_oriented
class ImplicitSolver:
    def __init__(self, grid:Grid, particles:Particles, config:Config ):
        self.grid = grid
        self.particles = particles
        self.dt = config.get("dt", 2e-3)
        self.solve_max_iter = config.get("solve_max_iter", 50)
        
        E = config.get("E", 4)
        nu = config.get("nu", 0.4)
        self.mu = E / (2*(1+nu))
        self.lam = E*nu/((1+nu)*(1-2*nu))
        self.v_grad=ti.field(ti.f32, grid.size**grid.dim * grid.dim,needs_grad=True)
        self.total_energy = ti.field(ti.f32, shape=(), needs_grad=True)
        
        solver_type = config.get("implicit_solver", "BFGS")
        if solver_type == "BFGS":
            self.optimizer = BFGS(self.compute_energy_grad, grid.size**grid.dim * grid.dim)
        else:
            self.optimizer = LBFGS(self.compute_energy_grad, grid.size**grid.dim * grid.dim)

    def solve(self):
        self.save_previous_velocity()
        self.set_initial_guess()
        self.optimizer.minimize(self.solve_max_iter)
        self.update_velocity()


    @ti.kernel
    def manual_particle_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()):
        for p in range(self.particles.n_particles):
            # 计算速度梯度
            vel_grad = ti.Matrix.zero(ti.f32, 2, 2)
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            fx = self.particles.x[p] * self.grid.inv_dx - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]

            # 遍历周围网格点
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                weight = 1.0
                for d in ti.static(range(2)):
                    weight *= w[offset[d]][d]

                # 获取网格点速度
                vidx = grid_idx.x * self.grid.size * 2 + grid_idx.y * 2
                v = ti.Vector([v_flat[vidx], v_flat[vidx+1]])

                # 计算速度梯度贡献
                vel_grad += 4 * self.dt * v.outer_product((offset - fx) * self.grid.inv_dx) * weight

            # 计算变形梯度导数
            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(ti.f32, 2) + vel_grad) @ F
            J = new_F.determinant()
            F_inv = new_F.inverse()
            F_inv_T = F_inv.transpose()

            # 导数
            dE_dvel_grad =(self.mu * (new_F - F_inv_T)*F.transpose() + self.lam * ti.log(J) * F_inv_T * F.transpose())*self.particles.p_vol

            # 将梯度分配到网格点
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                weight = 1.0
                for d in ti.static(range(2)):
                    weight *= w[offset[d]][d]

                # 计算当前offset的权重梯度
                dw_dv = 4 * self.dt * (offset - fx) * self.grid.inv_dx * weight

                # 累加梯度
                vidx = grid_idx.x * self.grid.size * 2 + grid_idx.y * 2
                grad_flat[vidx] += dE_dvel_grad[0,0] * dw_dv[0]
                grad_flat[vidx] += dE_dvel_grad[0,1] * dw_dv[1]
                grad_flat[vidx+1] += dE_dvel_grad[1,0] * dw_dv[0]
                grad_flat[vidx+1] += dE_dvel_grad[1,1] * dw_dv[1]

    @ti.kernel
    def manual_grid_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()):
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            vidx = i * self.grid.size * 2 + j * 2
            v = ti.Vector([v_flat[vidx], v_flat[vidx+1]])
            grad = self.grid.m[i,j] * (v - self.grid.v_prev[i,j])

            grad_flat[vidx] += grad[0]
            grad_flat[vidx+1] += grad[1]

    @ti.kernel
    def compute_particle_energy(self, v_flat: ti.template()):
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

            new_F = (ti.Matrix.identity(ti.f32, self.grid.dim) + vel_grad) @ self.particles.F[p]
            J = new_F.determinant()
            logJ = ti.log(J)

            energy = 0.5*self.mu*(new_F.norm_sqr() - self.grid.dim) - self.mu*logJ + 0.5*self.lam*logJ**2
            self.total_energy[None] += energy * self.particles.p_vol

    @ti.kernel
    def compute_grid_energy(self, v_flat: ti.template()):
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            vidx = i * self.grid.size * 2 + j * 2
            self.total_energy[None] += 0.5 * self.grid.m[i,j] * (ti.Vector([v_flat[vidx], v_flat[vidx+1]]) -self.grid.v_prev[i,j]).norm_sqr()


    def compute_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()) -> ti.f32:
        @ti.kernel
        def copy_field(a: ti.template(),b: ti.template()):
            for I in ti.grouped(a):
                a[I] = b[I]  
        
        copy_field(self.v_grad,v_flat)

#        with ti.ad.Tape(self.total_energy):
        self.total_energy[None] = 0.0
        self.compute_particle_energy(self.v_grad)
        self.compute_grid_energy(self.v_grad)
        self.manual_particle_energy_grad(self.v_grad,grad_flat)
        self.manual_grid_energy_grad(self.v_grad,grad_flat)

        #copy_field(grad_flat,self.v_grad.grad)

        return self.total_energy[None]  

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
            self.grid.apply_boundary_conditions_explicit()
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