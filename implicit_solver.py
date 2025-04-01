
import taichi as ti

from Geometry.Grid import Grid
from Geometry.Particles import Particles

from Optimizer.BFGS import BFGS
from Optimizer.LBFGS import LBFGS
from Optimizer.Newton import Newton

from Util.Config import Config

# ------------------ 隐式求解器模块 ------------------
@ti.data_oriented
class ImplicitSolver:
    def __init__(self, grid:Grid, particles:Particles, config:Config):
        self.grid = grid
        self.particles = particles

        self.float_type = ti.f32 if config.get("float_type", ti.f32) == "f32" else ti.f64

        self.dt = config.get("dt", 2e-3)
        self.solve_max_iter = config.get("solve_max_iter", 50)
        self.solve_init_iter = config.get("solve_init_iter", 10)
        
        E = config.get("E", 4)
        nu = config.get("nu", 0.4)
        self.mu = E / (2*(1+nu))
        self.lam = E*nu/((1+nu)*(1-2*nu))
        self.v_grad=ti.field(self.float_type, grid.size**grid.dim * grid.dim,needs_grad=True)
        self.grad_save=ti.field(self.float_type, grid.size**grid.dim * grid.dim)
        self.total_energy = ti.field(self.float_type, shape=(), needs_grad=True)
        self.grad_fn=None
        if config.get("use_auto_diff", True):
            self.grad_fn=self.compute_energy_grad_auto
        else:
            self.grad_fn=self.compute_energy_grad_manual
        
        if config.get("compare_diff", True):
            self.grad_fn=self.compute_energy_grad_diff

        solver_type = config.get("implicit_solver", "BFGS")

        self.iter_history = []

        eta = config.get("eta", 1)
        if solver_type == "BFGS":
            self.optimizer = BFGS(energy_fn=self.compute_energy,grad_fn=self.grad_fn, dim=grid.size**grid.dim * grid.dim,grad_normalizer=self.dt*self.particles.p_mass*self.particles.particles_per_grid,eta= eta,float_type=self.float_type)
        elif solver_type == "LBFGS":
            self.optimizer = LBFGS(self.grad_fn, grid.size**grid.dim * grid.dim, eta=eta,float_type=self.float_type)
        elif solver_type == "Newton":
            self.optimizer = Newton(energy_fn=self.compute_energy, grad_fn=self.grad_fn, hess_fn=self.compute_hess, dim=grid.size**grid.dim * grid.dim,grad_normalizer=self.dt*self.particles.area*self.particles.p_rho,eta= eta,float_type=self.float_type)
    def solve(self):
        self.save_previous_velocity()
        self.set_initial_guess()
        iter=self.optimizer.minimize(self.solve_max_iter, self.solve_init_iter)
        self.iter_history.append(iter)
        self.update_velocity()
        self.grid.set_boundary_v()
        return iter

    
    @ti.kernel
    def compute_particle_energy(self, v_flat: ti.template()):
        for p in range(self.particles.n_particles):
            vel_grad = ti.Matrix.zero(self.float_type, 2, 2)
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                vidx = grid_idx.x * self.grid.size * 2 + grid_idx.y * 2
                vel = ti.Vector([v_flat[vidx], v_flat[vidx+1]])
                vel_grad_= 4 * self.dt * vel.outer_product(self.particles.dwip[p, offset])
                vel_grad += vel_grad_

            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.grid.dim) + vel_grad) @ F
            J = new_F.determinant()
            logJ = ti.log(J)

            e1=0.5*self.mu*(new_F.norm_sqr() - self.grid.dim)
            e2=-self.mu*logJ
            e3=0.5*self.lam*(logJ**2)
            energy = e1 + e2 + e3
            self.total_energy[None] += energy * self.particles.p_vol



    @ti.kernel
    def compute_grid_energy(self, v_flat: ti.template()):
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            vidx = i * self.grid.size * 2 + j * 2
            self.total_energy[None] += 0.5 * self.grid.m[i,j] * (ti.Vector([v_flat[vidx], v_flat[vidx+1]]) - 
                                 self.grid.v_prev[i,j]).norm_sqr()

    @ti.kernel
    def manual_particle_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()):
        for p in range(self.particles.n_particles):
            # 计算速度梯度
            vel_grad = ti.Matrix.zero(self.float_type, 2, 2)
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)

            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                vidx = grid_idx.x * self.grid.size * 2 + grid_idx.y * 2
                vel = ti.Vector([v_flat[vidx], v_flat[vidx+1]])
                vel_grad += 4 * self.dt * vel.outer_product(self.particles.dwip[p, offset]) 

            # 计算变形梯度导数
            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.grid.dim) + vel_grad) @ F
            newJ = new_F.determinant()
            newF_T= new_F.transpose()
            newF_inv_T = newF_T.inverse()

            # 导数
            g1=self.mu * new_F  
            g2=-self.mu * newF_inv_T 
            g3=self.lam * ti.log(newJ) * newF_inv_T
            dE_dvel_grad =(g1+g2+g3)@ F.transpose()*self.particles.p_vol


            # 将梯度分配到网格点
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                vidx = grid_idx.x * self.grid.size * 2 + grid_idx.y * 2

                grad= 4*self.dt*dE_dvel_grad @ self.particles.dwip[p, offset]

                # 累加梯度
                ti.atomic_add(grad_flat[vidx], grad[0])
                ti.atomic_add(grad_flat[vidx+1], grad[1])

    @ti.kernel
    def manual_grid_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()):
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            vidx = i * self.grid.size * 2 + j * 2
            v = ti.Vector([v_flat[vidx], v_flat[vidx+1]])
            grad = self.grid.m[i,j] * (v - self.grid.v_prev[i,j])

            # 累加梯度
            ti.atomic_add(grad_flat[vidx], grad[0]) 
            ti.atomic_add(grad_flat[vidx+1], grad[1])

    @ti.kernel
    def manual_particle_energy_hess(self,v_flat: ti.template(),hess: ti.sparse_matrix_builder()):
        for p in range(self.particles.n_particles):
            # 计算速度梯度
            vel_grad = ti.Matrix.zero(self.float_type, 2, 2)
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)

            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                vidx = grid_idx.x * self.grid.size * 2 + grid_idx.y * 2
                vel = ti.Vector([v_flat[vidx], v_flat[vidx+1]])
                vel_grad += 4 * self.dt * vel.outer_product(self.particles.dwip[p, offset]) 


            # 计算变形梯度导数
            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.grid.dim) + vel_grad) @ F
            J = new_F.determinant()
            F_inv = new_F.inverse()
            F_inv_T = F_inv.transpose()
            


            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                FTw=4 * self.dt * F.transpose() @ self.particles.dwip[p, offset]
                FWWF=F_inv_T @ FTw.outer_product(FTw) @ F_inv 
                h1=self.mu *FTw.dot(FTw) * ti.Matrix.identity(self.float_type, self.grid.dim) 
                h2=self.mu * FWWF 
                h3=self.lam * (1-ti.log(J)) * FWWF
                hessian=(h1+h2+h3)*self.particles.p_vol
                # U ,sig, V = ti.svd(hessian)
                # for i in ti.static(range(self.grid.dim)):
                #     if sig[i,i] < 0:
                #         sig[i,i] = 0
                # hessian = U @ sig @ V.transpose()

                grid_idx = (base + offset) % self.grid.size
                vidx = grid_idx.x * self.grid.size * 2 + grid_idx.y * 2
                hess[vidx, vidx] += hessian[0,0]
                hess[vidx+1, vidx+1] += hessian[1,1]
                hess[vidx, vidx+1] += hessian[0,1]
                hess[vidx+1, vidx] += hessian[1,0]


    @ti.kernel
    def manual_grid_energy_hess(self,hess: ti. sparse_matrix_builder()):
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            vidx = i * self.grid.size * 2 + j * 2
            if self.grid.m[i,j] == 0 :
                hess[vidx, vidx] += 1
                hess[vidx+1, vidx+1] += 1
            else:
                hess[vidx, vidx] += self.grid.m[i,j]
                hess[vidx+1, vidx+1] += self.grid.m[i,j]

    def compute_energy(self, v_flat: ti.template()):
        self.grid.set_boundary_v_grid(v_flat)
        self.total_energy[None] = 0.0
        self.compute_particle_energy(v_flat)
        self.compute_grid_energy(v_flat)
        return self.total_energy[None]
    
    def compute_energy_grad_manual(self, v_flat: ti.template(), grad_flat: ti.template()):

        self.grid.set_boundary_v_grid(v_flat)

        self.total_energy[None] = 0.0
        self.compute_particle_energy(v_flat)
        self.compute_grid_energy(v_flat)

        grad_flat.fill(0.0)
        self.manual_particle_energy_grad(v_flat, grad_flat)
        self.manual_grid_energy_grad(v_flat, grad_flat)

        self.grid.set_boundary_grad(grad_flat)

        return self.total_energy[None]

    def compute_energy_grad_auto(self, v_flat: ti.template(), grad_flat: ti.template()):
        @ti.kernel
        def copy_field(a: ti.template(),b: ti.template()):
            for I in ti.grouped(a):
                a[I] = b[I]  
        
        copy_field(self.v_grad,v_flat)
        
        self.grid.set_boundary_v_grid(self.v_grad)
        
        grad_flat.fill(0.0)

        with ti.ad.Tape(loss=self.total_energy):
            self.compute_particle_energy(self.v_grad)
            self.compute_grid_energy(self.v_grad)

        copy_field(grad_flat,self.v_grad.grad)

        self.grid.set_boundary_grad(grad_flat)

        return self.total_energy[None]  
    
    def compute_energy_grad_diff(self, v_flat: ti.template(), grad_flat: ti.template()):
        #比较手动和自动求导的结果
        @ti.kernel
        def copy_field(a: ti.template(),b: ti.template()):
            for I in ti.grouped(a):
                a[I] = b[I]
        self.compute_energy_grad_auto(v_flat, grad_flat)
        copy_field(self.grad_save,grad_flat)
        self.compute_energy_grad_manual(v_flat, grad_flat)

        for i in range(self.grad_save.shape[0]):
            if ti.abs(self.grad_save[i]) > 1e-10:
                diff = (self.grad_save[i]-grad_flat[i])/self.grad_save[i]
                print(f"grad_auto: {self.grad_save[i]}, grad_manual: {grad_flat[i]}, diff: {diff}")

        copy_field(grad_flat,self.grad_save)

        return self.total_energy[None]

    def compute_hess(self, v_flat: ti.template(), hess: ti.sparse_matrix_builder()):
        self.grid.set_boundary_v_grid(v_flat)

        self.manual_particle_energy_hess(v_flat, hess)
        self.manual_grid_energy_hess(hess)


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
            for d in ti.static(range(self.grid.dim)):
                self.grid.v[i,j][d] = self.optimizer.x[idx+d]