
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

        self.dim = config.get("dim", 2)
        self.neighbor = (3,) * self.dim

        self.dt = config.get("dt", 2e-3)
        self.solve_max_iter = config.get("solve_max_iter", 50)
        self.solve_init_iter = config.get("solve_init_iter", 10)
        
        E = config.get("E", 4)
        nu = config.get("nu", 0.4)
        self.mu = E / (2*(1+nu))
        self.lam = E*nu/((1+nu)*(1-2*nu))
        self.v_grad=ti.field(self.float_type, grid.size**self.dim * self.dim,needs_grad=True)
        self.grad_save=ti.field(self.float_type, grid.size**self.dim * self.dim)
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
            self.optimizer = BFGS(energy_fn=self.compute_energy,grad_fn=self.grad_fn, dim=grid.size**self.dim * self.dim,grad_normalizer=self.dt*self.particles.p_mass*self.particles.particles_per_grid,eta= eta,float_type=self.float_type)
        elif solver_type == "LBFGS":
            self.optimizer = LBFGS(self.grad_fn, grid.size**self.dim * self.dim, eta=eta,float_type=self.float_type)
        elif solver_type == "Newton":
            self.optimizer = Newton(energy_fn=self.compute_energy, grad_fn=self.grad_fn, hess_fn=self.compute_hess, DBC_fn=self.set_hess_DBC,dim=grid.size**self.dim * self.dim,grad_normalizer=self.dt*self.particles.p_mass*self.particles.particles_per_grid,eta= eta,float_type=self.float_type)
    def solve(self):
        self.grid.set_boundary_v()
        self.set_initial_guess()
        iter=self.optimizer.minimize(self.solve_max_iter, self.solve_init_iter)
        self.iter_history.append(iter)
        self.update_velocity()
        return iter

    @ti.func
    def get_idx(self, I):
        idx = 0
        for d in ti.static(range(self.dim)):
            idx += I[d] * self.grid.size ** (self.dim-1-d) *self.dim
        return idx
    
    @ti.func
    def get_vel(self,v_flat,vidx):
        vel = ti.Vector.zero(self.float_type, self.dim)
        for d in ti.static(range(self.dim)):
            vel[d] = v_flat[vidx+d]
        return vel
    
    @ti.func
    def cal_vel_grad(self,v_flat,p):
        vel_grad = ti.Matrix.zero(self.float_type, self.dim, self.dim)
        base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
        for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
            grid_idx = (base + offset) % self.grid.size
            vidx = self.get_idx(grid_idx)
            vel = self.get_vel(v_flat,vidx)
            vel_grad += 4 * self.dt * vel.outer_product(self.particles.dwip[p, offset])
        return vel_grad
        
    
    @ti.kernel
    def compute_particle_energy(self, v_flat: ti.template()):
        for p in range(self.particles.n_particles):
            vel_grad = self.cal_vel_grad(v_flat,p)

            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.dim) + vel_grad) @ F
            J = new_F.determinant()
            logJ = ti.log(J)

            e1=0.5*self.mu*(new_F.norm_sqr() - self.dim)
            e2=-self.mu*logJ
            e3=0.5*self.lam*(logJ**2)
            energy = e1 + e2 + e3
            self.total_energy[None] += energy * self.particles.p_vol



    @ti.kernel
    def compute_grid_energy(self, v_flat: ti.template()):
        for I in ti.grouped(self.grid.v):
            vidx = self.get_idx(I)
            vel = self.get_vel(v_flat,vidx)
            self.total_energy[None] += 0.5 * self.grid.m[I] * (vel - self.grid.v_prev[I]).norm_sqr()

    @ti.kernel
    def manual_particle_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()):
        for p in range(self.particles.n_particles):
            # 计算速度梯度
            vel_grad = self.cal_vel_grad(v_flat,p)

            # 计算变形梯度导数
            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.dim) + vel_grad) @ F
            newJ = new_F.determinant()
            newF_T= new_F.transpose()
            newF_inv_T = newF_T.inverse()

            # 导数
            g1=self.mu * new_F  
            g2=-self.mu * newF_inv_T 
            g3=self.lam * ti.log(newJ) * newF_inv_T
            dE_dvel_grad =(g1+g2+g3)@ F.transpose()*self.particles.p_vol

            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            # 将梯度分配到网格点
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = (base + offset) % self.grid.size
                vidx = self.get_idx(grid_idx)

                grad= 4*self.dt*dE_dvel_grad @ self.particles.dwip[p, offset]

                # 累加梯度
                for d in ti.static(range(self.dim)):
                    ti.atomic_add(grad_flat[vidx+d], grad[d])

    @ti.kernel
    def manual_grid_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()):

        for I in ti.grouped(self.grid.v):
            vidx = self.get_idx(I)
            vel = self.get_vel(v_flat,vidx)
            grad = self.grid.m[I] * (vel - self.grid.v_prev[I])

            # 累加梯度
            for d in ti.static(range(self.dim)):
                ti.atomic_add(grad_flat[vidx+d], grad[d])

    @ti.kernel
    def manual_particle_energy_hess(self,v_flat: ti.template(),hess: ti.sparse_matrix_builder()):
        for p in range(self.particles.n_particles):
            # 计算速度梯度
            vel_grad = self.cal_vel_grad(v_flat,p)


            # 计算变形梯度导数
            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.dim) + vel_grad) @ F
            J = new_F.determinant()
            F_inv = new_F.inverse()
            F_inv_T = F_inv.transpose()
            
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int) 

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                FTw=4 * self.dt * F.transpose() @ self.particles.dwip[p, offset]
                FWWF=F_inv_T @ FTw.outer_product(FTw) @ F_inv 
                h1=self.mu *FTw.dot(FTw) * ti.Matrix.identity(self.float_type, self.dim) 
                h2=self.mu * FWWF 
                h3=self.lam * (1-ti.log(J)) * FWWF
                hessian=(h1+h2+h3)*self.particles.p_vol
                
                U ,sig, V = ti.svd(hessian)
                for i in ti.static(range(self.dim)):
                    if sig[i,i] < 0:
                        sig[i,i] = 0
                hessian = U @ sig @ V.transpose()

                grid_idx = (base + offset) % self.grid.size
                vidx = self.get_idx(grid_idx)
                for i in ti.static(range(self.dim)):
                    for j in ti.static(range(self.dim)):
                        hess[vidx+i, vidx+j] += hessian[i,j]


    @ti.kernel
    def manual_grid_energy_hess(self,hess: ti. sparse_matrix_builder()):
        for I in ti.grouped(self.grid.v):
            vidx= self.get_idx(I)
            m= 1 if self.grid.m[I] ==0 else self.grid.m[I]

            for d in ti.static(range(self.dim)):
                hess[vidx+d, vidx+d] += m

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

    def set_hess_DBC(self, hess):
        num_rows = self.grid.size**self.dim * self.dim
        hess1=ti.linalg.SparseMatrixBuilder(num_rows, num_rows, max_num_triplets=(num_rows)**self.dim, dtype=self.float_type)
        hess2=ti.linalg.SparseMatrixBuilder(num_rows, num_rows, max_num_triplets=num_rows, dtype=self.float_type)
        self.grid.get_boundary_hess(hess1,hess2)
        H1=hess1.build()
        H2=hess2.build()
        hess= hess * H1 + H2


    def save_previous_velocity(self):
        self.grid.v_prev.copy_from(self.grid.v)

    @ti.kernel
    def set_initial_guess(self):
        for I in ti.grouped(self.grid.v):
            idx=0
            for d in ti.static(range(self.dim)):
                idx += I[d] * self.grid.size ** (self.dim-1-d) *self.dim 
            for d in ti.static(range(self.dim)):
                self.optimizer.x[idx+d] = self.grid.v[I][d]

    @ti.kernel
    def update_velocity(self):
        for I in ti.grouped(self.grid.v):
            idx=0
            for d in ti.static(range(self.dim)):
                idx += I[d] * self.grid.size ** (self.dim-1-d) *self.dim 
            for d in ti.static(range(self.dim)):
                self.grid.v[I][d] = self.optimizer.x[idx+d]