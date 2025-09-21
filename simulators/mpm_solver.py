import taichi as ti
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Geometry.Grid import Grid
from Geometry.Particles import Particles

from Optimizer.BFGS import BFGS
from Optimizer.LBFGS import LBFGS
from Optimizer.Newton import Newton

from Util.Config import Config

# ------------------ MPM求解器模块 ------------------
@ti.data_oriented
class MPMSolver:
    def __init__(self, grid:Grid, particles:Particles, config:Config):
        self.grid = grid
        self.particles = particles

        self.float_type = ti.f32 if config.get("float_type", ti.f32) == "f32" else ti.f64

        self.dim = config.get("dim", 2)
        self.neighbor = (3,) * self.dim

        self.dt = config.get("dt", 2e-3)
        self.solve_max_iter = config.get("solve_max_iter", 50)
        self.solve_init_iter = config.get("solve_init_iter", 10)
        
        # 静力学求解选项
        self.static_solve = config.get("static_solve", False)
        
        # 保留兼容性的全局参数，但现在主要使用particles.material_params
        E = config.get("E", 4)
        nu = config.get("nu", 0.4)
        self.mu = E / (2*(1+nu))
        self.lam = E*nu/((1+nu)*(1-2*nu))

        self.damping = config.get("damping", 0.0)

        default_gravity = [0.0, 0.0] if self.dim == 2 else [0.0, 0.0, 0.0]
        gravity = config.get("gravity", default_gravity)
        self.gravity = ti.Vector(gravity)
        
        # 解析体积力配置
        self._parse_volume_forces(config.get("volume_forces", []))

        
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
            # 使用材料1的p_mass进行梯度归一化
            avg_p_mass = (self.particles.p_mass_1 + self.particles.p_mass_2) / 2.0
            self.optimizer = BFGS(energy_fn=self.compute_energy,grad_fn=self.grad_fn, dim=grid.size**self.dim * self.dim,grad_normalizer=self.dt*avg_p_mass*self.particles.particles_per_grid,eta= eta,float_type=self.float_type)
        elif solver_type == "LBFGS":
            self.optimizer = LBFGS(self.grad_fn, grid.size**self.dim * self.dim, eta=eta,float_type=self.float_type)
        elif solver_type == "Newton":
            # 使用材料1的p_mass进行梯度归一化
            avg_p_mass = (self.particles.p_mass_1 + self.particles.p_mass_2) / 2.0
            self.optimizer = Newton(energy_fn=self.compute_energy, grad_fn=self.grad_fn, hess_fn=self.compute_hess, DBC_fn=self.set_hess_DBC,dim=grid.size**self.dim * self.dim,grad_normalizer=self.dt*avg_p_mass*self.particles.particles_per_grid,eta= eta,float_type=self.float_type)

    def solve_implicit(self):
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
    def make_PSD(self, matrix):
        """
        Make a matrix positive semi-definite using eigenvalue decomposition
        Similar to the numpy version: set all negative eigenvalues to small positive values
        Supports both 2x2 and 3x3 matrices based on self.dim
        """
        # 特征值分解
        eigenvals, eigenvecs = ti.sym_eig(matrix)
        for i in ti.static(range(self.dim)):
            eigenvals[i] = ti.max(eigenvals[i], 0)
        
        # 重构正定矩阵：V @ diag(λ) @ V^T
        diag_vals = ti.Matrix.zero(self.float_type, self.dim, self.dim)
        for i in ti.static(range(self.dim)):
            diag_vals[i, i] = eigenvals[i]
    
        psd_matrix = eigenvecs @ diag_vals @ eigenvecs.transpose()
        return psd_matrix
    
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

            mu, lam = self.particles.get_material_params(p)

            e1=0.5*mu*(new_F.norm_sqr() - self.dim)
            e2=-mu*logJ
            e3=0.5*lam*(logJ**2)
            energy = e1 + e2 + e3
            weight = self.particles.get_particle_weight(p)
            self.total_energy[None] += energy * self.particles.p_vol * weight



    @ti.kernel
    def compute_grid_energy(self, v_flat: ti.template()):
        for I in ti.grouped(self.grid.v):
            vidx = self.get_idx(I)
            vel = self.get_vel(v_flat,vidx)
            
            # 惯性项 - 静力学求解时跳过
            if not ti.static(self.static_solve):
                self.total_energy[None] += 0.5 * self.grid.m[I] * (vel - self.grid.v_prev[I]).norm_sqr()

            # 计算势能
            pos = I * self.grid.dx + vel * self.dt
            total_external_force = self.gravity
            if self.n_volume_forces > 0:
                total_external_force += self._get_volume_force_at_position(pos)
            self.total_energy[None] -= total_external_force.dot(pos) * self.grid.m[I]

            # 计算阻尼
            if not ti.static(self.static_solve):
                self.total_energy[None] += 0.5 * self.damping * self.grid.m[I] * vel.norm_sqr()

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
            mu, lam = self.particles.get_material_params(p)
            g1=mu * new_F
            g2=-mu * newF_inv_T
            g3=lam * ti.log(newJ) * newF_inv_T
            weight = self.particles.get_particle_weight(p)
            dE_dvel_grad =(g1+g2+g3)@ F.transpose()*self.particles.p_vol * weight

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
            
            # 初始化梯度
            grad = ti.Vector.zero(self.float_type, self.dim)
            
            # 惯性项梯度 - 静力学求解时跳过
            if not ti.static(self.static_solve):
                grad += self.grid.m[I] * (vel - self.grid.v_prev[I])
            
            # 计算外部力（重力 + 体积力）
            pos = I * self.grid.dx + vel * self.dt
            total_external_force = self.gravity
            if self.n_volume_forces > 0:
                total_external_force += self._get_volume_force_at_position(pos)
            grad -= total_external_force * self.dt * self.grid.m[I]

            # 阻尼
            if not ti.static(self.static_solve):
                grad += self.damping * vel * self.grid.m[I] 


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

            mu, lam = self.particles.get_material_params(p)

            weight = self.particles.get_particle_weight(p)

            for offset in ti.grouped(ti.ndrange(*self.neighbor)):
                FTw=4 * self.dt * F.transpose() @ self.particles.dwip[p, offset]
                FWWF=F_inv_T @ FTw.outer_product(FTw) @ F_inv
                h1=mu *FTw.dot(FTw) * ti.Matrix.identity(self.float_type, self.dim)
                h2=mu * FWWF
                h3=lam * (1-ti.log(J)) * FWWF
                hessian=(h1+h2+h3)*self.particles.p_vol * weight

                hessian=self.make_PSD(hessian)

                grid_idx = (base + offset) % self.grid.size
                vidx = self.get_idx(grid_idx)

                for (i,j) in ti.static(ti.ndrange(self.dim, self.dim)):
                    if not self.grid.is_boundary_grid[grid_idx][i] and not self.grid.is_boundary_grid[grid_idx][j]:
                        hess[vidx+i, vidx+j] += hessian[i,j]
    @ti.kernel
    def manual_particle_energy_hess_1(self,v_flat: ti.template(),hess: ti.sparse_matrix_builder()):

        for p in range(self.particles.n_particles):
            # 计算速度梯度
            vel_grad = self.cal_vel_grad(v_flat,p)

            # 计算变形梯度
            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.dim) + vel_grad) @ F
            
            # 对new_F进行SVD分解 - 使用与NeoHookean.py相同的polar_svd逻辑
            U, sigma_mat, VT = ti.svd(new_F)
            
            # 确保行列式为正 - 按照NeoHookean.py的polar_svd逻辑
            if U.determinant() < 0:
                U[:, 1] = -U[:, 1]
                sigma_mat[1,1] = -sigma_mat[1,1]
            if VT.determinant() < 0:
                VT[1, :] = -VT[1, :]
                sigma_mat[1,1] = -sigma_mat[1,1]
            
            # 提取奇异值
            sigma = ti.Vector([sigma_mat[0,0], sigma_mat[1,1]])
            
            mu, lam = self.particles.get_material_params(p)
            weight = self.particles.get_particle_weight(p)
            
            # 计算d2Psi_div_dsigma2矩阵元素
            ln_sigma_prod = ti.log(sigma[0] * sigma[1])
            inv2_0 = 1.0 / (sigma[0] * sigma[0])
            d2Psi_dsigma2_00 = mu * (1 + inv2_0) - lam * inv2_0 * (ln_sigma_prod - 1)
            inv2_1 = 1.0 / (sigma[1] * sigma[1])
            d2Psi_dsigma2_11 = mu * (1 + inv2_1) - lam * inv2_1 * (ln_sigma_prod - 1)
            d2Psi_dsigma2_01 = lam / (sigma[0] * sigma[1])
            
            # 构建2x2 d2Psi_dsigma2矩阵并使用make_PSD
            d2Psi_dsigma2 = ti.Matrix([[d2Psi_dsigma2_00, d2Psi_dsigma2_01], 
                                       [d2Psi_dsigma2_01, d2Psi_dsigma2_11]])
            
            # 使用make_PSD函数确保正定性
            d2Psi_dsigma2_psd = self.make_PSD(d2Psi_dsigma2)
            d2Psi_dsigma2_00 = d2Psi_dsigma2_psd[0, 0]
            d2Psi_dsigma2_11 = d2Psi_dsigma2_psd[1, 1]
            d2Psi_dsigma2_01 = d2Psi_dsigma2_psd[0, 1]
            
            # 计算B矩阵
            sigma_prod = sigma[0] * sigma[1]
            B_left = (mu + (mu - lam * ti.log(sigma_prod)) / sigma_prod) / 2
            
            # 计算dPsi_div_dsigma
            inv0 = 1.0 / sigma[0]
            dPsi_dsigma_0 = mu * (sigma[0] - inv0) + lam * inv0 * ln_sigma_prod
            inv1 = 1.0 / sigma[1]
            dPsi_dsigma_1 = mu * (sigma[1] - inv1) + lam * inv1 * ln_sigma_prod
            
            B_right = (dPsi_dsigma_0 + dPsi_dsigma_1) / (2 * ti.max(sigma[0] + sigma[1], 1e-6))
            
            # 构建2x2 B矩阵并使用make_PSD
            B_matrix = ti.Matrix([[B_left + B_right, B_left - B_right], 
                                  [B_left - B_right, B_left + B_right]])
            
            # 使用make_PSD函数确保正定性
            B_psd = self.make_PSD(B_matrix)
            B_00 = B_psd[0, 0]
            B_01 = B_psd[0, 1]
            B_11 = B_psd[1, 1]
            
            # 构建M矩阵 (4x4)
            M = ti.Matrix.zero(self.float_type, 4, 4)
            M[0, 0] = d2Psi_dsigma2_00
            M[0, 3] = d2Psi_dsigma2_01
            M[1, 1] = B_00
            M[1, 2] = B_01
            M[2, 1] = B_01
            M[2, 2] = B_11
            M[3, 0] = d2Psi_dsigma2_01
            M[3, 3] = d2Psi_dsigma2_11
            
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            
            coeff = self.particles.p_vol * weight

            # 计算完整的4x4 Hessian矩阵 d²Ψ/dF² 然后转换为2x2网格速度Hessian
            dP_dF = ti.Matrix.zero(self.float_type, 4, 4)
            
            # 按照NeoHookean.py的公式构建4x4 Hessian
            for j in ti.static(range(self.dim)):
                for i in ti.static(range(self.dim)):
                    ij = j * self.dim + i
                    for s in ti.static(range(self.dim)):
                        for r in ti.static(range(self.dim)):
                            rs = s * self.dim + r
                            dP_dF[ij, rs] = M[0, 0] * U[i, 0] * VT[0, j] * U[r, 0] * VT[0, s] \
                                          + M[0, 3] * U[i, 0] * VT[0, j] * U[r, 1] * VT[1, s] \
                                          + M[1, 1] * U[i, 1] * VT[0, j] * U[r, 1] * VT[0, s] \
                                          + M[1, 2] * U[i, 1] * VT[0, j] * U[r, 0] * VT[1, s] \
                                          + M[2, 1] * U[i, 0] * VT[1, j] * U[r, 1] * VT[0, s] \
                                          + M[2, 2] * U[i, 0] * VT[1, j] * U[r, 0] * VT[1, s] \
                                          + M[3, 0] * U[i, 1] * VT[1, j] * U[r, 0] * VT[0, s] \
                                          + M[3, 3] * U[i, 1] * VT[1, j] * U[r, 1] * VT[1, s]
                            
            for offset in ti.grouped(ti.ndrange(*self.neighbor)):
                grid_idx = (base + offset) % self.grid.size
                vidx = self.get_idx(grid_idx)
                
                dwip = self.particles.dwip[p, offset]

                
                # --- 在每个 offset 循环内部替换掉你原来的四重求和 ---
                # 计算 FTw 向量（长度 = dim）
                FTw = 4.0 * self.dt * (F.transpose() @ dwip)  # shape (dim,)

                # 对每个速度分量对 (vi, vj) 计算 Hessian 块
                for vi in ti.static(range(self.dim)):
                    for vj in ti.static(range(self.dim)):
                        hess_val = 0.0
                        # 只需对 j, l 两个索引求和（dim 很小，一般是 2）
                        for j in ti.static(range(self.dim)):
                            ij = j * self.dim + vi   # 注意 ij 对应 dF 的位置 (fi=vi, fj=j)
                            for l in ti.static(range(self.dim)):
                                kl = l * self.dim + vj   # kl 对应 (fk=vj, fl=l)
                                hess_val += FTw[j] * dP_dF[ij, kl] * FTw[l]

                        # 应用系数并累加到全局稀疏 Hessian
                        hess_val *= coeff
                        if not self.grid.is_boundary_grid[grid_idx][vi] and not self.grid.is_boundary_grid[grid_idx][vj]:
                            hess[vidx + vi, vidx + vj] += hess_val
                


    @ti.kernel
    def manual_grid_energy_hess(self,hess: ti. sparse_matrix_builder()):
        for I in ti.grouped(self.grid.v):
            vidx= self.get_idx(I)
            m = 0.0
            # 惯性项 + 阻尼项
            if not ti.static(self.static_solve):
                m = self.grid.m[I] *(1+ self.damping)
            m= 1 if self.grid.m[I] == 0.0 else m

            for d in ti.static(range(self.dim)):
                if not self.grid.is_boundary_grid[I][d]:
                    hess[vidx+d, vidx+d] += m
                else:
                    hess[vidx+d, vidx+d] += 1.0

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
        hess1=ti.linalg.SparseMatrixBuilder(num_rows, num_rows, max_num_triplets=(num_rows)**2, dtype=self.float_type)
        hess2=ti.linalg.SparseMatrixBuilder(num_rows, num_rows, max_num_triplets=num_rows, dtype=self.float_type)
        self.grid.get_boundary_hess(hess1,hess2)
        H1=hess1.build()
        H2=hess2.build()
        # hess= hess * H1 + H2
        hess= hess + H2


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
    
    # =============== Volume Forces Support ===============
    def _parse_volume_forces(self, volume_forces_config):
        """解析体积力配置并创建Taichi字段"""
        self.n_volume_forces = len(volume_forces_config)
        
        # 创建Taichi字段存储体积力数据，即使没有体积力也要创建以避免访问错误
        field_size = max(1, self.n_volume_forces)
        self.volume_force_types = ti.field(ti.i32, shape=field_size)  # 0: rectangle, 1: ellipse
        self.volume_force_vectors = ti.Vector.field(self.dim, self.float_type, shape=field_size)
        
        # 为矩形力区域存储范围 [min_x, max_x, min_y, max_y, ...]
        max_range_params = self.dim * 2  # 每个维度的min和max
        self.volume_force_rect_ranges = ti.field(self.float_type, shape=(field_size, max_range_params))
        
        # 为椭圆力区域存储中心和半轴
        max_ellipse_params = self.dim * 2  # 中心坐标 + 半轴长度
        self.volume_force_ellipse_params = ti.field(self.float_type, shape=(field_size, max_ellipse_params))
        
        # 填充数据
        if self.n_volume_forces > 0:
            for i, force_config in enumerate(volume_forces_config):
                force_type = force_config.get("type", "rectangle")
                force_vector = force_config.get("force", [0.0] * self.dim)
                params = force_config.get("params", {})
                
                # 设置力向量
                self.volume_force_vectors[i] = ti.Vector(force_vector)
                
                if force_type == "rectangle":
                    self.volume_force_types[i] = 0
                    force_range = params.get("range", [])
                    for d in range(min(self.dim, len(force_range))):
                        if len(force_range[d]) >= 2:
                            self.volume_force_rect_ranges[i, d*2] = force_range[d][0]    # min
                            self.volume_force_rect_ranges[i, d*2+1] = force_range[d][1]  # max
                            
                elif force_type == "ellipse":
                    self.volume_force_types[i] = 1
                    center = params.get("center", [0.0] * self.dim)
                    semi_axes = params.get("semi_axes", [1.0] * self.dim)
                    for d in range(min(self.dim, len(center))):
                        self.volume_force_ellipse_params[i, d] = center[d]
                    for d in range(min(self.dim, len(semi_axes))):
                        self.volume_force_ellipse_params[i, self.dim + d] = semi_axes[d]
    
    @ti.func
    def _get_volume_force_at_position(self, pos):
        """获取位置处的体积力"""
        total_force = ti.Vector.zero(self.float_type, self.dim)
        for i in range(self.n_volume_forces):
            in_region = True
            if self.volume_force_types[i] == 0:  # rectangle
                for d in ti.static(range(self.dim)):
                    min_val = self.volume_force_rect_ranges[i, d*2]
                    max_val = self.volume_force_rect_ranges[i, d*2+1]
                    if pos[d] < min_val or pos[d] > max_val:
                        in_region = False
                        
            elif self.volume_force_types[i] == 1:  # ellipse
                sum_normalized = 0.0
                for d in ti.static(range(self.dim)):
                    center_d = self.volume_force_ellipse_params[i, d]
                    semi_axis_d = self.volume_force_ellipse_params[i, self.dim + d]
                    diff = pos[d] - center_d
                    sum_normalized += (diff / semi_axis_d)**2
                in_region = sum_normalized <= 1.0
            
            if in_region:
                total_force += self.volume_force_vectors[i]
                
        return total_force

    # =============== Explicit Solver Methods ===============
    def solve_explicit(self):
        """显式求解器"""
        self.compute_forces_explicit()
        self.update_velocity_explicit()
        self.grid.set_boundary_v()
        return 0
    
    @ti.kernel
    def compute_forces_explicit(self):
        """计算所有力并存储到网格力场中"""
        # 初始化网格力场
        for I in ti.grouped(self.grid.f):
            self.grid.f[I] = ti.Vector.zero(self.float_type, self.dim)
        
        # 第一步：计算应力力并分配到网格
        for p in range(self.particles.n_particles):
            Xp = self.particles.x[p] / self.grid.dx
            base = (Xp - 0.5).cast(int)

            U, sig, V = ti.svd(self.particles.F[p])
            if sig.determinant() < 0:
                sig[1,1] = -sig[1,1]
            self.particles.F[p] = U @ sig @ V.transpose()
            
            J = self.particles.F[p].determinant()
            logJ = ti.log(J)
            mu, lam = self.particles.get_material_params(p)
            cauchy = mu * (self.particles.F[p] @ self.particles.F[p].transpose()) + ti.Matrix.identity(self.float_type, self.dim) * (lam * logJ - mu)
            stress = -(self.particles.p_vol ) * cauchy

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = base + offset
                # 将应力力累加到网格力场
                ti.atomic_add(self.grid.f[grid_idx], 4 * stress @ self.particles.dwip[p,offset])
        
        # 第二步：在网格上计算外部力（重力、体积力、阻尼力）
        for I in ti.grouped(self.grid.f):
            if self.grid.m[I] > 1e-10:
                # 计算网格点的物理位置
                grid_pos = I * self.grid.dx
                
                # 添加重力
                self.grid.f[I] += self.gravity * self.grid.m[I]
                
                # 添加体积力
                if self.n_volume_forces > 0:
                    volume_force = self._get_volume_force_at_position(grid_pos)
                    self.grid.f[I] += volume_force * self.grid.m[I]
                
                # 添加阻尼力（基于当前速度）
                damping_force = -self.damping * self.grid.v[I] * self.grid.m[I]
                self.grid.f[I] += damping_force
    
    @ti.kernel  
    def update_velocity_explicit(self):
        """基于计算好的力更新速度"""
        for I in ti.grouped(self.grid.v):
            if self.grid.m[I] > 1e-10:
                # v = v + dt * f / m
                self.grid.v[I] += self.dt * self.grid.f[I] / self.grid.m[I]
    
    @ti.kernel
    def p2g_F_averaging(self, F_grid: ti.template(), m_grid: ti.template()):
        """P2G步骤：将粒子的F值传输到网格"""
        # 初始化网格
        for I in ti.grouped(F_grid):
            F_grid[I] = ti.Matrix.zero(self.float_type, self.dim, self.dim)
            m_grid[I] = 0.0
        
        # P2G: 将粒子的F值传输到网格
        for p in range(self.particles.n_particles):
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = (base + offset) % self.grid.size
                weight = self.particles.wip[p, offset]
                p_mass = self.particles.get_particle_mass(p)
                
                ti.atomic_add(m_grid[grid_idx], weight * p_mass)
                for i in ti.static(range(self.dim)):
                    for j in ti.static(range(self.dim)):
                        ti.atomic_add(F_grid[grid_idx][i,j], weight * p_mass * self.particles.F[p][i,j])
        
        # 归一化网格上的F值
        for I in ti.grouped(F_grid):
            if m_grid[I] > 1e-10:
                F_grid[I] /= m_grid[I]

    @ti.kernel
    def g2p_F_averaging(self, F_grid: ti.template()):
        """G2P步骤：将网格的平均F值传输回粒子"""
        for p in range(self.particles.n_particles):
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            new_F = ti.Matrix.zero(self.float_type, self.dim, self.dim)
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = (base + offset) % self.grid.size
                weight = self.particles.wip[p, offset]
                new_F += weight * F_grid[grid_idx]
            
            self.particles.F[p] = new_F

    def average_F_p2g_g2p(self):
        """通过p2g和g2p操作对F[p]进行平均"""
        # 创建临时网格字段存储F值
        F_grid = ti.Matrix.field(self.dim, self.dim, self.float_type, shape=(self.grid.size,)*self.dim)
        m_grid = ti.field(self.float_type, shape=(self.grid.size,)*self.dim)
        
        # 执行P2G和G2P
        self.p2g_F_averaging(F_grid, m_grid)
        self.g2p_F_averaging(F_grid)

    @ti.kernel
    def compute_stress_strain(self):
        """计算并存储所有粒子的应力和应变（仅在需要时调用）"""
        for p in range(self.particles.n_particles):
            # 重新计算应力
            U, sig, V = ti.svd(self.particles.F[p])
            if sig.determinant() < 0:
                sig[1,1] = -sig[1,1]
            F_corrected = U @ sig @ V.transpose()
            
            J = F_corrected.determinant()
            logJ = ti.log(J)
            mu, lam = self.particles.get_material_params(p)
            cauchy = mu * (F_corrected @ F_corrected.transpose()) + ti.Matrix.identity(self.float_type, self.dim) * (lam * logJ - mu)
            
            # 存储应力
            self.particles.stress[p] = cauchy
            
            # 计算并存储应变 (Green-Lagrange strain)
            F_transpose_F = F_corrected.transpose() @ F_corrected
            self.particles.strain[p] = 0.5 * (F_transpose_F - ti.Matrix.identity(self.float_type, self.dim))

    def compute_stress_strain_with_averaging(self):
        """计算应力应变前先对F[p]进行平均"""
        # 首先对F[p]进行一次p2g，g2p进行平均
        self.average_F_p2g_g2p()
        # 然后计算应力应变
        self.compute_stress_strain()