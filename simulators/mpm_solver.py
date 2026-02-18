import taichi as ti
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Geometry.Grid import Grid
from Geometry.Particles import Particles

from Optimizer.BFGS import BFGS
from Optimizer.LBFGS import LBFGS
from Optimizer.Newton import Newton
from Optimizer.CG import CG
from Optimizer.PCG import PCG

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

        # 最后100帧无限制max_iter的设置
        self.unlimited_iter_for_last_frames = False
        self.unlimited_max_iter = config.get("unlimited_max_iter", 10000)  # 最后100帧时的max_iter

        # 静力学求解选项
        self.static_solve = config.get("static_solve", False)

        # 弹性模型选项
        self.elasticity_model = config.get("elasticity_model", "neohookean")
        
        # 保留兼容性的全局参数，但现在主要使用particles.material_params
        E = config.get("E", 4)
        nu = config.get("nu", 0.4)
        self.mu = E / (2*(1+nu))
        self.lam = E*nu/((1+nu)*(1-2*nu))

        self.damping = config.get("damping", 0.0)

        default_gravity = [0.0, 0.0] if self.dim == 2 else [0.0, 0.0, 0.0]
        gravity = config.get("gravity", default_gravity)

        # 使用 ti.Vector.field 以支持运行时修改
        self.gravity = ti.Vector.field(self.dim, dtype=self.float_type, shape=())
        self.gravity[None] = gravity

        # 重力调度配置
        self.gravity_schedule = config.get("gravity_schedule", None)
        if self.gravity_schedule is not None:
            # 解析 gravity_schedule: [{"frame": 0, "gravity": [0, 0]}, ...]
            # 按帧数排序
            self.gravity_schedule = sorted(self.gravity_schedule, key=lambda x: x["frame"])
            print(f"使用重力调度: {len(self.gravity_schedule)} 个关键帧")

        self.current_frame_field = ti.field(ti.i32, shape=())  # 当前帧数字段
        self.current_frame_field[None] = 0
        
        # 解析体积力配置
        self._parse_volume_forces(config.get("volume_forces", []))

        # 体积力初始化将在调用initialize_volume_forces()时进行

        
        total_grid_points = grid.get_total_grid_points()
        self.v_grad=ti.field(self.float_type, total_grid_points * self.dim,needs_grad=True)
        self.grad_save=ti.field(self.float_type, total_grid_points * self.dim)
        self.total_energy = ti.field(self.float_type, shape=(), needs_grad=True)
        self.grad_fn=None
        if config.get("use_auto_diff", True):
            self.grad_fn=self.compute_energy_grad_auto
        else:
            self.grad_fn=self.compute_energy_grad_manual
        

        solver_type = config.get("implicit_solver", "BFGS")

        self.iter_history = []
        self.max_iter_history = 10000  # 最多保留 10000 次求解记录，防止无界增长

        eta = config.get("eta", 1)
        if solver_type == "BFGS":
            # 使用材料1的p_mass进行梯度归一化
            avg_p_mass = (self.particles.p_mass_1 + self.particles.p_mass_2) / 2.0
            self.optimizer = BFGS(energy_fn=self.compute_energy,grad_fn=self.grad_fn, dim=total_grid_points * self.dim,grad_normalizer=self.dt*avg_p_mass*self.particles.particles_per_grid,eta= eta,float_type=self.float_type)
        elif solver_type == "LBFGS":
            self.optimizer = LBFGS(self.grad_fn, total_grid_points * self.dim, eta=eta,float_type=self.float_type)
        elif solver_type == "Newton":
            # 使用材料1的p_mass进行梯度归一化
            avg_p_mass = (self.particles.p_mass_1 + self.particles.p_mass_2) / 2.0
            self.optimizer = Newton(energy_fn=self.compute_energy, grad_fn=self.grad_fn, hess_fn=self.compute_hess, DBC_fn=None,dim=total_grid_points * self.dim,grad_normalizer=self.dt*avg_p_mass*self.particles.particles_per_grid,eta= eta,float_type=self.float_type)
        elif solver_type == "CG":
            # 使用材料1的p_mass进行梯度归一化
            avg_p_mass = (self.particles.p_mass_1 + self.particles.p_mass_2) / 2.0
            self.optimizer = CG(energy_fn=self.compute_energy, grad_fn=self.grad_fn, hess_fn=self.compute_hess, DBC_fn=None,dim=total_grid_points * self.dim,grad_normalizer=self.dt*avg_p_mass*self.particles.particles_per_grid,eta= eta,float_type=self.float_type)
        elif solver_type == "PCG":
            # 使用材料1的p_mass进行梯度归一化
            avg_p_mass = (self.particles.p_mass_1 + self.particles.p_mass_2) / 2.0
            preconditioner_type = config.get("preconditioner_type", "diagonal")
            self.optimizer = PCG(energy_fn=self.compute_energy, grad_fn=self.grad_fn, hess_fn=self.compute_hess, DBC_fn=None,dim=total_grid_points * self.dim,grad_normalizer=self.dt*avg_p_mass*self.particles.particles_per_grid,eta= eta,float_type=self.float_type,preconditioner_type=preconditioner_type)

        # F矩阵备份，用于临时保存和恢复
        self.F_backup = None

        # 预分配F平均化所需的临时网格字段，避免内存泄漏
        if self.dim == 2:
            self.F_grid = ti.Matrix.field(self.dim, self.dim, self.float_type, shape=(self.grid.nx, self.grid.ny))
            self.m_grid = ti.field(self.float_type, shape=(self.grid.nx, self.grid.ny))
        else:
            self.F_grid = ti.Matrix.field(self.dim, self.dim, self.float_type, shape=(self.grid.size,)*self.dim)
            self.m_grid = ti.field(self.float_type, shape=(self.grid.size,)*self.dim)

    @ti.func
    def get_gravity(self):
        """
        获取当前帧的重力值（在kernel中调用）

        返回:
            当前帧应该使用的重力向量
        """
        # 返回 gravity field 的值
        # 实际的重力值会在 update_frame() 中根据调度更新
        return self.gravity[None]

    def update_frame(self, frame_number):
        """
        更新当前帧数（在Python侧调用）

        参数:
            frame_number: 当前帧数
        """
        self.current_frame_field[None] = frame_number

        # 如果配置了 gravity_schedule，根据当前帧数更新重力
        if self.gravity_schedule is not None:
            # 检查是否有任何调度条目匹配当前帧
            for schedule in self.gravity_schedule:
                if schedule["frame"] == frame_number:
                    # 切换到新的重力值
                    target_gravity = schedule["gravity"]
                    self.gravity[None] = target_gravity
                    print(f"帧 {frame_number}: 重力更新为 {target_gravity}")

    def save_F(self):
        """保存当前F矩阵"""
        import taichi as ti
        if self.F_backup is None:
            # 创建备份字段，与particles.F相同的形状和类型
            self.F_backup = ti.Matrix.field(self.dim, self.dim, self.float_type, shape=self.particles.n_particles)

        # 复制F矩阵到备份
        self.copy_F_to_backup()

    def restore_F(self):
        """恢复F矩阵"""
        if self.F_backup is not None:
            self.copy_backup_to_F()

    @ti.kernel
    def copy_F_to_backup(self):
        """将F矩阵复制到备份"""
        for p in range(self.particles.n_particles):
            self.F_backup[p] = self.particles.F[p]

    @ti.kernel
    def copy_backup_to_F(self):
        """将备份复制回F矩阵"""
        for p in range(self.particles.n_particles):
            self.particles.F[p] = self.F_backup[p]

    def set_unlimited_iter_mode(self, enable=True):
        """设置是否启用最后100帧的无限制迭代模式"""
        self.unlimited_iter_for_last_frames = enable
        if enable:
            print(f"启用最后100帧无限制迭代模式，max_iter: {self.unlimited_max_iter}")
        else:
            print(f"禁用无限制迭代模式，恢复正常max_iter: {self.solve_max_iter}")

    def solve_implicit(self):
        self.grid.set_boundary_v()
        # self.optimizer.resize(self.grid.get_total_grid_points() * self.dim)
        self.set_initial_guess()

        # 根据是否启用最后100帧模式选择max_iter
        max_iter = self.unlimited_max_iter if self.unlimited_iter_for_last_frames else self.solve_max_iter

        iter=self.optimizer.minimize(max_iter, self.solve_init_iter)
        self.iter_history.append(iter)

        # 限制 iter_history 大小，防止无界增长
        if len(self.iter_history) > self.max_iter_history:
            self.iter_history.pop(0)  # FIFO - 删除最旧的记录

        self.update_velocity()
        return iter

    @ti.func
    def get_idx(self, I):
        return self.grid.get_idx(I)

    @ti.func
    def get_compressed_idx(self, I):
        """获取压缩后的索引"""
        compressed_idx = self.grid.get_compressed_index(I)
        return compressed_idx * self.dim

    @ti.func
    def wrap_grid_idx(self, I):
        """处理边界的网格索引包装"""
        result = ti.Vector.zero(ti.i32, self.dim)
        result[0] = I[0] % self.grid.nx
        result[1] = I[1] % self.grid.ny
        if ti.static(self.dim == 3):
            result[2] = I[2] % self.grid.nz
        return result
    
    
    @ti.func
    def get_vel(self,v_flat,vidx):
        vel = ti.Vector.zero(self.float_type, self.dim)
        for d in ti.static(range(self.dim)):
            vel[d] = v_flat[vidx+d]
        return vel
    
    @ti.func
    def get_damping_force(self, I):
        v_prev = self.grid.v_prev[I]
        return -self.damping * v_prev

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
        base, fx = self.grid.particle_to_grid_base_and_fx(self.particles.x[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
            grid_idx = self.wrap_grid_idx(base + offset)
            vidx = self.get_idx(grid_idx)
            vel = self.get_vel(v_flat,vidx)
            vel_grad += 4 * self.dt * vel.outer_product(self.particles.dwip[p, offset])
        return vel_grad
        
    
    @ti.kernel
    def compute_particle_energy_neohookean(self, v_flat: ti.template()):
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
    def compute_particle_energy_linear(self, v_flat: ti.template()):
        for p in range(self.particles.n_particles):
            vel_grad = self.cal_vel_grad(v_flat,p)

            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.dim) + vel_grad) @ F

            # Linear elasticity: strain = 0.5*(F + F^T) - I
            I = ti.Matrix.identity(self.float_type, self.dim)
            eps = 0.5 * (new_F + new_F.transpose()) - I

            mu, lam = self.particles.get_material_params(p)

            # Energy = mu * tr(eps^2) + 0.5 * lambda * tr(eps)^2
            eps_sqr_trace = eps.norm_sqr()
            eps_trace = eps.trace()

            energy = mu * eps_sqr_trace + 0.5 * lam * (eps_trace ** 2)
            weight = self.particles.get_particle_weight(p)
            self.total_energy[None] += energy * self.particles.p_vol * weight

    @ti.kernel
    def compute_grid_energy(self, v_flat: ti.template()):
        if ti.static(self.grid.use_mapping):
            # 使用压缩映射时只遍历有质量的节点
            for compressed_idx in range(self.grid.get_active_grid_count()):
                I = self.grid.get_grid_indices_from_compressed(compressed_idx)
                vidx = compressed_idx * self.dim
                vel = self.get_vel(v_flat, vidx)

                # 惯性项 - 静力学求解时跳过
                if not ti.static(self.static_solve):
                    self.total_energy[None] += 0.5 * self.grid.m[I] * (vel - self.grid.v_prev[I]).norm_sqr()

                # 计算势能
                pos = ti.Vector([
                    I[0] * self.grid.dx_x + self.grid.offset[0],
                    I[1] * self.grid.dx_y + self.grid.offset[1]
                ]) + vel * self.dt
                total_external_force = self.get_gravity() + self.grid.volume_force[I]
                #阻尼力
                if not ti.static(self.static_solve):
                    total_external_force += self.get_damping_force(I)
                self.total_energy[None] -= total_external_force.dot(pos) * self.grid.m[I]
        else:
            # 原始方法：遍历所有节点
            for I in ti.grouped(self.grid.v):
                vidx = self.get_idx(I)
                vel = self.get_vel(v_flat,vidx)

                if self.grid.m[I] == 0.0:
                    continue

                # 惯性项 - 静力学求解时跳过
                if not ti.static(self.static_solve):
                    self.total_energy[None] += 0.5 * self.grid.m[I] * (vel - self.grid.v_prev[I]).norm_sqr()

                # 计算势能
                pos = I * ti.Vector([self.grid.dx_x, self.grid.dx_y]) + self.grid.offset + vel * self.dt
                total_external_force = self.get_gravity() + self.grid.volume_force[I]
                #阻尼力
                if not ti.static(self.static_solve):
                    total_external_force += self.get_damping_force(I)
                self.total_energy[None] -= total_external_force.dot(pos) * self.grid.m[I]


    @ti.kernel
    def manual_particle_energy_grad_neohookean(self, v_flat: ti.template(), grad_flat: ti.template()):
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

            base, fx = self.grid.particle_to_grid_base_and_fx(self.particles.x[p])
            # 将梯度分配到网格点
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = self.wrap_grid_idx(base + offset)
                vidx = self.get_idx(grid_idx)

                grad= 4*self.dt*dE_dvel_grad @ self.particles.dwip[p, offset]

                # 累加梯度
                for d in ti.static(range(self.dim)):
                    ti.atomic_add(grad_flat[vidx+d], grad[d])
    @ti.kernel
    def manual_particle_energy_grad_linear(self, v_flat: ti.template(), grad_flat: ti.template()):
        for p in range(self.particles.n_particles):
            # 计算速度梯度
            vel_grad = self.cal_vel_grad(v_flat,p)

            # 计算变形梯度
            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.dim) + vel_grad) @ F

            # Linear elasticity: strain = 0.5*(F + F^T) - I
            I = ti.Matrix.identity(self.float_type, self.dim)
            eps = 0.5 * (new_F + new_F.transpose()) - I

            mu, lam = self.particles.get_material_params(p)

            # Gradient: dE/dF = mu * (F + F^T - 2*I) + lambda * tr(eps) * I
            eps_trace = eps.trace()

            grad_F = 2 * mu * eps  + lam * eps_trace * I

            weight = self.particles.get_particle_weight(p)
            dE_dvel_grad = grad_F @ F.transpose() * self.particles.p_vol * weight

            base, fx = self.grid.particle_to_grid_base_and_fx(self.particles.x[p])
            # 将梯度分配到网格点
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = self.wrap_grid_idx(base + offset)
                vidx = self.get_idx(grid_idx)

                assert vidx >= 0, "Invalid grid index in gradient computation"
                grad = 4*self.dt*dE_dvel_grad @ self.particles.dwip[p, offset]
                    
                # 累加梯度
                for d in ti.static(range(self.dim)):
                    ti.atomic_add(grad_flat[vidx+d], grad[d])

    @ti.kernel
    def manual_grid_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()):
        if ti.static(self.grid.use_mapping):
            # 使用压缩映射时只遍历有质量的节点
            for compressed_idx in range(self.grid.get_active_grid_count()):
                I = self.grid.get_grid_indices_from_compressed(compressed_idx)
                vidx = compressed_idx * self.dim
                vel = self.get_vel(v_flat, vidx)

                # 初始化梯度
                grad = ti.Vector.zero(self.float_type, self.dim)

                # 惯性项梯度 - 静力学求解时跳过
                if not ti.static(self.static_solve):
                    grad += self.grid.m[I] * (vel - self.grid.v_prev[I])

                # 计算外部力（重力 + 体积力）
                total_external_force = self.get_gravity() + self.grid.volume_force[I]
                # 阻尼力
                if not ti.static(self.static_solve):
                    total_external_force += self.get_damping_force(I)
                grad -= total_external_force * self.dt * self.grid.m[I]

                # 写入梯度
                for d in ti.static(range(self.dim)):
                    ti.atomic_add(grad_flat[vidx+d], grad[d])
        else:
            # 原始方法：遍历所有节点
            for I in ti.grouped(self.grid.v):
                vidx = self.get_idx(I)
                vel = self.get_vel(v_flat,vidx)

                # 初始化梯度
                grad = ti.Vector.zero(self.float_type, self.dim)

                if self.grid.m[I] == 0.0:
                    for d in ti.static(range(self.dim)):
                        grad_flat[vidx+d]= 0.0
                    continue

                # 惯性项梯度 - 静力学求解时跳过
                if not ti.static(self.static_solve):
                    grad += self.grid.m[I] * (vel - self.grid.v_prev[I])

                # 计算外部力（重力 + 体积力）
                total_external_force = self.get_gravity() + self.grid.volume_force[I]
                # 阻尼力
                if not ti.static(self.static_solve):
                    total_external_force += self.get_damping_force(I)
                grad -= total_external_force * self.dt * self.grid.m[I]

                # 累加梯度
                for d in ti.static(range(self.dim)):
                    ti.atomic_add(grad_flat[vidx+d], grad[d])

    @ti.kernel
    def manual_particle_energy_hess_neohookean_1(self,v_flat: ti.template(),hess: ti.sparse_matrix_builder()):

        for p in range(self.particles.n_particles):
            # 计算速度梯度
            vel_grad = self.cal_vel_grad(v_flat,p)


            # 计算变形梯度导数
            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.dim) + vel_grad) @ F
            J = new_F.determinant()
            F_inv = new_F.inverse()
            F_inv_T = F_inv.transpose()
            
            base, fx = self.grid.particle_to_grid_base_and_fx(self.particles.x[p]) 

            mu, lam = self.particles.get_material_params(p)

            weight = self.particles.get_particle_weight(p)

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                FTw=4 * self.dt * F.transpose() @ self.particles.dwip[p, offset]
                FWWF=F_inv_T @ FTw.outer_product(FTw) @ F_inv
                h1=mu *FTw.dot(FTw) * ti.Matrix.identity(self.float_type, self.dim)
                h2=mu * FWWF
                h3=lam * (1-ti.log(J)) * FWWF
                hessian=(h1+h2+h3)*self.particles.p_vol * weight

                # hessian=self.make_PSD(hessian)

                grid_idx = self.wrap_grid_idx(base + offset)
                vidx = self.get_idx(grid_idx)

                for (i,j) in ti.static(ti.ndrange(self.dim, self.dim)):
                    if not self.grid.is_boundary_grid[grid_idx][i] and not self.grid.is_boundary_grid[grid_idx][j]:
                        hess[vidx+i, vidx+j] += hessian[i,j]
    @ti.kernel
    def manual_particle_energy_hess_neohookean(self,v_flat: ti.template(),hess: ti.sparse_matrix_builder()):

        for p in range(self.particles.n_particles):
            # 计算速度梯度
            vel_grad = self.cal_vel_grad(v_flat,p)

            # 计算变形梯度
            F = self.particles.F[p]
            new_F = (ti.Matrix.identity(self.float_type, self.dim) + vel_grad) @ F
            
            # 对new_F进行SVD分解 - 使用与NeoHookean.py相同的polar_svd逻辑
            U, sigma_mat, V = ti.svd(new_F)

            VT = V.transpose()
            
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
            # d2Psi_dsigma2_psd = d2Psi_dsigma2
            d2Psi_dsigma2_00 = d2Psi_dsigma2_psd[0, 0]
            d2Psi_dsigma2_11 = d2Psi_dsigma2_psd[1, 1]
            d2Psi_dsigma2_01 = d2Psi_dsigma2_psd[0, 1]
            
            # 计算B矩阵 - 按照C++参考实现
            sigma_prod = sigma[0] * sigma[1]
            middle = mu - lam * ti.log(sigma_prod)
            BLeftCoef = (mu + middle / (sigma[0] * sigma[1])) / 2.0

            # 计算dPsi_div_dsigma
            inv0 = 1.0 / sigma[0]
            dPsi_dsigma_0 = mu * (sigma[0] - inv0) + lam * inv0 * ln_sigma_prod
            inv1 = 1.0 / sigma[1]
            dPsi_dsigma_1 = mu * (sigma[1] - inv1) + lam * inv1 * ln_sigma_prod

            # 计算rightCoef - 按照C++参考实现
            rightCoef = (dPsi_dsigma_0 + dPsi_dsigma_1)
            sum_sigma = sigma[0] + sigma[1]
            eps = 1.0e-6
            if sum_sigma < eps:
                rightCoef /= 2 * eps
            else:
                rightCoef /= 2 * sum_sigma

            # 构建2x2 B矩阵并使用make_PSD - 按照C++参考实现
            B_matrix = ti.Matrix([[BLeftCoef + rightCoef, BLeftCoef - rightCoef],
                                  [BLeftCoef - rightCoef, BLeftCoef + rightCoef]])
            
            # 使用make_PSD函数确保正定性
            B_psd = self.make_PSD(B_matrix)
            # B_psd = B_matrix
            B_00 = B_psd[0, 0]
            B_01 = B_psd[0, 1]
            B_11 = B_psd[1, 1]
            
            # 构建M矩阵 (4x4)
            M = ti.Matrix.zero(self.float_type, 4, 4)
            M[0, 0] = d2Psi_dsigma2_00
            M[0, 2] = d2Psi_dsigma2_01
            M[1, 1] = B_00
            M[1, 3] = B_01
            M[3, 1] = B_01
            M[3, 3] = B_11
            M[2, 0] = d2Psi_dsigma2_01
            M[2, 2] = d2Psi_dsigma2_11
            
            base, fx = self.grid.particle_to_grid_base_and_fx(self.particles.x[p])
            
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
                                          + M[0, 2] * U[i, 0] * VT[0, j] * U[r, 1] * VT[1, s] \
                                          + M[2, 0] * U[i, 1] * VT[1, j] * U[r, 0] * VT[0, s] \
                                          + M[2, 2] * U[i, 1] * VT[1, j] * U[r, 1] * VT[1, s] \
                                          + M[1, 1] * U[i, 0] * VT[1, j] * U[r, 0] * VT[1, s] \
                                          + M[1, 3] * U[i, 0] * VT[1, j] * U[r, 1] * VT[0, s] \
                                          + M[3, 1] * U[i, 1] * VT[0, j] * U[r, 0] * VT[1, s] \
                                          + M[3, 3] * U[i, 1] * VT[0, j] * U[r, 1] * VT[0, s]

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = self.wrap_grid_idx(base + offset)
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
                        if vidx >= 0 and not self.grid.is_boundary_grid[grid_idx][vi] and not self.grid.is_boundary_grid[grid_idx][vj]:
                            # 添加边界检查和调试信息
                            row_idx = vidx + vi
                            col_idx = vidx + vj
                            if row_idx >= 0 and col_idx >= 0:  # 基本有效性检查
                                hess[row_idx, col_idx] += hess_val

    @ti.kernel
    def manual_particle_energy_hess_linear(self,hess: ti.sparse_matrix_builder()):
        for p in range(self.particles.n_particles):
            # 计算变形梯度导数
            F = self.particles.F[p]

            base, fx = self.grid.particle_to_grid_base_and_fx(self.particles.x[p])

            mu, lam = self.particles.get_material_params(p)
            weight = self.particles.get_particle_weight(p)

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                FTw = 4 * self.dt * F.transpose() @ self.particles.dwip[p, offset]
                FTw_norm_sqr = FTw.norm_sqr()
                # Simplified linear elasticity Hessian: mu*(I+I) + lambda*outer_product(I)
                # For linear elasticity, the Hessian is constant
                hessian = ti.Matrix.zero(self.float_type, self.dim, self.dim)
                for i in ti.static(range(self.dim)):
                    for j in ti.static(range(self.dim)):
                        if i == j:
                            hessian[i, j] = mu * FTw_norm_sqr
                        hessian[i, j] += (lam + mu) * FTw[i] * FTw[j]

                hessian *= self.particles.p_vol * weight
                grid_idx = self.wrap_grid_idx(base + offset)
                vidx = self.get_idx(grid_idx)

                # 添加边界检查
                for (i,j) in ti.static(ti.ndrange(self.dim, self.dim)):
                    if not self.grid.is_boundary_grid[grid_idx][i] and not self.grid.is_boundary_grid[grid_idx][j]:
                        matrix_i = vidx + i
                        matrix_j = vidx + j
                        hess_value = hessian[i,j]
                        hess[matrix_i, matrix_j] += hess_value
    @ti.kernel
    def manual_grid_energy_hess(self, hess: ti.sparse_matrix_builder()):
        if ti.static(self.grid.use_mapping):
            # 使用压缩映射时只遍历有质量的节点
            for compressed_idx in range(self.grid.get_active_grid_count()):
                I = self.grid.get_grid_indices_from_compressed(compressed_idx)
                vidx = compressed_idx * self.dim
                m_node = self.grid.m[I]

                # 惯性项 Hessian: m * I (不需要检查m_node==0，因为压缩映射只包含有质量节点)
                for d in ti.static(range(self.dim)):
                    matrix_idx = vidx + d
                    if not ti.static(self.static_solve) and not self.grid.is_boundary_grid[I][d]:
                        hess[matrix_idx, matrix_idx] += m_node
                    elif self.grid.is_boundary_grid[I][d]:
                        # 边界节点固定：加一个单位刚度
                        hess[matrix_idx, matrix_idx] += 1.0

        else:
            # 原始方法：遍历所有节点
            for I in ti.grouped(self.grid.v):
                vidx = self.get_idx(I)
                m_node = self.grid.m[I]

                if m_node == 0.0:
                    # 空节点：没有物理能量贡献，但为了数值稳定加一个单位刚度
                    for d in ti.static(range(self.dim)):
                       hess[vidx + d, vidx + d] += 1.0
                    continue

                # 惯性项 Hessian: m * I
                for d in ti.static(range(self.dim)):
                    if not ti.static(self.static_solve) and not self.grid.is_boundary_grid[I][d]:
                        hess[vidx + d, vidx + d] += m_node
                    elif self.grid.is_boundary_grid[I][d]:
                        # 边界节点固定：加一个单位刚度
                        hess[vidx + d, vidx + d] += 1.0


    def compute_energy(self, v_flat: ti.template()):
        """统一的能量计算方法，自动根据配置选择实现"""
        self.total_energy[None] = 0.0
        self.compute_particle_energy(v_flat)
        self.compute_grid_energy(v_flat)
        return self.total_energy[None]
    
    def compute_particle_energy(self, v_flat: ti.template()):
        if self.elasticity_model == "linear":
            self.compute_particle_energy_linear(v_flat)
        else:  # default to neohookean
            self.compute_particle_energy_neohookean(v_flat)
    
    def compute_energy_grad_manual(self, v_flat: ti.template(), grad_flat: ti.template()):

        grad_flat.fill(0.0)
        self.manual_particle_energy_grad(v_flat, grad_flat)
        self.manual_grid_energy_grad(v_flat, grad_flat)
        self.grid.set_boundary_grad(grad_flat)

        # return self.total_energy[None]

    def compute_energy_grad_auto(self, v_flat: ti.template(), grad_flat: ti.template()):
        @ti.kernel
        def copy_field(a: ti.template(),b: ti.template()):
            for I in ti.grouped(a):
                a[I] = b[I]  
        
        copy_field(self.v_grad,v_flat)
        
        grad_flat.fill(0.0)

        with ti.ad.Tape(loss=self.total_energy):
            self.compute_particle_energy(self.v_grad)
            self.compute_grid_energy(self.v_grad)

        copy_field(grad_flat,self.v_grad.grad)

        self.grid.set_boundary_grad(grad_flat)

        return self.total_energy[None]  
    

    def manual_particle_energy_grad(self, v_flat: ti.template(), grad_flat: ti.template()):
        if self.elasticity_model == "linear":
            self.manual_particle_energy_grad_linear(v_flat, grad_flat)
        else:  # default to neohookean
            self.manual_particle_energy_grad_neohookean(v_flat, grad_flat)

    def compute_hess(self, v_flat: ti.template(), hess: ti.sparse_matrix_builder()):
        
        self.manual_particle_energy_hess(v_flat, hess)
        self.manual_grid_energy_hess(hess)
        
    def manual_particle_energy_hess(self, v_flat: ti.template(), hess: ti.sparse_matrix_builder()):
        if self.elasticity_model == "linear":
            self.manual_particle_energy_hess_linear(hess)
        else:  # default to neohookean
            self.manual_particle_energy_hess_neohookean(v_flat, hess)



    def save_previous_velocity(self):
        self.grid.v_prev.copy_from(self.grid.v)

    @ti.kernel
    def set_initial_guess(self):
        if ti.static(self.grid.use_mapping):
            # 使用压缩映射时只设置有质量的节点
            for compressed_idx in range(self.grid.get_active_grid_count()):
                I = self.grid.get_grid_indices_from_compressed(compressed_idx)
                vidx = compressed_idx * self.dim
                for d in ti.static(range(self.dim)):
                    self.optimizer.x[vidx+d] = self.grid.v[I][d]
        else:
            # 原始方法：遍历所有节点
            for I in ti.grouped(self.grid.v):
                idx = self.grid.get_idx(I)
                for d in ti.static(range(self.dim)):
                    self.optimizer.x[idx+d] = self.grid.v[I][d]
                    
    @ti.kernel
    def update_velocity(self):
        if ti.static(self.grid.use_mapping):
            # 使用压缩映射时只更新有质量的节点
            for compressed_idx in range(self.grid.get_active_grid_count()):
                I = self.grid.get_grid_indices_from_compressed(compressed_idx)
                vidx = compressed_idx * self.dim
                for d in ti.static(range(self.dim)):
                    self.grid.v[I][d] = self.optimizer.x[vidx+d]
        else:
            # 原始方法：遍历所有节点
            for I in ti.grouped(self.grid.v):
                idx = self.grid.get_idx(I)
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

        # 记录每个体积力对应的实际粒子总质量
        self.volume_force_total_mass = ti.field(self.float_type, shape=field_size)
        # 记录每个体积力对应的理论质量（区域面积/体积 × 密度）
        self.volume_force_theoretical_mass = ti.field(self.float_type, shape=field_size)
        
        # 填充数据
        if self.n_volume_forces > 0:
            for i, force_config in enumerate(volume_forces_config):
                force_type = force_config.get("type", "rectangle")
                force_vector = force_config.get("force", [0.0] * self.dim)
                params = force_config.get("params", {})
                # 获取密度（默认使用第一个材料的密度）
                density = force_config.get("density", self.particles.material_params[0]['rho'] if self.particles.material_params else 1000.0)

                # 设置力向量
                self.volume_force_vectors[i] = ti.Vector(force_vector)
                
                if force_type == "rectangle":
                    self.volume_force_types[i] = 0
                    force_range = params.get("range", [])

                    # 计算矩形区域的面积/体积和理论质量
                    area_volume = 1.0
                    for d in range(min(self.dim, len(force_range))):
                        if len(force_range[d]) >= 2:
                            self.volume_force_rect_ranges[i, d*2] = force_range[d][0]    # min
                            self.volume_force_rect_ranges[i, d*2+1] = force_range[d][1]  # max
                            area_volume *= (force_range[d][1] - force_range[d][0])

                    self.volume_force_theoretical_mass[i] = area_volume * density
                            
                elif force_type == "ellipse":
                    self.volume_force_types[i] = 1
                    center = params.get("center", [0.0] * self.dim)
                    semi_axes = params.get("semi_axes", [1.0] * self.dim)
                    for d in range(min(self.dim, len(center))):
                        self.volume_force_ellipse_params[i, d] = center[d]
                    for d in range(min(self.dim, len(semi_axes))):
                        self.volume_force_ellipse_params[i, self.dim + d] = semi_axes[d]

                    # 计算椭圆区域的面积/体积和理论质量
                    import math
                    if self.dim == 2:
                        # 2D椭圆面积: π * a * b
                        area_volume = math.pi * semi_axes[0] * semi_axes[1]
                    elif self.dim == 3:
                        # 3D椭球体积: (4/3) * π * a * b * c
                        area_volume = (4.0/3.0) * math.pi * semi_axes[0] * semi_axes[1] * semi_axes[2]
                    else:
                        area_volume = 1.0  # 默认值

                    self.volume_force_theoretical_mass[i] = area_volume * density
    

    @ti.kernel
    def _mark_particles_with_volume_forces(self):
        """标记所有在体积力区域内的粒子并记录每个体积力的粒子总质量"""
        # 初始化每个体积力的总质量为0
        for i in range(self.n_volume_forces):
            self.volume_force_total_mass[i] = 0.0

        # 遍历所有粒子，标记体积力并累计质量
        for p in range(self.particles.n_particles):
            total_force = ti.Vector.zero(self.float_type, self.dim)
            pos = self.particles.x[p]

            # 根据材料ID获取粒子质量
            particle_mass = 0.0
            if self.particles.particle_material_id[p] == 0:
                particle_mass = self.particles.p_mass_1
            else:
                particle_mass = self.particles.p_mass_2

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
                    self.volume_force_total_mass[i] += particle_mass

            self.particles.volume_force[p] = total_force

    @ti.kernel
    def _apply_mass_normalization(self):
        """基于理论质量与实际质量的比值对体积力进行归一化"""
        # 计算并打印质量信息（仅用于调试）
        for i in range(self.n_volume_forces):
            theoretical_mass = self.volume_force_theoretical_mass[i]
            actual_mass = self.volume_force_total_mass[i]
            if actual_mass > 1e-12:  # 避免除零
                mass_ratio = theoretical_mass / actual_mass
                print(f"Volume force {i}: theoretical_mass={theoretical_mass:.6e}, actual_mass={actual_mass:.6e}, ratio={mass_ratio:.6f}")
            else:
                print(f"Volume force {i}: No particles in region")

        # 对每个粒子的体积力进行归一化
        for p in range(self.particles.n_particles):
            pos = self.particles.x[p]
            normalized_force = ti.Vector.zero(self.float_type, self.dim)

            # 检查粒子在哪些体积力区域内，并应用相应的归一化
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
                    theoretical_mass = self.volume_force_theoretical_mass[i]
                    actual_mass = self.volume_force_total_mass[i]

                    if actual_mass > 1e-12:  # 避免除零
                        mass_ratio = theoretical_mass / actual_mass
                        # 应用质量比归一化
                        normalized_force += self.volume_force_vectors[i] * mass_ratio
                    else:
                        # 如果没有实际质量，直接使用原始力
                        normalized_force += self.volume_force_vectors[i]

            self.particles.volume_force[p] = normalized_force

    def initialize_volume_forces(self):
        """初始化粒子体积力（在粒子数据生成后调用）"""
        if self.n_volume_forces > 0:
            self._mark_particles_with_volume_forces()
            self._apply_mass_normalization()

    def get_volume_force_masses(self):
        """获取每个体积力对应的粒子总质量"""
        if self.n_volume_forces > 0:
            masses = []
            for i in range(self.n_volume_forces):
                masses.append(self.volume_force_total_mass[i])
            return masses
        return []

    # =============== Explicit Solver Methods ===============
    def solve_explicit(self):
        """显式求解器"""
        self.compute_forces_explicit()
        self.update_velocity_explicit()
        self.grid.set_boundary_v()
        return 0
    
    @ti.func
    def compute_stress_neohookean(self, p):
        """计算Neo-Hookean模型的应力"""
        U, sig, V = ti.svd(self.particles.F[p])
        if sig.determinant() < 0:
            sig[1,1] = -sig[1,1]
        self.particles.F[p] = U @ sig @ V.transpose()

        J = self.particles.F[p].determinant()
        logJ = ti.log(J)
        mu, lam = self.particles.get_material_params(p)
        cauchy = mu * (self.particles.F[p] @ self.particles.F[p].transpose()) + ti.Matrix.identity(self.float_type, self.dim) * (lam * logJ - mu)
        stress = -(self.particles.p_vol) * cauchy
        return stress

    @ti.func
    def compute_stress_linear(self, p):
        """计算线性弹性模型的应力"""
        F = self.particles.F[p]
        I = ti.Matrix.identity(self.float_type, self.dim)
        eps = 0.5 * (F + F.transpose()) - I

        mu, lam = self.particles.get_material_params(p)
        eps_trace = eps.trace()

        # Cauchy stress: sigma = 2*mu*eps + lambda*tr(eps)*I
        cauchy = 2 * mu * eps + lam * eps_trace * I
        stress = -(self.particles.p_vol) * cauchy
        return stress

    @ti.func
    def compute_stress(self, p):
        """根据弹性模型计算应力"""
        if ti.static(self.elasticity_model == "linear"):
            return self.compute_stress_linear(p)
        else:  # default to neohookean
            return self.compute_stress_neohookean(p)

    @ti.kernel
    def compute_forces_explicit(self):
        """计算所有力并存储到网格力场中"""
        # 初始化网格力场
        for I in ti.grouped(self.grid.f):
            self.grid.f[I] = ti.Vector.zero(self.float_type, self.dim)

        # 第一步：计算应力力并分配到网格
        for p in range(self.particles.n_particles):
            base , fx = self.grid.particle_to_grid_base_and_fx(self.particles.x[p])

            stress = self.compute_stress(p)

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = base + offset
                # 将应力力累加到网格力场
                ti.atomic_add(self.grid.f[grid_idx], 4 * stress @ self.particles.dwip[p,offset])
        
        # 第二步：在网格上计算外部力（重力、体积力、阻尼力）
        for I in ti.grouped(self.grid.f):
            if self.grid.m[I] > 1e-10:
                # 添加重力和体积力
                total_external_force = self.get_gravity() + self.grid.volume_force[I]
                self.grid.f[I] += total_external_force * self.grid.m[I]
                
                # 添加阻尼力（基于当前速度）
                damping_force = -self.damping * self.grid.v[I] / self.grid.v[I].norm() * self.grid.f[I].norm() if self.grid.v[I].norm() > 1e-8 else ti.Vector.zero(self.float_type, self.dim)
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
            base, fx = self.grid.particle_to_grid_base_and_fx(self.particles.x[p])
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = self.wrap_grid_idx(base + offset)
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
            base, fx = self.grid.particle_to_grid_base_and_fx(self.particles.x[p])
            new_F = ti.Matrix.zero(self.float_type, self.dim, self.dim)
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = self.wrap_grid_idx(base + offset)
                weight = self.particles.wip[p, offset]
                new_F += weight * F_grid[grid_idx]
            
            self.particles.F[p] = new_F

    def average_F_p2g_g2p(self):
        """通过p2g和g2p操作对F[p]进行平均"""
        # 重用预分配的网格字段（避免每次调用都创建新fields导致内存泄漏）
        # 执行P2G和G2P
        self.p2g_F_averaging(self.F_grid, self.m_grid)
        self.g2p_F_averaging(self.F_grid)

    @ti.func
    def compute_stress_strain_neohookean(self, p):
        """计算Neo-Hookean模型的应力和应变"""
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

    @ti.func
    def compute_stress_strain_linear(self, p):
        """计算线性弹性模型的应力和应变"""
        F = self.particles.F[p]
        I = ti.Matrix.identity(self.float_type, self.dim)

        # 线性应变: eps = 0.5*(F + F^T) - I
        eps = 0.5 * (F + F.transpose()) - I

        mu, lam = self.particles.get_material_params(p)
        eps_trace = eps.trace()

        # Cauchy stress: sigma = 2*mu*eps + lambda*tr(eps)*I
        cauchy = 2 * mu * eps + lam * eps_trace * I

        # 存储应力和应变
        self.particles.stress[p] = cauchy
        self.particles.strain[p] = eps

    @ti.kernel
    def compute_stress_strain(self):
        """计算并存储所有粒子的应力和应变（仅在需要时调用）"""
        for p in range(self.particles.n_particles):
            if ti.static(self.elasticity_model == "linear"):
                self.compute_stress_strain_linear(p)
            else:  # default to neohookean
                self.compute_stress_strain_neohookean(p)

    def compute_stress_strain_with_averaging(self):
        """计算应力应变前先对F[p]进行平均"""
        # 保存原始F矩阵
        self.save_F()

        # 首先对F[p]进行一次p2g，g2p进行平均
        self.average_F_p2g_g2p()

        # 然后计算应力应变
        self.compute_stress_strain()

        # 恢复原始F矩阵
        self.restore_F()

    def get_grid_stress_from_F_grid_numpy(self):
        """基于当前F_grid/m_grid计算并返回网格Cauchy应力（numpy格式）"""
        F_grid = self.F_grid.to_numpy()
        m_grid = self.m_grid.to_numpy()

        # 默认使用material_id=0参数
        mat0 = self.particles.material_params.get(0, {})
        mu = float(mat0.get("mu", getattr(self.particles, "mu_1", self.mu)))
        lam = float(mat0.get("lambda", getattr(self.particles, "lam_1", self.lam)))

        stress_grid = np.zeros_like(F_grid)
        valid_mask = m_grid > 1e-10
        invalid_j_count = 0

        if np.any(valid_mask):
            F_valid = F_grid[valid_mask]  # (N, dim, dim)
            I = np.eye(self.dim, dtype=F_valid.dtype)

            if self.elasticity_model == "linear":
                eps = 0.5 * (F_valid + np.transpose(F_valid, (0, 2, 1))) - I
                eps_trace = np.trace(eps, axis1=1, axis2=2)
                sigma_valid = 2.0 * mu * eps + lam * eps_trace[:, None, None] * I
                stress_grid[valid_mask] = sigma_valid
            else:
                det_F = np.linalg.det(F_valid)
                positive_j_mask = det_F > 0.0
                invalid_j_count = int(np.count_nonzero(~positive_j_mask))

                sigma_valid = np.zeros_like(F_valid)
                if np.any(positive_j_mask):
                    F_pos = F_valid[positive_j_mask]
                    det_pos = det_F[positive_j_mask]
                    B = np.matmul(F_pos, np.transpose(F_pos, (0, 2, 1)))
                    sigma_pos = mu * B + (lam * np.log(det_pos) - mu)[:, None, None] * I
                    sigma_valid[positive_j_mask] = sigma_pos

                stress_grid[valid_mask] = sigma_valid

        offset = self.grid.offset
        offset_meta = [float(offset[d]) for d in range(self.dim)]

        meta = {
            "dim": int(self.dim),
            "nx": int(self.grid.nx),
            "ny": int(self.grid.ny),
            "offset": offset_meta,
            "dx_x": float(self.grid.dx_x),
            "dx_y": float(self.grid.dx_y),
            "material_id_used": 0,
            "mu": float(mu),
            "lambda": float(lam),
            "elasticity_model": self.elasticity_model,
            "valid_mass_threshold": 1e-10,
            "valid_grid_count": int(np.count_nonzero(valid_mask)),
            "invalid_j_count": int(invalid_j_count),
        }

        if self.dim == 3:
            meta["nz"] = int(self.grid.nz)
            meta["dx_z"] = float(self.grid.dx_z)

        return stress_grid, m_grid, meta
