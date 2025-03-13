import taichi as ti
import json
from Optimizer.BFGS import BFGS
from Optimizer.LBFGS import LBFGS

ti.init(arch=ti.vulkan,random_seed=114514)

@ti.data_oriented
class ImplicitMPM:
    def __init__(self, config_path):
        # 从配置文件加载参数
        with open(config_path) as f:
            self.cfg = json.load(f)
        
        # 初始化仿真参数
        self.dim = 2
        self.grid_size = self.cfg.get("grid_size", 16)
        self.dx = 1.0 / self.grid_size
        self.inv_dx = self.grid_size
        self.dt = self.cfg.get("dt", 2e-3)
        self.max_iter = self.cfg.get("max_iter", 1)
        self.implicit = self.cfg.get("implicit", True)
        
        # 材料参数
        self.E = self.cfg.get("E", 4)
        self.nu = self.cfg.get("nu", 0.4)
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.neighbor = (3,) * self.dim

        # 粒子参数
        self.p_rho = self.cfg.get("p_rho", 1)
        self.p_vol = self.dx**self.dim
        self.p_mass = self.p_vol * self.p_rho
        self.bound = self.cfg.get("bound", 2)
        self.init_vel_y = self.cfg.get("initial_velocity_y", -1)
        self.init_pos_range = self.cfg.get("initial_position_range", [0.3, 0.6])

        # 初始化场
        self.grid_v = ti.Vector.field(self.dim, ti.f32, (self.grid_size,)*2)
        self.grid_m = ti.field(ti.f32, (self.grid_size,)*2)
        self.grid_v_prev = ti.Vector.field(self.dim, ti.f32, (self.grid_size,)*2)
        
        # 粒子数据
        self.n_particles = self.grid_size**self.dim // 2**(self.dim - 1)
        self.x = ti.Vector.field(self.dim, ti.f32, self.n_particles)
        self.v = ti.Vector.field(self.dim, ti.f32, self.n_particles)
        self.C = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        self.F = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        self.temp_P = ti.Matrix.field(2, 2, ti.f32, self.n_particles)
        
        # # 隐式求解相关初始化
        # if self.implicit:
        self.grid_particles = ti.field(ti.i32, (self.grid_size, self.grid_size, 32))
        self.grid_wip = ti.field(ti.f32, (self.grid_size, self.grid_size, 32))
        self.grid_dwip = ti.Vector.field(2, ti.f32, (self.grid_size, self.grid_size, 32))
        self.grid_particles_count = ti.field(ti.i32, (self.grid_size, self.grid_size))
        if self.cfg.get("implicit_solver", "BFGS") == "BFGS":
            self.bfgs_solver = BFGS(self.compute_energy_grad, self.grid_size**2*self.dim)
        else:
            self.bfgs_solver = LBFGS(self.compute_energy_grad, self.grid_size**2*self.dim)

        self.init_particles()

    @ti.kernel
    def init_particles(self):
        #set random seed
        for p in range(self.n_particles):
            pos = ti.Vector.zero(ti.f32, self.dim)
            for d in ti.static(range(self.dim)):
                # 使用配置的初始位置范围
                pos[d] = ti.random() * (self.init_pos_range[1]-self.init_pos_range[0]) + self.init_pos_range[0]
            self.x[p] = pos
            self.v[p]= ti.Vector.zero(ti.f32, self.dim)
            self.v[p][1] = -1
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            
        # 初始化网格
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            self.grid_m[i,j] = 0.0
            self.grid_v[i,j] = [0.0, 0.0]
            self.grid_v_prev[i,j] = [0.0, 0.0]
            self.grid_particles_count[i,j] = 0

    @ti.kernel
    def build_neighbor_list(self):
        # 清空邻居列表
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            self.grid_particles_count[i,j] = 0
            
        # 构建粒子-网格邻居关系
        for p in range(self.n_particles):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = (base + offset) % self.grid_size
                cnt = ti.atomic_add(self.grid_particles_count[grid_idx], 1)
                if cnt < 32:
                    self.grid_particles[grid_idx.x, grid_idx.y, cnt] = p
                    weight = 1.0
                    for i in ti.static(range(self.dim)):
                        weight *= w[offset[i]][i]
                    dpos = (offset - fx) * self.inv_dx
                    self.grid_dwip[grid_idx.x, grid_idx.y, cnt] = dpos * weight

    @ti.kernel
    def compute_energy_grad(self, v_flatened: ti.template(), grad_flatened: ti.template()) -> ti.f32:
        total_energy = 0.0
        grad_flatened.fill(0.0)
        
        # 阶段1: 计算粒子的变形梯度和应力
        for p in range(self.n_particles):
            # 计算速度梯度（使用MLS）
            vel_grad = ti.Matrix.zero(ti.f32, 2, 2)
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = (base + offset) % self.grid_size
                weight = 1.0
                dpos = (offset - fx) *self.inv_dx   
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                
                vel = ti.Vector([v_flatened[grid_idx.x*self.grid_size*2 + grid_idx.y*2],
                 v_flatened[grid_idx.x*self.grid_size*2 + grid_idx.y*2 + 1]])
                vel_grad += 4*self.dt * vel.outer_product(dpos) * weight 
            
            # 更新变形梯度并处理反转
            new_F = (ti.Matrix.identity(ti.f32, self.dim) + vel_grad) @ self.F[p]
            J = new_F.determinant()
            logJ = ti.log(J)
            cauchy = self.mu * (new_F @ new_F.transpose()) + ti.Matrix.identity(float, self.dim) * (self.lam * logJ - self.mu)
            self.temp_P[p] = (self.p_vol * 4 ) * cauchy
            
            # 能量计算
            energy_density = 0.5 * self.mu * (new_F.norm_sqr() - self.dim) - self.mu * logJ
            energy_density += 0.5*self.lam*logJ**2
            total_energy += energy_density * self.p_vol
            
        # 阶段2: 计算网格梯度
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            if self.grid_m[i,j] == 0:
                continue
                
            # 动量梯度项
            momentum_grad = self.grid_m[i,j] * (ti.Vector([v_flatened[i*self.grid_size*2 + j*2]-self.grid_v_prev[i,j][0], 
                                                           v_flatened[i*self.grid_size*2 + j*2 + 1]-self.grid_v_prev[i,j][1]]))
            
            # 能量梯度项
            energy_grad = ti.Vector.zero(ti.f32, 2)
            for k in range(self.grid_particles_count[i,j]):
                p = self.grid_particles[i,j,k]
                energy_grad += self.temp_P[p] @ self.grid_dwip[i,j,k] *self.dt 
            
            grad_flatened[i*self.grid_size*2 + j*2] = momentum_grad[0] + energy_grad[0]
            grad_flatened[i*self.grid_size*2 + j*2 + 1] = momentum_grad[1] + energy_grad[1]
            total_energy += 0.5 * self.grid_m[i,j] * (ti.Vector([v_flatened[i*self.grid_size*2 + j*2]-self.grid_v_prev[i,j][0],
                                                                 v_flatened[i*self.grid_size*2 + j*2 + 1]-self.grid_v_prev[i,j][1]])).norm_sqr()
        
        return total_energy
    
    @ti.kernel
    def set_init_velocity(self, v_flatened: ti.template()):
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            v_flatened[i*self.grid_size*2 + j*2] = self.grid_v[i,j][0]
            v_flatened[i*self.grid_size*2 + j*2 + 1] = self.grid_v[i,j][1]

    def solve_implicit(self):
        self.build_neighbor_list()
        
        # 保存初始速度
        self.grid_v_prev.copy_from(self.grid_v)

        # 初始化速度场
        self.set_init_velocity(self.bfgs_solver.x)
    
        #self.bfgs_solver.check_gradient()
        # 使用L-BFGS求解
        self.bfgs_solver.minimize()
        @ti.kernel
        def update_velocity():
            for i, j in ti.ndrange(self.grid_size, self.grid_size):
                self.grid_v[i,j] = [self.bfgs_solver.x[i*self.grid_size*2 + j*2],
                                    self.bfgs_solver.x[i*self.grid_size*2 + j*2 + 1]]
        
        update_velocity()

    @ti.kernel
    def grid_boundary_conditions(self):
        for I in ti.grouped(self.grid_v):
            cond = (I < self.bound) & (self.grid_v[I] < 0) | (I > self.grid_size - self.bound) & (self.grid_v[I] > 0)
            self.grid_v[I] = ti.select(cond, 0, self.grid_v[I])
        
    @ti.kernel
    def P2G(self):
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            self.grid_m[i,j] = 0.0
            self.grid_v[i,j] = [0.0, 0.0]
            
        for p in range(self.n_particles):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
            
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                dpos = (offset - fx) * self.dx
                grid_idx = (base + offset) % self.grid_size
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.grid_m[grid_idx] += weight * self.p_mass
                self.grid_v[grid_idx] += weight * self.p_mass * (self.v[p] + self.C[p] @ dpos)
        
        # 速度场归一化
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            if self.grid_m[i,j] > 0:
                self.grid_v[i,j] /= self.grid_m[i,j]

    @ti.kernel
    def G2P(self):
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
                g_v = self.grid_v[base + offset] 
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx**2
                
            self.v[p] = new_v
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.dt * self.C[p]) @ self.F[p]
            self.C[p] = new_C
    
    @ti.kernel
    def advection(self):
        for p in self.x:
            pass
            self.x[p] += self.dt * self.v[p]
            # self.x[p] += ti.Vector([0,self.dt])
    @ti.kernel
    def solve_explicit(self):
        for p in self.x:
            Xp = self.x[p] / self.dx
            base = (Xp - 0.5).cast(int)
            fx = Xp - base.cast(float)
            
            # 三次B样条权重
            w =[0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            

            U, sig, V = ti.svd(self.F[p])
            if sig.determinant() < 0:
                sig[1,1] = -sig[1,1]
            self.F[p] = U @ sig @ V.transpose()
            
            # 计算应力 (Neo-Hookean)
            J = self.F[p].determinant()
            cauchy = self.mu * (self.F[p] @ self.F[p].transpose()) + ti.Matrix.identity(float, self.dim) * (self.lam * ti.log(J) - self.mu)
            stress = -(self.dt * self.p_vol * 4 / self.dx**2) * cauchy

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.grid_v[base + offset] += weight * stress @ dpos / self.grid_m[base + offset]

    def step(self):
        for _ in range(self.max_iter):
            self.P2G()
            if self.implicit:
                self.solve_implicit()
            else:
                self.solve_explicit()
            self.grid_boundary_conditions()
            self.G2P()
            self.advection()
        
# 使用示例
if __name__ == "__main__":
    mpm = ImplicitMPM("config.json")
    gui = ti.GUI("Implicit MPM", res=800)
    
    while gui.running:
        mpm.step()
        gui.circles(mpm.x.to_numpy(), radius=1.5, color=0x66CCFF)
        gui.show()
