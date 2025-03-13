import taichi as ti


@ti.data_oriented
class MPMBase:
    def __init__(self, config):
        # 基础参数 (保持与原始类兼容)
        self.dim = config['dim']
        self.n_grid = config['n_grid']
        self.dx = 1.0 / self.n_grid
        self.dt = config.get('dt', 1e-4)
        self.E = config.get('youngs_modulus', 400)
        self.nu = config.get('poisson_ratio', 0.49)
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.p_rho = config.get('density', 1)
        self.p_vol = (self.dx * 0.5)**2
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = ti.Vector([0, -9.8] if self.dim == 2 else [0, -9.8, 0])
        self.bound = 3
        self.neighbor = (3,) * self.dim

        # 网格场 (由子类实现)
        self.grid_v = None
        self.grid_m = None

    @ti.kernel
    def reset_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
            self.grid_m[I] = 0


@ti.data_oriented
class MPMSubDomain(MPMBase):
    def __init__(self, config, global_fields, bounds):
        """
        config: 子域配置
        global_fields: 全局粒子数据引用 {
            'x': VectorField,
            'v': VectorField,
            'C': MatrixField,
            'J': ScalarField
        }
        bounds: 子域空间范围 [[x_min, x_max], [y_min, y_max]]
        """
        super().__init__(config)
        MPMSubDomain 
        # 绑定全局粒子数据
        self.x = global_fields['x']
        self.v = global_fields['v']
        self.C = global_fields['C']
        self.F = global_fields['F']
        
        # 初始化子域网格
        grid_shape = tuple([self.n_grid]*self.dim)
        self.grid_momt = ti.Vector.field(self.dim, ti.f32, shape=grid_shape)
        self.grid_m = ti.field(ti.f32, shape=grid_shape)
        
        # 子域空间参数
        self.bounds = ti.Vector.field(2, ti.f32, shape=self.dim)
        for d in ti.static(range(self.dim)):
            self.bounds[d] = ti.Vector(bounds[d])

    @ti.func
    def is_in_domain(self, pos):
        """判断位置是否在本子域影响范围内"""
        in_domain = True
        for d in ti.static(range(self.dim)):
            in_domain &= (pos[d] >= self.bounds[d][0])
            in_domain &= (pos[d] <= self.bounds[d][1])
        return in_domain

    @ti.kernel
    def p2g(self):
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
    def update_grid_v(self):
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
                             ti.Matrix.identity(float, self.dim) * self.lam * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4) * stress / self.dx**2
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.grid_v[base + offset] += weight * stress @ dpos / self.grid_m[base + offset]
        """只更新本子域内的网格"""
        for I in ti.grouped(self.grid_m):
            pos = I * self.dx
            if not self.is_in_domain(pos):
                continue
            
            # 边界处理
            cond = (I < self.bound) & (self.grid_momt[I] < 0) | (I > self.n_grid - self.bound) & (self.grid_momt[I] > 0)
            self.grid_momt[I] = ti.select(cond, 0, self.grid_momt[I])

    @ti.kernel
    def g2p(self):
        """只更新本子域内的粒子"""
        for p in self.x:
            pos = self.x[p]
            if not self.is_in_domain(pos):
                continue
            
            # 原始G2P计算逻辑
            Xp = pos / self.dx
            base = (Xp - 0.5).cast(int)
            fx = Xp - base.cast(float)
            
            # 三次B样条权重
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            
            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            
            # 网格插值
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i] 
                g_v = self.grid_momt[base + offset] / (self.grid_m[base + offset] + 1e-10)
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx**2
                
            # 更新粒子
            self.v[p] = new_v
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.dt * self.C[p]) @ self.F[p]
            self.C[p] = new_C

    # @ti.kernel
    # def get_boundary_data(self, 
    #                      other_grid_momt: ti.types.ndarray(ndim=2)):
    #                     #  other_bound: ti.types.ndarray(ndim=1)):
    #     """
    #     参数说明：
    #     other_grid_momt: ti.Vector.field(2, float, shape=(n_grid, n_grid)) 的动量场
    #     other_bound: ti.Vector.field(2, float, shape=(2)) 的边界范围
    #     """
    #     for I in ti.grouped(self.grid_momt):
    #         pos = I * self.dx
    #         if not self.is_in_domain(pos):
    #             continue
            
    #         # 检查是否在对方域内
    #         in_other_domain = True
    #         for d in ti.static(range(self.dim)):
    #             in_other_domain &= (pos[d] >= 0.4)
    #             in_other_domain &= (pos[d] <= 0.6)

    #         if in_other_domain:
    #             self.grid_momt[I] = other_grid_momt[I]

        
@ti.data_oriented
class DomainDecompositionMPM:
    def __init__(self, config):
        ti.init(arch=ti.vulkan)
        
        # 全局粒子数据
        self.dim = config['dim']
        self.max_steps = config.get('max_steps', 10)
        self.total_particles = config['total_particles']
        self.dt= config.get('dt', 1e-4)
        self.x = ti.Vector.field(self.dim, ti.f32, shape=self.total_particles)
        self.v = ti.Vector.field(self.dim, ti.f32, shape=self.total_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, ti.f32, shape=self.total_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.total_particles)
        
        # 创建子域
        self.subdomains:list[MPMSubDomain] = []
        domain_config = {
            'dim': self.dim,
            'n_grid': config['n_grid'],
            'youngs_modulus': config['youngs_modulus']
        }
        global_fields = {
            'x': self.x,
            'v': self.v,
            'C': self.C,
            'F': self.F
        }
        
        # 示例：水平划分为两个子域
        self.subdomains.append(MPMSubDomain(
            domain_config,
            global_fields,
            bounds=[[0.0, 1.0], [0.0, 0.6]]  # x范围,y范围
        ))
        self.subdomains.append(MPMSubDomain(
            domain_config,
            global_fields,
            bounds=[[0.0, 1.0], [0.4, 1.0]]
        ))
        
        # 初始化粒子
        self.init_particles()

    @ti.kernel
    def init_particles(self):
        for p in range(self.total_particles):
            pos = ti.Vector.zero(ti.f32, self.dim)
            for d in ti.static(range(self.dim)):
                pos[d] = ti.random() * 0.5 + 0.25
            self.x[p] = pos
            self.v[p] = ti.Vector.zero(ti.f32, self.dim)
            self.v[p][1] = -2
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)

    @ti.kernel
    def advect(self):
        for p in self.x:
            self.x[p] += self.v[p] * self.dt
    
    @ti.kernel
    def exchange_boundary_data(self):
        for I in ti.grouped(self.subdomains[0].grid_momt):
            pos = I * self.subdomains[0].dx
            if self.subdomains[0].is_in_domain(pos) and self.subdomains[1].is_in_domain(pos):
                tmp = self.subdomains[0].grid_momt[I]
                self.subdomains[0].grid_momt[I] = self.subdomains[1].grid_momt[I]
                self.subdomains[1].grid_momt[I] = tmp
    def step(self):
        for sd in self.subdomains:
            sd.reset_grid()
            sd.p2g()
        for _ in range(4):
            for sd in self.subdomains:
                sd.update_grid_v()
            self.exchange_boundary_data()
        self.advect()

    def render(self):
        gui = ti.GUI("Domain Decomposition MPM", (800, 800))
        while gui.running:
            for _ in range(self.max_steps):
                self.step()
            particles = self.x.to_numpy()
            gui.circles(particles, radius=1.5, color=0x66CCFF)
            gui.show()

# 运行示例
if __name__ == "__main__":
    config = {
        'dim': 2,
        'dt': 1e-4,
        'n_grid': 64,
        'total_particles': 8192,
        'youngs_modulus': 30
    }
    simulator = DomainDecompositionMPM(config)
    simulator.render()
