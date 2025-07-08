from implicit_mpm import *

from Util.Recorder import *

import matplotlib.pyplot as plt

from Util.Project import project

@ti.data_oriented
class MPM_Schwarz:
    def __init__(self, main_config: Config):
        """
        基于Schwarz域分解的双域MPM求解器
        
        参数：
            main_config: 包含Domain1和Domain2配置的全局配置
        """
        # 提取子域配置并创建独立配置对象
        domain1_config = Config(data=main_config.get("Domain1", {}))
        domain2_config = Config(data=main_config.get("Domain2", {}))
        common_particles_config = main_config.get("Common_Particles", None)

        # 如果有公共粒子配置，则创建公共粒子实例
        common_particles = Particles(common_particles_config) if common_particles_config else None
        
        # 初始化两个子域MPM实例
        self.Domain1 = ImplicitMPM(domain1_config,common_particles)
        self.Domain2 = ImplicitMPM(domain2_config,common_particles)

        if self.Domain1.dt >= self.Domain2.dt:
            self.BigTimeDomain = self.Domain1
            self.SmallTimeDomain = self.Domain2
        else:
            self.BigTimeDomain = self.Domain2
            self.SmallTimeDomain = self.Domain1

        # 小时间步长的domain的时间步长需要调整为大时间步长的整数倍
        ratio = self.BigTimeDomain.dt / self.SmallTimeDomain.dt
        import math
        if not math.isclose(ratio, round(ratio), rel_tol=1e-8):
            print("big time domain dt:", self.BigTimeDomain.dt,
                  "small time domain dt:", self.SmallTimeDomain.dt,
                  "ratio:", ratio)
            raise ValueError("Big time domain dt must be a multiple of small time domain dt.")
        
        self.do_small_advect = main_config.get("do_small_advect", True)  # 是否执行小时间步长域的粒子自由运动
        
        # 分配临时数组用于保存网格速度
        self.BigTimeDomainTempGridV = ti.Vector.field(self.BigTimeDomain.grid.dim, self.BigTimeDomain.float_type, (self.BigTimeDomain.grid.size,)*self.BigTimeDomain.grid.dim)
        self.SmallTimeDomainTempGridV = ti.Vector.field(self.SmallTimeDomain.grid.dim, self.SmallTimeDomain.float_type, (self.SmallTimeDomain.grid.size,)*self.SmallTimeDomain.grid.dim)
        self.SmallTimeDomainBoundaryVLast = ti.Vector.field(self.SmallTimeDomain.grid.dim, self.SmallTimeDomain.float_type, (self.SmallTimeDomain.grid.size,)*self.SmallTimeDomain.grid.dim)
        self.SmallTimeDomainBoundaryVNext = ti.Vector.field(self.SmallTimeDomain.grid.dim, self.SmallTimeDomain.float_type, (self.SmallTimeDomain.grid.size,)*self.SmallTimeDomain.grid.dim)

        #临时数组用于保存粒子数据
        self.SmallTimeDomainTempParticlesX = ti.Vector.field(self.SmallTimeDomain.particles.dim, self.SmallTimeDomain.float_type, self.SmallTimeDomain.particles.n_particles)
        self.SmallTimeDomainTempParticlesV = ti.Vector.field(self.SmallTimeDomain.particles.dim, self.SmallTimeDomain.float_type, self.SmallTimeDomain.particles.n_particles)
        self.SmallTimeDomainTempParticlesF = ti.Matrix.field(self.SmallTimeDomain.particles.dim,self.SmallTimeDomain.particles.dim, self.SmallTimeDomain.float_type, self.SmallTimeDomain.particles.n_particles)
        self.SmallTimeDomainTempParticlesC = ti.Matrix.field(self.SmallTimeDomain.particles.dim,self.SmallTimeDomain.particles.dim, self.SmallTimeDomain.float_type, self.SmallTimeDomain.particles.n_particles)

        # 其他公共参数初始化
        self.max_schwarz_iter = main_config.get("max_schwarz_iter", 1)  # Schwarz迭代次数
        self.steps=main_config.get("steps", 10)  # 迭代步数
        self.recorder = None
        self.visualize_grid = main_config.get("visualize_grid", False)
        if main_config.get("record_frames", 0) > 0:
            lines_begin = None
            lines_end = None
            lines_color = None
            if self.visualize_grid:
                lines_begin_1 = self.Domain1.get_grid_lines_begin()
                lines_end_1 = self.Domain1.get_grid_lines_end()
                lines_begin_2 = self.Domain2.get_grid_lines_begin()
                lines_end_2 = self.Domain2.get_grid_lines_end()
                lines_begin = np.concatenate([lines_begin_1, lines_begin_2])
                lines_end = np.concatenate([lines_end_1, lines_end_2])
                lines_color_1 = np.full(len(lines_begin_1), 0x66CCFF, dtype=np.uint32)
                lines_color_2 = np.full(len(lines_begin_2), 0xED553B, dtype=np.uint32)
                lines_color = np.concatenate([lines_color_1, lines_color_2])
            self.recorder = ParticleRecorder(
            palette=np.array([
                0x66CCFF,  # 域1普通粒子
                0xED553B,  # 域2普通粒子 
                0xFFFFFF   # 边界粒子
            ], dtype=np.uint32),
            max_frames=main_config.get("record_frames", 60),
            lines_begin=lines_begin,
            lines_end=lines_end,
            lines_color=lines_color
        )

        self.use_mass_boundary = main_config.get("use_mass_boundary", False)
            
        self.residuals = []

        self.gui=ti.GUI("Implicit MPM Schwarz", res=800)


    def exchange_boundary_conditions(self):
        """
        设置边界条件
        """
        self.project_to_big_time_domain_boundary(self.SmallTimeDomain.grid.v, self.BigTimeDomain.grid.boundary_v)
        self.project_to_small_time_domain_boundary(self.BigTimeDomain.grid.v, self.SmallTimeDomain.grid.boundary_v)

    def save_small_time_domain_particles(self):
        """
        保存小时间步长域的粒子数据到临时数组
        """
        self.SmallTimeDomainTempParticlesX.copy_from(self.SmallTimeDomain.particles.x)
        self.SmallTimeDomainTempParticlesV.copy_from(self.SmallTimeDomain.particles.v)
        self.SmallTimeDomainTempParticlesF.copy_from(self.SmallTimeDomain.particles.F)
        self.SmallTimeDomainTempParticlesC.copy_from(self.SmallTimeDomain.particles.C)

    def restore_small_time_domain_particles(self):
        """
        恢复小时间步长域的粒子数据从临时数组
        """
        self.SmallTimeDomain.particles.x.copy_from(self.SmallTimeDomainTempParticlesX)
        self.SmallTimeDomain.particles.v.copy_from(self.SmallTimeDomainTempParticlesV)
        self.SmallTimeDomain.particles.F.copy_from(self.SmallTimeDomainTempParticlesF)
        self.SmallTimeDomain.particles.C.copy_from(self.SmallTimeDomainTempParticlesC)

    @ti.kernel
    def project_to_big_time_domain_boundary(self, from_boundary_v: ti.template(), to_boundary_v: ti.template()):
        """
        将小时间步长域的速度投影到大时间步长域的边界
        """
        for I in ti.grouped(to_boundary_v):
            if self.BigTimeDomain.grid.is_particle_boundary_grid[I]:
                m = 0.0
                is_SmallTimeDomain_boundary = False
                x = ((I * self.BigTimeDomain.grid.dx)*self.BigTimeDomain.scale + self.BigTimeDomain.offset)- self.SmallTimeDomain.offset
                x = x / self.SmallTimeDomain.scale
                base = (x * self.SmallTimeDomain.grid.inv_dx - 0.5).cast(int)
                fx = x * self.SmallTimeDomain.grid.inv_dx - base.cast(float)
                w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
                to_boundary_v[I] = ti.Vector.zero(self.BigTimeDomain.grid.float_type, self.BigTimeDomain.grid.dim)

                for offset in ti.static(ti.grouped(ti.ndrange(*self.BigTimeDomain.particles.neighbor))):
                    weight = 1.0
                    for d in ti.static(range(self.BigTimeDomain.grid.dim)):
                        weight *= w[offset[d]][d]
                    if self.SmallTimeDomain.grid.is_particle_boundary_grid[base + offset]:
                        is_SmallTimeDomain_boundary = True
                    m += weight * self.SmallTimeDomain.grid.m[base + offset]
                    to_boundary_v[I] += weight * from_boundary_v[base + offset]* self.SmallTimeDomain.grid.m[base + offset]

                BigTimeDomain_set_boundary = self.BigTimeDomain.grid.is_particle_boundary_grid[I] and m > 1e-10
                BigTimeDomain_set_boundary = BigTimeDomain_set_boundary and (not is_SmallTimeDomain_boundary or self.BigTimeDomain.grid.m[I] <m)

                if BigTimeDomain_set_boundary:
                    self.BigTimeDomain.grid.is_boundary_grid[I] = [1]*self.BigTimeDomain.grid.dim
                    to_boundary_v[I] = to_boundary_v[I] / m

    @ti.kernel
    def project_to_small_time_domain_boundary(self, from_boundary_v: ti.template(), to_boundary_v: ti.template()):
        """
        将大时间步长域的速度投影到小时间步长域的边界
        """
        for I in ti.grouped(to_boundary_v):
            if self.SmallTimeDomain.grid.is_particle_boundary_grid[I]:
                m = 0.0
                is_BigTimeDomain_boundary = False
                x = ((I * self.SmallTimeDomain.grid.dx)*self.SmallTimeDomain.scale + self.SmallTimeDomain.offset)- self.BigTimeDomain.offset
                x = x / self.BigTimeDomain.scale
                base = (x * self.BigTimeDomain.grid.inv_dx - 0.5).cast(int)
                fx = x * self.BigTimeDomain.grid.inv_dx - base.cast(float)
                w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
                to_boundary_v[I] = ti.Vector.zero(self.SmallTimeDomain.grid.float_type, self.SmallTimeDomain.grid.dim)

                for offset in ti.static(ti.grouped(ti.ndrange(*self.SmallTimeDomain.particles.neighbor))):
                    weight = 1.0
                    for d in ti.static(range(self.SmallTimeDomain.grid.dim)):
                        weight *= w[offset[d]][d]
                    if self.BigTimeDomain.grid.is_particle_boundary_grid[base + offset]:
                        is_BigTimeDomain_boundary = True
                    m += weight * self.BigTimeDomain.grid.m[base + offset]
                    to_boundary_v[I] += weight * from_boundary_v[base + offset]* self.BigTimeDomain.grid.m[base + offset]

                SmallTimeDomain_set_boundary = self.SmallTimeDomain.grid.is_particle_boundary_grid[I] and m > 1e-10
                SmallTimeDomain_set_boundary = SmallTimeDomain_set_boundary and (not is_BigTimeDomain_boundary or self.SmallTimeDomain.grid.m[I] <m)

                if SmallTimeDomain_set_boundary:
                    self.SmallTimeDomain.grid.is_boundary_grid[I] = [1]*self.SmallTimeDomain.grid.dim
                    to_boundary_v[I] = to_boundary_v[I] / m

    @ti.kernel
    def linp(self,dest: ti.template(), a: ti.template(), b: ti.template(), ratio: ti.f32):
        """
        将a和b的线性组合存储到dest中
        """
        for I in ti.grouped(dest):
            dest[I] = a[I] * (1 - ratio) + b[I] * ratio

    @ti.kernel
    def apply_average_grid_v(self):
        """
        将网格速度平均到粒子上
        """
        for I in ti.grouped(self.Domain1.grid.v):
            if self.Domain1.grid.m[I] > 0 and self.Domain2.grid.m[I] > 0:
                self.Domain1.grid.v[I] = (self.Domain1.grid.v[I] + self.Domain2.grid.v[I]) / 2.0
                self.Domain2.grid.v[I] = self.Domain1.grid.v[I]

    @ti.kernel
    def check_grid_v_residual(self) -> ti.f32:
        """
        计算残差
        """
        residual = 0.0
        cnt = 0
        for I in ti.grouped(self.Domain1.grid.v):
            if self.Domain1.grid.m[I] > 0 and self.Domain2.grid.m[I] > 0:
                ti.atomic_add(residual,(self.Domain1.grid.v[I]-self.Domain2.grid.v[I]).norm())
                ti.atomic_add(cnt, 1)

        return residual / cnt
    
    
    def step(self):
        for _ in range(self.steps):
            """
            执行一步Schwarz迭代
            """
            self.BigTimeDomain.pre_p2g()
            self.SmallTimeDomain.pre_p2g()

            # 1.P2G: 将粒子的质量和动量传递到网格上
            self.BigTimeDomain.p2g()
            self.SmallTimeDomain.p2g()

            self.BigTimeDomain.post_p2g()
            self.SmallTimeDomain.post_p2g()

            # 记录上一步网格速度
            self.BigTimeDomainTempGridV.copy_from(self.BigTimeDomain.grid.v)
            self.SmallTimeDomainTempGridV.copy_from(self.SmallTimeDomain.grid.v)

            self.save_small_time_domain_particles()

            residuals = []

            self.exchange_boundary_conditions()
            self.SmallTimeDomainBoundaryVLast.copy_from(self.SmallTimeDomain.grid.boundary_v)
            self.SmallTimeDomainBoundaryVNext.copy_from(self.SmallTimeDomain.grid.boundary_v)

            # 2.迭代求解两个子域
            for i in range(self.max_schwarz_iter):
                print(f"Schwarz Iteration {i}/{self.max_schwarz_iter}")

                # residuals.append(self.check_grid_v_residual())
            
                # self.exchange_boundary_conditions()

                timesteps = int(self.BigTimeDomain.dt // self.SmallTimeDomain.dt)
                self.BigTimeDomain.solve()

                for j in range(timesteps):
                    self.linp(self.SmallTimeDomain.grid.boundary_v, self.SmallTimeDomainBoundaryVLast, self.SmallTimeDomainBoundaryVNext, (j+1) / timesteps)
                    self.SmallTimeDomain.solve()
                    self.SmallTimeDomain.g2p(self.SmallTimeDomain.dt)
                    if self.do_small_advect:
                        self.SmallTimeDomain.particles.advect(self.SmallTimeDomain.dt)
                        self.SmallTimeDomain.pre_p2g()
                        self.SmallTimeDomain.p2g()
                        self.SmallTimeDomain.post_p2g()
                    else:
                        self.SmallTimeDomain.solver.save_previous_velocity()

                if i < self.max_schwarz_iter - 1:
                    self.exchange_boundary_conditions()
                    self.SmallTimeDomainBoundaryVNext.copy_from(self.SmallTimeDomain.grid.boundary_v) 
                    if self.do_small_advect:
                        self.restore_small_time_domain_particles()
                    else:
                        self.SmallTimeDomain.grid.v_prev.copy_from(self.SmallTimeDomainTempGridV)

                if not self.do_small_advect:
                    self.restore_small_time_domain_particles()

            # self.apply_average_grid_v()

            self.residuals.append(residuals)

            # 3.G2P: 将网格的速度传递回粒子上
            # 4.粒子自由运动
            self.BigTimeDomain.g2p(self.BigTimeDomain.dt)
            self.BigTimeDomain.particles.advect(self.BigTimeDomain.dt)
            

            if not self.do_small_advect:
                self.SmallTimeDomain.g2p(self.BigTimeDomain.dt)
                self.SmallTimeDomain.particles.advect(self.BigTimeDomain.dt)

    def render(self):
        transformed_x1 = self.BigTimeDomain.particles.x.to_numpy() * self.BigTimeDomain.scale + self.BigTimeDomain.offset
        transformed_x2 = self.SmallTimeDomain.particles.x.to_numpy() * self.SmallTimeDomain.scale + self.SmallTimeDomain.offset

        if self.BigTimeDomain.particles.dim == 3:
            transformed_x1 = project(transformed_x1, self.BigTimeDomain.particles.dim)
            transformed_x2 = project(transformed_x2, self.SmallTimeDomain.particles.dim)

        self.gui.circles(transformed_x1, radius=1.5, color=0x068587)
        self.gui.circles(transformed_x2, radius=1.5, color=0xED553B)

        #绘制边界粒子
        self.gui.circles(transformed_x1[self.BigTimeDomain.particles.is_boundary_particle.to_numpy().astype(bool)], radius=1.5, color=0x66CCFF)
        self.gui.circles(transformed_x2[self.SmallTimeDomain.particles.is_boundary_particle.to_numpy().astype(bool)], radius=1.5, color=0x66CCFF)

        #绘制grid网格
        if self.visualize_grid:
            self.gui.lines(np.array(self.BigTimeDomain.grid.get_lines_begin()), np.array(self.BigTimeDomain.grid.get_lines_end()), radius=0.8, color=0x068587)
            self.gui.lines(np.array(self.SmallTimeDomain.grid.get_lines_begin()), np.array(self.SmallTimeDomain.grid.get_lines_end()), radius=0.8, color=0xED553B)

        self.gui.show()
        # 合并两域粒子数据
        if self.recorder is None:
            return
        print("Frame", len(self.recorder.frame_data))
        all_pos = np.concatenate([
            transformed_x1,
            transformed_x2
        ])
        
        # 生成颜色索引 (0:域1普通, 1:域2普通, 2:边界)
        d1_colors = np.where(
            self.BigTimeDomain.particles.is_boundary_particle.to_numpy(),
            2, 0
        )
        d2_colors = np.where(
            self.SmallTimeDomain.particles.is_boundary_particle.to_numpy(),
            2, 1
        )
        all_colors = np.concatenate([d1_colors, d2_colors]).astype(np.uint32)
        
        # 捕获帧
        self.recorder.capture(all_pos, all_colors)



# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # 读取配置文件
    cfg = Config(path="config/schwarz_2d.json")
    float_type=ti.f32 if cfg.get("float_type", "f32") == "f32" else ti.f64        
    arch=cfg.get("arch", "cpu")
    if arch == "cuda":
        arch = ti.cuda
    elif arch == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, default_fp=float_type, device_memory_GB=20)
    
    # 创建Schwarz域分解MPM实例
    mpm = MPM_Schwarz(cfg)
    
    while mpm.gui.running:
        mpm.step()
        
        mpm.render()
        
        # 自动停止条件
        if len(mpm.recorder.frame_data) >= mpm.recorder.max_frames:
            break
    
    
    #绘制最后1组residuals
    for i in range(1):
        plt.plot(mpm.residuals[-i-1])
    plt.ylabel('Residual')
    plt.xlabel('Iteration')
    plt.xscale('log')  # X轴对数化
    plt.yscale('log')  # Y轴对数化
    plt.show()

    if mpm.recorder is None:
        exit()
    print("Playback finished.")
    # 播放录制动画
    mpm.recorder.play(loop=True, fps=60)