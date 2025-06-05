from implicit_mpm import *

from Util.Recorder import *

import matplotlib.pyplot as plt

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
        
        # 其他公共参数初始化
        self.max_schwarz_iter = main_config.get("max_schwarz_iter", 1)  # Schwarz迭代次数
        self.steps=main_config.get("steps", 10)  # 迭代步数
        self.recorder = None
        if main_config.get("record_frames", 0) > 0:
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

    @ti.kernel
    def exchange_boundary_conditions(self):
        """
        设置边界条件
        """
        # for I in ti.grouped(self.Domain1.grid.v):
        #     Domain1_set_boundary = self.Domain1.grid.is_particle_boundary_grid[I] and self.Domain2.grid.m[I] > 0
        #     Domain2_set_boundary = self.Domain2.grid.is_particle_boundary_grid[I] and self.Domain1.grid.m[I] > 0
        #     if self.use_mass_boundary:
        #         Domain1_set_boundary = Domain1_set_boundary and (not Domain2_set_boundary or self.Domain1.grid.m[I] <self.Domain2.grid.m[I])
        #         Domain2_set_boundary = Domain2_set_boundary and (not Domain1_set_boundary or self.Domain2.grid.m[I] < self.Domain1.grid.m[I])
        #     else: 
        #         Domain1_set_boundary = Domain1_set_boundary and not self.Domain2.grid.is_particle_boundary_grid[I]
        #         Domain2_set_boundary = Domain2_set_boundary and not self.Domain1.grid.is_particle_boundary_grid[I]

        #     if Domain1_set_boundary:
        #         self.Domain1.grid.is_boundary_grid[I] = [1]*self.Domain1.grid.dim
        #         self.Domain1.grid.boundary_v[I] = self.Domain2.grid.v[I]

        #     if Domain2_set_boundary:
        #         self.Domain2.grid.is_boundary_grid[I] = [1]*self.Domain2.grid.dim
        #         self.Domain2.grid.boundary_v[I] = self.Domain1.grid.v[I]
        for I in ti.grouped(self.Domain1.grid.v):
            if self.Domain1.grid.is_particle_boundary_grid[I]:
                m = 0.0
                is_Domain2_boundary = False
                x = ((I * self.Domain1.grid.dx)*self.Domain1.scale + self.Domain1.offset)- self.Domain2.offset
                x = x / self.Domain2.scale
                base = (x * self.Domain2.grid.inv_dx - 0.5).cast(int)
                fx = x * self.Domain2.grid.inv_dx - base.cast(float)
                w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
                self.Domain1.grid.boundary_v[I] = ti.Vector.zero(self.Domain1.grid.float_type, self.Domain1.grid.dim)

                for offset in ti.static(ti.grouped(ti.ndrange(*self.Domain1.particles.neighbor))):
                    weight = 1.0
                    for d in ti.static(range(self.Domain1.grid.dim)):
                        weight *= w[offset[d]][d]
                    if self.Domain2.grid.is_particle_boundary_grid[base + offset]:
                        is_Domain2_boundary = True
                    m += weight * self.Domain2.grid.m[base + offset]
                    self.Domain1.grid.boundary_v[I] += weight * self.Domain2.grid.v[base + offset]* self.Domain2.grid.m[base + offset]

                Domain1_set_boundary = self.Domain1.grid.is_particle_boundary_grid[I] and m > 1e-10
                Domain1_set_boundary = Domain1_set_boundary and (not is_Domain2_boundary or self.Domain1.grid.m[I] <m)

                if Domain1_set_boundary:
                    self.Domain1.grid.is_boundary_grid[I] = [1]*self.Domain1.grid.dim
                    self.Domain1.grid.boundary_v[I] = self.Domain1.grid.boundary_v[I] / m

        for I in ti.grouped(self.Domain2.grid.v):
            if self.Domain2.grid.is_particle_boundary_grid[I]:
                m = 0.0
                is_Domain1_boundary = False
                x = ((I * self.Domain2.grid.dx)*self.Domain2.scale + self.Domain2.offset)- self.Domain1.offset
                x = x / self.Domain1.scale
                base = (x * self.Domain1.grid.inv_dx - 0.5).cast(int)
                fx = x * self.Domain1.grid.inv_dx - base.cast(float)
                w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
                self.Domain2.grid.boundary_v[I] = ti.Vector.zero(self.Domain2.grid.float_type, self.Domain2.grid.dim)

                for offset in ti.static(ti.grouped(ti.ndrange(*self.Domain2.particles.neighbor))):
                    weight = 1.0
                    for d in ti.static(range(self.Domain2.grid.dim)):
                        weight *= w[offset[d]][d]
                    if self.Domain1.grid.is_particle_boundary_grid[base + offset]:
                        is_Domain1_boundary = True
                    m += weight * self.Domain1.grid.m[base + offset]
                    self.Domain2.grid.boundary_v[I] += weight * self.Domain1.grid.v[base + offset]* self.Domain1.grid.m[base + offset]

                Domain2_set_boundary = self.Domain2.grid.is_particle_boundary_grid[I] and m > 1e-10
                Domain2_set_boundary = Domain2_set_boundary and (not is_Domain1_boundary or self.Domain2.grid.m[I] < m)

                if Domain2_set_boundary:
                    self.Domain2.grid.is_boundary_grid[I] = [1]*self.Domain2.grid.dim
                    self.Domain2.grid.boundary_v[I] = self.Domain2.grid.boundary_v[I] / m

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
            self.Domain1.pre_p2g()
            self.Domain2.pre_p2g()

            # 1.P2G: 将粒子的质量和动量传递到网格上
            self.Domain1.p2g()
            self.Domain2.p2g()

            self.Domain1.solver.save_previous_velocity()
            self.Domain2.solver.save_previous_velocity()

            residuals=[]

            self.Domain1.grid.apply_boundary_conditions()
            self.Domain2.grid.apply_boundary_conditions()

            # 2.迭代求解两个子域
            for i in range(self.max_schwarz_iter):
                print(f"Schwarz Iteration {i}/{self.max_schwarz_iter}")

                # residuals.append(self.check_grid_v_residual())
            
                self.exchange_boundary_conditions()

                self.Domain1.solve()
                self.Domain2.solve()
                
            # self.apply_average_grid_v()

            self.residuals.append(residuals)

            # 3.G2P: 将网格的速度传递回粒子上
            self.Domain1.g2p()
            self.Domain2.g2p()

            # 4.粒子自由运动
            self.Domain1.particles.advect(self.Domain1.dt)
            self.Domain2.particles.advect(self.Domain2.dt)

    def render(self):
        transformed_x1 = self.Domain1.particles.x.to_numpy() * self.Domain1.scale + self.Domain1.offset
        transformed_x2 = self.Domain2.particles.x.to_numpy() * self.Domain2.scale + self.Domain2.offset
        self.gui.circles(transformed_x1, radius=1.5, color=0x068587)
        self.gui.circles(transformed_x2, radius=1.5, color=0xED553B)

        #绘制边界粒子
        self.gui.circles(transformed_x1[self.Domain1.particles.is_boundary_particle.to_numpy().astype(bool)], radius=1.5, color=0x66CCFF)
        self.gui.circles(transformed_x2[self.Domain2.particles.is_boundary_particle.to_numpy().astype(bool)], radius=1.5, color=0x66CCFF)

        #绘制grid网格
        self.gui.lines(np.array(self.Domain1.get_grid_lines_begin()),np.array(self.Domain1.get_grid_lines_end()), radius=0.8, color=0x068587)
        self.gui.lines(np.array(self.Domain2.get_grid_lines_begin()),np.array(self.Domain2.get_grid_lines_end()), radius=0.8, color=0xED553B)
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
            self.Domain1.particles.is_boundary_particle.to_numpy(),
            2, 0
        )
        d2_colors = np.where(
            self.Domain2.particles.is_boundary_particle.to_numpy(),
            2, 1
        )
        all_colors = np.concatenate([d1_colors, d2_colors]).astype(np.uint32)
        
        # 捕获帧
        self.recorder.capture(all_pos, all_colors)



# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # 读取配置文件
    cfg = Config(path="config/schwarz.json")
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