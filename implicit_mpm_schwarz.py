from implicit_mpm import *
from domain_manager import DomainManager
from boundary_exchanger import BoundaryExchanger
from particle_state_manager import ParticleStateManager

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
        # 初始化域管理器
        self.domain_manager = DomainManager(main_config)
        
        # 获取域的引用（保持向后兼容）
        self.Domain1 = self.domain_manager.domain1
        self.Domain2 = self.domain_manager.domain2
        self.BigTimeDomain = self.domain_manager.big_time_domain
        self.SmallTimeDomain = self.domain_manager.small_time_domain
        
        # 初始化边界交换器
        self.boundary_exchanger = BoundaryExchanger(
            self.BigTimeDomain, self.SmallTimeDomain
        )
        
        # 初始化粒子状态管理器
        self.particle_state_manager = ParticleStateManager(
            self.SmallTimeDomain.particles
        )
        
        self.do_small_advect = main_config.get("do_small_advect", True)  # 是否执行小时间步长域的粒子自由运动

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
        self.boundary_exchanger.exchange_boundary_conditions()

    def save_small_time_domain_particles(self):
        """
        保存小时间步长域的粒子数据到临时数组
        """
        self.particle_state_manager.save_particle_state()

    def restore_small_time_domain_particles(self):
        """
        恢复小时间步长域的粒子数据从临时数组
        """
        self.particle_state_manager.restore_particle_state()


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

            # 1.P2G: 将粒子的质量和动量传递到网格上
            self.domain_manager.pre_step()

            # 记录上一步网格速度
            self.boundary_exchanger.save_grid_velocities()

            self.save_small_time_domain_particles()

            residuals = []

            self.exchange_boundary_conditions()
            self.boundary_exchanger.save_boundary_velocities()

            # 2.迭代求解两个子域
            for i in range(self.max_schwarz_iter):
                print(f"Schwarz Iteration {i}/{self.max_schwarz_iter}")

                # residuals.append(self.check_grid_v_residual())
            
                # self.exchange_boundary_conditions()

                timesteps = self.domain_manager.get_timestep_ratio()
                self.BigTimeDomain.solve()

                for j in range(timesteps):
                    self.boundary_exchanger.interpolate_boundary_velocity((j+1) / timesteps)
                    self.SmallTimeDomain.solve()
                    self.SmallTimeDomain.g2p(self.SmallTimeDomain.dt)
                    if self.do_small_advect:
                        self.SmallTimeDomain.particles.advect(self.SmallTimeDomain.dt)
                        if j < timesteps - 1:
                            self.SmallTimeDomain.pre_p2g_sub()
                            self.SmallTimeDomain.p2g()
                            self.SmallTimeDomain.post_p2g()
                    else:
                        self.SmallTimeDomain.solver.save_previous_velocity()

                if i < self.max_schwarz_iter - 1:
                    self.exchange_boundary_conditions()
                    self.boundary_exchanger.small_time_domain_boundary_v_next.copy_from(self.SmallTimeDomain.grid.boundary_v) 
                    if self.do_small_advect:
                        self.restore_small_time_domain_particles()                        
                        self.SmallTimeDomain.pre_p2g_sub()
                        self.SmallTimeDomain.p2g()
                        self.SmallTimeDomain.post_p2g()
                    else:
                        self.SmallTimeDomain.grid.v_prev.copy_from(self.boundary_exchanger.small_time_domain_temp_grid_v)

                if not self.do_small_advect:
                    self.restore_small_time_domain_particles()

            # self.apply_average_grid_v()

            self.residuals.append(residuals)

            # 3.G2P: 将网格的速度传递回粒子上
            # 4.粒子自由运动
            self.domain_manager.finalize_step()
            

            if not self.do_small_advect:
                self.SmallTimeDomain.g2p(self.BigTimeDomain.dt)
                self.SmallTimeDomain.particles.advect(self.BigTimeDomain.dt)

    def render(self):
        transformed_x1 = self.Domain1.particles.x.to_numpy() * self.Domain1.scale + self.Domain1.offset
        transformed_x2 = self.Domain2.particles.x.to_numpy() * self.Domain2.scale + self.Domain2.offset

        if self.Domain1.particles.dim == 3:
            transformed_x1 = project(transformed_x1, self.Domain1.particles.dim)
            transformed_x2 = project(transformed_x2, self.Domain2.particles.dim)

        self.gui.circles(transformed_x1, radius=1.5, color=0x068587)
        self.gui.circles(transformed_x2, radius=1.5, color=0xED553B)

        #绘制边界粒子
        self.gui.circles(transformed_x1[self.Domain1.particles.is_boundary_particle.to_numpy().astype(bool)], radius=1.5, color=0x66CCFF)
        self.gui.circles(transformed_x2[self.Domain2.particles.is_boundary_particle.to_numpy().astype(bool)], radius=1.5, color=0x66CCFF)

        #绘制grid网格
        if self.visualize_grid:
            self.gui.lines(np.array(self.Domain1.grid.get_lines_begin()), np.array(self.Domain1.grid.get_lines_end()), radius=0.8, color=0x068587)
            self.gui.lines(np.array(self.Domain2.grid.get_lines_begin()), np.array(self.Domain2.grid.get_lines_end()), radius=0.8, color=0xED553B)

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