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
        
        # 初始化两个子域MPM实例
        self.Domain1 = ImplicitMPM(domain1_config)
        self.Domain2 = ImplicitMPM(domain2_config)
        
        # 其他公共参数初始化
        self.max_schwarz_iter = main_config.get("max_schwarz_iter", 1)  # Schwarz迭代次数
        self.steps=main_config.get("steps", 10)  # 迭代步数
        self.recorder = None
        if main_config.get("record_frames", 0) > 0:
            self.recorder = ParticleRecorder(
            palette=np.array([
                0x66CCFF,  # 域1普通粒子
                0xED553B,  # 域2普通粒子 
                0xFFFFFF   # 边界粒子
            ], dtype=np.uint32),
            max_frames=main_config.get("record_frames", 60)
        )
            
        self.residuals = []
        self.iter_history1 = []
        self.iter_history2 = []

    @ti.kernel
    def exchange_boundary_conditions(self):
        """
        设置边界条件
        """
        for I in ti.grouped(self.Domain1.grid.v):
            if self.Domain1.grid.is_particle_boundary_grid[I] and self.Domain2.grid.m[I] > 0:
                self.Domain1.grid.is_boundary_grid[I] = [1]*self.Domain1.grid.dim
                self.Domain1.grid.boundary_v[I] = self.Domain2.grid.v[I]

        for I in ti.grouped(self.Domain2.grid.v):
            if self.Domain2.grid.is_particle_boundary_grid[I] and self.Domain1.grid.m[I] > 0:
                self.Domain2.grid.is_boundary_grid[I] = [1]*self.Domain2.grid.dim
                self.Domain2.grid.boundary_v[I] = self.Domain1.grid.v[I]

    @ti.kernel
    def check_grid_v_residual(self) -> ti.f32:
        """
        计算残差
        """
        residual = 0.0
        cnt = 0
        for I in ti.grouped(self.Domain1.grid.v):
            if self.Domain1.grid.m[I] > 0 and self.Domain2.grid.m[I] > 0:
                residual += (self.Domain1.grid.v[I]-self.Domain2.grid.v[I]).norm()
                cnt += 1
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

            residuals=[]
            iters1=[]
            iters2=[]

            # 2.迭代求解两个子域
            for _ in range(self.max_schwarz_iter):

                residuals.append(self.check_grid_v_residual())
            
                self.exchange_boundary_conditions()

                self.Domain1.grid.apply_boundary_conditions()
                self.Domain2.grid.apply_boundary_conditions()

                iter1=self.Domain1.solve()
                iter2=self.Domain2.solve()

                iters1.append(iter1)
                iters2.append(iter2)

            self.residuals.append(residuals)
            self.iter_history1.append(iters1)
            self.iter_history2.append(iters2)

            # 3.G2P: 将网格的速度传递回粒子上
            self.Domain1.g2p()
            self.Domain2.g2p()

            # 4.粒子自由运动
            self.Domain1.particles.advect(self.Domain1.dt)
            self.Domain2.particles.advect(self.Domain2.dt)



# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # 读取配置文件
    cfg = Config(path="config/schwarz.json")
    arch=cfg.get("arch", "cpu")
    if arch == "cuda":
        arch = ti.cuda
    elif arch == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, device_memory_GB=20)
    
    # 创建Schwarz域分解MPM实例
    mpm = MPM_Schwarz(cfg)

    gui = ti.GUI("Implicit MPM", res=800)
    
    while gui.running:
        mpm.step()
        gui.circles(mpm.Domain1.particles.x.to_numpy(), radius=1.5, color=0x068587)
        gui.circles(mpm.Domain2.particles.x.to_numpy(), radius=1.5, color=0xED553B)

        #绘制边界粒子
        gui.circles(mpm.Domain1.particles.x.to_numpy()[mpm.Domain1.particles.is_boundary_particle.to_numpy().astype(bool)], radius=1.5, color=0x66CCFF)
        gui.circles(mpm.Domain2.particles.x.to_numpy()[mpm.Domain2.particles.is_boundary_particle.to_numpy().astype(bool)], radius=1.5, color=0x66CCFF)
        gui.show()
        # 合并两域粒子数据
        if mpm.recorder is None:
            continue
        print("Frame", len(mpm.recorder.frame_data))
        all_pos = np.concatenate([
            mpm.Domain1.particles.x.to_numpy(),
            mpm.Domain2.particles.x.to_numpy()
        ])
        
        # 生成颜色索引 (0:域1普通, 1:域2普通, 2:边界)
        d1_colors = np.where(
            mpm.Domain1.particles.is_boundary_particle.to_numpy(),
            2, 0
        )
        d2_colors = np.where(
            mpm.Domain2.particles.is_boundary_particle.to_numpy(),
            2, 1
        )
        all_colors = np.concatenate([d1_colors, d2_colors]).astype(np.uint32)
        
        # 捕获帧
        mpm.recorder.capture(all_pos, all_colors)
        
        # 自动停止条件
        if len(mpm.recorder.frame_data) >= mpm.recorder.max_frames:
            break
    
    
    # 绘制最后5组residuals
    for i in range(5):
        frame = len(mpm.residuals) - i - 1
        plt.plot(mpm.residuals[-i-1], label=f'frame {frame}')
    plt.ylabel('Residual')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

    #iters1
    for i in range(5):
        frame = len(mpm.iter_history1) - i - 1
        plt.plot(mpm.iter_history1[-i-1], label=f'frame {frame}')
    plt.ylabel('Iterations1')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

    #iters2
    for i in range(5):
        frame = len(mpm.iter_history2) - i - 1
        plt.plot(mpm.iter_history2[-i-1], label=f'frame {frame}')
    plt.ylabel('Iterations2')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

    if mpm.recorder is None:
        exit()
    print("Playback finished.")
    # 播放录制动画
    mpm.recorder.play(loop=True, fps=60)