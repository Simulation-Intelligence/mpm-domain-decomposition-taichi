from implicit_mpm import *
from tools.domain_manager import DomainManager
from tools.boundary_exchanger import BoundaryExchanger
from tools.particle_state_manager import ParticleStateManager

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
                        self.SmallTimeDomain.grid.v.copy_from(self.boundary_exchanger.small_time_domain_temp_grid_v)

                if not self.do_small_advect:
                    self.restore_small_time_domain_particles()

            # self.apply_average_grid_v()

            self.residuals.append(residuals)

            # 3.G2P: 将网格的速度传递回粒子上
            # 4.粒子自由运动
            self.domain_manager.finalize_step(self.do_small_advect)

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
    
    def save_stress_strain_data(self, frame_number):
        """保存两个域的应力和应变数据"""
        import json
        import os
        
        # 计算两个域的应力和应变
        print("正在计算Domain1的应力和应变...")
        self.Domain1.solver.compute_stress_strain()
        
        print("正在计算Domain2的应力和应变...")
        self.Domain2.solver.compute_stress_strain()
        
        # 获取两个域的数据
        stress_data1 = self.Domain1.particles.stress.to_numpy()
        strain_data1 = self.Domain1.particles.strain.to_numpy()
        positions1 = self.Domain1.particles.x.to_numpy()
        
        stress_data2 = self.Domain2.particles.stress.to_numpy()
        strain_data2 = self.Domain2.particles.strain.to_numpy()
        positions2 = self.Domain2.particles.x.to_numpy()
        
        # 过滤掉位置为(0,0)的粒子 - Domain1
        if self.Domain1.dim == 2:
            # 2D情况：过滤(0,0)位置
            valid_mask1 = ~((positions1[:, 0] == 0.0) & (positions1[:, 1] == 0.0))
        else:
            # 3D情况：过滤(0,0,0)位置
            valid_mask1 = ~((positions1[:, 0] == 0.0) & (positions1[:, 1] == 0.0) & (positions1[:, 2] == 0.0))
        
        # 应用过滤掩码 - Domain1
        filtered_stress_data1 = stress_data1[valid_mask1]
        filtered_strain_data1 = strain_data1[valid_mask1]
        filtered_positions1 = positions1[valid_mask1]
        
        # 过滤掉位置为(0,0)的粒子 - Domain2
        if self.Domain2.dim == 2:
            # 2D情况：过滤(0,0)位置
            valid_mask2 = ~((positions2[:, 0] == 0.0) & (positions2[:, 1] == 0.0))
        else:
            # 3D情况：过滤(0,0,0)位置
            valid_mask2 = ~((positions2[:, 0] == 0.0) & (positions2[:, 1] == 0.0) & (positions2[:, 2] == 0.0))
        
        # 应用过滤掩码 - Domain2
        filtered_stress_data2 = stress_data2[valid_mask2]
        filtered_strain_data2 = strain_data2[valid_mask2]
        filtered_positions2 = positions2[valid_mask2]
        
        print(f"Domain1: Filtered out {np.sum(~valid_mask1)} particles at origin position")
        print(f"Domain1: Saving data for {len(filtered_positions1)} particles")
        print(f"Domain2: Filtered out {np.sum(~valid_mask2)} particles at origin position") 
        print(f"Domain2: Saving data for {len(filtered_positions2)} particles")
        
        # 创建统一的输出目录结构
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = "experiment_results"
        output_dir = os.path.join(base_output_dir, f"schwarz_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 分别保存两个域的数据
        # Domain1数据
        np.save(f"{output_dir}/domain1_stress_frame_{frame_number}.npy", filtered_stress_data1)
        np.save(f"{output_dir}/domain1_strain_frame_{frame_number}.npy", filtered_strain_data1)
        np.save(f"{output_dir}/domain1_positions_frame_{frame_number}.npy", filtered_positions1)
        
        # Domain2数据
        np.save(f"{output_dir}/domain2_stress_frame_{frame_number}.npy", filtered_stress_data2)
        np.save(f"{output_dir}/domain2_strain_frame_{frame_number}.npy", filtered_strain_data2)
        np.save(f"{output_dir}/domain2_positions_frame_{frame_number}.npy", filtered_positions2)
        
        # 计算两个域的von Mises应力
        def compute_von_mises_stress(stress_data, dim):
            von_mises_stress = []
            for i in range(stress_data.shape[0]):
                if dim == 2:
                    s = stress_data[i]
                    # 2D von Mises应力
                    von_mises = np.sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2)
                else:
                    # 3D von Mises应力
                    s = stress_data[i]
                    von_mises = np.sqrt(0.5*((s[0,0]-s[1,1])**2 + (s[1,1]-s[2,2])**2 + (s[2,2]-s[0,0])**2) + 3*(s[0,1]**2 + s[1,2]**2 + s[2,0]**2))
                von_mises_stress.append(von_mises)
            return von_mises_stress
        
        von_mises_stress1 = compute_von_mises_stress(filtered_stress_data1, self.Domain1.dim)
        von_mises_stress2 = compute_von_mises_stress(filtered_stress_data2, self.Domain2.dim)
        
        # 保存统计信息
        stats = {
            "frame": frame_number,
            "domain1": {
                "n_particles": int(filtered_stress_data1.shape[0]),
                "n_particles_total": int(stress_data1.shape[0]),
                "n_particles_filtered": int(np.sum(~valid_mask1)),
                "dimension": int(self.Domain1.dim),
                "von_mises_stress": {
                    "min": float(np.min(von_mises_stress1)),
                    "max": float(np.max(von_mises_stress1)),
                    "mean": float(np.mean(von_mises_stress1)),
                    "std": float(np.std(von_mises_stress1))
                }
            },
            "domain2": {
                "n_particles": int(filtered_stress_data2.shape[0]),
                "n_particles_total": int(stress_data2.shape[0]),
                "n_particles_filtered": int(np.sum(~valid_mask2)),
                "dimension": int(self.Domain2.dim),
                "von_mises_stress": {
                    "min": float(np.min(von_mises_stress2)),
                    "max": float(np.max(von_mises_stress2)),
                    "mean": float(np.mean(von_mises_stress2)),
                    "std": float(np.std(von_mises_stress2))
                }
            }
        }
        
        with open(f"{output_dir}/stress_strain_stats_frame_{frame_number}.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"两域应力应变数据已保存到 {output_dir}/")
        print(f"Domain1 von Mises应力范围: {stats['domain1']['von_mises_stress']['min']:.3e} - {stats['domain1']['von_mises_stress']['max']:.3e}")
        print(f"Domain1 平均von Mises应力: {stats['domain1']['von_mises_stress']['mean']:.3e}")
        print(f"Domain2 von Mises应力范围: {stats['domain2']['von_mises_stress']['min']:.3e} - {stats['domain2']['von_mises_stress']['max']:.3e}")
        print(f"Domain2 平均von Mises应力: {stats['domain2']['von_mises_stress']['mean']:.3e}")
        
        return output_dir



# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # 读取配置文件
    cfg = Config(path="config/schwarz_2d_test3.json")
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
    
    frame_count = 0
    while mpm.gui.running:
        mpm.step()
        
        mpm.render()
        
        frame_count += 1

        # 自动停止条件
        if len(mpm.recorder.frame_data) >= mpm.recorder.max_frames:
            # 在最后一帧记录应力和应变数据
            break
    
    
    # #绘制最后1组residuals
    # for i in range(1):
    #     plt.plot(mpm.residuals[-i-1])
    # plt.ylabel('Residual')
    # plt.xlabel('Iteration')
    # plt.xscale('log')  # X轴对数化
    # plt.yscale('log')  # Y轴对数化
    # plt.show()
    
    #
    print("记录最终帧的应力和应变数据...")
    mpm.save_stress_strain_data(frame_count)
    
    if mpm.recorder is None:
        exit()
    print("Playback finished.")
    # 播放录制动画
    mpm.recorder.play(loop=True, fps=60)