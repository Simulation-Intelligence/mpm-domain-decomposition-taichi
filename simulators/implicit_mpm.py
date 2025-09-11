import taichi as ti

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Geometry.Grid import Grid
from Geometry.Particles import Particles

from mpm_solver import MPMSolver

from Util.Config import Config

from Util.Recorder import *

from Util.Project import project



# ------------------ 主模拟器 ------------------
@ti.data_oriented
class ImplicitMPM:
    def __init__(self, config:Config,common_particles:Particles=None):
        self.cfg = config

        self.float_type = ti.f32 if config.get("float_type", "f32") == "f32" else ti.f64

        self.dim = config.get("dim", 2)
    
        self.grid = Grid(
            config
        )
        self.particles = Particles(self.cfg,common_particles=common_particles)
        self.implicit = self.cfg.get("implicit", True)
        self.max_iter = self.cfg.get("max_iter", 1)
        self.dt = self.cfg.get("dt", 2e-3)
        
        # 静力学求解配置
        self.static_solve = self.cfg.get("static_solve", False)
        if self.static_solve and not self.implicit:
            raise ValueError("静力学求解要求使用隐式方法 (implicit=True)")
        if self.static_solve:
            print("启用静力学求解模式")

        self.scale = self.cfg.get("scale", 1.0)
        self.offset = self.cfg.get("offset", (0, 0))

        
        
        # 材料参数现在通过particles.material_params访问

        self.neighbor = (3,) * self.dim
        
        # 统一创建solver，用于参数管理和求解
        self.solver = MPMSolver(self.grid, self.particles, self.cfg)


        # 其他公共参数初始化
        self.max_schwarz_iter = config.get("max_schwarz_iter", 1)  # Schwarz迭代次数
        self.recorder = None
        if config.get("record_frames", 0) > 0:
            self.recorder = ParticleRecorder(
            palette=np.array([
                0x66CCFF,  # 域1普通粒子
                0xFFFFFF
            ], dtype=np.uint32),
            max_frames=config.get("record_frames", 60)
        )

        self.gui= ti.GUI("Implicit MPM", res=800)
    
        

    def solve(self):
        if self.implicit:
            return self.solver.solve_implicit()
        else:
            return self.solver.solve_explicit()

    def pre_p2g(self):
        self.grid.clear()
        self.particles.build_neighbor_list()

    def pre_p2g_sub(self):
        self.grid.clear_sub()
        self.particles.build_neighbor_list()

    def post_p2g(self):
        self.solver.save_previous_velocity()
        self.grid.apply_boundary_conditions()

    def step(self):
        for _ in range(self.max_iter):
            self.pre_p2g()
            self.p2g()

            if self.implicit:
                self.solver.save_previous_velocity()

            self.grid.apply_boundary_conditions()
            
            self.solve()
            
            self.g2p(self.dt)
            
            # 静力学求解时不进行advect
            if not self.static_solve:
                self.particles.advect(self.dt)

    @ti.kernel
    def p2g(self):
        for p in range(self.particles.n_particles):
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            fx = self.particles.x[p] * self.grid.inv_dx - base.cast(float)
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = (base + offset) % self.grid.size
                dpos = (offset - fx) * self.grid.dx
                p_mass = self.particles.get_particle_mass(p)
                self.grid.m[grid_idx] += self.particles.wip[p,offset] * p_mass
                self.grid.v[grid_idx] += self.particles.wip[p,offset] * p_mass * (self.particles.v[p] + self.particles.C[p] @ dpos)
                if self.particles.is_boundary_particle[p]:
                    self.grid.is_particle_boundary_grid[grid_idx] = 1

        for I in ti.grouped(self.grid.m):
            if self.grid.m[I] > 1e-10:
                self.grid.v[I] /= self.grid.m[I]

    @ti.kernel
    def g2p(self, dt: ti.f32):
        for p in self.particles.x:
            Xp = self.particles.x[p] / self.grid.dx
            base = (Xp - 0.5).cast(int)

            new_v = ti.Vector.zero(self.float_type, self.dim)
            new_C = ti.Matrix.zero(self.float_type, self.dim, self.dim)
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = base + offset
                g_v = self.grid.v[grid_idx]
                new_v += g_v*self.particles.wip[p, offset]
                new_C += 4*  g_v.outer_product(self.particles.dwip[p, offset])
                
            self.particles.v[p] = new_v
            if self.implicit:
                self.particles.F[p] = (ti.Matrix.identity(self.float_type, self.grid.dim) + dt * new_C) @ self.particles.F[p]
            else:
                self.particles.F[p] = (ti.Matrix.identity(self.float_type, self.grid.dim) + dt * self.particles.C[p]) @ self.particles.F[p]
            self.particles.C[p] = new_C

    @ti.kernel
    def update_F(self, dt: ti.f32):
        for p in self.particles.x:
            Xp = self.particles.x[p] / self.grid.dx
            base = (Xp - 0.5).cast(int)

            new_C = ti.Matrix.zero(self.float_type, self.dim, self.dim)

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbor))):
                grid_idx = base + offset
                g_v = self.grid.v[grid_idx]
                new_C += 4*  g_v.outer_product(self.particles.dwip[p, offset])

            if self.implicit:
                self.particles.F[p] = (ti.Matrix.identity(self.float_type, self.grid.dim) + dt * new_C) @ self.particles.F[p]
            else:
                self.particles.F[p] = (ti.Matrix.identity(self.float_type, self.grid.dim) + dt * self.particles.C[p]) @ self.particles.F[p]

    #returns the grid lines for visualization
    def get_grid_lines_begin(self):
        lines_begin = []
        for i in range(self.grid.size):
            lines_begin.append(((i+0.5)*self.grid.dx*self.scale + self.offset[0], self.offset[1]))
            lines_begin.append((self.offset[0], (i+0.5)*self.grid.dx*self.scale + self.offset[1]))
        return lines_begin

    def get_grid_lines_end(self):
        lines_end = []
        for i in range(self.grid.size):
            lines_end.append(((i+0.5)*self.grid.dx*self.scale + self.offset[0], self.scale + self.offset[1]))
            lines_end.append((self.scale + self.offset[0], (i+0.5)*self.grid.dx*self.scale + self.offset[1]))
        return lines_end
    
    def render(self):
        x_numpy = self.particles.x.to_numpy()

        if self.dim == 3:
            x_numpy= project(x_numpy)

        self.gui.circles(x_numpy, radius=1.5, color=0x66CCFF)
        self.gui.show()

        if self.recorder is None:
            return

        print("Recording frame: ", len(self.recorder.frame_data) + 1)
        self.recorder.capture(
            x_numpy,
            self.particles.is_boundary_particle.to_numpy().astype(np.uint32)
        )
    
    def _is_point_in_single_region(self, point, region_config):
        """检查点是否在单个区域内"""
        region_type = region_config.get("type", "rectangle")
        params = region_config.get("params", {})
        
        if region_type == "rectangle":
            rect_range = params.get("range", [])
            if len(rect_range) != self.dim:
                return False
            
            for d in range(self.dim):
                if point[d] < rect_range[d][0] or point[d] > rect_range[d][1]:
                    return False
            return True
            
        elif region_type == "ellipse":
            center = params.get("center", [0.5] * self.dim)
            semi_axes = params.get("semi_axes", [0.1] * self.dim)
            
            if len(center) != self.dim or len(semi_axes) != self.dim:
                return False
            
            sum_normalized = 0.0
            for d in range(self.dim):
                diff = point[d] - center[d]
                sum_normalized += (diff / semi_axes[d])**2
            return sum_normalized <= 1.0
        
        return False

    def _weight_gauss_data_to_grid(self):
        """将高斯积分点的应力应变数据按权重加权到网格位置"""
        from Geometry.GaussQuadrature import GaussQuadrature
        import numpy as np
        
        # 获取粒子数据
        particle_stress = self.particles.stress.to_numpy()
        particle_strain = self.particles.strain.to_numpy()
        particle_positions = self.particles.x.to_numpy()
        
        # 获取高斯积分点配置
        ppg = self.particles.particles_per_grid
        grid_size = self.particles.grid_size
        dx = 1.0 / grid_size
        
        try:
            n_1d = GaussQuadrature.validate_particles_per_grid(ppg)
        except ValueError:
            # 如果不是完全平方数，使用最接近的
            sqrt_n = int(ppg ** 0.5)
            if sqrt_n * sqrt_n < ppg:
                sqrt_n += 1
            n_1d = min(sqrt_n, 10)
        
        # 获取高斯积分点的权重
        gauss_positions, gauss_weights = GaussQuadrature.get_2d_grid_points_and_weights(n_1d, dx)
        
        # 创建网格数据存储
        grid_stress = {}
        grid_strain = {}
        grid_weights = {}
        
        # 遍历所有粒子，将它们的数据加权到对应的网格中心
        for p_idx, pos in enumerate(particle_positions):
            x, y = pos[0], pos[1]
            
            # 找到粒子所属的网格
            grid_i = int(x / dx)
            grid_j = int(y / dx)
            
            # 确保网格索引在有效范围内
            if 0 <= grid_i < grid_size and 0 <= grid_j < grid_size:
                grid_center_x = grid_i * dx
                grid_center_y = grid_j * dx
                
                # 找到粒子在该网格中对应的高斯积分点索引
                gauss_idx = -1
                min_dist = float('inf')
                for g_idx, g_pos in enumerate(gauss_positions):
                    expected_x = grid_center_x + g_pos[0]
                    expected_y = grid_center_y + g_pos[1]
                    dist = (x - expected_x)**2 + (y - expected_y)**2
                    if dist < min_dist:
                        min_dist = dist
                        gauss_idx = g_idx
                
                if gauss_idx >= 0:
                    # 使用对应的高斯积分权重
                    weight = gauss_weights[gauss_idx]
                    grid_key = (grid_i, grid_j)
                    
                    if grid_key not in grid_stress:
                        grid_stress[grid_key] = np.zeros_like(particle_stress[p_idx])
                        grid_strain[grid_key] = np.zeros_like(particle_strain[p_idx])
                        grid_weights[grid_key] = 0.0
                    
                    # 加权累加
                    grid_stress[grid_key] += particle_stress[p_idx] * weight
                    grid_strain[grid_key] += particle_strain[p_idx] * weight
                    grid_weights[grid_key] += weight
        
        # 归一化并生成最终的网格中心数据
        final_positions = []
        final_stress = []
        final_strain = []
        
        for (grid_i, grid_j), total_weight in grid_weights.items():
            if total_weight > 1e-10:  # 避免除零
                # 网格中心位置
                center_x = grid_i * dx
                center_y = grid_j * dx
                final_positions.append([center_x, center_y])
                
                # 归一化的应力应变
                final_stress.append(grid_stress[(grid_i, grid_j)] / total_weight)
                final_strain.append(grid_strain[(grid_i, grid_j)] / total_weight)
        
        print(f"高斯积分点加权完成: {len(particle_positions)} 个粒子 -> {len(final_positions)} 个网格点")
        
        return np.array(final_stress), np.array(final_strain), np.array(final_positions)

    def _is_point_in_regions(self, point, regions_config):
        """检查点是否在指定区域内（支持多个区域）"""
        # 如果是单个区域配置（字典），转换为列表
        if isinstance(regions_config, dict):
            regions_config = [regions_config]
        
        # 检查点是否在任意一个区域内
        for region in regions_config:
            if self._is_point_in_single_region(point, region):
                return True
        
        return False

    def save_stress_strain_data(self, frame_number):
        """保存最终帧的应力和应变数据"""
        import json
        import os
        from datetime import datetime
        
        # 计算当前状态的应力和应变
        print("正在计算应力和应变...")
        self.solver.compute_stress_strain_with_averaging()
        
        # 检查采样方式，如果是高斯采样则进行加权处理
        if self.particles.sampling_method == "gauss":
            print("检测到高斯积分点采样，进行权重加权...")
            stress_data, strain_data, positions = self._weight_gauss_data_to_grid()
        else:
            # 获取原始粒子的应力和应变数据
            stress_data = self.particles.stress.to_numpy()
            strain_data = self.particles.strain.to_numpy()
            positions = self.particles.x.to_numpy()
        
        # 过滤掉位置为(0,0)的粒子
        if self.dim == 2:
            # 2D情况：过滤(0,0)位置
            valid_mask = ~((positions[:, 0] == 0.0) & (positions[:, 1] == 0.0))
        else:
            # 3D情况：过滤(0,0,0)位置
            valid_mask = ~((positions[:, 0] == 0.0) & (positions[:, 1] == 0.0) & (positions[:, 2] == 0.0))
        
        # 检查是否有应力输出区域配置
        stress_output_regions = self.cfg.get("stress_output_regions", None)
        # 为了向后兼容，也检查单数形式
        if stress_output_regions is None:
            stress_output_regions = self.cfg.get("stress_output_region", None)
        
        if stress_output_regions is not None:
            print("应用应力输出区域过滤...")
            # 应用区域过滤
            region_mask = np.array([self._is_point_in_regions(pos, stress_output_regions) for pos in positions])
            valid_mask = valid_mask & region_mask
            print(f"区域过滤后剩余 {np.sum(region_mask)} / {len(positions)} 个粒子")
        
        # 应用过滤掩码
        filtered_stress_data = stress_data[valid_mask]
        filtered_strain_data = strain_data[valid_mask]
        filtered_positions = positions[valid_mask]
        
        print(f"Filtered out {np.sum(~valid_mask)} particles at origin position")
        print(f"Saving data for {len(filtered_positions)} particles")
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = "stress_strain_output"
        timestamped_dir = f"frame_{frame_number}_{timestamp}"
        output_dir = os.path.join(base_output_dir, timestamped_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为numpy文件（用于可视化）
        np.save(f"{output_dir}/stress_frame_{frame_number}.npy", filtered_stress_data)
        np.save(f"{output_dir}/strain_frame_{frame_number}.npy", filtered_strain_data)
        np.save(f"{output_dir}/positions_frame_{frame_number}.npy", filtered_positions)
        
        # 计算应力标量值（von Mises应力）- 使用过滤后的数据
        von_mises_stress = []
        for i in range(filtered_stress_data.shape[0]):
            if self.dim == 2:
                s = filtered_stress_data[i]
                # 2D von Mises应力
                von_mises = np.sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2)
            else:
                # 3D von Mises应力
                s = filtered_stress_data[i]
                von_mises = np.sqrt(0.5*((s[0,0]-s[1,1])**2 + (s[1,1]-s[2,2])**2 + (s[2,2]-s[0,0])**2) + 3*(s[0,1]**2 + s[1,2]**2 + s[2,0]**2))
            von_mises_stress.append(von_mises)
        
        # 保存统计信息
        stats = {
            "frame": frame_number,
            "n_particles": int(filtered_stress_data.shape[0]),
            "n_particles_total": int(stress_data.shape[0]),
            "n_particles_filtered": int(np.sum(~valid_mask)),
            "dimension": int(self.dim),
            "stress_output_regions": stress_output_regions,
            "von_mises_stress": {
                "min": float(np.min(von_mises_stress)),
                "max": float(np.max(von_mises_stress)),
                "mean": float(np.mean(von_mises_stress)),
                "std": float(np.std(von_mises_stress))
            }
        }
        
        with open(f"{output_dir}/stress_strain_stats_frame_{frame_number}.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"应力应变数据已保存到 {output_dir}/")
        print(f"von Mises应力范围: {stats['von_mises_stress']['min']:.3e} - {stats['von_mises_stress']['max']:.3e}")
        print(f"平均von Mises应力: {stats['von_mises_stress']['mean']:.3e}")
        
        return output_dir

    def run_static_solve(self):
        """执行静力学求解 - 只进行一步求解直到受力平衡"""
        print("开始静力学求解...")
        self.step()  # 执行一步求解
        print("静力学求解完成")
        
        # 保存结果
        self.save_stress_strain_data(1)
        
        # 渲染最终结果
        self.render()

if __name__ == "__main__":

    cfg=Config("config/config_2d.json")
    float_type=ti.f32 if cfg.get("float_type", "f32") == "f32" else ti.f64
    arch=cfg.get("arch", "cpu")
    if arch == "cuda":
        arch = ti.cuda
    elif arch == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, default_fp=float_type, device_memory_GB=20)
    mpm = ImplicitMPM(cfg)

    # 检查是否是静力学求解模式
    if mpm.static_solve:
        mpm.run_static_solve()
        # 保持GUI开启以显示结果
        while mpm.gui.running:
            mpm.gui.show()
    else:
        # 原有的动态求解模式
        i = 0
        while mpm.gui.running:
            mpm.step()
            
            mpm.render()

            i += 1

            # 自动停止条件
            if i >= mpm.recorder.max_frames:
                break

        # 在最后一帧记录应力和应变数据
        print("记录最终帧的应力和应变数据...")
        mpm.save_stress_strain_data(i)

        if mpm.recorder is None:
            exit()
        print("Playback finished.")
        # 播放录制动画
        mpm.recorder.play(loop=True, fps=60)