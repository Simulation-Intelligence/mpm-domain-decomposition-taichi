# tools/performance_stats.py
"""
Schwarz 域分解方法的性能统计与可视化工具
"""

import time
import numpy as np
from copy import deepcopy
# 使用非交互式后端避免 macOS 上的 matplotlib 错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os


class SchwarzPerformanceStats:
    """Schwarz 域分解性能统计收集器"""

    def __init__(self, max_frames_to_keep=1000):
        """
        初始化性能统计

        参数:
            max_frames_to_keep: 最多保留的帧数（防止内存泄漏）
        """
        self.max_frames_to_keep = max_frames_to_keep
        self.reset()

    def reset(self):
        """重置所有统计"""
        # 每帧数据
        self.frame_data = []

        # 当前帧临时数据
        self._init_current_frame()
        self._frame_start_time = 0

    def _init_current_frame(self):
        """初始化当前帧数据结构"""
        self.current_frame = {
            'schwarz_iters': 0,
            'converged': False,
            'residuals': [],
            'big_domain_newton_iters': [],      # 每次 schwarz 迭代的牛顿数
            'small_domain_newton_iters': [],    # 每次 schwarz 迭代的牛顿数（累积所有子步）
            'small_domain_substep_newton_iters': [],  # 每个子步的牛顿数详情
            'big_domain_solve_time': [],
            'small_domain_solve_time': [],
            'boundary_exchange_time': [],
            'p2g_time': 0,
            'g2p_time': 0,
            'total_frame_time': 0,
        }

    def start_frame(self):
        """开始新的一帧"""
        self._init_current_frame()
        self._frame_start_time = time.perf_counter()

    def record_schwarz_iteration(self,
                                  residual: float = 0.0,
                                  big_newton_iters: int = 0,
                                  small_newton_iters: int = 0,
                                  small_substep_iters: list = None,
                                  big_solve_time: float = 0,
                                  small_solve_time: float = 0,
                                  boundary_time: float = 0):
        """
        记录一次 Schwarz 迭代的数据

        Args:
            residual: Schwarz 残差
            big_newton_iters: 大域牛顿迭代次数
            small_newton_iters: 小域牛顿迭代次数（所有子步累积）
            small_substep_iters: 小域每个子步的牛顿迭代次数列表
            big_solve_time: 大域求解时间
            small_solve_time: 小域求解时间
            boundary_time: 边界交换时间
        """
        self.current_frame['schwarz_iters'] += 1
        self.current_frame['residuals'].append(residual)
        self.current_frame['big_domain_newton_iters'].append(big_newton_iters)
        self.current_frame['small_domain_newton_iters'].append(small_newton_iters)
        if small_substep_iters:
            self.current_frame['small_domain_substep_newton_iters'].append(small_substep_iters)
        self.current_frame['big_domain_solve_time'].append(big_solve_time)
        self.current_frame['small_domain_solve_time'].append(small_solve_time)
        self.current_frame['boundary_exchange_time'].append(boundary_time)

    def end_frame(self, converged: bool = True):
        """结束当前帧，保存数据"""
        self.current_frame['total_frame_time'] = time.perf_counter() - self._frame_start_time
        self.current_frame['converged'] = converged
        # 使用 deepcopy 确保嵌套列表被完全复制，避免内存泄漏
        self.frame_data.append(deepcopy(self.current_frame))

        # 限制 frame_data 大小，防止内存泄漏
        if len(self.frame_data) > self.max_frames_to_keep:
            # 只保留最近的 max_frames_to_keep 帧
            self.frame_data = self.frame_data[-self.max_frames_to_keep:]

    def record_p2g_time(self, elapsed: float):
        """记录 P2G 时间"""
        self.current_frame['p2g_time'] = elapsed

    def record_g2p_time(self, elapsed: float):
        """记录 G2P 时间"""
        self.current_frame['g2p_time'] = elapsed

    # ============ 数据获取方法 ============

    def get_schwarz_iter_counts(self):
        """获取每帧的 Schwarz 迭代次数"""
        return [f['schwarz_iters'] for f in self.frame_data]

    def get_total_newton_iters_per_frame(self):
        """获取每帧的总牛顿迭代次数"""
        big = [sum(f['big_domain_newton_iters']) for f in self.frame_data]
        small = [sum(f['small_domain_newton_iters']) for f in self.frame_data]
        return big, small

    def get_average_schwarz_iters(self, last_n: int = None):
        """获取平均 Schwarz 迭代次数"""
        counts = self.get_schwarz_iter_counts()
        if last_n and len(counts) > last_n:
            counts = counts[-last_n:]
        return np.mean(counts) if counts else 0

    # ============ 可视化方法 ============

    def plot_newton_iters_vs_schwarz_average(self, save_path: str = None, show: bool = True):
        """
        绘制：牛顿迭代数 vs Schwarz 迭代次数（所有帧平均）

        对所有帧的数据进行聚合，计算每个 Schwarz 迭代位置的平均牛顿迭代次数
        """
        if not self.frame_data:
            print("No data to plot")
            return

        # 找到最大 Schwarz 迭代次数
        max_schwarz = max(f['schwarz_iters'] for f in self.frame_data)

        if max_schwarz == 0:
            print("No Schwarz iterations to plot")
            return

        # 收集每个 Schwarz 迭代位置的牛顿迭代数据
        big_by_schwarz_idx = [[] for _ in range(max_schwarz)]
        small_by_schwarz_idx = [[] for _ in range(max_schwarz)]

        for frame in self.frame_data:
            for i, big_n in enumerate(frame['big_domain_newton_iters']):
                big_by_schwarz_idx[i].append(big_n)
            for i, small_n in enumerate(frame['small_domain_newton_iters']):
                small_by_schwarz_idx[i].append(small_n)

        # 计算每个位置的均值和标准差
        schwarz_iters = []
        big_means = []
        big_stds = []
        small_means = []
        small_stds = []
        sample_counts = []

        for i in range(max_schwarz):
            if len(big_by_schwarz_idx[i]) > 0:
                schwarz_iters.append(i + 1)
                big_means.append(np.mean(big_by_schwarz_idx[i]))
                big_stds.append(np.std(big_by_schwarz_idx[i]))
                small_means.append(np.mean(small_by_schwarz_idx[i]))
                small_stds.append(np.std(small_by_schwarz_idx[i]))
                sample_counts.append(len(big_by_schwarz_idx[i]))

        schwarz_iters = np.array(schwarz_iters)
        big_means = np.array(big_means)
        big_stds = np.array(big_stds)
        small_means = np.array(small_means)
        small_stds = np.array(small_stds)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：平均值带误差带
        ax1.fill_between(schwarz_iters, big_means - big_stds, big_means + big_stds,
                         color='blue', alpha=0.2)
        ax1.plot(schwarz_iters, big_means, 'b-o', label='Big Domain', linewidth=2, markersize=6)

        ax1.fill_between(schwarz_iters, small_means - small_stds, small_means + small_stds,
                         color='red', alpha=0.2)
        ax1.plot(schwarz_iters, small_means, 'r-s', label='Small Domain', linewidth=2, markersize=6)

        ax1.set_xlabel('Schwarz Iteration', fontsize=12)
        ax1.set_ylabel('Newton Iterations (mean ± std)', fontsize=12)
        ax1.set_title(f'Average Newton Iterations vs Schwarz Iteration\n(Aggregated over {len(self.frame_data)} frames)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(schwarz_iters)

        # 右图：样本数量（每个 Schwarz 迭代位置有多少帧的数据）
        ax2.bar(schwarz_iters, sample_counts, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Schwarz Iteration', fontsize=12)
        ax2.set_ylabel('Number of Frames', fontsize=12)
        ax2.set_title('Sample Count per Schwarz Iteration\n(How many frames reached this iteration)', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(schwarz_iters)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        # 打印统计
        print(f"\n{'='*60}")
        print("Average Newton Iterations per Schwarz Iteration (All Frames)")
        print(f"{'='*60}")
        print(f"{'Schwarz Iter':<15}{'Big Domain':<20}{'Small Domain':<20}{'# Frames':<10}")
        print(f"{'-'*60}")
        for i, (si, bm, bs, sm, ss, sc) in enumerate(zip(
            schwarz_iters, big_means, big_stds, small_means, small_stds, sample_counts)):
            print(f"{int(si):<15}{bm:.2f} ± {bs:.2f}       {sm:.2f} ± {ss:.2f}       {sc}")
        print(f"{'='*60}")

    def plot_newton_iters_vs_schwarz(self, frame_idx: int = -1, save_path: str = None, show: bool = True):
        """
        绘制：牛顿迭代数 vs Schwarz 迭代次数（单帧）
        """
        if not self.frame_data:
            print("No data to plot")
            return

        frame = self.frame_data[frame_idx]
        schwarz_iters = range(1, frame['schwarz_iters'] + 1)

        if frame['schwarz_iters'] == 0:
            print(f"Frame {frame_idx} has no Schwarz iterations")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(schwarz_iters, frame['big_domain_newton_iters'],
                'b-o', label='Big Domain (dt_large)', linewidth=2, markersize=8)
        ax.plot(schwarz_iters, frame['small_domain_newton_iters'],
                'r-s', label='Small Domain (dt_small, accumulated)', linewidth=2, markersize=8)

        ax.set_xlabel('Schwarz Iteration', fontsize=12)
        ax.set_ylabel('Newton Iterations', fontsize=12)
        actual_frame_idx = frame_idx if frame_idx >= 0 else len(self.frame_data) + frame_idx
        ax.set_title(f'Newton Iterations vs Schwarz Iteration (Frame {actual_frame_idx})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, frame['schwarz_iters'] + 1))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_schwarz_convergence(self, frame_idx: int = -1, save_path: str = None, show: bool = True):
        """
        绘制：Schwarz 残差收敛曲线（单帧）
        """
        if not self.frame_data:
            print("No data to plot")
            return

        frame = self.frame_data[frame_idx]
        residuals = frame['residuals']

        if len(residuals) == 0:
            print(f"Frame {frame_idx} has no residual data")
            return

        # 过滤掉 0 或负值（对数坐标不能显示）
        valid_residuals = [(i+1, r) for i, r in enumerate(residuals) if r > 0]
        if not valid_residuals:
            print("No valid residuals to plot")
            return

        iters, vals = zip(*valid_residuals)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogy(iters, vals, 'g-o', linewidth=2, markersize=8)

        ax.set_xlabel('Schwarz Iteration', fontsize=12)
        ax.set_ylabel('Residual (log scale)', fontsize=12)
        actual_frame_idx = frame_idx if frame_idx >= 0 else len(self.frame_data) + frame_idx
        ax.set_title(f'Schwarz Convergence (Frame {actual_frame_idx})', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_schwarz_iters_over_frames(self, window_size: int = 10, save_path: str = None, show: bool = True):
        """
        绘制：Schwarz 迭代次数随帧变化 + 滑动平均
        """
        if not self.frame_data:
            print("No data to plot")
            return

        schwarz_counts = self.get_schwarz_iter_counts()
        frames = range(len(schwarz_counts))

        # 计算滑动平均
        moving_avg = []
        for i in range(len(schwarz_counts)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(schwarz_counts[start:i+1]))

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(frames, schwarz_counts, 'b-', alpha=0.5, linewidth=1, label='Per Frame')
        ax.plot(frames, moving_avg, 'r-', linewidth=2,
                label=f'Moving Average (window={window_size})')

        # 标记未收敛的帧
        unconverged = [i for i, f in enumerate(self.frame_data) if not f['converged']]
        if unconverged:
            ax.scatter(unconverged, [schwarz_counts[i] for i in unconverged],
                      c='red', s=50, marker='x', label='Unconverged', zorder=5)

        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Schwarz Iterations', fontsize=12)
        ax.set_title('Schwarz Iterations Over Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        # 打印统计信息
        print(f"Total frames: {len(schwarz_counts)}")
        print(f"Average Schwarz iters: {np.mean(schwarz_counts):.2f}")
        print(f"Std Schwarz iters: {np.std(schwarz_counts):.2f}")
        print(f"Max Schwarz iters: {max(schwarz_counts)}")
        print(f"Min Schwarz iters: {min(schwarz_counts)}")
        print(f"Unconverged frames: {len(unconverged)}")

    def plot_total_newton_iters_over_frames(self, save_path: str = None, show: bool = True):
        """
        绘制：每帧的总牛顿迭代数（两域合计，堆叠图）
        """
        if not self.frame_data:
            print("No data to plot")
            return

        big_newton, small_newton = self.get_total_newton_iters_per_frame()
        total_newton = [b + s for b, s in zip(big_newton, small_newton)]

        frames = range(len(self.frame_data))

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.stackplot(frames, big_newton, small_newton,
                     labels=['Big Domain', 'Small Domain'],
                     colors=['#3498db', '#e74c3c'], alpha=0.7)
        ax.plot(frames, total_newton, 'k-', linewidth=1.5, label='Total')

        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Total Newton Iterations', fontsize=12)
        ax.set_title('Newton Iterations per Frame (Stacked)', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_convergence_rate(self, frame_idx: int = -1, save_path: str = None, show: bool = True):
        """
        绘制：收敛率 (r_{k+1} / r_k)
        """
        if not self.frame_data:
            print("No data to plot")
            return

        frame = self.frame_data[frame_idx]
        residuals = [r for r in frame['residuals'] if r > 0]

        if len(residuals) < 2:
            print("Not enough residuals to compute convergence rate")
            return

        rates = [residuals[i+1] / residuals[i] for i in range(len(residuals)-1)]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(range(2, len(residuals) + 1), rates, 'b-o', linewidth=2, markersize=8)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Rate = 1 (no convergence)')
        ax.axhline(y=0.5, color='g', linestyle=':', alpha=0.7, label='Rate = 0.5 (good)')

        ax.set_xlabel('Schwarz Iteration', fontsize=12)
        ax.set_ylabel('Convergence Rate (r_{k+1}/r_k)', fontsize=12)
        actual_frame_idx = frame_idx if frame_idx >= 0 else len(self.frame_data) + frame_idx
        ax.set_title(f'Schwarz Convergence Rate (Frame {actual_frame_idx})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, min(2, max(rates) * 1.2)])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_time_breakdown(self, save_path: str = None, show: bool = True):
        """
        绘制：时间开销分解
        """
        if not self.frame_data:
            print("No data to plot")
            return

        big_times = [sum(f['big_domain_solve_time']) for f in self.frame_data]
        small_times = [sum(f['small_domain_solve_time']) for f in self.frame_data]
        boundary_times = [sum(f['boundary_exchange_time']) for f in self.frame_data]
        total_times = [f['total_frame_time'] for f in self.frame_data]

        frames = range(len(self.frame_data))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：时间随帧变化
        ax1.plot(frames, big_times, 'b-', label='Big Domain Solve', alpha=0.7)
        ax1.plot(frames, small_times, 'r-', label='Small Domain Solve', alpha=0.7)
        ax1.plot(frames, total_times, 'k-', label='Total Frame Time', linewidth=2)
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel('Time (s)', fontsize=12)
        ax1.set_title('Solve Time per Frame', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 右图：饼图（总体）
        total_big = sum(big_times)
        total_small = sum(small_times)
        total_boundary = sum(boundary_times)
        other = max(0, sum(total_times) - total_big - total_small - total_boundary)

        sizes = [total_big, total_small, total_boundary, other]
        labels = [f'Big Domain\n{total_big:.2f}s',
                  f'Small Domain\n{total_small:.2f}s',
                  f'Boundary Exchange\n{total_boundary:.2f}s',
                  f'Other\n{other:.2f}s']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']

        # 只显示非零的部分
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0.001]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Time Distribution', fontsize=14)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_newton_iter_distribution(self, save_path: str = None, show: bool = True):
        """
        绘制：牛顿迭代次数分布直方图
        """
        if not self.frame_data:
            print("No data to plot")
            return

        all_big_iters = []
        all_small_iters = []
        for f in self.frame_data:
            all_big_iters.extend(f['big_domain_newton_iters'])
            all_small_iters.extend(f['small_domain_newton_iters'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 大域
        if all_big_iters:
            ax1.hist(all_big_iters, bins=range(0, max(all_big_iters)+2),
                    color='#3498db', alpha=0.7, edgecolor='black')
            ax1.axvline(np.mean(all_big_iters), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_big_iters):.1f}')
            ax1.set_xlabel('Newton Iterations', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Big Domain Newton Iterations Distribution', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 小域
        if all_small_iters:
            ax2.hist(all_small_iters, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(all_small_iters), color='blue', linestyle='--',
                       label=f'Mean: {np.mean(all_small_iters):.1f}')
            ax2.set_xlabel('Newton Iterations (accumulated)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Small Domain Newton Iterations Distribution', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_all_frames_residuals(self, max_frames: int = 20, save_path: str = None, show: bool = True):
        """
        绘制：多帧的残差收敛曲线（叠加）
        """
        if not self.frame_data:
            print("No data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # 选择要显示的帧
        n_frames = len(self.frame_data)
        if n_frames <= max_frames:
            frame_indices = range(n_frames)
        else:
            # 均匀采样
            frame_indices = np.linspace(0, n_frames-1, max_frames, dtype=int)

        cmap = plt.cm.viridis

        for i, idx in enumerate(frame_indices):
            frame = self.frame_data[idx]
            residuals = [r for r in frame['residuals'] if r > 0]
            if len(residuals) > 0:
                color = cmap(i / len(frame_indices))
                ax.semilogy(range(1, len(residuals)+1), residuals,
                           '-o', color=color, alpha=0.6, markersize=4, linewidth=1)

        ax.set_xlabel('Schwarz Iteration', fontsize=12)
        ax.set_ylabel('Residual (log scale)', fontsize=12)
        ax.set_title(f'Schwarz Convergence Curves ({len(frame_indices)} frames)', fontsize=14)
        ax.grid(True, alpha=0.3)

        # 添加 colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_frames-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Frame Index', fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.frame_data:
            print("No data collected")
            return

        print("=" * 70)
        print("               SCHWARZ PERFORMANCE SUMMARY REPORT")
        print("=" * 70)

        n_frames = len(self.frame_data)
        schwarz_counts = self.get_schwarz_iter_counts()
        big_newton, small_newton = self.get_total_newton_iters_per_frame()
        total_newton_big = sum(big_newton)
        total_newton_small = sum(small_newton)

        print(f"\n{'='*30} Basic Statistics {'='*30}")
        print(f"  Total frames simulated:           {n_frames}")
        print(f"  Total Schwarz iterations:         {sum(schwarz_counts)}")
        print(f"  Total Newton iterations (Big):    {total_newton_big}")
        print(f"  Total Newton iterations (Small):  {total_newton_small}")
        print(f"  Total Newton iterations:          {total_newton_big + total_newton_small}")

        print(f"\n{'='*30} Schwarz Iterations {'='*28}")
        print(f"  Mean per frame:    {np.mean(schwarz_counts):.2f}")
        print(f"  Std per frame:     {np.std(schwarz_counts):.2f}")
        print(f"  Min per frame:     {min(schwarz_counts)}")
        print(f"  Max per frame:     {max(schwarz_counts)}")

        # 未收敛帧
        unconverged = [i for i, f in enumerate(self.frame_data) if not f['converged']]
        print(f"  Unconverged frames: {len(unconverged)}")
        if unconverged and len(unconverged) <= 10:
            print(f"    Indices: {unconverged}")
        elif unconverged:
            print(f"    First 10: {unconverged[:10]}...")

        print(f"\n{'='*30} Newton Iterations {'='*29}")
        print(f"  Big Domain:")
        print(f"    Mean per Schwarz iter: {np.mean([i for f in self.frame_data for i in f['big_domain_newton_iters']]):.2f}")
        print(f"    Mean per frame:        {np.mean(big_newton):.2f}")
        print(f"  Small Domain:")
        print(f"    Mean per Schwarz iter: {np.mean([i for f in self.frame_data for i in f['small_domain_newton_iters']]):.2f}")
        print(f"    Mean per frame:        {np.mean(small_newton):.2f}")

        print(f"\n{'='*30} Timing {'='*40}")
        total_time = sum(f['total_frame_time'] for f in self.frame_data)
        big_solve_time = sum(sum(f['big_domain_solve_time']) for f in self.frame_data)
        small_solve_time = sum(sum(f['small_domain_solve_time']) for f in self.frame_data)

        print(f"  Total simulation time:     {total_time:.2f} s")
        print(f"  Average time per frame:    {total_time/n_frames:.4f} s")
        print(f"  Big domain solve time:     {big_solve_time:.2f} s ({100*big_solve_time/total_time:.1f}%)")
        print(f"  Small domain solve time:   {small_solve_time:.2f} s ({100*small_solve_time/total_time:.1f}%)")

        print("=" * 70)

        return {
            'n_frames': n_frames,
            'total_schwarz_iters': sum(schwarz_counts),
            'mean_schwarz_iters': np.mean(schwarz_counts),
            'total_newton_big': total_newton_big,
            'total_newton_small': total_newton_small,
            'total_time': total_time,
            'unconverged_frames': len(unconverged),
        }

    def save_to_file(self, filepath: str):
        """保存统计数据到 JSON 文件"""
        data = {
            'frame_data': self.frame_data,
            'summary': self.generate_summary_report() if self.frame_data else {}
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Statistics saved to {filepath}")

    def save_all_plots(self, output_dir: str, show: bool = False):
        """保存所有图表到指定目录"""
        os.makedirs(output_dir, exist_ok=True)

        if not self.frame_data:
            print("No data to plot")
            return

        print(f"Saving plots to {output_dir}...")

        # 1. Schwarz 迭代次数随帧变化
        self.plot_schwarz_iters_over_frames(
            save_path=os.path.join(output_dir, 'schwarz_iters_over_frames.png'),
            show=show
        )

        # 2. 总牛顿迭代数
        self.plot_total_newton_iters_over_frames(
            save_path=os.path.join(output_dir, 'total_newton_iters.png'),
            show=show
        )

        # 3. 最后一帧的牛顿迭代 vs Schwarz
        self.plot_newton_iters_vs_schwarz(
            frame_idx=-1,
            save_path=os.path.join(output_dir, 'newton_vs_schwarz_last_frame.png'),
            show=show
        )

        # 3b. 所有帧平均的牛顿迭代 vs Schwarz
        self.plot_newton_iters_vs_schwarz_average(
            save_path=os.path.join(output_dir, 'newton_vs_schwarz_average.png'),
            show=show
        )

        # 4. 最后一帧的收敛曲线
        self.plot_schwarz_convergence(
            frame_idx=-1,
            save_path=os.path.join(output_dir, 'convergence_last_frame.png'),
            show=show
        )

        # 5. 收敛率
        self.plot_convergence_rate(
            frame_idx=-1,
            save_path=os.path.join(output_dir, 'convergence_rate_last_frame.png'),
            show=show
        )

        # 6. 时间分解
        self.plot_time_breakdown(
            save_path=os.path.join(output_dir, 'time_breakdown.png'),
            show=show
        )

        # 7. 牛顿迭代分布
        self.plot_newton_iter_distribution(
            save_path=os.path.join(output_dir, 'newton_iter_distribution.png'),
            show=show
        )

        # 8. 多帧残差曲线
        self.plot_all_frames_residuals(
            save_path=os.path.join(output_dir, 'all_frames_residuals.png'),
            show=show
        )

        # 确保所有图形都被关闭，避免 matplotlib 后端错误
        plt.close('all')

        print(f"All plots saved to {output_dir}")


# ============================================================================
# 单域模拟器性能统计类
# ============================================================================

class SingleDomainPerformanceStats:
    """单域 MPM 模拟器性能统计收集器"""

    def __init__(self, max_frames_to_keep=1000):
        """
        初始化性能统计

        参数:
            max_frames_to_keep: 最多保留的帧数（防止内存泄漏）
        """
        self.max_frames_to_keep = max_frames_to_keep
        self.reset()

    def reset(self):
        """重置所有统计"""
        self.frame_data = []
        self._frame_start_time = 0

    def start_frame(self):
        """开始新的一帧"""
        self._frame_start_time = time.perf_counter()

    def record_frame(self, newton_iters: int = 0, solve_time: float = 0):
        """
        记录一帧的数据

        Args:
            newton_iters: 牛顿迭代次数
            solve_time: 求解时间
        """
        total_time = time.perf_counter() - self._frame_start_time
        self.frame_data.append({
            'newton_iters': newton_iters,
            'solve_time': solve_time,
            'total_frame_time': total_time,
        })

        # 限制 frame_data 大小，防止内存泄漏
        if len(self.frame_data) > self.max_frames_to_keep:
            self.frame_data = self.frame_data[-self.max_frames_to_keep:]

    # ============ 数据获取方法 ============

    def get_newton_iter_counts(self):
        """获取每帧的牛顿迭代次数"""
        return [f['newton_iters'] for f in self.frame_data]

    def get_average_newton_iters(self, last_n: int = None):
        """获取平均牛顿迭代次数"""
        counts = self.get_newton_iter_counts()
        if last_n and len(counts) > last_n:
            counts = counts[-last_n:]
        return np.mean(counts) if counts else 0

    # ============ 可视化方法 ============

    def plot_newton_iters_over_frames(self, window_size: int = 10, save_path: str = None, show: bool = True):
        """
        绘制：牛顿迭代次数随帧变化 + 滑动平均
        """
        if not self.frame_data:
            print("No data to plot")
            return

        newton_counts = self.get_newton_iter_counts()
        frames = range(len(newton_counts))

        # 计算滑动平均
        moving_avg = []
        for i in range(len(newton_counts)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(newton_counts[start:i+1]))

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(frames, newton_counts, 'b-', alpha=0.5, linewidth=1, label='Per Frame')
        ax.plot(frames, moving_avg, 'r-', linewidth=2,
                label=f'Moving Average (window={window_size})')

        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Newton Iterations', fontsize=12)
        ax.set_title('Newton Iterations Over Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        # 打印统计信息
        print(f"Total frames: {len(newton_counts)}")
        print(f"Average Newton iters: {np.mean(newton_counts):.2f}")
        print(f"Std Newton iters: {np.std(newton_counts):.2f}")
        print(f"Max Newton iters: {max(newton_counts)}")
        print(f"Min Newton iters: {min(newton_counts)}")

    def plot_newton_iter_distribution(self, save_path: str = None, show: bool = True):
        """
        绘制：牛顿迭代次数分布直方图
        """
        if not self.frame_data:
            print("No data to plot")
            return

        newton_counts = self.get_newton_iter_counts()

        fig, ax = plt.subplots(figsize=(10, 6))

        if newton_counts:
            bins = range(0, max(newton_counts) + 2)
            ax.hist(newton_counts, bins=bins, color='#3498db', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(newton_counts), color='red', linestyle='--',
                       label=f'Mean: {np.mean(newton_counts):.1f}')
            ax.set_xlabel('Newton Iterations', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Newton Iterations Distribution', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_time_per_frame(self, save_path: str = None, show: bool = True):
        """
        绘制：每帧时间开销
        """
        if not self.frame_data:
            print("No data to plot")
            return

        solve_times = [f['solve_time'] for f in self.frame_data]
        total_times = [f['total_frame_time'] for f in self.frame_data]
        frames = range(len(self.frame_data))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：时间随帧变化
        ax1.plot(frames, solve_times, 'b-', label='Solve Time', alpha=0.7)
        ax1.plot(frames, total_times, 'k-', label='Total Frame Time', linewidth=2)
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel('Time (s)', fontsize=12)
        ax1.set_title('Time per Frame', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：饼图（总体）
        total_solve = sum(solve_times)
        total_all = sum(total_times)
        other = max(0, total_all - total_solve)

        sizes = [total_solve, other]
        labels = [f'Solve\n{total_solve:.2f}s', f'Other\n{other:.2f}s']
        colors = ['#3498db', '#95a5a6']

        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0.001]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Time Distribution', fontsize=14)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_newton_vs_time(self, save_path: str = None, show: bool = True):
        """
        绘制：牛顿迭代次数 vs 求解时间（散点图）
        """
        if not self.frame_data:
            print("No data to plot")
            return

        newton_counts = self.get_newton_iter_counts()
        solve_times = [f['solve_time'] * 1000 for f in self.frame_data]  # 转换为毫秒

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(newton_counts, solve_times, alpha=0.5, c='blue', s=20)

        # 添加线性拟合
        if len(newton_counts) > 1:
            # 检查是否有足够的数据变化来进行拟合
            if np.std(newton_counts) > 1e-10:  # 有变化
                try:
                    z = np.polyfit(newton_counts, solve_times, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(newton_counts), max(newton_counts), 100)
                    ax.plot(x_line, p(x_line), 'r--', linewidth=2,
                           label=f'Linear fit: {z[0]:.2f}ms/iter')
                except np.linalg.LinAlgError:
                    # 拟合失败，跳过
                    print("Warning: Could not fit Newton iterations vs solve time (insufficient variation)")
            else:
                print("Warning: Newton iterations have no variation, skipping linear fit")

        ax.set_xlabel('Newton Iterations', fontsize=12)
        ax.set_ylabel('Solve Time (ms)', fontsize=12)
        ax.set_title('Newton Iterations vs Solve Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        # 打印统计
        if len(newton_counts) > 0:
            avg_time_per_newton = np.mean(solve_times) / np.mean(newton_counts) if np.mean(newton_counts) > 0 else 0
            print(f"Average time per Newton iteration: {avg_time_per_newton:.2f} ms")

    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.frame_data:
            print("No data collected")
            return

        print("=" * 70)
        print("           SINGLE DOMAIN PERFORMANCE SUMMARY REPORT")
        print("=" * 70)

        n_frames = len(self.frame_data)
        newton_counts = self.get_newton_iter_counts()
        total_newton = sum(newton_counts)

        print(f"\n{'='*30} Basic Statistics {'='*30}")
        print(f"  Total frames simulated:      {n_frames}")
        print(f"  Total Newton iterations:     {total_newton}")

        print(f"\n{'='*30} Newton Iterations {'='*29}")
        print(f"  Mean per frame:    {np.mean(newton_counts):.2f}")
        print(f"  Std per frame:     {np.std(newton_counts):.2f}")
        print(f"  Min per frame:     {min(newton_counts)}")
        print(f"  Max per frame:     {max(newton_counts)}")

        print(f"\n{'='*30} Timing {'='*40}")
        total_time = sum(f['total_frame_time'] for f in self.frame_data)
        solve_time = sum(f['solve_time'] for f in self.frame_data)

        print(f"  Total simulation time:     {total_time:.2f} s")
        print(f"  Average time per frame:    {total_time/n_frames:.4f} s")
        print(f"  Total solve time:          {solve_time:.2f} s ({100*solve_time/total_time:.1f}%)")
        if total_newton > 0:
            print(f"  Time per Newton iter:      {solve_time/total_newton*1000:.2f} ms")

        print("=" * 70)

        return {
            'n_frames': n_frames,
            'total_newton_iters': total_newton,
            'mean_newton_iters': np.mean(newton_counts),
            'total_time': total_time,
            'solve_time': solve_time,
        }

    def save_to_file(self, filepath: str):
        """保存统计数据到 JSON 文件"""
        data = {
            'frame_data': self.frame_data,
            'summary': self.generate_summary_report() if self.frame_data else {}
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Statistics saved to {filepath}")

    def save_all_plots(self, output_dir: str, show: bool = False):
        """保存所有图表到指定目录"""
        os.makedirs(output_dir, exist_ok=True)

        if not self.frame_data:
            print("No data to plot")
            return

        print(f"Saving plots to {output_dir}...")

        # 1. 牛顿迭代次数随帧变化
        try:
            self.plot_newton_iters_over_frames(
                save_path=os.path.join(output_dir, 'newton_iters_over_frames.png'),
                show=show
            )
        except Exception as e:
            print(f"Warning: Failed to plot newton_iters_over_frames: {e}")

        # 2. 牛顿迭代分布
        try:
            self.plot_newton_iter_distribution(
                save_path=os.path.join(output_dir, 'newton_iter_distribution.png'),
                show=show
            )
        except Exception as e:
            print(f"Warning: Failed to plot newton_iter_distribution: {e}")

        # 3. 时间开销
        try:
            self.plot_time_per_frame(
                save_path=os.path.join(output_dir, 'time_per_frame.png'),
                show=show
            )
        except Exception as e:
            print(f"Warning: Failed to plot time_per_frame: {e}")

        # 4. 牛顿迭代 vs 时间
        try:
            self.plot_newton_vs_time(
                save_path=os.path.join(output_dir, 'newton_vs_time.png'),
                show=show
            )
        except Exception as e:
            print(f"Warning: Failed to plot newton_vs_time: {e}")

        # 确保所有图形都被关闭
        plt.close('all')

        print(f"Plots saved to {output_dir} (some may have been skipped due to errors)")
