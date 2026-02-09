"""
内存使用分析工具
用于追踪 MPM 模拟过程中的内存使用情况
"""

import tracemalloc
import psutil
import os
import time
from collections import defaultdict
import json

class MemoryProfiler:
    """内存使用分析器"""

    def __init__(self, enabled=True, snapshot_top_n=10):
        """
        初始化内存分析器

        参数:
            enabled: 是否启用内存分析
            snapshot_top_n: 记录前N个最大内存分配
        """
        self.enabled = enabled
        self.snapshot_top_n = snapshot_top_n
        self.process = psutil.Process(os.getpid())

        # 记录数据
        self.checkpoints = []
        self.frame_data = []

        # 起始状态
        self.start_time = None
        self.start_memory_rss = None
        self.start_memory_vms = None

        if self.enabled:
            tracemalloc.start()
            self._record_start_state()

    def _record_start_state(self):
        """记录初始状态"""
        mem_info = self.process.memory_info()
        self.start_time = time.time()
        self.start_memory_rss = mem_info.rss / 1024 / 1024  # MB
        self.start_memory_vms = mem_info.vms / 1024 / 1024  # MB

    def checkpoint(self, label: str):
        """
        记录一个检查点

        参数:
            label: 检查点标签
        """
        if not self.enabled:
            return

        mem_info = self.process.memory_info()
        current_rss = mem_info.rss / 1024 / 1024  # MB
        current_vms = mem_info.vms / 1024 / 1024  # MB

        # 获取 tracemalloc 快照
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        # 记录前N个最大的内存分配
        top_allocations = []
        for stat in top_stats[:self.snapshot_top_n]:
            top_allocations.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })

        checkpoint_data = {
            'label': label,
            'timestamp': time.time() - self.start_time,
            'rss_mb': current_rss,
            'vms_mb': current_vms,
            'rss_delta_mb': current_rss - self.start_memory_rss,
            'vms_delta_mb': current_vms - self.start_memory_vms,
            'python_current_mb': tracemalloc.get_traced_memory()[0] / 1024 / 1024,
            'python_peak_mb': tracemalloc.get_traced_memory()[1] / 1024 / 1024,
            'top_allocations': top_allocations
        }

        self.checkpoints.append(checkpoint_data)

        # 打印到控制台
        print(f"\n[MemProfile] {label}")
        print(f"  RSS: {current_rss:.2f} MB (Δ {current_rss - self.start_memory_rss:+.2f} MB)")
        print(f"  VMS: {current_vms:.2f} MB (Δ {current_vms - self.start_memory_vms:+.2f} MB)")
        print(f"  Python traced: {checkpoint_data['python_current_mb']:.2f} MB (peak: {checkpoint_data['python_peak_mb']:.2f} MB)")

    def record_frame(self, frame_number: int, stage_data: dict = None):
        """
        记录帧级别的内存使用

        参数:
            frame_number: 帧号
            stage_data: 各阶段的内存数据（可选）
        """
        if not self.enabled:
            return

        mem_info = self.process.memory_info()
        current_rss = mem_info.rss / 1024 / 1024  # MB
        current_vms = mem_info.vms / 1024 / 1024  # MB

        frame_record = {
            'frame': frame_number,
            'timestamp': time.time() - self.start_time,
            'rss_mb': current_rss,
            'vms_mb': current_vms,
            'rss_delta_mb': current_rss - self.start_memory_rss,
            'python_current_mb': tracemalloc.get_traced_memory()[0] / 1024 / 1024,
        }

        if stage_data:
            frame_record['stages'] = stage_data

        self.frame_data.append(frame_record)

        # 每100帧打印一次摘要
        if frame_number % 100 == 0 and frame_number > 0:
            last_100_frames = self.frame_data[-100:]
            rss_start = last_100_frames[0]['rss_mb']
            rss_end = last_100_frames[-1]['rss_mb']
            rss_growth = rss_end - rss_start

            print(f"\n[MemProfile] Frame {frame_number} - Last 100 frames:")
            print(f"  RSS growth: {rss_growth:.2f} MB ({rss_growth/100:.4f} MB/frame)")
            print(f"  Current RSS: {current_rss:.2f} MB (total Δ {current_rss - self.start_memory_rss:+.2f} MB)")

    def analyze_frame_memory_leak(self, window_size=100):
        """
        分析帧级别的内存泄漏

        参数:
            window_size: 滑动窗口大小

        返回:
            包含分析结果的字典
        """
        if not self.enabled or len(self.frame_data) < window_size * 2:
            return None

        # 计算每个窗口的内存增长率
        windows = []
        for i in range(0, len(self.frame_data) - window_size, window_size):
            window_frames = self.frame_data[i:i+window_size]
            rss_start = window_frames[0]['rss_mb']
            rss_end = window_frames[-1]['rss_mb']
            growth = rss_end - rss_start
            growth_per_frame = growth / window_size

            windows.append({
                'start_frame': window_frames[0]['frame'],
                'end_frame': window_frames[-1]['frame'],
                'growth_mb': growth,
                'growth_per_frame_mb': growth_per_frame
            })

        # 检测是否有持续增长
        avg_growth = sum(w['growth_per_frame_mb'] for w in windows) / len(windows)

        result = {
            'total_frames': len(self.frame_data),
            'window_size': window_size,
            'windows': windows,
            'average_growth_per_frame_mb': avg_growth,
            'has_memory_leak': avg_growth > 0.01  # 如果每帧增长超过10KB则可能有泄漏
        }

        return result

    def save_report(self, output_path: str):
        """
        保存分析报告

        参数:
            output_path: 输出文件路径
        """
        if not self.enabled:
            return

        # 分析内存泄漏
        leak_analysis = self.analyze_frame_memory_leak()

        report = {
            'summary': {
                'start_time': self.start_time,
                'total_duration_sec': time.time() - self.start_time,
                'start_memory_rss_mb': self.start_memory_rss,
                'end_memory_rss_mb': self.frame_data[-1]['rss_mb'] if self.frame_data else self.start_memory_rss,
                'total_memory_growth_mb': (self.frame_data[-1]['rss_mb'] - self.start_memory_rss) if self.frame_data else 0,
                'total_frames': len(self.frame_data)
            },
            'checkpoints': self.checkpoints,
            'frame_data': self.frame_data,
            'leak_analysis': leak_analysis
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n[MemProfile] Report saved to: {output_path}")

        # 打印摘要
        print("\n" + "="*60)
        print("MEMORY PROFILE SUMMARY")
        print("="*60)
        print(f"Total frames: {report['summary']['total_frames']}")
        print(f"Duration: {report['summary']['total_duration_sec']:.2f} seconds")
        print(f"Memory growth: {report['summary']['total_memory_growth_mb']:.2f} MB")

        if leak_analysis and leak_analysis['has_memory_leak']:
            print(f"\n⚠️  POTENTIAL MEMORY LEAK DETECTED!")
            print(f"Average growth per frame: {leak_analysis['average_growth_per_frame_mb']:.4f} MB/frame")
            print(f"Projected growth per 1000 frames: {leak_analysis['average_growth_per_frame_mb'] * 1000:.2f} MB")

        print("\nCheckpoints:")
        for cp in self.checkpoints:
            print(f"  {cp['label']:30s}: RSS Δ {cp['rss_delta_mb']:+8.2f} MB")

        print("="*60 + "\n")

    def stop(self):
        """停止内存分析"""
        if self.enabled:
            tracemalloc.stop()


class FrameStageProfiler:
    """帧内各阶段的内存分析器"""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.process = psutil.Process(os.getpid())
        self.stages = {}
        self.current_stage = None
        self.stage_start_rss = None

    def start_stage(self, stage_name: str):
        """开始记录一个阶段"""
        if not self.enabled:
            return

        mem_info = self.process.memory_info()
        self.current_stage = stage_name
        self.stage_start_rss = mem_info.rss / 1024 / 1024  # MB

    def end_stage(self):
        """结束当前阶段的记录"""
        if not self.enabled or self.current_stage is None:
            return

        mem_info = self.process.memory_info()
        current_rss = mem_info.rss / 1024 / 1024  # MB
        delta = current_rss - self.stage_start_rss

        self.stages[self.current_stage] = {
            'rss_mb': current_rss,
            'delta_mb': delta
        }

        self.current_stage = None
        self.stage_start_rss = None

    def get_stages(self):
        """获取所有阶段的数据"""
        return self.stages.copy()

    def reset(self):
        """重置阶段数据"""
        self.stages = {}
        self.current_stage = None
        self.stage_start_rss = None
