import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 数据目录
data_dir = Path("experiment_results/case2/diff_reso")
output_dir = Path("experiment_results/case2/analysis_figures")
output_dir.mkdir(parents=True, exist_ok=True)

# 存储结果
results = []

# 遍历所有 reso 目录
for reso_dir in sorted(data_dir.glob("reso_*")):
    reso_value = int(reso_dir.name.split("_")[1])
    stats_file = reso_dir / "performance_stats" / "stats_data.json"

    if not stats_file.exists():
        print(f"Warning: {stats_file} not found, skipping...")
        continue

    try:
        with open(stats_file, 'r') as f:
            data = json.load(f)

        # 计算总时间和各 domain 时间
        total_time = 0
        big_domain_time = 0
        small_domain_time = 0

        for frame in data['frame_data']:
            total_time += frame['total_frame_time']

            # 累加所有 Schwarz 迭代的时间
            if 'big_domain_solve_time' in frame:
                big_domain_time += sum(frame['big_domain_solve_time'])
            if 'small_domain_solve_time' in frame:
                small_domain_time += sum(frame['small_domain_solve_time'])

        results.append({
            'reso': reso_value,
            'total_time': total_time,
            'big_domain_time': big_domain_time,
            'small_domain_time': small_domain_time,
            'num_frames': len(data['frame_data'])
        })

        print(f"Reso {reso_value}: Total={total_time:.2f}s, Big={big_domain_time:.2f}s, Small={small_domain_time:.2f}s, Frames={len(data['frame_data'])}")

    except Exception as e:
        print(f"Error processing {stats_file}: {e}")
        continue

# 按分辨率排序
results.sort(key=lambda x: x['reso'])

# 提取数据
resos = [r['reso'] for r in results]
total_times = [r['total_time'] for r in results]
big_domain_times = [r['big_domain_time'] for r in results]
small_domain_times = [r['small_domain_time'] for r in results]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 总时间随分辨率变化
ax1 = axes[0, 0]
ax1.plot(resos, total_times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Resolution', fontsize=12)
ax1.set_ylabel('Total Time (s)', fontsize=12)
ax1.set_title('Total Simulation Time vs Resolution', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(min(resos)-5, max(resos)+5)

# 图2: 两个 domain 时间对比
ax2 = axes[0, 1]
ax2.plot(resos, big_domain_times, 'o-', linewidth=2, markersize=8, label='Big Domain', color='#A23B72')
ax2.plot(resos, small_domain_times, 's-', linewidth=2, markersize=8, label='Small Domain', color='#F18F01')
ax2.set_xlabel('Resolution', fontsize=12)
ax2.set_ylabel('Solve Time (s)', fontsize=12)
ax2.set_title('Domain Solve Times vs Resolution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(min(resos)-5, max(resos)+5)

# 图3: 时间占比堆叠图
ax3 = axes[1, 0]
ax3.bar(resos, big_domain_times, label='Big Domain', color='#A23B72', alpha=0.8)
ax3.bar(resos, small_domain_times, bottom=big_domain_times, label='Small Domain', color='#F18F01', alpha=0.8)
ax3.set_xlabel('Resolution', fontsize=12)
ax3.set_ylabel('Time (s)', fontsize=12)
ax3.set_title('Stacked Domain Times vs Resolution', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# 图4: 时间占比百分比
ax4 = axes[1, 1]
total_solve_times = [big + small for big, small in zip(big_domain_times, small_domain_times)]
big_ratios = [big/total*100 if total > 0 else 0 for big, total in zip(big_domain_times, total_solve_times)]
small_ratios = [small/total*100 if total > 0 else 0 for small, total in zip(small_domain_times, total_solve_times)]

ax4.plot(resos, big_ratios, 'o-', linewidth=2, markersize=8, label='Big Domain %', color='#A23B72')
ax4.plot(resos, small_ratios, 's-', linewidth=2, markersize=8, label='Small Domain %', color='#F18F01')
ax4.set_xlabel('Resolution', fontsize=12)
ax4.set_ylabel('Percentage of Solve Time (%)', fontsize=12)
ax4.set_title('Domain Time Distribution vs Resolution', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(min(resos)-5, max(resos)+5)
ax4.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(output_dir / 'timing_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_dir / 'timing_analysis.png'}")

# 额外绘制: log-log 图来分析复杂度
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Log-log plot for total time
ax_log1 = axes2[0]
ax_log1.loglog(resos, total_times, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Total Time')
ax_log1.set_xlabel('Resolution (log scale)', fontsize=12)
ax_log1.set_ylabel('Total Time (s, log scale)', fontsize=12)
ax_log1.set_title('Total Time vs Resolution (Log-Log)', fontsize=13, fontweight='bold')
ax_log1.grid(True, alpha=0.3, which='both')
ax_log1.legend(fontsize=11)

# 拟合幂律关系
log_resos = np.log(resos)
log_times = np.log(total_times)
coeffs = np.polyfit(log_resos, log_times, 1)
slope = coeffs[0]
ax_log1.text(0.05, 0.95, f'Slope ≈ {slope:.2f}\nComplexity: O(n^{slope:.2f})',
             transform=ax_log1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Log-log plot for domain times
ax_log2 = axes2[1]
ax_log2.loglog(resos, big_domain_times, 'o-', linewidth=2, markersize=8, label='Big Domain', color='#A23B72')
ax_log2.loglog(resos, small_domain_times, 's-', linewidth=2, markersize=8, label='Small Domain', color='#F18F01')
ax_log2.set_xlabel('Resolution (log scale)', fontsize=12)
ax_log2.set_ylabel('Solve Time (s, log scale)', fontsize=12)
ax_log2.set_title('Domain Times vs Resolution (Log-Log)', fontsize=13, fontweight='bold')
ax_log2.grid(True, alpha=0.3, which='both')
ax_log2.legend(fontsize=11)

# 拟合各 domain 的幂律关系
coeffs_big = np.polyfit(log_resos, np.log(big_domain_times), 1)
coeffs_small = np.polyfit(log_resos, np.log(small_domain_times), 1)
ax_log2.text(0.05, 0.95, f'Big Domain slope ≈ {coeffs_big[0]:.2f}\nSmall Domain slope ≈ {coeffs_small[0]:.2f}',
             transform=ax_log2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'timing_analysis_loglog.png', dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_dir / 'timing_analysis_loglog.png'}")

# 保存数据到 CSV 文件
import csv
csv_file = output_dir / 'timing_data.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Resolution', 'Total Time (s)', 'Big Domain Time (s)', 'Small Domain Time (s)', 'Num Frames'])
    for r in results:
        writer.writerow([r['reso'], r['total_time'], r['big_domain_time'], r['small_domain_time'], r['num_frames']])
print(f"Data saved to: {csv_file}")

print("\nAnalysis complete!")
