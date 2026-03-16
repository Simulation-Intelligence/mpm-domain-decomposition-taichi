#!/usr/bin/env python3
"""绘制实验四（悬臂梁）h/w vs gamma 结果图"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='绘制悬臂梁 h/w vs gamma 结果图')
    parser.add_argument('--data', default='useful_results/test4_beam/integrated_error_data.json',
                        help='数据文件路径 (integrated_error_data.json)')
    parser.add_argument('--output', default=None,
                        help='输出图片路径（默认与data同目录下的 hw_vs_gamma.pdf）')
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    gamma = np.array(data['gamma'])
    hw = np.array(data['h/w'])

    # 参考线的 gamma 范围（稍宽于数据范围）
    g_ref = np.logspace(np.log10(gamma.min() * 0.5), np.log10(gamma.max() * 2), 200)
    hw_linear = g_ref / 8            # H/W = gamma/8
    hw_sqrt = np.sqrt(g_ref / 2)     # H/W = (gamma/2)^(1/2)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(gamma, hw, 'bo-', linewidth=2, markersize=8, label='MPM (单域)')
    ax.loglog(g_ref, hw_linear, 'r--', linewidth=1.8, label=r'$h/w = \gamma/8$')
    ax.loglog(g_ref, hw_sqrt,   'r:',  linewidth=1.8, label=r'$h/w = (\gamma/2)^{1/2}$')

    ax.set_xlabel(r'$\gamma = 12\rho g L^3(1-\nu^2)/(Eh^2)$', fontsize=12)
    ax.set_ylabel(r'$h/w$', fontsize=12)
    ax.set_title('悬臂梁尖端偏转 (实验四)', fontsize=13)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)

    plt.tight_layout()

    out = args.output or os.path.join(os.path.dirname(os.path.abspath(args.data)), 'hw_vs_gamma.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"图片已保存: {out}")
    plt.close()


if __name__ == '__main__':
    main()
