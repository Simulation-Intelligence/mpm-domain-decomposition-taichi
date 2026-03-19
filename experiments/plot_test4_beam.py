#!/usr/bin/env python3
"""Plot cantilever beam h/w vs gamma results."""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.plot_style import apply_cmame_style
apply_cmame_style()


def main():
    parser = argparse.ArgumentParser(description='Plot cantilever beam h/w vs gamma')
    parser.add_argument('--data', default='useful_results/test4_beam/integrated_error_data.json',
                        help='Path to integrated_error_data.json')
    parser.add_argument('--output', default=None,
                        help='Output path (default: hw_vs_gamma.pdf in data directory)')
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    gamma = np.array(data['gamma'])
    hw = np.array(data['h/w'])

    g_ref = np.logspace(np.log10(gamma.min() * 0.5), np.log10(gamma.max() * 2), 200)
    hw_linear = g_ref / 8
    hw_sqrt = np.sqrt(g_ref / 2)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.loglog(gamma, hw, 'o-', color='#b2182b', linewidth=2, markersize=7,
              markerfacecolor='none', markeredgewidth=1.5, label='MPM (single domain)')
    ax.loglog(g_ref, hw_linear, '--', color='black', linewidth=1.5,
              label=r'$h/w = \gamma/8$')
    ax.loglog(g_ref, hw_sqrt, ':', color='black', linewidth=1.5,
              label=r'$h/w = (\gamma/2)^{1/2}$')

    ax.set_xlabel(r'$\gamma = 12\rho g L^3(1-\nu^2)/(Eh^2)$')
    ax.set_ylabel(r'$h/w$')
    ax.minorticks_on()
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    plt.tight_layout()

    out = args.output or os.path.join(os.path.dirname(os.path.abspath(args.data)), 'hw_vs_gamma.pdf')
    plt.savefig(out)
    print(f'Saved: {out}')
    plt.close()


if __name__ == '__main__':
    main()
