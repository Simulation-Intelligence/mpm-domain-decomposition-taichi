"""
CMAME publication style for matplotlib.

Usage:
    from tools.plot_style import apply_cmame_style
    apply_cmame_style()

Or if running from the experiments/ directory:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from tools.plot_style import apply_cmame_style
    apply_cmame_style()
"""

import matplotlib.pyplot as plt
from contextlib import contextmanager

# CMAME color palette
COLORS = [
    '#3182bd',  # dark blue
    '#b2182b',  # dark red
    '#2ca02c',  # green
    '#8c510a',  # brown
    '#756bb1',  # purple
    '#e5735c',  # salmon
    '#1f77b4',  # mpl blue
    '#636363',  # gray
]

# Analytical solution color
COLOR_ANALYTICAL = '#b2182b'

# Primary MPM / single-domain color
COLOR_MPM = '#3182bd'

# Secondary domain color
COLOR_DUAL = '#b2182b'


def apply_cmame_style():
    """Apply CMAME publication-quality rcParams."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{lmodern}",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


PAPER_TEXT_FONTSIZE = 16


def paper_font_rcparams(base_fontsize=PAPER_TEXT_FONTSIZE):
    """Return rcParams tuned for body-text-matching figure fonts."""
    return {
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "figure.titlesize": base_fontsize,
        "legend.title_fontsize": base_fontsize,
    }


def apply_paper_font_style(base_fontsize=PAPER_TEXT_FONTSIZE):
    """Apply CMAME-compatible paper font size settings for all labels/ticks."""
    plt.rcParams.update(paper_font_rcparams(base_fontsize))


@contextmanager
def paper_font_context(base_fontsize=PAPER_TEXT_FONTSIZE):
    with plt.rc_context(paper_font_rcparams(base_fontsize)):
        yield
