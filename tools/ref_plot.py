import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 0. 全局设置：强制纯正 LaTeX 渲染与顶刊规范
# ==========================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    # 核心修正：注入 LaTeX 宏包，确保字体（特别是数学符号）极其纯正
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{lmodern}", 
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,       # 顶部显示刻度
    "ytick.right": True,     # 右侧显示刻度
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "legend.frameon": False, # 图例无边框
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
})

# ==========================================
# 图 1: 接触压力与间隙距离 (Contact pressure vs Gap distance)
# ==========================================
def plot_figure_1():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
    x = np.linspace(0, 0.005, 500)
    y1 = 200 * np.exp(-x / 0.00008)  
    y2 = 200 * np.exp(-x / 0.00025)  
    y3 = 200 * np.exp(-x / 0.0006)   
    
    ax.plot(x, y1, color='#5a0015', linewidth=2.0, label=r'$\hat{d} = 0.001$ m')
    ax.plot(x, y2, color='#b4182b', linewidth=2.0, label=r'$\hat{d} = 0.002$ m')
    ax.plot(x, y3, color='#e5735c', linewidth=2.0, label=r'$\hat{d} = 0.004$ m')
    
    ax.set_xlim(0, 0.005)
    ax.set_ylim(0, 200)
    
    # 强制修正刻度显示，防止出现科学计数法或多余小数
    ax.set_xticks([0, 0.001, 0.002, 0.003, 0.004, 0.005])
    ax.set_xticklabels(['0', '0.001', '0.002', '0.003', '0.004', '0.005'])
    ax.set_yticks([0, 100, 200])
    ax.set_yticklabels(['0', '100', '200'])
    
    ax.set_xlabel(r'Gap distance (m)', labelpad=5)
    ax.set_ylabel(r'Contact pressure (kPa)', labelpad=5)
    
    ax.legend(loc='upper right')
    plt.savefig('Figure1_ContactPressure.pdf')
    plt.close()

# ==========================================
# 图 2: 罚函数参数曲线与极值点标注
# ==========================================
def plot_figure_2():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
    x = np.linspace(0, 1, 200)
    y = 70 * (x - 0.376)**2 + 5.03
    
    ax.plot(x, y, color='#b2182b', linewidth=2.0)
    ax.axhline(y=1.2, color='black', linewidth=1.5)
    
    x_min, y_min = 0.376, 5.03
    ax.plot(x_min, y_min, marker='o', color='black', markersize=6)
    ax.plot([x_min, x_min], [0, y_min], linestyle='--', color='gray')
    ax.plot([0, x_min], [y_min, y_min], linestyle='--', color='gray')
    
    ax.text(x_min - 0.12, y_min + 0.8, r'$g(0.376) \approx 5.03$', fontsize=11)
    ax.text(0.5, 2.3, r'A typical value of $E/(p_{N,0})_{\mathrm{optimal}}(\hat{d}/h)$', 
            fontsize=11, ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 15)
    
    # 强制去除两端的 .0
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_yticklabels(['0', '5', '10', '15'])
    
    ax.set_xlabel(r'$d_0/\hat{d}$', labelpad=5)
    ax.set_ylabel(r'$g(d_0/\hat{d})$', labelpad=5)
    
    plt.savefig('Figure2_OptimalParameter.pdf')
    plt.close()

# ==========================================
# 图 3: 双子图并排 (牛顿法收敛性对比)
# ==========================================
def plot_figure_3():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    iters_left = np.arange(0, 15)
    res_smooth_left = np.array([1, 1e-1, 8e-2, 7e-2, 1e-2, 6e-3, 5e-3, 1.5e-3, 5e-4, 8e-6, 5e-9, np.nan, np.nan, np.nan, np.nan])
    res_penalty_left = np.array([1, 2e-1, 1e-1, 6e-2, 6.5e-2, 7e-2, 1.5e-1, 1.6e-1, 1.8e-1, 2.8e-1, 4e-1, 4.5e-1, 5.5e-1, 6e-1, 7e-1])
    
    iters_right = np.arange(0, 15)
    res_smooth_right = np.array([1, 2e-1, 1e-1, 7e-2, 2.5e-2, 2e-2, 1.2e-2, 7e-3, 5e-3, 1e-3, 2e-4, 5e-6, 2e-9, np.nan, np.nan])
    res_penalty_right = np.array([1, 2.5e-1, 5e-2, 3e-2, 2.5e-2, 3e-2, 4e-2, 5e-2, 1.2e-1, 2e-1, 4e-1, 1.2, 1.3, 1.5, 1.0])

    titles = [r'(a) $E_{\mathrm{hard}}/E_{\mathrm{soft}} = 10^1$', 
              r'(b) $E_{\mathrm{hard}}/E_{\mathrm{soft}} = 10^7$']
    
    for i, ax in enumerate(axes):
        # 绘制时强调用空心点 markerfacecolor='none'
        ax.plot(iters_left if i==0 else iters_right, 
                res_smooth_left if i==0 else res_smooth_right, 
                marker='o', linestyle='-', color='#b2182b', 
                markerfacecolor='none', label='Smoothed friction')
        
        ax.plot(iters_left if i==0 else iters_right, 
                res_penalty_left if i==0 else res_penalty_right, 
                marker='s', linestyle='-', color='#3182bd', 
                markerfacecolor='none', label='Penalty')
        
        ax.set_yscale('log')
        ax.set_xlim(0, 15)
        ax.set_ylim(1e-10, 1e2)
        
        # 强制格式化 X 轴刻度，避免出现 0.0, 5.0
        ax.set_xticks([0, 5, 10, 15])
        ax.set_xticklabels(['0', '5', '10', '15'])
        
        ax.set_xlabel('Iteration', labelpad=5)
        ax.set_ylabel('Relative residual norm (log scale)', labelpad=5)
        ax.set_title(titles[i], loc='left', pad=10)
        
        ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('Figure3_Convergence.pdf')
    plt.close()

if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    print("三张图均已应用原汁原味的 LaTeX 字体并修复刻度，已保存为 PDF 文件！")