import numpy as np

def calculate_eshelby_stress(x, y, E1, nu1, E2, nu2, a, sigma_inf_xx, sigma_inf_yy, sigma_inf_xy=0.0, initial_F=None):
    """
    计算Eshelby夹杂问题中点(x,y)处的应力解析解
    """

    # ========== Step 1: 计算基体材料的Lamé常数 ==========
    mu1 = E1 / (2 * (1 + nu1))
    lambda1 = E1 * nu1 / ((1 + nu1) * (1 - 2 * nu1))

    # ========== Step 2: 计算等效本征应变 ==========
    eps_star_xx, eps_star_yy, eps_star_xy = calculate_equivalent_eigenstrain(initial_F)
    
    # ========== Step 3: 计算几何参数 ==========
    r = np.sqrt(x**2 + y**2)
    
    # 避免除零
    if r < 1e-12:
        r = 1e-12
        
    rho = a / r  # ρ = a/|r|
    e1 = x / r   # e1 = x/|r|  
    e2 = y / r   # e2 = y/|r|
    
    # 特征函数 χ(r)
    if r <= a:
        chi = 1.0  # 夹杂内部
        inside_inclusion = True
    else:
        chi = 0.0  # 夹杂外部
        inside_inclusion = False
    
    # ========== Step 4: 计算Eshelby张量 ==========
    S_tensor = calculate_eshelby_tensor_corrected(inside_inclusion, rho, e1, e2, nu1)
    
    # ========== Step 5: 构建弹性张量 ==========
    C_tensor = construct_elasticity_tensor(lambda1, mu1)
    
    # ========== Step 6: 计算应力 ==========
    # σᵢⱼᵉ = Cᵢⱼₖₗ(Sₖₗₚᵩ ε*ₚᵩ - χ(r) ε*ₖₗ)
    
    # 本征应变张量（2x2矩阵形式）
    eps_star = np.array([[eps_star_xx, eps_star_xy],
                         [eps_star_xy, eps_star_yy]])
    
    # 计算 Sₖₗₚᵩ ε*ₚᵩ
    S_eps_star = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            for p in range(2):
                for q in range(2):
                    S_eps_star[i,j] += S_tensor[i,j,p,q] * eps_star[p,q]
    
    # 计算弹性应变: εᵉₖₗ = Sₖₗₚᵩ ε*ₚᵩ - χ(r) ε*ₖₗ
    eps_elastic = S_eps_star - chi * eps_star
    
    # 计算应力: σᵢⱼ = Cᵢⱼₖₗ εᵉₖₗ
    sigma = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    sigma[i,j] += C_tensor[i,j,k,l] * eps_elastic[k,l]
    
    return sigma[0,0], sigma[1,1], sigma[0,1]

def calculate_equivalent_eigenstrain(initial_F):
    """计算等效本征应变 - 使用初始F矩阵的逆"""

    # 使用初始F矩阵的逆作为本征应变
    # initial_F 应该是一个2x2的numpy数组
    F_inv = np.linalg.inv(initial_F)

    # 从F^(-1)计算本征应变：ε* = I - F^(-1)
    I = np.eye(2)  # 单位矩阵
    eps_star_matrix = I - F_inv

    eps_star_xx = eps_star_matrix[0, 0]
    eps_star_yy = eps_star_matrix[1, 1]
    eps_star_xy = eps_star_matrix[0, 1]

    return eps_star_xx, eps_star_yy, eps_star_xy

def calculate_eshelby_tensor_corrected(inside_inclusion, rho, e1, e2, nu):
    """计算4阶Eshelby张量 - 修正版"""
    
    S = np.zeros((2, 2, 2, 2))
    
    if inside_inclusion:
        # 夹杂内部 (公式中的S^{I,∞})
        factor1 = (3 - 4*nu) / (8*(1 - nu))
        factor2 = (4*nu - 1) / (8*(1 - nu))
        
        # S₁₁₁₁ = S₂₂₂₂
        S[0,0,0,0] = S[1,1,1,1] = factor1
        # S₁₁₂₂ = S₂₂₁₁  
        S[0,0,1,1] = S[1,1,0,0] = factor2
        # S₁₂₁₂ = S₁₂₂₁ = S₂₁₁₂ = S₂₁₂₁
        S[0,1,0,1] = S[0,1,1,0] = S[1,0,0,1] = S[1,0,1,0] = factor1
        
    else:
        # 夹杂外部 (公式中的S^{E,∞}) - 修正版
        factor = rho**2 / (8*(1 - nu))
        
        # S₁₁₁₁
        S[0,0,0,0] = factor * (
            (rho**2 + 4*nu - 2) +                    # δ₁₁δ₁₁项
            4*(1 - rho**2)*e1*e1 +                   # δ₁₁r₁r₁项 
            (rho**2 - 4*nu + 2) +                    # δ₁₁δ₁₁项
            4*(1 - 2*nu - rho**2)*e1*e1 +            # δ₁₁r₁r₁项
            4*(nu - rho**2)*e1*e1 +                  # δ₁₁e₁e₁项
            8*(3*rho**2 - 2)*e1*e1*e1*e1             # e₁e₁e₁e₁项
        )
        
        # S₂₂₂₂  
        S[1,1,1,1] = factor * (
            (rho**2 + 4*nu - 2) +                    
            4*(1 - rho**2)*e2*e2 +                   
            (rho**2 - 4*nu + 2) +                    
            4*(1 - 2*nu - rho**2)*e2*e2 +            
            4*(nu - rho**2)*e2*e2 +                  
            8*(3*rho**2 - 2)*e2*e2*e2*e2             
        )
        
        # S₁₁₂₂
        S[0,0,1,1] = factor * (
            (rho**2 + 4*nu - 2) +                    
            4*(1 - rho**2)*e2*e2 +                   
            0 +                                      
            4*(1 - 2*nu - rho**2)*e1*e1 +            
            4*(nu - rho**2)*e2*e2 +                  
            8*(3*rho**2 - 2)*e1*e1*e2*e2             
        )
        
        # S₂₂₁₁ 
        S[1,1,0,0] = factor * (
            (rho**2 + 4*nu - 2) +                    
            4*(1 - rho**2)*e1*e1 +                   
            0 +                                      
            4*(1 - 2*nu - rho**2)*e2*e2 +            
            4*(nu - rho**2)*e1*e1 +                  
            8*(3*rho**2 - 2)*e2*e2*e1*e1             
        )
        
        # S₁₂₁₂ (剪切分量)
        S[0,1,0,1] = S[0,1,1,0] = S[1,0,0,1] = S[1,0,1,0] = factor * (
            0 +                                      
            0 +                                      
            (rho**2 - 4*nu + 2) +                    
            0 +                                      
            4*(nu - rho**2)*e1*e2 +                  
            8*(3*rho**2 - 2)*e1*e2*e1*e2             
        )
    
    return S

def construct_elasticity_tensor(lam, mu):
    """构建4阶弹性张量"""
    C = np.zeros((2, 2, 2, 2))
    
    # Cᵢⱼₘₙ = λδᵢⱼδₘₙ + μ(δᵢₘδⱼₙ + δᵢₙδⱼₘ)
    for i in range(2):
        for j in range(2):
            for m in range(2):
                for n in range(2):
                    # λδᵢⱼδₘₙ项
                    if i == j and m == n:
                        C[i,j,m,n] += lam
                    
                    # μδᵢₘδⱼₙ项
                    if i == m and j == n:
                        C[i,j,m,n] += mu
                        
                    # μδᵢₙδⱼₘ项
                    if i == n and j == m:
                        C[i,j,m,n] += mu
    
    return C

def calculate_stress_concentration_factor(x, y, E1, nu1, E2, nu2, a, initial_F, sigma_inf=1.0):
    """
    计算应力集中系数（归一化应力）

    返回: 应力集中系数 = σ(x,y) / σ_∞
    """
    sigma_xx, sigma_yy, sigma_xy = calculate_eshelby_stress(
        x, y, E1, nu1, E2, nu2, a, sigma_inf, sigma_inf, 0.0, initial_F)

    concentration_factor_xx = sigma_xx / sigma_inf
    concentration_factor_yy = sigma_yy / sigma_inf
    concentration_factor_xy = sigma_xy / sigma_inf

    return concentration_factor_xx, concentration_factor_yy, concentration_factor_xy

def test_special_cases():
    """测试特殊情况（简化输出）"""
    print("=== 特殊情况测试 ===")

    a = 0.08
    sigma_inf = 1.0  # 归一化应力

    # 测试1：单位初始F矩阵
    print("\n测试1: 单位初始F矩阵")
    E1 = E2 = 1e6
    nu1 = nu2 = 0.3
    initial_F = np.array([[1.0, 0.0], [0.0, 1.0]])

    kx, ky, kxy = calculate_stress_concentration_factor(0.0, 0.0, E1, nu1, E2, nu2, a, initial_F, sigma_inf)
    print(f"  夹杂中心应力集中系数: Kx={kx:.3f}, Ky={ky:.3f}")
    print(f"  期望值: 接近1.000 (单位F矩阵)")

    # 测试2：拉伸变形的初始F
    print("\n测试2: 拉伸变形的初始F")
    E1 = 1e6
    E2 = 1e5
    nu1 = nu2 = 0.3
    initial_F = np.array([[2.0, 0.0], [0.0, 1.0]])  # x方向拉伸2倍

    kx, ky, kxy = calculate_stress_concentration_factor(0.0, 0.0, E1, nu1, E2, nu2, a, initial_F, sigma_inf)
    print(f"  夹杂中心应力集中系数: Kx={kx:.3f}, Ky={ky:.3f}")
    print(f"  物理意义: x方向预拉伸，应产生相应应力分布")

    # 测试3：剪切变形的初始F
    print("\n测试3: 剪切变形的初始F")
    E1 = 1e6
    E2 = 1e2
    nu1 = nu2 = 0.3
    initial_F = np.array([[1.0, 0.1], [0.1, 1.0]])  # 轻微剪切

    kx, ky, kxy = calculate_stress_concentration_factor(0.0, 0.0, E1, nu1, E2, nu2, a, initial_F, sigma_inf)
    print(f"  夹杂中心应力集中系数: Kx={kx:.3f}, Ky={ky:.3f}, Kxy={kxy:.3f}")
    print(f"  物理意义: 预剪切变形，应产生剪应力")

    # 测试4：空间分布检查
    print("\n测试4: 空间分布检查 (拉伸F)")
    E1 = 1e6
    E2 = 1e5
    nu1 = nu2 = 0.3
    initial_F = np.array([[2.0, 0.0], [0.0, 1.0]])

    test_points = [
        (0.0, 0.0, "夹杂中心"),
        (a, 0.0, "x轴界面"),
        (0.0, a, "y轴界面"),
        (2*a, 0.0, "近场"),
        (5*a, 0.0, "远场")
    ]

    print("  位置           应力集中系数Kx    应力集中系数Ky")
    print("  " + "-"*50)
    for x, y, desc in test_points:
        kx, ky, kxy = calculate_stress_concentration_factor(x, y, E1, nu1, E2, nu2, a, initial_F, sigma_inf)
        r = np.sqrt(x**2 + y**2)
        location = "内" if r <= a else "外"
        print(f"  {desc:12s} ({location}):  {kx:8.3f}        {ky:8.3f}")

    print(f"\n远场期望值: 1.000")

def generate_stress_field_data(E1, nu1, E2, nu2, a, initial_F, nx=51, ny=51, domain_size=5):
    """
    生成应力场数据用于验证

    参数:
    nx, ny: 网格点数
    domain_size: 域大小（以夹杂半径为单位）

    返回: x_grid, y_grid, stress_concentration_xx, stress_concentration_yy
    """

    # 创建网格
    x_range = np.linspace(-domain_size*a, domain_size*a, nx)
    y_range = np.linspace(-domain_size*a, domain_size*a, ny)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # 计算应力集中系数
    kx_field = np.zeros_like(x_grid)
    ky_field = np.zeros_like(y_grid)

    for i in range(nx):
        for j in range(ny):
            x, y = x_grid[j,i], y_grid[j,i]
            kx, ky, kxy = calculate_stress_concentration_factor(x, y, E1, nu1, E2, nu2, a, initial_F)
            kx_field[j,i] = kx
            ky_field[j,i] = ky

    return x_grid, y_grid, kx_field, ky_field

# ========== 主函数测试 ==========
if __name__ == "__main__":
    test_special_cases()
    
    print("\n" + "="*60)
    print("使用建议:")
    print("1. 对于MPM验证，使用归一化的应力集中系数")
    print("2. 重点关注应力分布模式，而不是绝对数值")
    print("3. 使用 calculate_stress_concentration_factor() 函数")
    print("4. 用 generate_stress_field_data() 生成验证数据")