import numpy as np

def calculate_eshelby_stress(x, y, E1, nu1, E2, nu2, a, sigma_inf_xx, sigma_inf_yy, sigma_inf_xy=0.0):
    """
    计算Eshelby夹杂问题中点(x,y)处的应力解析解
    """
    
    # ========== Step 1: 计算基体材料的Lamé常数 ==========
    mu1 = E1 / (2 * (1 + nu1))
    lambda1 = E1 * nu1 / ((1 + nu1) * (1 - 2 * nu1))
    
    # ========== Step 2: 计算等效本征应变 ==========
    eps_star_xx, eps_star_yy, eps_star_xy = calculate_equivalent_eigenstrain(
        E1, nu1, E2, nu2, sigma_inf_xx, sigma_inf_yy, sigma_inf_xy)
    
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

def calculate_equivalent_eigenstrain(E1, nu1, E2, nu2, sigma_inf_xx, sigma_inf_yy, sigma_inf_xy):
    """计算等效本征应变"""
    
    # 平面应变条件下的Eshelby张量分量（夹杂内部，使用基体材料泊松比）
    nu = nu1  # 应该使用基体材料的泊松比
    S1111 = S2222 = (3 - 4*nu) / (8*(1 - nu))  
    S1122 = S2211 = (4*nu - 1) / (8*(1 - nu))  
    S1212 = (3 - 4*nu) / (8*(1 - nu))
    
    # 计算远场应变（基体材料）
    eps_inf_xx = (sigma_inf_xx - nu1 * sigma_inf_yy) / E1
    eps_inf_yy = (sigma_inf_yy - nu1 * sigma_inf_xx) / E1
    eps_inf_xy = sigma_inf_xy / (2 * E1 / (2*(1 + nu1)))  # σ_xy / (2μ)
    
    # 材料性质差异因子
    alpha_normal = (1/E1 - 1/E2) / (1 - S1111) if abs(1 - S1111) > 1e-12 else 0
    beta_normal = (nu2/E2 - nu1/E1) / (1 - S2222) if abs(1 - S2222) > 1e-12 else 0
    alpha_shear = (1/E1 - 1/E2) * (1 + nu1) * (1 + nu2) / ((1 - S1212) * 2) if abs(1 - S1212) > 1e-12 else 0
    
    # 计算所需的本征应变
    eps_star_xx = alpha_normal * sigma_inf_xx + beta_normal * sigma_inf_yy
    eps_star_yy = beta_normal * sigma_inf_xx + alpha_normal * sigma_inf_yy
    eps_star_xy = alpha_shear * sigma_inf_xy
    
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

def calculate_stress_concentration_factor(x, y, E1, nu1, E2, nu2, a, sigma_inf=1.0):
    """
    计算应力集中系数（归一化应力）
    
    返回: 应力集中系数 = σ(x,y) / σ_∞
    """
    sigma_xx, sigma_yy, sigma_xy = calculate_eshelby_stress(
        x, y, E1, nu1, E2, nu2, a, sigma_inf, sigma_inf, 0.0)
    
    concentration_factor_xx = sigma_xx / sigma_inf
    concentration_factor_yy = sigma_yy / sigma_inf
    concentration_factor_xy = sigma_xy / sigma_inf
    
    return concentration_factor_xx, concentration_factor_yy, concentration_factor_xy

def test_special_cases():
    """测试特殊情况（简化输出）"""
    print("=== 特殊情况测试 ===")
    
    a = 0.08
    sigma_inf = 1.0  # 归一化应力
    
    # 测试1：均匀材料（E1 = E2）
    print("\n测试1: 均匀材料 (E₁ = E₂)")
    E1 = E2 = 1e6
    nu1 = nu2 = 0.3
    
    kx, ky, kxy = calculate_stress_concentration_factor(0.0, 0.0, E1, nu1, E2, nu2, a, sigma_inf)
    print(f"  夹杂中心应力集中系数: Kx={kx:.3f}, Ky={ky:.3f}")
    print(f"  期望值: 1.000 (均匀材料)")
    print(f"  误差: {abs(kx - 1.0):.6f}")
    
    # 测试2：中等刚度比
    print("\n测试2: 中等刚度比 (E₂/E₁ = 0.1)")
    E1 = 1e6
    E2 = 1e5
    nu1 = nu2 = 0.3
    
    kx, ky, kxy = calculate_stress_concentration_factor(0.0, 0.0, E1, nu1, E2, nu2, a, sigma_inf)
    print(f"  夹杂中心应力集中系数: Kx={kx:.3f}, Ky={ky:.3f}")
    print(f"  物理意义: 软夹杂，应力应该较小")
    
    # 测试3：极端刚度比
    print("\n测试3: 极端刚度比 (E₂/E₁ = 0.0001)")
    E1 = 1e6
    E2 = 1e2
    nu1 = nu2 = 0.3
    
    kx, ky, kxy = calculate_stress_concentration_factor(0.0, 0.0, E1, nu1, E2, nu2, a, sigma_inf)
    print(f"  夹杂中心应力集中系数: Kx={kx:.3f}, Ky={ky:.3f}")
    print(f"  物理意义: 接近空洞，应力应该很小")
    
    # 测试4：空间分布检查
    print("\n测试4: 空间分布检查 (E₂/E₁ = 0.1)")
    E1 = 1e6
    E2 = 1e5
    nu1 = nu2 = 0.3
    
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
        kx, ky, kxy = calculate_stress_concentration_factor(x, y, E1, nu1, E2, nu2, a, sigma_inf)
        r = np.sqrt(x**2 + y**2)
        location = "内" if r <= a else "外"
        print(f"  {desc:12s} ({location}):  {kx:8.3f}        {ky:8.3f}")
    
    print(f"\n远场期望值: 1.000")

def generate_stress_field_data(E1, nu1, E2, nu2, a, nx=51, ny=51, domain_size=5):
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
            kx, ky, kxy = calculate_stress_concentration_factor(x, y, E1, nu1, E2, nu2, a)
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