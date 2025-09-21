import numpy as np

def calculate_eshelby_stress(x, y, E1, nu1, E2, nu2, a, sigma_inf_xx, sigma_inf_yy, sigma_inf_xy=0.0):
    """
    计算Eshelby夹杂问题中点(x,y)处的应力解析解
    
    参数:
    x, y: 场点坐标
    E1, nu1: 基体材料的杨氏模量和泊松比
    E2, nu2: 夹杂材料的杨氏模量和泊松比  
    a: 夹杂半径
    sigma_inf_xx, sigma_inf_yy, sigma_inf_xy: 远场应力分量
    
    返回:
    sigma_xx, sigma_yy, sigma_xy: 该点的应力分量
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
    S_tensor = calculate_eshelby_tensor(inside_inclusion, rho, e1, e2, nu1)
    
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
    nu = nu1  # 应该使用基体材料的泊松比，不是固定的0.25
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

def calculate_eshelby_tensor(inside_inclusion, rho, e1, e2, nu):
    """计算4阶Eshelby张量"""
    
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
        # 夹杂外部 (公式中的S^{E,∞})
        factor = rho**2 / (8*(1 - nu))
        
        # 各项系数
        coeff1 = rho**2 + 4*nu - 2
        coeff2 = 4*(1 - rho**2)
        coeff3 = rho**2 - 4*nu + 2
        coeff4 = 4*(1 - 2*nu - rho**2)
        coeff5 = 4*(nu - rho**2)
        coeff6 = 8*(3*rho**2 - 2)
        
        # S₁₁₁₁
        S[0,0,0,0] = factor * (coeff1 + coeff2*e1*e1 + coeff3 + coeff4*e1*e1 + 
                               coeff5*(e1*e1 + e1*e1) + coeff6*e1*e1*e1*e1)
        
        # S₂₂₂₂  
        S[1,1,1,1] = factor * (coeff1 + coeff2*e2*e2 + coeff3 + coeff4*e2*e2 + 
                               coeff5*(e2*e2 + e2*e2) + coeff6*e2*e2*e2*e2)
        
        # S₁₁₂₂
        S[0,0,1,1] = factor * (coeff1 + coeff2*e2*e2 + coeff4*e1*e1 + 
                               coeff5*(e2*e2 + e2*e2) + coeff6*e1*e1*e2*e2)
        
        # S₂₂₁₁
        S[1,1,0,0] = factor * (coeff1 + coeff2*e1*e1 + coeff4*e2*e2 + 
                               coeff5*(e1*e1 + e1*e1) + coeff6*e2*e2*e1*e1)
        
        # 剪切分量 S₁₂₁₂
        S[0,1,0,1] = S[0,1,1,0] = S[1,0,0,1] = S[1,0,1,0] = factor * (
            coeff3 + coeff4*e1*e2 + coeff5*(e1*e2 + e1*e2) + coeff6*e1*e1*e2*e2)
    
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

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 材料参数
    E1, nu1 = 70e9, 0.3    # 基体 (Pa)
    E2, nu2 = 200e9, 0.3   # 夹杂 (Pa)
    a = 1.0                # 夹杂半径 (m)
    
    # 远场应力 (Pa)
    sigma_inf_xx = 1e6     # 1 MPa拉伸
    sigma_inf_yy = 0.0
    sigma_inf_xy = 0.0
    
    # 计算网格点的应力
    x_coords = np.linspace(-3*a, 3*a, 101)
    y_coords = np.linspace(-3*a, 3*a, 101)
    
    # 示例：计算单点应力
    x, y = 0.5*a, 0.0  # 夹杂内部点
    sigma_xx, sigma_yy, sigma_xy = calculate_eshelby_stress(
        x, y, E1, nu1, E2, nu2, a, sigma_inf_xx, sigma_inf_yy, sigma_inf_xy)
    
    print(f"在点({x:.2f}, {y:.2f})处的应力:")
    print(f"σ_xx = {sigma_xx/1e6:.3f} MPa")
    print(f"σ_yy = {sigma_yy/1e6:.3f} MPa") 
    print(f"σ_xy = {sigma_xy/1e6:.3f} MPa")
    
    # 批量计算可以用以下方式：
    # for x in x_coords:
    #     for y in y_coords:
    #         sigma_xx, sigma_yy, sigma_xy = calculate_eshelby_stress(
    #             x, y, E1, nu1, E2, nu2, a, sigma_inf_xx, sigma_inf_yy, sigma_inf_xy)