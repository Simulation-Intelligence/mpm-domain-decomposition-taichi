import numpy as np
import matplotlib.pyplot as plt

def calculate_hertz_contact_pressure(x, y, R, E, nu, p_applied, contact_center_y=1.0):
    """
    计算Hertz接触问题中点(x,y)处的接触压力
    
    参数:
    x, y: 场点坐标 (m)
    R: 圆柱半径 (m)
    E: 杨氏模量 (Pa)
    nu: 泊松比
    p_applied: 施加的均布压力 (Pa)
    contact_center_y: 接触区域中心y坐标 (m)
    
    返回:
    pressure: 该点的接触压力 (Pa)
    """
    
    # 计算接触半宽度 b
    b = calculate_contact_width(R, E, nu, p_applied)
    
    # 相对于接触中心的坐标
    x_rel = x
    y_rel = y - contact_center_y
    
    # 判断是否在接触区域内
    if abs(x_rel) <= b and y_rel <= 0:  # 接触区域在y <= contact_center_y
        # Hertz接触压力分布
        pressure = calculate_hertz_pressure_distribution(x_rel, R, E, nu, p_applied, b)
    else:
        pressure = 0.0
    
    return pressure

def calculate_contact_width(R, E, nu, p_applied):
    """
    计算Hertz接触的接触半宽度
    
    根据公式: b = 2√(2R²p(1-ν²)/(Eπ))
    """
    b = 2 * np.sqrt(2 * R**2 * p_applied * (1 - nu**2) / (E * np.pi))
    return b

def calculate_hertz_pressure_distribution(x, R, E, nu, p_applied, b):
    """
    计算Hertz接触压力分布
    
    根据公式: p = (4Kp/πb²)√(b² - x²)
    其中 K 需要通过力平衡确定
    """
    
    # 确保在接触区域内
    if abs(x) > b:
        return 0.0
    
    # 计算压力分布的形状函数
    shape_factor = np.sqrt(b**2 - x**2)
    
    # 通过力平衡确定压力幅值
    # 总的法向力应该等于施加的压力乘以投影面积
    # ∫p(x)dx from -b to b = p_applied * (某个有效面积)
    
    # 对于半圆形压力分布: ∫√(b²-x²)dx from -b to b = πb²/2
    # 因此: p_max * πb²/2 = F_total
    # 需要根据具体问题确定总力
    
    # 简化计算：假设最大压力使得平均压力等于施加压力的某个倍数
    p_max = 4* R * p_applied / np.pi / b 
    
    pressure = p_max * shape_factor / b
    
    return pressure

def calculate_hertz_pressure_at_y_zero(x, R, E, nu, p_applied):
    """
    计算y=0位置的Hertz接触压力分布

    参数:
    x: x坐标数组或单个值 (m)
    R: 圆柱半径 (m)
    E: 杨氏模量 (Pa)
    nu: 泊松比
    p_applied: 施加的均布压力 (Pa)

    返回:
    pressure: y=0位置的接触压力 (Pa)
    """

    # 计算接触半宽度
    b = calculate_contact_width(R, E, nu, p_applied)

    # 如果输入是数组
    if hasattr(x, '__len__'):
        pressure = np.zeros_like(x)
        for i, x_val in enumerate(x):
            if abs(x_val) <= b:
                pressure[i] = calculate_hertz_pressure_distribution(x_val, R, E, nu, p_applied, b)
            else:
                pressure[i] = 0.0
    else:
        # 如果输入是单个值
        if abs(x) <= b:
            pressure = calculate_hertz_pressure_distribution(x, R, E, nu, p_applied, b)
        else:
            pressure = 0.0

    return pressure

def generate_contact_pressure_field(R, E, nu, p_applied, nx=101, ny=51, 
                                  x_range=(-20, 20), y_range=(-5, 5), 
                                  contact_center_y=1.0):
    """
    生成接触压力场数据
    
    参数:
    nx, ny: 网格点数
    x_range, y_range: 计算域范围 (m)
    contact_center_y: 接触区域中心y坐标 (m)
    
    返回: x_grid, y_grid, pressure_field
    """
    
    # 创建网格
    x_coords = np.linspace(x_range[0], x_range[1], nx)
    y_coords = np.linspace(y_range[0], y_range[1], ny)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    # 计算压力场
    pressure_field = np.zeros_like(x_grid)
    
    for i in range(nx):
        for j in range(ny):
            x, y = x_grid[j,i], y_grid[j,i]
            pressure_field[j,i] = calculate_hertz_contact_pressure(
                x, y, R, E, nu, p_applied, contact_center_y)
    
    return x_grid, y_grid, pressure_field

def plot_contact_pressure_distribution(R, E, nu, p_applied, contact_center_y=1.0):
    """
    绘制接触压力分布
    """
    
    # 计算接触参数
    b = calculate_contact_width(R, E, nu, p_applied)
    
    # 生成x轴上的压力分布
    x_contact = np.linspace(-b, b, 100)
    pressure_contact = np.array([
        calculate_hertz_pressure_distribution(x, R, E, nu, p_applied, b) 
        for x in x_contact
    ])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：接触区域的压力分布
    ax1.plot(x_contact, pressure_contact/1000, 'b-', linewidth=2, label='Hertz压力分布')
    ax1.axhline(y=p_applied/1000, color='r', linestyle='--', label=f'施加压力 = {p_applied/1000:.1f} kPa')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('接触压力 (kPa)')
    ax1.set_title('Hertz接触压力分布')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-b*1.2, b*1.2)
    
    # 右图：二维压力场
    x_grid, y_grid, pressure_field = generate_contact_pressure_field(
        R, E, nu, p_applied, nx=81, ny=41, 
        x_range=(-15, 15), y_range=(-2, 4), 
        contact_center_y=contact_center_y)
    
    im = ax2.contourf(x_grid, y_grid, pressure_field/1000, levels=20, cmap='viridis')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('二维接触压力场')
    ax2.set_aspect('equal')
    
    # 绘制圆柱轮廓
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = R * np.cos(theta)
    circle_y = R * np.sin(theta) + contact_center_y + R
    ax2.plot(circle_x, circle_y, 'w-', linewidth=2, label='圆柱边界')
    
    # 绘制接触区域
    contact_x = np.linspace(-b, b, 100)
    contact_y = np.full_like(contact_x, contact_center_y)
    ax2.plot(contact_x, contact_y, 'r-', linewidth=3, label='接触区域')
    
    ax2.legend()
    plt.colorbar(im, ax=ax2, label='接触压力 (kPa)')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def validate_hertz_solution(R, E, nu, p_applied):
    """
    验证Hertz解的基本性质
    """
    
    print("=== Hertz接触问题验证 ===")
    print(f"圆柱半径 R = {R} m")
    print(f"杨氏模量 E = {E:.1f} Pa")
    print(f"泊松比 ν = {nu}")
    print(f"施加压力 p = {p_applied/1000:.1f} kPa")
    print("-" * 40)
    
    # 计算接触参数
    b, p_max = calculate_contact_parameters(R, E, nu, p_applied)
    
    print(f"计算结果:")
    print(f"  接触半宽度 b = {b:.3f} m")
    print(f"  最大接触压力 p_max = {p_max/1000:.2f} kPa")
    print(f"  压力集中系数 = {p_max/p_applied:.2f}")
    
    # 验证力平衡
    # 计算总的接触力
    x_integration = np.linspace(-b, b, 1000)
    pressure_integration = np.array([
        calculate_hertz_pressure_distribution(x, R, E, nu, p_applied, b) 
        for x in x_integration
    ])
    
    # 数值积分
    dx = x_integration[1] - x_integration[0]
    total_force_per_unit_length = np.sum(pressure_integration) * dx
    
    print(f"\n力平衡验证:")
    print(f"  数值积分总力/单位长度 = {total_force_per_unit_length/1000:.2f} kN/m")
    
    # 理论上，对于椭圆分布：∫p(x)dx = π/2 * p_max * b
    theoretical_force = np.pi/2 * p_max * b
    print(f"  理论总力/单位长度 = {theoretical_force/1000:.2f} kN/m")
    print(f"  相对误差 = {abs(total_force_per_unit_length - theoretical_force)/theoretical_force*100:.2f}%")
    
    # 检查边界条件
    print(f"\n边界条件检查:")
    pressure_at_edge = calculate_hertz_contact_pressure(b, 1.0, R, E, nu, p_applied)
    pressure_outside = calculate_hertz_contact_pressure(b*1.1, 1.0, R, E, nu, p_applied)
    print(f"  接触边界处压力 = {pressure_at_edge:.6f} Pa")
    print(f"  接触区域外压力 = {pressure_outside:.6f} Pa")
    
    return b, p_max

def test_hertz_contact():
    """
    测试Hertz接触问题
    """
    
    # 问题参数（基于图片中的设置）
    R = 8.0          # 圆柱半径 (m)
    E = 2e5         # 杨氏模量 (Pa) - 典型混凝土值
    nu = 0.3         # 泊松比
    p_applied = 1000 # 施加压力 (Pa = 1 kPa)
    
    # 验证解析解
    b, p_max = validate_hertz_solution(R, E, nu, p_applied)
    
    # 绘制压力分布
    print(f"\n正在绘制压力分布图...")
    fig = plot_contact_pressure_distribution(R, E, nu, p_applied)
    
    # 测试不同位置的压力值
    print(f"\n不同位置的压力值:")
    test_points = [
        (0.0, 1.0, "接触中心"),
        (b/2, 1.0, "接触区域中部"),
        (b*0.9, 1.0, "接触边缘附近"),
        (b*1.1, 1.0, "接触区域外"),
        (0.0, 2.0, "圆柱内部"),
        (0.0, 0.0, "基础内部")
    ]
    
    for x, y, desc in test_points:
        pressure = calculate_hertz_contact_pressure(x, y, R, E, nu, p_applied)
        print(f"  {desc:12s}: p = {pressure/1000:.3f} kPa")
    
    return R, E, nu, p_applied, b, p_max

def generate_validation_data(R, E, nu, p_applied, filename=None):
    """
    生成用于MPM验证的数据
    """
    
    # 生成高分辨率的压力场数据
    x_grid, y_grid, pressure_field = generate_contact_pressure_field(
        R, E, nu, p_applied, nx=201, ny=101, 
        x_range=(-20, 20), y_range=(-5, 5))
    
    # 计算接触参数
    b, p_max = calculate_contact_parameters(R, E, nu, p_applied)
    
    validation_data = {
        'x_grid': x_grid,
        'y_grid': y_grid, 
        'pressure_field': pressure_field,
        'contact_width': b,
        'max_pressure': p_max,
        'applied_pressure': p_applied,
        'material_properties': {'R': R, 'E': E, 'nu': nu}
    }
    
    if filename:
        np.savez(filename, **validation_data)
        print(f"验证数据已保存到: {filename}")
    
    return validation_data

# ========== 主函数 ==========
if __name__ == "__main__":
    # 运行测试
    test_results = test_hertz_contact()
    
    print("\n" + "="*60)
    print("使用建议:")
    print("1. 用于MPM验证时，重点关注接触区域的压力分布")
    print("2. 验证MPM是否能正确捕捉压力集中效应")
    print("3. 检查接触边界处的压力梯度")
    print("4. 使用 generate_validation_data() 生成验证数据集")
    
    # 生成验证数据
    print("\n正在生成验证数据...")
    R, E, nu, p_applied, b, p_max = test_results
    validation_data = generate_validation_data(R, E, nu, p_applied, 
                                             "hertz_contact_validation.npz")