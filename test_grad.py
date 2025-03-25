from implicit_mpm import *


def test_gradient_derivative():
    ti.init(arch=ti.cuda, debug=True)

    # 测试配置
    grid_size = 8
    dim = 2
    dt = 0.001
    n_particles = 2

    # 初始化网格
    grid = Grid(size=grid_size, dim=dim, bound=2)
    grid.dx = 1.0 / grid_size
    grid.inv_dx = grid_size

    # 初始化场变量
    grid.m = ti.field(dtype=ti.f32, shape=(grid_size, grid_size))
    grid.v_prev = ti.Vector.field(dim, dtype=ti.f32, shape=(grid_size, grid_size))
    grid.v = ti.Vector.field(dim, dtype=ti.f32, shape=(grid_size, grid_size))

    # 随机初始化网格质量
    @ti.kernel
    def init_grid_mass():
        for i, j in grid.m:
            grid.m[i, j] = ti.random() + 0.5  # 确保质量不为零

    init_grid_mass()

    # 初始化粒子
    particles = Particles(Config(data={}), grid_size)

    # 随机初始化粒子位置和权重
    @ti.kernel
    def init_particles():
        for p in range(n_particles):
            particles.x[p] = [ti.random(), ti.random()]
            for i, j in ti.static(ti.ndrange(3, 3)):
                particles.dwip[p, i, j] = ti.Vector([ti.random(), ti.random()])

    particles.initialize()
    init_particles()

    # 初始化求解器
    config = Config(data={
        "dt": dt,
        "E": 4.0,
        "nu": 0.4,
        "solve_max_iter": 1,
        "implicit_solver": "BFGS"
    })
    solver = ImplicitSolver(grid, particles, config)
    n_vars = grid.size**dim * dim

    # 随机生成速度场
    @ti.kernel
    def init_velocity(v_flat: ti.template()):
        for i in range(n_vars):
            v_flat[i] = ti.random() - 0.5

    init_velocity(solver.v_grad)

    # 自动求导计算梯度
    auto_grad = ti.field(ti.f32, shape=n_vars)
    solver.compute_energy_grad_auto(solver.v_grad, auto_grad)

    # 手动计算梯度
    manual_grad = ti.field(ti.f32, shape=n_vars)
    solver.compute_energy_grad_manual(solver.v_grad, manual_grad)

    # 比较结果
    max_error = 0.0
    for i in range(n_vars):
        print (auto_grad[i],manual_grad[i])
        error = abs(auto_grad[i] - manual_grad[i])
        max_error = max(max_error, error)
        if error > 1e-4:
            print(f"显著差异 @ {i}: Auto={auto_grad[i]:.6f}, Manual={manual_grad[i]:.6f}, Δ={error:.6f}")

    print(f"最大梯度误差: {max_error:.10f}")
    assert max_error < 1e-4, "手动梯度与自动求导结果不一致"

    # 手动计算hessian
    manual_hessian = ti.sparse_matrix_builder()
    solver.compute_hess(solver.v_grad, manual_hessian)

    # finite difference hessian
    h = 1e-6
    finite_diff_hessian = ti.field(ti.f32, shape=(n_vars, n_vars))
    finite_diff_hessian.fill(0.0)
    new_grad1 = ti.field(ti.f32, shape=n_vars)
    new_grad2 = ti.field(ti.f32, shape=n_vars)
    for i in range(n_vars):
        solver.v_grad[i] += h
        new_grad1.fill(0.0)
        new_grad2.fill(0.0)
        solver.compute_energy_grad_auto(solver.v_grad, new_grad1)
        solver.v_grad[i] -= 2 * h
        solver.compute_energy_grad_auto(solver.v_grad, new_grad2)
        for j in range(n_vars):
            finite_diff_hessian[i, j] = (new_grad1[j]- new_grad2[j]) / h/2
        solver.v_grad[i] += h

    # 比较结果
    max_error = 0.0
    for i in range(n_vars):
        for j in range(n_vars):
            error = abs(manual_hessian[i, j] - finite_diff_hessian[i, j])
            max_error = max(max_error, error)
            if error > 1e-4:
                print(f"显著差异 @ {i},{j}: Manual={manual_hessian[i, j]:.6f}, FD={finite_diff_hessian[i, j]:.6f}, Δ={error:.6f}")
    print(f"最大Hessian误差: {max_error:.6f}")
    assert max_error < 1e-4, "手动Hessian与有限差分结果不一致"

if __name__ == "__main__":
    test_gradient_derivative()