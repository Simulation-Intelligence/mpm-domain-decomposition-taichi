import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulators.implicit_mpm import *
from simulators.mpm_solver import MPMSolver

config = Config(data={
        "dt": 0.001,
        "float_type": "f64",
        "elasticity_model": "neohookean",  # "neohookean" or "linear"
        "dim": 2,
        "E": 4.0,
        "nu": 0.4,
        "solve_max_iter": 1,
        "gravity": [0.0, 8.0],
        "implicit_solver": "BFGS"
    })
float_type = ti.f32 if config.get("float_type", "f32") == "f32" else ti.f64
ti.init(arch=ti.cpu, default_fp=float_type, device_memory_GB=20,random_seed=114514)

# 测试配置
grid_size = 8
dim = config.get("dim", 2)
dt = 0.001
n_particles = 2
# 初始化网格
grid = Grid(config)
grid.dx = 1.0 / grid_size
grid.inv_dx = grid_size

# 初始化场变量
shape = (grid_size, grid_size, dim) if dim == 3 else (grid_size, grid_size)
grid.m = ti.field(dtype=float_type, shape=shape)
grid.v_prev = ti.Vector.field(dim, dtype=float_type, shape=shape)
grid.v = ti.Vector.field(dim, dtype=float_type, shape=shape)

# 随机初始化网格质量
@ti.kernel
def init_grid_mass():
    for I in ti.grouped(grid.m):
        grid.m[I] = ti.random() * 0.1 + 0.1

init_grid_mass()

    # 初始化粒子
particles = Particles(config)

    # 随机初始化粒子位置和权重
@ti.kernel
def init_particles():
    for p in range(n_particles):
        for d in ti.static(range(dim)):
            particles.x[p][d] = ti.random() * 0.5 + 0.25
            particles.v[p] = ti.Vector([ti.random() for _ in range(dim)])
            for q in ti.static(range(dim)):
                particles.F[p][d,q] = ti.random() * 1.0 + 0.5

# init_particles()
particles.build_neighbor_list()

# 初始化求解器
solver = MPMSolver(grid, particles, config)
n_vars = grid.size**dim * dim

# 随机生成速度场
@ti.kernel
def init_velocity(v_flat: ti.template()):
    for i in range(n_vars):
        v_flat[i] = ti.random() - 0.5

init_velocity(solver.v_grad)

def test_gradient_derivative():

    # 手动计算梯度
    manual_grad = ti.field(float_type, shape=n_vars)
    solver.compute_energy_grad_manual(solver.v_grad, manual_grad)

    # 比较与有限差分结果
    h = 1e-3
    finite_diff_grad = ti.field(float_type, shape=n_vars)
    for i in range(n_vars):
        solver.v_grad[i] += h
        e1 = solver.compute_energy(solver.v_grad)
        solver.v_grad[i] -= 2 * h
        e2 = solver.compute_energy(solver.v_grad)
        finite_diff_grad[i] = (e1 - e2) / (2 * h)
        solver.v_grad[i] += h
    max_error = 0.0
    for i in range(n_vars):
        error = abs(finite_diff_grad[i] - manual_grad[i]) / abs(finite_diff_grad[i] + 1e-10)
        max_error = max(max_error, error)
        idx1 = i/2 // grid.size
        idx2 = i/2 % grid.size
        if error > 1e-4:
            print(f"x:{idx1},y:{idx2},显著差异 @ {i}: FiniteDiff={finite_diff_grad[i]:.4e}, Manual={manual_grad[i]:.4e}, Δ={error:.4e}")
        elif finite_diff_grad[i] != 0:
            print(f"x:{idx1},y:{idx2},小差异 @ {i}: FiniteDiff={finite_diff_grad[i]:.4e}, Manual={manual_grad[i]:.4e}, Δ={error:.4e}")

    print(f"最大梯度相对误差: {max_error:.10f}")
    assert max_error < 1e-4, "手动梯度与有限差分结果不一致"

def test_hessian_derivative():
    # 手动计算hessian
    manual_hessian = ti.linalg.SparseMatrixBuilder(n_vars, n_vars, max_num_triplets=n_vars*n_vars, dtype=float_type)
    solver.compute_hess(v_flat=solver.v_grad, hess=manual_hessian)
    H = manual_hessian.build()


    # finite difference hessian
    h = 1e-3
    finite_diff_hessian = ti.field(float_type, shape=(n_vars, n_vars))
    finite_diff_hessian.fill(0.0)
    new_grad1 = ti.field(float_type, shape=n_vars)
    new_grad2 = ti.field(float_type, shape=n_vars)
    for i in range(n_vars):
        solver.v_grad[i] += h
        new_grad1.fill(0.0)
        new_grad2.fill(0.0)
        solver.compute_energy_grad_manual(solver.v_grad, new_grad1)
        solver.v_grad[i] -= 2 * h
        solver.compute_energy_grad_manual(solver.v_grad, new_grad2)
        for j in range(n_vars):
            finite_diff_hessian[i, j] = (new_grad1[j]- new_grad2[j]) / h/2
        solver.v_grad[i] += h

    # 比较结果
    max_error = 0.0
    for i in range(n_vars):
        for j in range(n_vars):
            if abs(H[i, j]) < 1e-8 and abs(finite_diff_hessian[i, j]) < 1e-8 or H[i, j] == 0.0:
                continue
            error = abs(H[i, j] - finite_diff_hessian[i, j]) / (abs(H[i, j])+abs(finite_diff_hessian[i, j]) + 1e-10)
            max_error = max(max_error, error)
            if error > 1e-4 :
                print(f"显著差异 @ {i},{j}: Manual={H[i, j]:.4e}, FD={finite_diff_hessian[i, j]:.4e}, Δ={error:.4e}")
            else:
                print(f"小差异 @ {i},{j}: Manual={H[i, j]:.4e}, FD={finite_diff_hessian[i, j]:.4e}, Δ={error:.4e}")
    print(f"最大相对Hessian误差: {max_error:.4e}")
    assert max_error < 1e-4, "手动Hessian与有限差分结果不一致"

if __name__ == "__main__":
    # test_gradient_derivative()
    test_hessian_derivative()