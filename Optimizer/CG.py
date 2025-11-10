import taichi as ti
import numpy as np
import time
import matplotlib.pyplot as plt

@ti.data_oriented
class CG:
    def __init__(self, energy_fn, grad_fn, hess_fn, DBC_fn=None, dim=3, alpha=0.0, beta=0.6, eta=1e-3, grad_normalizer=1.0, float_type=ti.f32):
        """
        共轭梯度(CG)优化器，API与Newton完全兼容

        参数:
            energy_fn: 能量函数
            grad_fn: 梯度函数
            hess_fn: Hessian函数 (CG可以用于预条件)
            DBC_fn: 边界条件函数 (可选)
            dim: 维度
            alpha: 线搜索参数
            beta: 线搜索回退率
            eta: 收敛容差
            grad_normalizer: 梯度归一化因子
            float_type: 浮点类型
        """
        self.dim = dim
        self.energy_fn = energy_fn
        self.grad_fn = grad_fn
        self.hess_fn = hess_fn  # CG中可用于预条件
        self.DBC_fn = DBC_fn
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.grad_normalizer = grad_normalizer
        self.float_type = float_type

        # 参数和梯度存储
        self.x = ti.field(self.float_type, shape=dim)
        self.grad = ti.field(self.float_type, shape=dim)
        self.grad_old = ti.field(self.float_type, shape=dim)
        self.temp_x = ti.field(self.float_type, shape=dim)
        self.d = ti.field(self.float_type, shape=dim)  # 搜索方向
        self.r = ti.field(self.float_type, shape=dim)  # 残差
        self.p = ti.field(self.float_type, shape=dim)  # 共轭方向
        self.f0 = 0.0

        # 历史记录
        self.f_his = []
        self.time_his = []

    @ti.kernel
    def copy_field(self, src: ti.template(), dst: ti.template()):
        """复制field"""
        for i in range(self.dim):
            dst[i] = src[i]

    @ti.kernel
    def dot_product(self, a: ti.template(), b: ti.template()) -> ti.f64:
        """计算两个向量的点积"""
        result = 0.0
        for i in range(self.dim):
            result += a[i] * b[i]
        return result

    @ti.kernel
    def vector_add_scaled(self, result: ti.template(), a: ti.template(),
                         b: ti.template(), scale_b: ti.f64):
        """计算 result = a + scale_b * b"""
        for i in range(self.dim):
            result[i] = a[i] + scale_b * b[i]

    @ti.kernel
    def scale_vector(self, vec: ti.template(), scale: ti.f64):
        """将向量缩放"""
        for i in range(self.dim):
            vec[i] *= scale

    @ti.kernel
    def grad_inf_norm(self) -> ti.f64:
        """计算梯度的无穷范数"""
        n = 0.0
        for i in range(self.dim):
            ti.atomic_max(n, ti.abs(self.grad[i]))
        return n

    def line_search(self, direction):
        """线搜索"""
        alpha = 1.0

        # 计算方向导数
        g0 = self.dot_product(self.grad, direction)

        if g0 >= 0:
            print("Warning: Not a descent direction! g0:", g0)
            return 0.0

        # Armijo线搜索
        while alpha > 1e-6:
            self.vector_add_scaled(self.temp_x, self.x, direction, alpha)
            f_new = self.energy_fn(self.temp_x)
            if f_new <= self.f0 + self.alpha * alpha * g0:
                break
            alpha *= self.beta

        return alpha

    def minimize(self, max_iter=200, init_iter=50):
        """
        CG优化主循环

        参数:
            max_iter: 最大迭代次数
            init_iter: 初始迭代次数 (为了API兼容性，CG中不使用)

        返回:
            迭代次数
        """
        start_time = time.time()

        # 初始化：计算初始梯度
        self.f0 = self.energy_fn(self.x)
        self.grad_fn(self.x, self.grad)

        # r = -grad (残差等于负梯度)
        self.scale_vector(self.grad, -1.0)
        self.copy_field(self.grad, self.r)
        self.copy_field(self.r, self.p)  # p = r (初始搜索方向)

        # 恢复梯度符号
        self.scale_vector(self.grad, -1.0)

        for it in range(max_iter):
            # 计算当前能量和梯度
            self.f0 = self.energy_fn(self.x)
            self.grad_fn(self.x, self.grad)

            print(f"CG Iteration {it}, Energy: {self.f0:.4e}")

            # 检查收敛
            g_norm = self.grad_inf_norm() / self.grad_normalizer
            print(f"Grad norm: {g_norm:.4e}")

            if g_norm < self.eta:
                print(f"CG Converged at iteration {it}")
                self.f_his.append(self.f0)
                self.time_his.append(time.time() - start_time)
                return it

            # 更新残差 r = -grad
            self.scale_vector(self.grad, -1.0)
            self.copy_field(self.grad, self.r)
            self.scale_vector(self.grad, -1.0)  # 恢复梯度符号

            # 计算 r^T * r
            rsold = self.dot_product(self.r, self.r)

            if it == 0:
                # 第一次迭代，p = r
                self.copy_field(self.r, self.p)
            else:
                # 计算beta = (r^T * r) / (r_old^T * r_old)
                if rsold > 1e-16:
                    beta = rsold / self.rsold_prev
                    # p = r + beta * p
                    self.vector_add_scaled(self.p, self.r, self.p, beta)
                else:
                    self.copy_field(self.r, self.p)

            # 保存当前rsold
            self.rsold_prev = rsold

            # 使用p作为搜索方向进行线搜索
            self.copy_field(self.p, self.d)
            alpha = self.line_search(self.d)
            print(f"Step size: {alpha:.4e}")

            # 更新参数 x = x + alpha * p
            self.vector_add_scaled(self.x, self.x, self.p, alpha)

            # 记录历史
            self.f_his.append(self.f0)
            self.time_his.append(time.time() - start_time)

            # 每20步重启CG避免数值累积误差
            if (it + 1) % 100 == 0:
                print("CG restart to avoid numerical errors")
                self.grad_fn(self.x, self.grad)
                self.scale_vector(self.grad, -1.0)
                self.copy_field(self.grad, self.r)
                self.copy_field(self.r, self.p)
                self.scale_vector(self.grad, -1.0)

        print(f"CG reached maximum iterations ({max_iter})")
        return max_iter

# 示例使用 (与Newton.py相同的测试案例)
if __name__ == "__main__":
    float_type = ti.f64
    ti.init(arch=ti.cpu, default_fp=float_type, device_memory_GB=20)
    dim = 30000

    @ti.kernel
    def quadratic_energy(x: ti.template()) -> float_type:
        f = 0.0
        for i in range(x.shape[0]):
            f += x[i] ** 2
        return f

    # 能量函数和梯度函数
    @ti.kernel
    def quadratic_energy_grad(x: ti.template(), grad: ti.template()) -> float_type:
        f = 0.0
        for i in range(x.shape[0]):
            f += x[i] ** 2
            grad[i] = 2 * x[i]
        return f

    @ti.kernel
    def quadratic_hess(x: ti.template(), H: ti.types.sparse_matrix_builder()):
        for i in range(x.shape[0]):
            H[i, i] += 2.0

    @ti.kernel
    def rosenbrock(x: ti.template()) -> float_type:
        f_total = 0.0
        for i in range(x.shape[0]):
            if i % 3 == 0:
                x1, x2, x3 = x[i], x[i+1], x[i+2]
                f_total += (3 - x1)**2 + 7*(x2 - x1**2)**2 + 9*(x3 - x1 - x2**2)**2
        return f_total

    @ti.kernel
    def rosenbrock_grad(x: ti.template(), grad: ti.template()) -> float_type:
        f_total = 0.0
        for i in range(x.shape[0]):
            if i % 3 == 0:
                x1, x2, x3 = x[i], x[i+1], x[i+2]
                # 能量计算
                f_total += (3 - x1)**2 + 7*(x2 - x1**2)**2 + 9*(x3 - x1 - x2**2)**2
                # 梯度计算
                grad[i] = 2*(x1 - 3) + 28*(x1**2 - x2)*x1 + 18*(-x3 + x1 + x2**2)
                grad[i+1] = 14*(x2 - x1**2) + 18*(x3 - x1 - x2**2)*(-2*x2)
                grad[i+2] = 18*(x3 - x1 - x2**2)
        return f_total

    @ti.kernel
    def rosenbrock_hess(x: ti.template(), H: ti.sparse_matrix_builder()):
        # CG中不需要Hessian，但为了API兼容性保留
        for i in range(x.shape[0]):
            H[i, i] += 1.0

    # 初始化优化器
    optimizer = CG(energy_fn=rosenbrock,
                   grad_fn=rosenbrock_grad,
                   hess_fn=rosenbrock_hess,
                   dim=dim,
                   float_type=float_type)

    x_np = np.ones(dim)
    # 设置初始值
    optimizer.x.from_numpy(x_np)

    # 执行优化
    iter_count = optimizer.minimize(max_iter=200)

    print(f"Final parameters (first 10): {optimizer.x.to_numpy()[:10]}")
    print(f"Converged in {iter_count} iterations")

    # 绘制能量变化
    plt.plot(optimizer.f_his)
    plt.title("Energy History - CG")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.show()