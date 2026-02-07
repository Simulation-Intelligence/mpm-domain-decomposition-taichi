import taichi as ti
import numpy as np
import time
import matplotlib.pyplot as plt
import gc

@ti.data_oriented
class PCG:
    def __init__(self, energy_fn, grad_fn, hess_fn, DBC_fn=None, dim=3, alpha=0.0, beta=0.6, eta=1e-3, grad_normalizer=1.0, float_type=ti.f32, preconditioner_type="diagonal"):
        """
        预条件共轭梯度(PCG)优化器，API与Newton完全兼容

        参数:
            energy_fn: 能量函数
            grad_fn: 梯度函数
            hess_fn: Hessian函数 (用于构造预条件子)
            DBC_fn: 边界条件函数 (可选)
            dim: 维度
            alpha: 线搜索参数
            beta: 线搜索回退率
            eta: 收敛容差
            grad_normalizer: 梯度归一化因子
            float_type: 浮点类型
            preconditioner_type: 预条件子类型 ("diagonal", "identity", "jacobi")
        """
        self.dim = dim
        self.energy_fn = energy_fn
        self.grad_fn = grad_fn
        self.hess_fn = hess_fn
        self.DBC_fn = DBC_fn
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.grad_normalizer = grad_normalizer
        self.float_type = float_type
        self.preconditioner_type = preconditioner_type

        # 参数和梯度存储
        self.x = ti.field(self.float_type, shape=dim)
        self.grad = ti.field(self.float_type, shape=dim)
        self.temp_x = ti.field(self.float_type, shape=dim)

        # PCG特有字段
        self.r = ti.field(self.float_type, shape=dim)      # 残差
        self.z = ti.field(self.float_type, shape=dim)      # 预条件残差
        self.p = ti.field(self.float_type, shape=dim)      # 搜索方向
        self.Ap = ti.field(self.float_type, shape=dim)     # A*p

        # 预条件子 (对角矩阵)
        self.M_inv = ti.field(self.float_type, shape=dim)  # M^(-1)的对角元素

        self.f0 = 0.0
        self.built_preconditioner = False

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
    def apply_preconditioner(self, r: ti.template(), z: ti.template()):
        """应用预条件子: z = M^(-1) * r"""
        for i in range(self.dim):
            z[i] = self.M_inv[i] * r[i]

    @ti.kernel
    def grad_inf_norm(self) -> ti.f64:
        """计算梯度的无穷范数"""
        n = 0.0
        for i in range(self.dim):
            ti.atomic_max(n, ti.abs(self.grad[i]))
        return n

    def build_preconditioner(self):
        """构造预条件子"""
        if self.built_preconditioner:
            return

        # print(f"Building {self.preconditioner_type} preconditioner...")

        if self.preconditioner_type == "identity":
            self._build_identity_preconditioner()
        elif self.preconditioner_type == "diagonal":
            self._build_diagonal_preconditioner()
        elif self.preconditioner_type == "jacobi":
            self._build_jacobi_preconditioner()
        else:
            print(f"Unknown preconditioner type: {self.preconditioner_type}, using identity")
            self._build_identity_preconditioner()

        self.built_preconditioner = True
        # print("Preconditioner built successfully")

    @ti.kernel
    def _build_identity_preconditioner(self):
        """构造单位预条件子 M^(-1) = I"""
        for i in range(self.dim):
            self.M_inv[i] = 1.0

    def _build_diagonal_preconditioner(self):
        """构造对角预条件子 M^(-1) = diag(H)^(-1)"""
        # 构建Hessian矩阵获取对角元素
        H_builder = ti.linalg.SparseMatrixBuilder(self.dim, self.dim, max_num_triplets=self.dim * 90, dtype=self.float_type)
        self.hess_fn(self.x, H_builder)
        H = H_builder.build()

        # 提取对角元素
        H_diag = self._extract_diagonal(H)

        # 清理 Taichi 对象，防止内存泄漏
        del H, H_builder
        # 注意：gc.collect() 太频繁会影响性能，已移至外层每1000帧调用

        # 构造预条件子的逆
        self._build_diagonal_inverse(H_diag)

    @ti.kernel
    def _build_diagonal_inverse(self, H_diag: ti.types.ndarray()):
        """从Hessian对角元素构造预条件子的逆"""
        for i in range(self.dim):
            diag_val = H_diag[i]
            if ti.abs(diag_val) > 1e-12:
                self.M_inv[i] = 1.0 / diag_val
            else:
                self.M_inv[i] = 1.0  # 避免除零，退化为单位预条件子

    def _extract_diagonal(self, H):
        """从稀疏矩阵提取对角元素"""
        H_diag = np.zeros(self.dim, dtype=np.float32 if self.float_type == ti.f32 else np.float64)

        try:
            # 使用Taichi SparseMatrix的元素访问接口A[i,j]
            for i in range(self.dim):
                diag_val = H[i, i]  # 直接访问对角元素
                if abs(diag_val) > 1e-12:
                    H_diag[i] = diag_val
                else:
                    H_diag[i] = 1.0  # 避免除零，设为1.0

        except Exception as e:
            # 如果提取失败，使用单位预条件子
            print(f"Warning: Failed to extract diagonal ({e}), using identity preconditioner")
            H_diag.fill(1.0)

        return H_diag

    def _build_jacobi_preconditioner(self):
        """构造Jacobi预条件子（加权对角预条件子）"""
        # 与对角预条件子类似，但添加一些正则化
        self._build_diagonal_preconditioner()

        # 应用阻尼系数提高稳定性
        damping = 0.1
        self._apply_jacobi_damping(damping)

    @ti.kernel
    def _apply_jacobi_damping(self, damping: ti.f64):
        """对Jacobi预条件子应用阻尼"""
        for i in range(self.dim):
            # M^(-1) = (1-damping) * M^(-1) + damping * I
            self.M_inv[i] = (1.0 - damping) * self.M_inv[i] + damping

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
        PCG优化主循环

        参数:
            max_iter: 最大迭代次数
            init_iter: 初始迭代次数 (为了API兼容性，PCG中不使用)

        返回:
            迭代次数
        """
        start_time = time.time()

        # 构造预条件子
        self.build_preconditioner()

        # 初始化：计算初始能量和梯度
        self.f0 = self.energy_fn(self.x)
        self.grad_fn(self.x, self.grad)

        # 初始化PCG
        # r = -grad (残差)
        self.scale_vector(self.grad, -1.0)
        self.copy_field(self.grad, self.r)
        self.scale_vector(self.grad, -1.0)  # 恢复梯度符号

        # z = M^(-1) * r (预条件残差)
        self.apply_preconditioner(self.r, self.z)

        # p = z (初始搜索方向)
        self.copy_field(self.z, self.p)

        # 主循环
        for it in range(max_iter):
            # 计算当前能量和梯度
            self.f0 = self.energy_fn(self.x)
            self.grad_fn(self.x, self.grad)

            print(f"PCG Iteration {it}, Energy: {self.f0:.4e}")

            # 检查收敛
            g_norm = self.grad_inf_norm() / self.grad_normalizer
            print(f"Grad norm: {g_norm:.4e}")

            if g_norm < self.eta:
                print(f"PCG Converged at iteration {it}")
                self.f_his.append(self.f0)
                self.time_his.append(time.time() - start_time)
                return it

            # 更新残差 r = -grad
            self.scale_vector(self.grad, -1.0)
            self.copy_field(self.grad, self.r)
            self.scale_vector(self.grad, -1.0)  # 恢复梯度符号

            # 应用预条件子 z = M^(-1) * r
            self.apply_preconditioner(self.r, self.z)

            # 计算 r^T * z
            rzold = self.dot_product(self.r, self.z) if it > 0 else 0.0
            rz = self.dot_product(self.r, self.z)

            if it == 0:
                # 第一次迭代，p = z
                self.copy_field(self.z, self.p)
            else:
                # 计算beta = (r^T * z) / (r_old^T * z_old)
                if rzold > 1e-16:
                    beta = rz / rzold
                    # p = z + beta * p
                    self.vector_add_scaled(self.p, self.z, self.p, beta)
                else:
                    self.copy_field(self.z, self.p)

            # 保存当前rz用于下次迭代
            rzold = rz

            # 线搜索
            alpha = self.line_search(self.p)
            print(f"Step size: {alpha:.4e}")

            # 更新参数 x = x + alpha * p
            self.vector_add_scaled(self.x, self.x, self.p, alpha)

            # 记录历史
            self.f_his.append(self.f0)
            self.time_his.append(time.time() - start_time)

            # 每30步重新构建预条件子避免数值累积误差
            if (it + 1) % 30 == 0:
                print("PCG: Rebuilding preconditioner to avoid numerical errors")
                self.built_preconditioner = False
                self.build_preconditioner()

        print(f"PCG reached maximum iterations ({max_iter})")
        return max_iter

# 示例使用和测试
if __name__ == "__main__":
    float_type = ti.f64
    ti.init(arch=ti.cpu, default_fp=float_type, device_memory_GB=20)
    dim = 10000

    @ti.kernel
    def quadratic_energy(x: ti.template()) -> float_type:
        f = 0.0
        for i in range(x.shape[0]):
            # 构造一个病态的二次函数
            scale = 1.0 + 100.0 * (i % 10) / 10.0  # 条件数约为100
            f += scale * x[i] ** 2
        return f

    @ti.kernel
    def quadratic_energy_grad(x: ti.template(), grad: ti.template()) -> float_type:
        f = 0.0
        for i in range(x.shape[0]):
            scale = 1.0 + 100.0 * (i % 10) / 10.0
            f += scale * x[i] ** 2
            grad[i] = 2 * scale * x[i]
        return f

    @ti.kernel
    def quadratic_hess(x: ti.template(), H: ti.types.sparse_matrix_builder()):
        for i in range(x.shape[0]):
            scale = 1.0 + 100.0 * (i % 10) / 10.0
            H[i, i] += 2 * scale

    @ti.kernel
    def rosenbrock(x: ti.template()) -> float_type:
        f_total = 0.0
        for i in range(x.shape[0]):
            if i % 3 == 0 and i + 2 < x.shape[0]:
                x1, x2, x3 = x[i], x[i+1], x[i+2]
                f_total += (3 - x1)**2 + 7*(x2 - x1**2)**2 + 9*(x3 - x1 - x2**2)**2
        return f_total

    @ti.kernel
    def rosenbrock_grad(x: ti.template(), grad: ti.template()) -> float_type:
        f_total = 0.0
        for i in range(x.shape[0]):
            if i % 3 == 0 and i + 2 < x.shape[0]:
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
        for i in range(x.shape[0]):
            if i % 3 == 0 and i + 2 < x.shape[0]:
                # 简化的Hessian近似（对角占优）
                H[i, i] += 100.0
                H[i+1, i+1] += 50.0
                H[i+2, i+2] += 20.0

    # 测试不同的预条件子
    preconditioner_types = ["identity", "diagonal", "jacobi"]

    for prec_type in preconditioner_types:
        print(f"\n{'='*50}")
        print(f"Testing PCG with {prec_type} preconditioner")
        print(f"{'='*50}")

        # 初始化优化器
        optimizer = PCG(energy_fn=quadratic_energy,
                       grad_fn=quadratic_energy_grad,
                       hess_fn=quadratic_hess,
                       dim=dim,
                       float_type=float_type,
                       preconditioner_type=prec_type,
                       eta=1e-6)

        # 设置初始值
        x_np = np.random.random(dim) * 10  # 随机初始值
        optimizer.x.from_numpy(x_np)

        # 执行优化
        iter_count = optimizer.minimize(max_iter=100)

        final_x = optimizer.x.to_numpy()
        final_energy = optimizer.energy_fn(optimizer.x)

        print(f"Final energy: {final_energy:.6e}")
        print(f"Converged in {iter_count} iterations")
        print(f"Solution norm: {np.linalg.norm(final_x):.6e}")

    print("\nPCG testing completed!")