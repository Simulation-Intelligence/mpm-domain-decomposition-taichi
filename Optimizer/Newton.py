import taichi as ti
import numpy as np
import time
import matplotlib.pyplot as plt

@ti.data_oriented
class Newton:
    def __init__(self, energy_fn, grad_fn, hess_fn, DBC_fn=None,dim=3, alpha=0, beta=0.6, eta=1e-3, grad_normalizer=1.0,float_type=ti.f32):
        self.dim = ti.field(ti.i32, shape=())
        self.dim[None] = dim
        self.energy_fn = energy_fn
        self.grad_fn = grad_fn
        self.hess_fn = hess_fn
        self.DBC_fn = DBC_fn
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.grad_normalizer = grad_normalizer
        self.float_type = float_type

        # 参数和梯度存储
        self.x = ti.field(self.float_type, shape=dim) 
        self.grad = ti.field(self.float_type, shape=dim)
        self.temp_x = ti.field(self.float_type, shape=dim)
        self.d = ti.field(self.float_type, shape=dim)
        self.b = ti.ndarray(self.float_type, shape=dim)
        self.f0 =0.0

        # 创建可重用的SparseMatrixBuilder
        self.H_builder = ti.linalg.SparseMatrixBuilder(dim, dim, max_num_triplets=int(dim**2* 0.1), dtype=self.float_type)

        # 历史记录
        self.f_his = []
        self.time_his = []

    @ti.func
    def get_dimension(self):
        """返回当前优化器的维度"""
        return self.dim[None]

    def resize(self, new_dim):
        """重新申请指定维度的向量和稀疏矩阵"""
        self.dim[None] = new_dim

        # 重新申请ndarray
        self.b = ti.ndarray(self.float_type, shape=new_dim)

        # 重新创建SparseMatrixBuilder
        self.H_builder = ti.linalg.SparseMatrixBuilder(new_dim, new_dim, max_num_triplets=int(new_dim**2* 0.5), dtype=self.float_type)

        print(f"Newton optimizer resized to dimension: {new_dim}")
        print(f"SparseMatrixBuilder created with shape: ({new_dim}, {new_dim})")

    def line_search(self):
        alpha = 1.0
            
        @ti.kernel
        def calc_g0(d:ti.types.ndarray()) -> self.float_type:
            g = 0.0
            for i in range(self.get_dimension()):
                g += self.grad[i] * d[i]
            return g
        
        g0 = calc_g0(self.d)

        if g0 >= 0:
            print("Warning: Not a descent direction! g0:", g0)
            raise RuntimeError("Not a descent direction")

        @ti.kernel
        def update_temp(a: self.float_type,d :ti.types.ndarray()):
            for i in range(self.get_dimension()):
                self.temp_x[i] = self.x[i] + a * d[i]

        while alpha > 1e-4:
            update_temp(alpha,self.d)
            f_new = self.energy_fn(self.temp_x)
            if f_new <= self.f0+ self.alpha  * alpha * g0:
                break
            alpha *= self.beta
        return alpha

    def minimize(self, max_iter=200, init_iter=50):
        start_time = time.time()
        for it in range(max_iter):
            # 计算当前能量和梯度
            self.f0 = self.energy_fn(self.x)
            self.grad_fn(self.x, self.grad)

            # 检查收敛
            @ti.kernel
            def get_inf_norm(x: ti.template()) -> self.float_type:
                n = 0.0
                for i in range(self.get_dimension()):
                    ti.atomic_max(n, ti.abs(x[i]))
                return n

            g_norm = get_inf_norm(self.grad) / self.grad_normalizer

            if it % 100 == 0:
                print(f"Iteration {it}, grad norm: {g_norm:.4e}, Energy: {self.f0:.4e}")

            if g_norm < self.eta:
                print(f"Newton Converged at iteration {it}")
                print("x_inf_norm:", get_inf_norm(self.x))
                print(f"Final Energy: {self.f0:.4e}")
                return it

            # 构建Hessian矩阵 (重用builder)
            # self.H_builder = ti.linalg.SparseMatrixBuilder(self.dim[None], self.dim[None], max_num_triplets=int(self.dim[None]**2), dtype=self.float_type)
            self.hess_fn(self.x, self.H_builder)
            H = self.H_builder.build()

            # 构建右端项
            @ti.kernel
            def fill_b(b:ti.types.ndarray(),grad: ti.template()):
                for i in range(self.get_dimension()):
                    b[i] = -grad[i]
            fill_b(self.b,self.grad)


            # 求解线性系统
            solver = ti.linalg.SparseSolver(solver_type="LDLT",dtype=self.float_type)
            try:
                solver.analyze_pattern(H)
                solver.factorize(H)
                self.d = solver.solve(self.b)
            except RuntimeError:
                self.d = self.b
                print("Solver failed, resetting to gradient descent")

            # 线搜索
            alpha = self.line_search()
            # print(f"Step size: {alpha:.4e}")

            # 更新参数
            @ti.kernel
            def update_x(a: self.float_type, d: ti.types.ndarray()):
                for i in range(self.get_dimension()):
                    self.x[i] += a * d[i]
            update_x(alpha, self.d)

            

            # 记录历史
            self.f_his.append(self.f0)
            self.time_his.append(time.time() - start_time)

        # 达到最大迭代次数，返回 max_iter
        return max_iter

# 示例使用
if __name__ == "__main__":
    float_type = ti.f64
    ti.init(arch=ti.cpu, default_fp=float_type, device_memory_GB=20)
    dim =30000

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
        for i in range(x.shape[0]):
            if i % 3 == 0:
                # 对角线元素
                hess=ti.Matrix.zero(float_type, 3, 3)
                hess[0, 0] =  28*(3*x[i]**2 - x[i+1]) + 20
                hess[1, 1] = 14 +36*x[i]-36*x[i+2] +108*x[i+1]**2
                hess[2, 2] = 18
                # 非对角线元素
                hess[0, 1] = -28*x[i] +36*x[i+1]
                hess[0,2] = -18
                hess[1, 0] = -28*x[i]+36*x[i+1]
                hess[1, 2] = -36*x[i+1]
                hess[2, 0] = -18
                hess[2, 1] = -36*x[i+1]
                #正定化
                U, sig, V = ti.svd(hess)
                for j in ti.static(range(3)):
                    if sig[j,j] < 0:
                        sig[j,j] =  0
                hess = U @ sig @ V.transpose()

                # 填充Hessian矩阵
                for j in range(3):
                    for k in range(3):
                        H[i+ j, i+ k] += hess[j, k]
    # 初始化优化器
    optimizer = Newton(energy_fn=rosenbrock,
                      grad_fn=rosenbrock_grad,
                      hess_fn=rosenbrock_hess,
                      dim=dim,
                      float_type=float_type)
    
    x_np= np.ones(dim)
    # 设置初始值
    optimizer.x.from_numpy(x_np)
    
    # # 梯度检查
    # optimizer.check_gradient()
    
    # 执行优化
    optimizer.minimize(max_iter=200)

    print(f"Final parameters: {optimizer.x.to_numpy()}")
    
    # 绘制能量变化
    plt.plot(optimizer.f_his)
    plt.title("Energy History")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.show()