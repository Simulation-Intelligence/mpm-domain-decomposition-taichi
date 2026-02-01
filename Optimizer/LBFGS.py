import taichi as ti
import numpy as np
import time
import matplotlib.pyplot as plt



@ti.data_oriented
class LBFGS:
    def __init__(self, energy_fn, dim=3, alpha=1.0, beta=0.5, eta=1e-2, m=15):
        self.dim = dim
        self.m = m  # 历史窗口大小
        self.energy = energy_fn
        self.alpha = alpha  # 线搜索参数
        self.beta = beta    # 线搜索衰减率
        self.eta = eta      # 收敛阈值

        # 参数和梯度存储
        self.x = ti.field(ti.f32, shape=dim)
        self.grad = ti.field(ti.f32, shape=dim)
        self.temp_x = ti.field(ti.f32, shape=dim)
        self.temp_grad = ti.field(ti.f32, shape=dim)
        self.grad_prev = ti.field(ti.f32, shape=dim)
        self.d = ti.field(ti.f32, shape=dim)  # 搜索方向

        # LBFGS历史记录
        self.s = ti.field(ti.f32, shape=(dim, m))  # x的变化量
        self.y = ti.field(ti.f32, shape=(dim, m))  # 梯度的变化量
        self.rho = ti.field(ti.f32, shape=m)       # 1/(y^T s)
        self.alpha_arr = ti.field(ti.f32, shape=m) # 避免与线搜索alpha重名
        self.current_idx = ti.field(ti.i32, shape=())  # 当前历史索引
        self.gamma_k = ti.field(ti.f32, shape=())      # 缩放因子
        self.q = ti.field(ti.f32, shape=dim)           # 搜索方向中间变量
        self.x_old = ti.field(ti.f32, shape=dim)       # 旧参数
        self.grad_old = ti.field(ti.f32, shape=dim)    # 旧梯度

        self.current_idx[None] = 0
        self.gamma_k[None] = 1.0

        # 历史记录
        self.f_his = []
        self.time_his = []

    def check_gradient(self, x_test=None, h=1e-5, tol=1e-5):
        # 实现与BFGS相同，此处省略以节省空间
        pass

    @ti.kernel
    def init_history(self):
        for i, j in self.s:
            self.s[i, j] = 0.0
            self.y[i, j] = 0.0
        for i in range(self.m):
            self.rho[i] = 0.0
            self.alpha_arr[i] = 0.0

    @ti.kernel
    def compute_dir(self, grad: ti.template(), q: ti.template()) -> ti.f32:
        dir = 0.0
        for i in range(self.dim):
            dir += grad[i] * q[i]
        return dir

    @ti.kernel
    def compute_alpha_rho_reverse(self, current_idx: ti.i32, iter: ti.i32):
        for idx in range(iter):
            i = iter - 1 - idx  # 反向处理
            storage_idx = (current_idx - 1 - i) % self.m
            s_dot_y = 0.0
            for j in range(self.dim):
                s_dot_y += self.s[j, storage_idx] * self.y[j, storage_idx]
            rho_val = 1.0 / s_dot_y if s_dot_y != 0 else 0.0
            self.rho[storage_idx] = rho_val

            s_dot_q = 0.0
            for j in range(self.dim):
                s_dot_q += self.s[j, storage_idx] * self.q[j]
            alpha_val = rho_val * s_dot_q
            self.alpha_arr[storage_idx] = alpha_val

            # 更新 q = q - alpha_val * y
            for j in range(self.dim):
                self.q[j] -= alpha_val * self.y[j, storage_idx]

    @ti.kernel
    def compute_beta_forward(self, current_idx: ti.i32, iter: ti.i32):
        for i in range(iter):
            storage_idx = (current_idx - 1 - i) % self.m
            rho_val = self.rho[storage_idx]
            y_dot_q = 0.0
            for j in range(self.dim):
                y_dot_q += self.y[j, storage_idx] * self.q[j]
            beta = rho_val * y_dot_q

            alpha_val = self.alpha_arr[storage_idx]
            # 更新 q = q + (alpha_val - beta) * s
            for j in range(self.dim):
                self.q[j] += (alpha_val - beta) * self.s[j, storage_idx]

    @ti.kernel
    def scale_q(self, gamma: ti.f32):
        for i in range(self.dim):
            self.q[i] *= gamma

    @ti.kernel
    def update_history(self, s_temp: ti.template(), y_temp: ti.template(), current_idx: ti.i32):
        idx = current_idx % self.m
        for i in range(self.dim):
            self.s[i, idx] = s_temp[i]
            self.y[i, idx] = y_temp[i]

    @ti.kernel
    def compute_gamma_k(self, s_temp: ti.template(), y_temp: ti.template()) -> ti.f32:
        s_dot_y = 0.0
        y_dot_y = 0.0
        for i in range(self.dim):
            s_dot_y += s_temp[i] * y_temp[i]
            y_dot_y += y_temp[i] * y_temp[i]
        return s_dot_y / y_dot_y if y_dot_y != 0 else 1.0

    def line_search(self) -> ti.f32:
        alpha = 1.0
        f0 = self.energy(self.x, self.grad)
        
        @ti.kernel
        def calc_g0() -> ti.f32:
            g = 0.0
            for i in range(self.dim):
                g += self.grad[i] * self.d[i]
            return g
        g0 = calc_g0()

        @ti.kernel
        def update_temp_x(a: ti.f32):
            for i in range(self.dim):
                self.temp_x[i] = self.x[i] + a * self.d[i]

        while alpha > 1e-6:
            update_temp_x(alpha)
            f_new = self.energy(self.temp_x, self.temp_grad)
            if f_new <= f0 + self.alpha * alpha * g0:
                break
            alpha *= self.beta
        return alpha

    def minimize(self, max_iter=50):
        start_time = time.time()
        current_idx = 0

        # 初始梯度计算
        f = self.energy(self.x, self.grad)
        self.grad_old.copy_from(self.grad)

        for k in range(max_iter):
            f = self.energy(self.x, self.grad)
            self.f_his.append(f)
            print(f"Iteration {k}: Energy = {f}")

            # 检查梯度收敛
            @ti.kernel
            def grad_norm() -> ti.f32:
                n = 0.0
                for i in range(self.dim):
                    n += self.grad[i] ** 2
                return ti.sqrt(n)
            
            grad_n = grad_norm()
            print(f"Grad norm: {grad_n}")
            if grad_n < self.eta:
                print(f"Converged at iteration {k}")
                return k

            # 初始化搜索方向
            self.q.copy_from(self.grad)
            iter = min(k, self.m)

            if iter > 0:
                self.compute_alpha_rho_reverse(self.current_idx[None], iter)
                self.scale_q(self.gamma_k[None])
                self.compute_beta_forward(self.current_idx[None], iter)

            # 检查方向是否为下降方向
            dir = self.compute_dir(self.grad, self.q)
            if dir <= 0:
                print("Resetting to gradient direction")
                self.q.copy_from(self.grad)
                self.gamma_k[None] = 1.0

            # 设置搜索方向d为负q
            @ti.kernel
            def set_d():
                for i in range(self.dim):
                    self.d[i] = -self.q[i]
            set_d()

            # 线搜索
            alpha = self.line_search()
            print(f"Alpha: {alpha}")

            # 保存旧状态
            self.x_old.copy_from(self.x)
            self.grad_old.copy_from(self.grad)

            # 更新x
            @ti.kernel
            def update_x(alpha: ti.f32):
                for i in range(self.dim):
                    self.x[i] += alpha * self.d[i]
            update_x(alpha)

            # 计算新梯度
            self.energy(self.x, self.grad)

            # 计算s和y
            @ti.kernel
            def compute_s_y(s_temp: ti.template(), y_temp: ti.template()):
                for i in range(self.dim):
                    s_temp[i] = self.x[i] - self.x_old[i]
                    y_temp[i] = self.grad[i] - self.grad_old[i]
            
            s_temp = ti.field(ti.f32, self.dim)
            y_temp = ti.field(ti.f32, self.dim)
            compute_s_y(s_temp, y_temp)

            # 更新gamma_k
            gamma_k = self.compute_gamma_k(s_temp, y_temp)
            self.gamma_k[None] = gamma_k if gamma_k > 0 else 1.0

            # 更新历史记录
            self.update_history(s_temp, y_temp, self.current_idx[None])
            self.current_idx[None] = (self.current_idx[None] + 1) % self.m

            self.time_his.append(time.time() - start_time)

        # 达到最大迭代次数
        return max_iter


# 测试示例（与BFGS相同）
if __name__ == "__main__":
    dim = 9

    ti.init(arch=ti.vulkan)

    @ti.kernel  
    def rosenbrock_energy(x: ti.template(),  grad: ti.template()) -> ti.f32:
        """支持梯度引用传递的能量函数实现"""
        f_term_total = 0.0
        for i in range(x.shape[0]):
            if i % 3 == 0:
                x1, x2, x3 = x[i], x[i+1], x[i+2]

                # 能量计算 
                f_term = (3 - x1)**2 + 7*(x2 - x1**2)**2 + 9*(x3 - x1 - x2**2)**2 

                # 梯度计算 
                grad[i] = 2*(x1 - 3) + 28*(x1**2 - x2)*x1 + 18*(-x3 + x1 + x2**2)
                grad[i+1] = 14*(x2 - x1**2) + 18*(x3 - x1 - x2**2)*(-2*x2)
                grad[i+2] = 18*(x3 - x1 - x2**2)

                f_term_total += f_term
        return f_term_total 

    # 初始化参数
    x_init = np.zeros(dim, dtype=np.float32)
    lbfgs = LBFGS(rosenbrock_energy, dim=dim)
    lbfgs.x.from_numpy(x_init)
    
    # 运行优化
    lbfgs.minimize(max_iter=200)
    print("Optimized x:", lbfgs.x.to_numpy())