import taichi as ti 
import numpy as np 
import time 
import matplotlib.pyplot  as plt 
 

 
@ti.data_oriented  
class BFGS:
    def __init__(self, energy_fn,grad_fn, dim=3, alpha=0.01, beta=0.6, eta=20,grad_normalizer=1.0):
        self.dim  = dim 
        self.energy_fn  = energy_fn 
        self.grad_fn = grad_fn
        self.alpha  = alpha 
        self.beta  = beta 
        self.eta  = eta
        self.grad_normalizer=grad_normalizer
 
        # 全部改为标量场存储 
        self.x = ti.field(ti.f32,  shape=dim)       # 参数向量 
        self.grad  = ti.field(ti.f32,  shape=dim)    # 梯度向量
        self.temp_x  = ti.field(ti.f32,  shape=dim)   # 临时参数向量
        self.temp_grad  = ti.field(ti.f32,  shape=dim) # 临时梯度向量
        self.grad_prev  = ti.field(ti.f32,  shape=dim) # 上一次梯度向量 
        self.H = ti.field(ti.f32,  shape=(dim, dim)) # Hessian近似矩阵 
        self.d = ti.field(ti.f32,  shape=dim)       # 搜索方向 
        self.dx = ti.field(ti.f32,  shape=dim)      # 临时参数变化量
        self.dg = ti.field(ti.f32,  shape=dim)      # 临时梯度变化量
 
        # 历史记录 
        self.f_his = []
        self.time_his  = []

        self.init_Hessian() 

    @ti.kernel  
    def update_hessian(self):
        
        # 计算 H * dg 
        Hdg = ti.Vector.zero(ti.f32,  self.dim) 
        for i in range(self.dim): 
            for j in range(self.dim): 
                Hdg[i] += self.H[i, j] * self.dg[j]
        
        # 计算各项系数 
        dgdx = 0.0 
        for i in range(self.dim): 
            dgdx += self.dg[i] * self.dx[i]
        
        Hdg_norm = 0.0 
        for i in range(self.dim): 
            Hdg_norm += Hdg[i] * self.dg[i]
        
        # 更新Hessian矩阵 
        for i in range(self.dim): 
            for j in range(self.dim): 
                term1 = (1 + Hdg_norm / dgdx) * self.dx[i] * self.dx[j] / dgdx 
                term2 = (Hdg[i] *self.dx[j] + Hdg[j] * self.dx[i]) / dgdx 
                self.H[i, j] = self.H[i, j] + term1 - term2 

    @ti.kernel  
    def init_Hessian(self):
        for i, j in ti.ndrange(self.dim,  self.dim): 
            self.H[i, j] = 1.0 if i == j else 0.0 
 
    def line_search(self) -> ti.f32:
        alpha = 1.0 
        f0 = self.grad_fn(self.x,  self.grad)
        g0 = 0.0 
        @ti.kernel
        def calc_g0()->ti.f32:
            g = 0.0
            for i in range(self.dim):
                g += self.grad[i] * self.d[i]
            return g
        g0=calc_g0()
        @ti.kernel
        def calc_temp_x(a: ti.f32):
            for i in range(self.dim): 
                self.temp_x[i] = self.x[i] + a* self.d[i]
        while alpha > 1e-6:
            calc_temp_x(alpha)
            f_new = self.energy_fn(self.temp_x)
            if f_new <= f0 + self.alpha  * alpha * g0:
                break 
            alpha *= self.beta  
        return alpha 
 
    def minimize(self,max_iter=200,init_iter=50):
        # 初始化参数 
        
        # start_time = time.time() 
        for _ in range(max_iter):
            # 计算当前能量和梯度 
            f = self.grad_fn(self.x,  self.grad)
            print ("Iteration:", _)
            print(f"Energy: {f}")
            # 检查收敛  
            @ti.kernel
            def calc_grad_norm() -> ti.f32:
                grad_norm = 0.0
                for i in range(self.dim): 
                    grad_norm += self.grad[i]**2
                return grad_norm
            grad_norm=calc_grad_norm() / self.grad_normalizer ** 2 
            print(f"Grad norm: {ti.sqrt(grad_norm)}")
            if grad_norm < self.eta**2 :
                print(f"Converged at iteration {_}")
                break 
            
            # 计算搜索方向 d = -H * grad 
            @ti.kernel
            def calc_d():
                for i in range(self.dim): 
                    self.d[i] = 0.0
                    for j in range(self.dim): 
                        self.d[i] -= self.H[i, j] * self.grad[j]
            calc_d()

            # 线搜索 
            alpha = self.line_search() 
            print("Alpha:", alpha)
            # 保存旧梯度 
            @ti.kernel
            def save_grad():
                for i in range(self.dim): 
                    self.grad_prev[i] = self.grad[i]
            save_grad()
            
            @ti.kernel
            def update_x():
                for i in range(self.dim): 
                    self.dx[i] = alpha * self.d[i]
                    self.x[i] += self.dx[i]
            update_x()
            
            # 计算新梯度
            self.grad_fn(self.x,  self.grad)
            @ti.kernel
            def calc_dg():
                for i in range(self.dim): 
                    self.dg[i] = self.grad[i] - self.grad_prev[i]
            calc_dg()
            # 更新Hessian 
            self.update_hessian() 

            if _ % init_iter == 0:
                self.init_Hessian()
            # 记录历史 
            # self.f_his.append(f) 
            # self.time_his.append(time.time()  - start_time)
        
 
    def check_gradient(self, x_test=None, h=1e-5, tol=1e-5):
        # 确定测试点
        if x_test is None:
            x_test_np = self.x.to_numpy().copy()
        else:
            x_test_np = np.array(x_test, dtype=np.float32).copy()

        # 将测试点复制到临时变量temp_x中，避免影响当前优化状态
        self.temp_x.from_numpy(x_test_np)

        # 计算解析梯度
        f_analytic = self.grad_fn(self.temp_x, self.temp_grad)
        grad_analytic = self.temp_grad.to_numpy().copy()
        
        #如果全0则不计算
        if np.all(grad_analytic == 0):
            print("\033[91mAnalytic Gradient is zero. Skipping gradient check.\033[0m")
            return True

        # 计算数值梯度
        grad_num = np.zeros_like(x_test_np, dtype=np.float32)
        for i in range(self.dim):
            # 正向扰动
            x_plus = x_test_np.copy()
            x_plus[i] += h
            self.temp_x.from_numpy(x_plus)
            f_plus = self.grad_fn(self.temp_x, self.temp_grad)

            # 负向扰动
            x_minus = x_test_np.copy()
            x_minus[i] -= h
            self.temp_x.from_numpy(x_minus)
            f_minus = self.grad_fn(self.temp_x, self.temp_grad)

            # 中心差分
            grad_num[i] = (f_plus - f_minus) / (2 * h)

        # 计算梯度差异
        diff = np.abs(grad_analytic - grad_num)
        rel_diff = diff / (np.abs(grad_num) + 1e-8)  # 避免除以零

        max_abs_diff = np.max(diff)
        max_rel_diff = np.max(rel_diff)

        print("Numerical Gradient:\n", grad_num)
        print("Analytic Gradient:\n", grad_analytic)
        print("Max Absolute Difference:", max_abs_diff)
        print("Max Relative Difference:", max_rel_diff)

        # 检查是否通过
        if max_abs_diff > tol or max_rel_diff > tol:
            print("\033[91mGradient check failed. Potential errors in gradient computation.\033[0m")
            return False
        else:
            print("\033[92mGradient check passed.\033[0m")
            return True
# 使用示例（需要修改能量函数接口）
if __name__ == "__main__":
    dim = 3
    ti.init(arch=ti.cuda) 
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
    
    @ti.kernel
    def quadratic_energy(x: ti.template(), grad: ti.template()) -> ti.f32:
        """支持梯度引用传递的能量函数实现"""
        f_term_total = 0.0
        for i in range(x.shape[0]):
            f_term = x[i]**2
            grad[i] = 2 * x[i]
            f_term_total += f_term
        return f_term_total
 
    bfgs = BFGS(rosenbrock_energy, dim=dim)
    bfgs.check_gradient()
    bfgs.minimize(max_iter=1000)
    
    print("Optimized result:", bfgs.x.to_numpy())
    plt.plot(bfgs.f_his) 
    plt.ylabel('Energy') 
    plt.xlabel('Iteration') 
    plt.show() 