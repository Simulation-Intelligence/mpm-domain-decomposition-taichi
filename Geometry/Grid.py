import taichi as ti



# ------------------ 网格模块 ------------------
@ti.data_oriented
class Grid:
    def __init__(self, size, dim, bound,float_type=ti.f32):
        self.size = size
        self.dim = dim
        self.bound = bound
        self.dx = 1.0 / size
        self.inv_dx = size
        self.float_type = float_type
        
        self.v = ti.Vector.field(dim, self.float_type, (size,)*dim)
        self.m = ti.field(self.float_type, (size,)*dim)
        self.v_prev = ti.Vector.field(dim, self.float_type, (size,)*dim)
        
        # 隐式求解相关
        self.particles = ti.field(ti.i32, (size, size, 32))
        self.wip = ti.field(self.float_type, (size, size, 32))
        self.dwip = ti.Vector.field(dim, self.float_type, (size, size, 32))
        self.particle_count = ti.field(ti.i32, (size, size))

        # 边界条件
        self.is_boundary_grid = ti.Vector.field(dim, ti.i32, (size,)*dim)
        self.boundary_v = ti.Vector.field(dim, self.float_type, (size,)*dim)
        self.is_particle_boundary_grid = ti.field(ti.i32, (size,)*dim)

    @ti.kernel
    def apply_boundary_conditions_explicit(self):
        for I in ti.grouped(self.v):
            cond = (I < self.bound) & (self.v[I] < 0) | (I > self.size - self.bound) & (self.v[I] > 0)
            self.v[I] = ti.select(cond, 0, self.v[I])

    @ti.kernel
    def apply_boundary_conditions(self):
        for I in ti.grouped(self.v):
            cond = (I < self.bound) | (I > self.size - self.bound)
            is_boundary =False
            
            for d in ti.static(range(self.dim)):
                if cond[d]:
                    is_boundary = True

            if is_boundary:
                self.is_boundary_grid[I] = cond
                self.boundary_v[I] = ti.select(cond, 0, self.v[I])

    @ti.kernel
    def set_boundary_v_grid(self,v_grad: ti.template()):
        for I in ti.grouped(self.is_boundary_grid):
            for d in ti.static(range(self.dim)):
                if self.is_boundary_grid[I][d]:
                    idx= I[0] * self.size * 2 + I[1] * 2 + d
                    v_grad[idx] = self.boundary_v[I][d]
            
    @ti.kernel
    def set_boundary_grad(self,grad: ti.template()):
        for I in ti.grouped(self.is_boundary_grid):
            for d in ti.static(range(self.dim)):
                if self.is_boundary_grid[I][d]:
                    idx= I[0] * self.size * 2 + I[1] * 2 + d
                    grad[idx] = 0.0 

    @ti.kernel
    def get_boundary_hess(self,hess1:ti.sparse_matrix_builder(),hess2:ti.sparse_matrix_builder()):
        for I in ti.grouped(self.is_boundary_grid):
            for d1 in ti.static(range(self.dim)):
                idx1= I[0] * self.size * 2 + I[1] * 2 + d1
                if not self.is_boundary_grid[I][d1] :
                    for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
                        for d2 in ti.static(range(self.dim)):
                            idx2= i * self.size * 2 + j * 2 + d2
                            if not self.is_boundary_grid[i, j][d2]:
                                hess1[idx1, idx2] +=1.0
                else:
                    hess2[idx1, idx1] += 1.0


    @ti.kernel
    def set_boundary_v(self):
        for I in ti.grouped(self.is_boundary_grid):
            self.v[I]=ti.select(self.is_boundary_grid[I],self.boundary_v[I],self.v[I])
    
    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.v):
            self.m[I] = 0.0
            self.v[I] = [0.0]*self.dim
            self.is_particle_boundary_grid[I] = 0
            self.is_boundary_grid[I] = [0]*self.dim