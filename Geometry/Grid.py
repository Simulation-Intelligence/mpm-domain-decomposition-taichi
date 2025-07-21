import taichi as ti

from Util.Config import Config



# ------------------ 网格模块 ------------------
@ti.data_oriented
class Grid:
    def __init__(self, config:Config):
        self.size = config.get("grid_size", 16)
        self.dim = config.get("dim", 2)
        self.bound = config.get("bound", 2)
        self.dx = 1.0 / self.size
        self.inv_dx = self.size
        self.float_type = ti.f32 if config.get("float_type", "f32") == "f32" else ti.f64

        dim = self.dim
        size = self.size

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
        default_DBC_range = [[0.0, 0.0], [0.0, 0.0]] if self.dim == 2 else [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        DBC_range = config.get("DBC_range", [default_DBC_range])
        self.num_DBC = len(DBC_range)
        if self.num_DBC > 0:
            self.DBC_range = ti.Vector.field(2, self.float_type, shape=(self.num_DBC, self.dim))
            for i in range(self.num_DBC):
                for d in range(dim):
                    self.DBC_range[i, d] = ti.Vector(DBC_range[i][d])


    @ti.func
    def get_idx(self, I):
        idx = 0
        for d in ti.static(range(self.dim)):
            idx += I[d] * self.size ** (self.dim-1-d) *self.dim
        return idx
    
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
            
            if self.num_DBC >0:
                pos = I * self.dx
                is_DBC = True
                for i in range(self.num_DBC):
                    for d in ti.static(range(self.dim)):
                        if not (self.DBC_range[i, d][0] <= pos[d] <= self.DBC_range[i, d][1]):
                            is_DBC = False
                if is_DBC:
                    self.is_boundary_grid[I] = ti.Vector([1]*self.dim)
                    self.boundary_v[I] = ti.Vector([0.0]*self.dim)
                    


    @ti.kernel
    def set_boundary_v_grid(self,v_grad: ti.template()):
        for I in ti.grouped(self.is_boundary_grid):
            idx = self.get_idx(I)
            for d in ti.static(range(self.dim)):
                if self.is_boundary_grid[I][d]:
                    v_grad[idx+ d] = self.boundary_v[I][d]
            
    @ti.kernel
    def set_boundary_grad(self,grad: ti.template()):
        for I in ti.grouped(self.is_boundary_grid):
            idx = self.get_idx(I)
            for d in ti.static(range(self.dim)):
                if self.is_boundary_grid[I][d]:
                    grad[idx+d] = 0.0 

    @ti.kernel
    def get_boundary_hess(self,hess1:ti.sparse_matrix_builder(),hess2:ti.sparse_matrix_builder()):
        for I in ti.grouped(self.is_boundary_grid):
            idx1 = self.get_idx(I)
            for d1 in ti.static(range(self.dim)):
                # if self.is_boundary_grid[I][d1]:
                hess2[idx1 + d1, idx1 + d1] += 1.0
                # if not self.is_boundary_grid[I][d1] :
                #     if ti.static(self.dim ==2):
                #         for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
                #             for d2 in ti.static(range(self.dim)):
                #                 idx2= i * self.size * 2 + j * 2 + d2
                #                 if not self.is_boundary_grid[i, j][d2]:
                #                     hess1[idx1+d1 , idx2] +=1.0
                #     elif ti.static(self.dim ==3):
                #         for i, j, k in ti.static(ti.ndrange(self.dim, self.dim, self.dim)):
                #             for d2 in ti.static(range(self.dim)):
                #                 idx2 = i * self.size**2 * 3 + j * self.size * 3 + k * 3 + d2
                #                 if not self.is_boundary_grid[i, j, k][d2]:
                #                     hess1[idx1+d1 , idx2] +=1.0
                                
                # else:
                #     hess2[idx1+d1, idx1+d1] += 1.0

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

    @ti.kernel
    def clear_sub(self):
        for I in ti.grouped(self.v):
            self.m[I] = 0.0
            self.v[I] = [0.0]*self.dim

