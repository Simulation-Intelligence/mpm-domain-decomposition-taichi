import taichi as ti

from Util.Config import Config



# ------------------ 网格模块 ------------------
@ti.data_oriented
class Grid:
    def __init__(self, config:Config):
        self.dim = config.get("dim", 2)
        self.bound = config.get("bound", 2)

        # 支持新的网格配置格式
        if self.dim == 2:
            # 2D情况
            if "grid_nx" in config.data and "grid_ny" in config.data:
                self.nx = config.get("grid_nx", 16)
                self.ny = config.get("grid_ny", 16)
                self.size = max(self.nx, self.ny)  # 为了兼容性保留size
            else:
                # 兼容旧的grid_size配置
                self.size = config.get("grid_size", 16)
                self.nx = self.size
                self.ny = self.size

            # 2D情况下为兼容性定义nz
            self.nz = 1

            # 支持自定义域尺寸
            self.domain_width = config.get("domain_width", 1.0)
            self.domain_height = config.get("domain_height", 1.0)
            self.domain_depth = 1.0  # 2D情况下不使用

        else:
            # 3D情况
            if "grid_nx" in config.data and "grid_ny" in config.data and "grid_nz" in config.data:
                self.nx = config.get("grid_nx", 16)
                self.ny = config.get("grid_ny", 16)
                self.nz = config.get("grid_nz", 16)
                self.size = max(self.nx, self.ny, self.nz)  # 为了兼容性保留size
            else:
                # 兼容旧的grid_size配置
                self.size = config.get("grid_size", 16)
                self.nx = self.size
                self.ny = self.size
                self.nz = self.size

            # 支持自定义域尺寸
            self.domain_width = config.get("domain_width", 1.0)
            self.domain_height = config.get("domain_height", 1.0)
            self.domain_depth = config.get("domain_depth", 1.0)

        # 获取网格偏移
        self.offset = config.get("offset", [0.0, 0.0] if self.dim == 2 else [0.0, 0.0, 0.0])

        # 计算网格间距
        self.dx_x = self.domain_width / self.nx
        self.dx_y = self.domain_height / self.ny
        self.inv_dx_x = self.nx / self.domain_width
        self.inv_dx_y = self.ny / self.domain_height

        if self.dim == 3:
            self.dx_z = self.domain_depth / self.nz
            self.inv_dx_z = self.nz / self.domain_depth
            self.dx = min(self.dx_x, self.dx_y, self.dx_z)
            self.inv_dx = max(self.inv_dx_x, self.inv_dx_y, self.inv_dx_z)
        else:
            self.dx = min(self.dx_x, self.dx_y)
            self.inv_dx = max(self.inv_dx_x, self.inv_dx_y)

        self.float_type = ti.f32 if config.get("float_type", "f32") == "f32" else ti.f64

        # 根据维度配置field大小
        if self.dim == 2:
            grid_shape = (self.nx, self.ny)
        else:
            grid_shape = (self.nx, self.ny, self.nz)

        self.v = ti.Vector.field(self.dim, self.float_type, grid_shape)
        self.m = ti.field(self.float_type, grid_shape)
        self.v_prev = ti.Vector.field(self.dim, self.float_type, grid_shape)
        self.f = ti.Vector.field(self.dim, self.float_type, grid_shape)  # 力场
        self.volume_force = ti.Vector.field(self.dim, self.float_type, grid_shape)  # 体积力场

        # 隐式求解相关
        if self.dim == 2:
            self.particles = ti.field(ti.i32, (self.nx, self.ny, 32))
            self.wip = ti.field(self.float_type, (self.nx, self.ny, 32))
            self.dwip = ti.Vector.field(self.dim, self.float_type, (self.nx, self.ny, 32))
            self.particle_count = ti.field(ti.i32, (self.nx, self.ny))
        else:
            # 3D情况使用nx, ny, nz
            self.particles = ti.field(ti.i32, (self.nx, self.ny, self.nz, 32))
            self.wip = ti.field(self.float_type, (self.nx, self.ny, self.nz, 32))
            self.dwip = ti.Vector.field(self.dim, self.float_type, (self.nx, self.ny, self.nz, 32))
            self.particle_count = ti.field(ti.i32, (self.nx, self.ny, self.nz))

        # 边界条件
        self.is_boundary_grid = ti.Vector.field(self.dim, ti.i32, grid_shape)
        self.boundary_v = ti.Vector.field(self.dim, self.float_type, grid_shape)
        self.is_particle_boundary_grid = ti.field(ti.i32, grid_shape)

        # X方向速度约束选项
        self.constrain_x_velocity = config.get("constrain_x_velocity", False)

        default_DBC_range = [[0.0, 0.0], [0.0, 0.0]] if self.dim == 2 else [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        DBC_range = config.get("DBC_range", [default_DBC_range])
        self.num_DBC = len(DBC_range)
        if self.num_DBC > 0:
            self.DBC_range = ti.Vector.field(2, self.float_type, shape=(self.num_DBC, self.dim))
            for i in range(self.num_DBC):
                for d in range(self.dim):
                    self.DBC_range[i, d] = ti.Vector(DBC_range[i][d])

        # DBC边界速度配置
        default_boundary_v = [0.0] * self.dim
        DBC_boundary_v = config.get("DBC_boundary_v", [default_boundary_v] * self.num_DBC)
        if self.num_DBC > 0:
            self.DBC_boundary_v = ti.Vector.field(self.dim, self.float_type, shape=self.num_DBC)
            for i in range(self.num_DBC):
                for d in range(self.dim):
                    if i < len(DBC_boundary_v):
                        self.DBC_boundary_v[i][d] = DBC_boundary_v[i][d] if d < len(DBC_boundary_v[i]) else 0.0
                    else:
                        self.DBC_boundary_v[i][d] = 0.0

        # Mapping功能：压缩有质量的grid节点
        max_grid_points = self.get_total_grid_points()
        self.active_grid_count = ti.field(ti.i32, shape=())  # 有质量的grid数量
        self.grid_to_compressed = ti.field(ti.i32, shape=grid_shape)  # grid index -> 压缩index (-1表示无效)
        self.compressed_to_grid = ti.Vector.field(self.dim, ti.i32, shape=max_grid_points)  # 压缩index -> grid indices

        # 初始化mapping为无效值
        self.grid_to_compressed.fill(-1)

        # 配置是否使用mapping
        self.use_mapping = config.get("use_mapping", False)

    def get_total_grid_points(self):
        """获取网格点总数"""
        if self.dim == 2:
            return self.nx * self.ny
        else:
            return self.nx * self.ny * self.nz

    @ti.func
    def get_idx(self, I):
        """根据配置自动选择原始索引或压缩映射索引"""
        if ti.static(self.use_mapping):
            # 使用压缩映射
            compressed_idx = self.get_compressed_index(I)
            res = compressed_idx * self.dim
            if compressed_idx < 0:
                res = -1
            return res  # 无效的grid节点
        else:
            # 使用原始索引计算
            idx = 0
            if ti.static(self.dim == 2):
                idx = I[0] * self.ny * self.dim + I[1] * self.dim
            else:
                # 3D情况保持原有计算方式，使用size
                for d in ti.static(range(self.dim)):
                    idx += I[d] * self.size ** (self.dim-1-d) *self.dim
            return idx

    @ti.func
    def get_grid_pos(self, I):
        """获取网格点的物理位置"""
        pos = ti.Vector.zero(self.float_type, self.dim)
        pos[0] = I[0] * self.dx_x
        pos[1] = I[1] * self.dx_y
        if ti.static(self.dim == 3):
            pos[2] = I[2] * self.dx_z
        return pos

    def build_mapping(self):
        """构建有质量grid节点的mapping"""
        # 重置mapping
        self.grid_to_compressed.fill(-1)
        self.active_grid_count[None] = 0

        # 重新建立映射
        if self.dim == 2:
            self._build_mapping_2d()
        else:
            self._build_mapping_3d()

        print(f"Active grid count: {self.active_grid_count[None]} / {self.get_total_grid_points()}")
        return self.active_grid_count[None]

    @ti.kernel
    def _count_active_grids_2d(self) -> ti.i32:
        """统计2D有质量的grid数量"""
        count = 0
        for i, j in self.m:
            if i==21 and j==15:
                print(self.m[i,j])
            if self.m[i, j] > 0:  # 使用合理的质量阈值，避免数值误差
                ti.atomic_add(count, 1)
        return count

    @ti.kernel
    def _count_active_grids_3d(self) -> ti.i32:
        """统计3D有质量的grid数量"""
        count = 0
        for i, j, k in self.m:
            if self.m[i, j, k] > 0:  # 使用合理的质量阈值，避免数值误差
                ti.atomic_add(count, 1)
        return count

    @ti.kernel
    def _build_mapping_2d(self):
        """使用原子操作避免竞争条件"""
        for i, j in self.m:
            if self.m[i, j] > 0:  # 使用合理的质量阈值，避免数值误差
                # 使用原子操作获取唯一的compressed索引
                compressed_idx = ti.atomic_add(self.active_grid_count[None], 1)
                # 建立双向映射
                self.grid_to_compressed[i, j] = compressed_idx
                if i ==21 and j==15:
                    print(f"Mapping grid (21,15) to compressed index {compressed_idx}")
                self.compressed_to_grid[compressed_idx] = ti.Vector([i, j])

    @ti.kernel
    def _build_mapping_3d(self):
        """使用原子操作避免竞争条件"""
        for i, j, k in self.m:
            if self.m[i, j, k] > 0:  # 使用合理的质量阈值，避免数值误差
                # 使用原子操作获取唯一的compressed索引
                compressed_idx = ti.atomic_add(self.active_grid_count[None], 1)
                # 建立双向映射
                self.grid_to_compressed[i, j, k] = compressed_idx
                self.compressed_to_grid[compressed_idx] = ti.Vector([i, j, k])

    @ti.func
    def get_active_grid_count(self):
        """获取有质量的grid节点数量"""
        return self.active_grid_count[None]

    @ti.func
    def get_compressed_index(self, I):
        """获取grid索引对应的压缩索引"""
        if ti.static(self.dim == 2):
            return self.grid_to_compressed[I[0], I[1]]
        else:
            return self.grid_to_compressed[I[0], I[1], I[2]]

    @ti.func
    def get_grid_indices_from_compressed(self, compressed_idx):
        """从压缩索引获取grid索引"""
        return self.compressed_to_grid[compressed_idx]

    @ti.func
    def pos_to_grid_index(self, pos):
        """将物理位置转换为网格索引（支持矩形网格）"""
        grid_pos = ti.Vector.zero(self.float_type, self.dim)
        grid_pos[0] = pos[0] * self.inv_dx_x
        grid_pos[1] = pos[1] * self.inv_dx_y
        if ti.static(self.dim == 3):
            grid_pos[2] = pos[2] * self.inv_dx_z
        return grid_pos

    @ti.func
    def particle_to_grid_base_and_fx(self, particle_pos):
        """计算粒子到网格的基础索引和分数部分（MPM专用）"""
        # 使用矩形网格的正确dx_x和dx_y
        base = ti.Vector.zero(ti.i32, self.dim)
        fx = ti.Vector.zero(self.float_type, self.dim)

        base[0] = ti.cast(particle_pos[0] * self.inv_dx_x - 0.5, ti.i32)
        base[1] = ti.cast(particle_pos[1] * self.inv_dx_y - 0.5, ti.i32)
        fx[0] = ti.cast(particle_pos[0] * self.inv_dx_x - base[0], self.float_type)
        fx[1] = ti.cast(particle_pos[1] * self.inv_dx_y - base[1], self.float_type)

        if ti.static(self.dim == 3):
            base[2] = ti.cast(particle_pos[2] * self.inv_dx_z - 0.5, ti.i32)
            fx[2] = ti.cast(particle_pos[2] * self.inv_dx_z - base[2], self.float_type)

        return base, fx

    @ti.func
    def get_dpos_from_offset_and_fx(self, offset, fx):
        """根据偏移和分数部分计算相对位置（MPM专用）"""
        dpos = ti.Vector.zero(self.float_type, self.dim)
        dpos[0] = (offset[0] - fx[0]) * self.dx_x
        dpos[1] = (offset[1] - fx[1]) * self.dx_y

        if ti.static(self.dim == 3):
            dpos[2] = (offset[2] - fx[2]) * self.dx_z

        return dpos

    @ti.kernel
    def apply_boundary_conditions_explicit(self):
        for I in ti.grouped(self.v):
            cond = ti.Vector.zero(ti.i32, self.dim)
            for d in ti.static(range(self.dim)):
                grid_size = self.get_grid_size(d)
                cond[d] = (I[d] < self.bound) & (self.v[I][d] < 0) | (I[d] > grid_size - self.bound) & (self.v[I][d] > 0)

            for d in ti.static(range(self.dim)):
                if cond[d]:
                    self.v[I][d] = 0

    @ti.func
    def get_grid_size(self, d):
        """获取指定维度的网格大小"""
        result = 0
        if d == 0:
            result = self.nx
        elif d == 1:
            result = self.ny
        elif d == 2:
            result = self.nz
        return result

    @ti.kernel
    def apply_boundary_conditions(self):
        for I in ti.grouped(self.v):
            cond = ti.Vector.zero(ti.i32, self.dim)
            for d in ti.static(range(self.dim)):
                grid_size = self.get_grid_size(d)
                cond[d] = (I[d] < self.bound) | (I[d] > grid_size - self.bound)

            is_boundary = False
            for d in ti.static(range(self.dim)):
                if cond[d]:
                    is_boundary = True

            if is_boundary:
                self.is_boundary_grid[I] = cond
                for d in ti.static(range(self.dim)):
                    if cond[d]:
                        self.boundary_v[I][d] = 0
                    else:
                        self.boundary_v[I][d] = self.v[I][d]

            # X方向速度约束：如果启用，则强制所有网格点的x方向速度为0
            if ti.static(self.constrain_x_velocity):
                self.is_boundary_grid[I][0] = 1
                self.boundary_v[I][0] = 0.0

            if self.num_DBC > 0:
                pos = self.get_grid_pos(I)
                applied_DBC = False
                for i in range(self.num_DBC):
                    if not applied_DBC:
                        is_DBC = True
                        for d in ti.static(range(self.dim)):
                            if not (self.DBC_range[i, d][0] <= pos[d] <= self.DBC_range[i, d][1]):
                                is_DBC = False
                        if is_DBC:
                            self.is_boundary_grid[I] = ti.Vector([1]*self.dim)
                            self.boundary_v[I] = self.DBC_boundary_v[i]
                            applied_DBC = True
                    


    @ti.kernel
    def set_boundary_grad(self,grad: ti.template()):
        if ti.static(self.use_mapping):
            # 使用压缩映射时只遍历有质量的节点
            for compressed_idx in range(self.get_active_grid_count()):
                I = self.get_grid_indices_from_compressed(compressed_idx)
                vidx = compressed_idx * self.dim
                for d in ti.static(range(self.dim)):
                    if self.is_boundary_grid[I][d]:
                        grad[vidx+d] = 0.0
        else:
            # 原始方法：遍历所有边界节点
            for I in ti.grouped(self.is_boundary_grid):
                idx = self.get_idx(I)
                for d in ti.static(range(self.dim)):
                    if self.is_boundary_grid[I][d]:
                        grad[idx+d] = 0.0 
                    
    @ti.kernel
    def set_boundary_v(self):
        for I in ti.grouped(self.is_boundary_grid):
            self.v[I]=ti.select(self.is_boundary_grid[I],self.boundary_v[I],self.v[I])
    
    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.v):
            self.m[I] = 0.0
            self.v[I] = [0.0]*self.dim
            self.volume_force[I] = [0.0]*self.dim
            self.is_particle_boundary_grid[I] = 0
            self.is_boundary_grid[I] = [0]*self.dim

    @ti.kernel
    def clear_sub(self):
        for I in ti.grouped(self.v):
            self.m[I] = 0.0
            self.v[I] = [0.0]*self.dim

    def get_lines_begin(self):
        """获取网格线的起始点，用于可视化"""
        lines_begin = []
        if self.dim == 2:
            # 垂直线
            for i in range(self.nx + 1):
                x = i * self.dx_x + self.offset[0]
                lines_begin.append([x, self.offset[1]])
            # 水平线
            for j in range(self.ny + 1):
                y = j * self.dx_y + self.offset[1]
                lines_begin.append([self.offset[0], y])
        return lines_begin

    def get_lines_end(self):
        """获取网格线的结束点，用于可视化"""
        lines_end = []
        if self.dim == 2:
            # 垂直线
            for i in range(self.nx + 1):
                x = i * self.dx_x + self.offset[0]
                lines_end.append([x, self.offset[1] + self.domain_height])
            # 水平线
            for j in range(self.ny + 1):
                y = j * self.dx_y + self.offset[1]
                lines_end.append([self.offset[0] + self.domain_width, y])
        return lines_end

    @ti.kernel
    def compute_max_velocity_magnitude(self) -> ti.f64:
        """计算所有网格点速度的最大模长"""
        max_v_mag = 0.0
        for I in ti.grouped(self.v):
            v_mag = 0.0
            for d in ti.static(range(self.dim)):
                v_mag += self.v[I][d] ** 2
            v_mag = ti.sqrt(v_mag)
            max_v_mag = ti.max(max_v_mag, v_mag)
        return max_v_mag

