import taichi as ti

from Geometry.Grid import Grid
from Geometry.Particles import Particles

from implicit_solver import ImplicitSolver

from Util.Config import Config

from Util.Recorder import *



# ------------------ 主模拟器 ------------------
@ti.data_oriented
class ImplicitMPM:
    def __init__(self, config:Config):
        self.cfg = config

        self.float_type = ti.f32 if config.get("float_type", "f32") == "f32" else ti.f64
    
        self.grid = Grid(
            size=self.cfg.get("grid_size", 16),
            dim=2,
            bound=self.cfg.get("bound", 2),
            float_type=self.float_type
        )
        self.particles = Particles(self.cfg)
        self.implicit = self.cfg.get("implicit", True)
        self.max_iter = self.cfg.get("max_iter", 1)
        self.dt = self.cfg.get("dt", 2e-3)

        E = self.cfg.get("E", 4)
        nu = self.cfg.get("nu", 0.4)
        self.mu = E / (2 * (1 + nu))  # 现在作为实例属性
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        if self.implicit:
            self.solver = ImplicitSolver(self.grid, self.particles, self.cfg)


        # 其他公共参数初始化
        self.max_schwarz_iter = config.get("max_schwarz_iter", 1)  # Schwarz迭代次数
        self.recorder = None
        if config.get("record_frames", 0) > 0:
            self.recorder = ParticleRecorder(
            palette=np.array([
                0x66CCFF,  # 域1普通粒子
                0xFFFFFF
            ], dtype=np.uint32),
            max_frames=config.get("record_frames", 60)
        )
        self.particles.initialize()

        self.gui= ti.GUI("Implicit MPM", res=800)

    def solve(self):
        if self.implicit:
            return self.solver.solve()
        else:
            self.solve_explicit()
            self.grid.set_boundary_v()
            return 0

    @ti.kernel
    def build_neighbor_list(self):
        for p in range(self.particles.n_particles):
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            fx = self.particles.x[p] * self.grid.inv_dx - base.cast(float)
            w = [0.5*(1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5*(fx - 0.5)**2]
            
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                weight = 1.0
                for d in ti.static(range(self.grid.dim)):
                    weight *= w[offset[d]][d]
                dpos = (offset - fx) * self.grid.inv_dx
                self.particles.wip[p, offset] = weight
                self.particles.dwip[p, offset] = weight * dpos

    def pre_p2g(self):
        self.grid.clear()
        self.particles.build_neighbor_list()
        
    def step(self):
        for _ in range(self.max_iter):
            self.pre_p2g()
            self.p2g()

            if self.implicit:
                self.solver.save_previous_velocity()

            self.grid.apply_boundary_conditions()
            
            self.solve()
            
            self.g2p()
            self.particles.advect(self.dt)

    @ti.kernel
    def p2g(self):
        for p in range(self.particles.n_particles):
            base = (self.particles.x[p] * self.grid.inv_dx - 0.5).cast(int)
            fx = self.particles.x[p] * self.grid.inv_dx - base.cast(float)
            
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = (base + offset) % self.grid.size
                dpos = (offset - fx) * self.grid.dx
                self.grid.m[grid_idx] += self.particles.wip[p,offset] * self.particles.p_mass
                self.grid.v[grid_idx] += self.particles.wip[p,offset] * self.particles.p_mass * (self.particles.v[p] + self.particles.C[p] @ dpos)
                if self.particles.is_boundary_particle[p]:
                    self.grid.is_particle_boundary_grid[grid_idx] = 1

        
        for i, j in ti.ndrange(self.grid.size, self.grid.size):
            if self.grid.m[i,j] > 1e-10:
                self.grid.v[i,j] /= self.grid.m[i,j]

    @ti.kernel
    def g2p(self):
        for p in self.particles.x:
            Xp = self.particles.x[p] / self.grid.dx
            base = (Xp - 0.5).cast(int)

            new_v = ti.Vector.zero(self.float_type, 2)
            new_C = ti.Matrix.zero(self.float_type, 2, 2)
            
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = base + offset
                g_v = self.grid.v[grid_idx]
                new_v += g_v*self.particles.wip[p, offset]
                new_C += 4*  g_v.outer_product(self.particles.dwip[p, offset])
                
            self.particles.v[p] = new_v
            self.particles.F[p] = (ti.Matrix.identity(self.float_type, self.grid.dim) + self.dt * self.particles.C[p]) @ self.particles.F[p]
            self.particles.C[p] = new_C


    @ti.kernel
    def solve_explicit(self):
        for p in range(self.particles.n_particles):
            Xp = self.particles.x[p] / self.grid.dx
            base = (Xp - 0.5).cast(int)

            U, sig, V = ti.svd(self.particles.F[p])
            if sig.determinant() < 0:
                sig[1,1] = -sig[1,1]
            self.particles.F[p] = U @ sig @ V.transpose()
            
            J = self.particles.F[p].determinant()
            logJ = ti.log(J)
            cauchy = self.mu * (self.particles.F[p] @ self.particles.F[p].transpose()) + ti.Matrix.identity(self.float_type, 2) * (self.lam * logJ - self.mu)
            stress = -(self.particles.p_vol ) * cauchy

            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                grid_idx = base + offset
                self.grid.v[grid_idx] +=4* self.dt * stress @ self.particles.dwip[p,offset] / self.grid.m[grid_idx] 

    def render(self):
        self.gui.circles(self.particles.x.to_numpy(), radius=1.5, color=0x66CCFF)
        self.gui.show()

        if self.recorder is None:
            return

        print("Recording frame: ", len(self.recorder.frame_data) + 1)
        self.recorder.capture(
            self.particles.x.to_numpy(),
            self.particles.is_boundary_particle.to_numpy().astype(np.uint32)
        )

# 使用示例
if __name__ == "__main__":

    cfg=Config("config/config.json")
    float_type=ti.f32 if cfg.get("float_type", "f32") == "f32" else ti.f64        
    arch=cfg.get("arch", "cpu")
    if arch == "cuda":
        arch = ti.cuda
    elif arch == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, default_fp=float_type, device_memory_GB=20)
    mpm = ImplicitMPM(cfg)
    
    while mpm.gui.running:
        mpm.step()
        
        mpm.render()

        # 自动停止条件
        if len(mpm.recorder.frame_data) >= mpm.recorder.max_frames:
            break

    if mpm.recorder is None:
        exit()
    print("Playback finished.")
    # 播放录制动画
    mpm.recorder.play(loop=True, fps=60)