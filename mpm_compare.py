import matplotlib.pyplot as plt

from implicit_mpm import *

from implicit_mpm_schwarz import *

@ti.data_oriented
class MPM_Compare:
    def __init__(self,config1:Config,config2:Config):
        
        self.single_domain_mpm=ImplicitMPM(config1)
        self.schwarz_mpm=MPM_Schwarz(config2)

        self.gird_v_diff = []

    def step(self):
        self.single_domain_mpm.step()
        self.schwarz_mpm.step()

    def render(self):
        self.single_domain_mpm.render()
        self.schwarz_mpm.render()

    @ti.kernel
    def compute_grid_v_diff(self) -> ti.f32:

        cnt=0
        residual = 0.0

        for I in ti.grouped(self.single_domain_mpm.grid.v):
            if self.single_domain_mpm.grid.m[I] > 0:
                ti.atomic_add(cnt, 1)
                m1=self.schwarz_mpm.Domain1.grid.m[I] > 0.0
                m2=self.schwarz_mpm.Domain2.grid.m[I] > 0.0
                if m1 and m2:
                    avg_v = (self.schwarz_mpm.Domain1.grid.v[I] + self.schwarz_mpm.Domain2.grid.v[I]) * 0.5
                    ti.atomic_add(residual, (self.single_domain_mpm.grid.v[I] - avg_v).norm())
                elif m1:
                    ti.atomic_add(residual, (self.single_domain_mpm.grid.v[I] - self.schwarz_mpm.Domain1.grid.v[I]).norm())
                elif m2:
                    ti.atomic_add(residual, (self.single_domain_mpm.grid.v[I] - self.schwarz_mpm.Domain2.grid.v[I]).norm())

        residual /= cnt
        print(f"Grid Velocity Difference: {residual}")
        return residual

if __name__ == "__main__":

    cfg1 = Config("config/config.json")
    cfg2 = Config(path="config/schwarz.json")

    float_type=ti.f32 if cfg1.get("float_type", "f32") == "f32" else ti.f64
    arch=cfg1.get("arch", "cpu")
    if arch == "cuda":
        arch = ti.cuda
    elif arch == "vulkan":
        arch = ti.vulkan
    else:
        arch = ti.cpu

    ti.init(arch=arch, default_fp=float_type, device_memory_GB=20)

    mpm_compare = MPM_Compare(cfg1,cfg2)

    max_frames = 60

    while mpm_compare.single_domain_mpm.gui.running:
        mpm_compare.step()
        mpm_compare.gird_v_diff.append(mpm_compare.compute_grid_v_diff())

        mpm_compare.render()

        # 自动停止条件
        if len(mpm_compare.single_domain_mpm.recorder.frame_data) > max_frames:
            break

    # 绘制网格速度差异随时间的变化
    plt.figure(figsize=(10, 6))
    plt.plot(mpm_compare.gird_v_diff, label='Grid Velocity Difference')
    plt.show()

    #绘制录制的帧
    mpm_compare.single_domain_mpm.recorder.play(loop=True, fps=60)
    mpm_compare.schwarz_mpm.recorder.play(loop=True, fps=60)