import taichi as ti
from simulators.implicit_mpm import *

@ti.data_oriented
class DomainManager:
    """管理两个MPM域的类，负责域的初始化和基本操作"""
    
    def __init__(self, main_config: Config):
        # 提取子域配置并创建独立配置对象
        domain1_config = Config(data=main_config.get("Domain1", {}))
        domain2_config = Config(data=main_config.get("Domain2", {}))
        common_particles_config = main_config.get("Common_Particles", None)

        # 如果有公共粒子配置，则创建公共粒子实例
        common_particles = Particles(common_particles_config) if common_particles_config else None
        
        # 初始化两个子域MPM实例
        self.domain1 = ImplicitMPM(domain1_config, common_particles)
        self.domain2 = ImplicitMPM(domain2_config, common_particles)

        # 确定大小时间步长域
        self._setup_time_domains()
        
    def _setup_time_domains(self):
        """设置大小时间步长域"""
        if self.domain1.dt >= self.domain2.dt:
            self.big_time_domain = self.domain1
            self.small_time_domain = self.domain2
        else:
            self.big_time_domain = self.domain2
            self.small_time_domain = self.domain1

        # 验证时间步长比例
        ratio = self.big_time_domain.dt / self.small_time_domain.dt
        import math
        if not math.isclose(ratio, round(ratio), rel_tol=1e-8):
            print("big time domain dt:", self.big_time_domain.dt,
                  "small time domain dt:", self.small_time_domain.dt,
                  "ratio:", ratio)
            raise ValueError("Big time domain dt must be a multiple of small time domain dt.")
    
    def get_timestep_ratio(self):
        """获取时间步长比例"""
        return int(self.big_time_domain.dt // self.small_time_domain.dt)
    
    def pre_step(self):
        """执行步进前的准备工作"""
        self.big_time_domain.pre_p2g()
        self.small_time_domain.pre_p2g()
        
        self.big_time_domain.p2g()
        self.small_time_domain.p2g()
        
        self.big_time_domain.post_p2g()
        self.small_time_domain.post_p2g()

    def finalize_step(self, do_small_advect):
        """完成步进后的工作"""
        self.big_time_domain.g2p(self.big_time_domain.dt)
        self.big_time_domain.particles.advect(self.big_time_domain.dt)

        if not do_small_advect:
            self.small_time_domain.g2p(self.big_time_domain.dt)
            self.small_time_domain.particles.advect(self.big_time_domain.dt)