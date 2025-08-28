import taichi as ti

@ti.data_oriented
class ParticleStateManager:
    """管理粒子状态保存和恢复的工具类"""
    
    def __init__(self, particles):
        self.particles = particles
        
        # 临时数组用于保存粒子数据
        self.temp_particles_x = ti.Vector.field(
            particles.dim, particles.float_type, particles.n_particles
        )
        self.temp_particles_v = ti.Vector.field(
            particles.dim, particles.float_type, particles.n_particles
        )
        self.temp_particles_f = ti.Matrix.field(
            particles.dim, particles.dim, particles.float_type, particles.n_particles
        )
        self.temp_particles_c = ti.Matrix.field(
            particles.dim, particles.dim, particles.float_type, particles.n_particles
        )
    
    def save_particle_state(self):
        """保存粒子数据到临时数组"""
        self.temp_particles_x.copy_from(self.particles.x)
        self.temp_particles_v.copy_from(self.particles.v)
        self.temp_particles_f.copy_from(self.particles.F)
        self.temp_particles_c.copy_from(self.particles.C)
    
    def restore_particle_state(self):
        """从临时数组恢复粒子数据"""
        self.particles.x.copy_from(self.temp_particles_x)
        self.particles.v.copy_from(self.temp_particles_v)
        self.particles.F.copy_from(self.temp_particles_f)
        self.particles.C.copy_from(self.temp_particles_c)