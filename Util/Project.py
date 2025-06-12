import numpy as np


def project(particles_np):
    phi, theta = np.radians(28), np.radians(32)
    particles_np = particles_np - 0.5
    x, y, z = particles_np[:,0], particles_np[:,1], particles_np[:,2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    projected = np.stack([u, v], axis=1) + 0.5
    return projected