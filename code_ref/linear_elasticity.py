import numpy as np

def linear_elasticity_energy(F, mu, lam):
    I = np.eye(F.shape[0])
    eps = 0.5 * (F + F.T) - I
    energy = mu * np.sum(eps * eps) + 0.5 * lam * (np.trace(eps) ** 2)
    return energy

def linear_elasticity_gradient(F, mu, lam):
    I = np.eye(F.shape[0])
    eps = 0.5 * (F + F.T) - I
    grad = mu * (F + F.T - 2 * I) + lam * np.trace(eps) * I
    return grad

def linear_elasticity_hessian(F, mu, lam):
    # Hessian is a 4th-order tensor H[i,j,k,l]
    dim = F.shape[0]
    H = np.zeros((dim, dim, dim, dim))
    I = np.eye(dim)
    
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    # mu term
                    H[i,j,k,l] += mu * (int(i==k and j==l) + int(i==l and j==k))
                    # lambda term
                    H[i,j,k,l] += lam * 0.5 * (int(i==j) * int(k==l))
    return H
