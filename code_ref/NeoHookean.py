import utils
import numpy as np
import math

def polar_svd(F):
    [U, s, VT] = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U[:, 1] = -U[:, 1]
        s[1] = -s[1]
    if np.linalg.det(VT) < 0:
        VT[1, :] = -VT[1, :]
        s[1] = -s[1]
    return [U, s, VT]

def dPsi_div_dsigma(s, mu, lam):
    ln_sigma_prod = math.log(s[0] * s[1])
    inv0 = 1.0 / s[0]
    dPsi_dsigma_0 = mu * (s[0] - inv0) + lam * inv0 * ln_sigma_prod
    inv1 = 1.0 / s[1]
    dPsi_dsigma_1 = mu * (s[1] - inv1) + lam * inv1 * ln_sigma_prod
    return [dPsi_dsigma_0, dPsi_dsigma_1]

def d2Psi_div_dsigma2(s, mu, lam):
    ln_sigma_prod = math.log(s[0] * s[1])
    inv2_0 = 1 / (s[0] * s[0])
    d2Psi_dsigma2_00 = mu * (1 + inv2_0) - lam * inv2_0 * (ln_sigma_prod - 1)
    inv2_1 = 1 / (s[1] * s[1])
    d2Psi_dsigma2_11 = mu * (1 + inv2_1) - lam * inv2_1 * (ln_sigma_prod - 1)
    d2Psi_dsigma2_01 = lam / (s[0] * s[1])
    return [[d2Psi_dsigma2_00, d2Psi_dsigma2_01], [d2Psi_dsigma2_01, d2Psi_dsigma2_11]]

def B_left_coef(s, mu, lam):
    sigma_prod = s[0] * s[1]
    return (mu + (mu - lam * math.log(sigma_prod)) / sigma_prod) / 2

def Psi(F, mu, lam):
    J = np.linalg.det(F)
    lnJ = math.log(J)
    return mu / 2 * (np.trace(np.transpose(F).dot(F)) - 2) - mu * lnJ + lam / 2 * lnJ * lnJ

def dPsi_div_dF(F, mu, lam):
    FinvT = np.transpose(np.linalg.inv(F))
    return mu * (F - FinvT) + lam * math.log(np.linalg.det(F)) * FinvT

def d2Psi_div_dF2(F, mu, lam):
    [U, sigma, VT] = polar_svd(F)

    Psi_sigma_sigma = utils.make_PSD(d2Psi_div_dsigma2(sigma, mu, lam))

    B_left = B_left_coef(sigma, mu, lam)
    Psi_sigma = dPsi_div_dsigma(sigma, mu, lam)
    B_right = (Psi_sigma[0] + Psi_sigma[1]) / (2 * max(sigma[0] + sigma[1], 1e-6))
    B = utils.make_PSD([[B_left + B_right, B_left - B_right], [B_left - B_right, B_left + B_right]])

    M = np.array([[0, 0, 0, 0]] * 4)
    M[0, 0] = Psi_sigma_sigma[0, 0]
    M[0, 3] = Psi_sigma_sigma[0, 1]
    M[1, 1] = B[0, 0]
    M[1, 2] = B[0, 1]
    M[2, 1] = B[1, 0]
    M[2, 2] = B[1, 1]
    M[3, 0] = Psi_sigma_sigma[1, 0]
    M[3, 3] = Psi_sigma_sigma[1, 1]

    dP_div_dF = np.array([[0, 0, 0, 0]] * 4)
    for j in range(0, 2):
        for i in range(0, 2):
            ij = j * 2 + i
            for s in range(0, 2):
                for r in range(0, 2):
                    rs = s * 2 + r
                    dP_div_dF[ij, rs] = M[0, 0] * U[i, 0] * VT[0, j] * U[r, 0] * VT[0, s] \
                        + M[0, 3] * U[i, 0] * VT[0, j] * U[r, 1] * VT[1, s] \
                        + M[1, 1] * U[i, 1] * VT[0, j] * U[r, 1] * VT[0, s] \
                        + M[1, 2] * U[i, 1] * VT[0, j] * U[r, 0] * VT[1, s] \
                        + M[2, 1] * U[i, 0] * VT[1, j] * U[r, 1] * VT[0, s] \
                        + M[2, 2] * U[i, 0] * VT[1, j] * U[r, 0] * VT[1, s] \
                        + M[3, 0] * U[i, 1] * VT[1, j] * U[r, 0] * VT[0, s] \
                        + M[3, 3] * U[i, 1] * VT[1, j] * U[r, 1] * VT[1, s]
    return dP_div_dF


def val(x, e, vol, IB, mu, lam):
    sum = 0.0
    for i in range(0, len(e)):
        F = deformation_grad(x, e[i], IB[i])
        sum += vol[i] * Psi(F, mu[i], lam[i])
    return sum

def grad(x, e, vol, IB, mu, lam):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(e)):
        F = deformation_grad(x, e[i], IB[i])
        P = vol[i] * dPsi_div_dF(F, mu[i], lam[i])
        g_local = dPsi_div_dx(P, IB[i])
        for j in range(0, 3):
            g[e[i][j]] += g_local[j]
    return g

def hess(x, e, vol, IB, mu, lam):
    IJV = [[0] * (len(e) * 36), [0] * (len(e) * 36), np.array([0.0] * (len(e) * 36))]
    for i in range(0, len(e)):
        F = deformation_grad(x, e[i], IB[i])
        dP_div_dF = vol[i] * d2Psi_div_dF2(F, mu[i], lam[i])
        local_hess = d2Psi_div_dx2(dP_div_dF, IB[i])
        for xI in range(0, 3):
            for xJ in range(0, 3):
                for dI in range(0, 2):
                    for dJ in range(0, 2):
                        ind = i * 36 + (xI * 3 + xJ) * 4 + dI * 2 + dJ
                        IJV[0][ind] = e[i][xI] * 2 + dI
                        IJV[1][ind] = e[i][xJ] * 2 + dJ
                        IJV[2][ind] = local_hess[xI * 2 + dI, xJ * 2 + dJ]
    return IJV
