import sys
import numpy as np
import random
from numpy.linalg import norm
import math

sys.path.insert(1, '../lib')
from numpy.linalg import svd
from utils import *


def X_generate_SV(D, N, K, U0, K_o, U_o, P_o, SNR, OSR, center_flag):

    zmax = 2
    z = np.random.uniform(0,zmax,K)
    Smax = 10
    S0 = Smax * np.exp(-z)

    # print('S0: ', S0)

    V_noise = norm(S0)**2/(D*N*10**(SNR/10))
    V_outlier = 10 ** (OSR / 10) * norm(S0)**2 /(K_o*N*P_o)

    V0 = rand_orthonormal(N, K)
    X0 = U0 @ np.diag(S0)@V0.T  # Random rank-K matrix in subspace of U_0
    Noise =  np.sqrt(V_noise) * np.random.randn(D, N)
    X = X0 + Noise  # Add noise
    U2, S2, V2, X2 = l2svd(X, K)
    # print('S2 noisy: ', S2)
    X_o = np.copy(X)
    idx_list = range(N)
    for i in range(int(P_o * N)):
        # if random.uniform(0, 1) < P_o:  # Set probability P_o to fill in outlier
        j = random.choice(idx_list)
        # X_o[:, j] = X_o[:, j] + np.sqrt(V_outlier) * np.random.randn(1, K_o) @ U_o.T
        X_o[:, j] = np.sqrt(V_outlier) * np.random.randn(1, K_o) @ U_o.T
    # print('SNR: ',norm(X0)**2/norm(Noise)**2)
    # print('OSR: ', norm(X_o - X)**2/norm(X)**2)
    # Center data
    if center_flag:
        for i in range(D):
            X_o[i, :] = X_o[i, :] - np.mean(X_o[i, :])
            X0[i, :] = X0[i, :] - np.mean(X0[i, :])
    return X, X_o
