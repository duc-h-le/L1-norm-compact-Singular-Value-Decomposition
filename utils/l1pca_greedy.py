"""
This code implements the suboptimal greedy L1-PCA  algorithm in
Kwak, Nojun. "Principal component analysis based on L1-norm maximization." IEEE transactions on pattern analysis and machine intelligence 30, no. 9 (2008): 1672-1680.
"""
import numpy as np
from numpy.linalg import svd
from utils import *

def mysign(X):
    B = np.sign(X)
    D = B.shape[0]
    N = B.shape[1]
    for i in range(D):
        for j in range(N):
            if B[i, j] == 0:
                B[i, j] = 1
    return B

def L1pca_1(X, num_init):
    D = X.shape[0]
    N = X.shape[1]
    metopt = -np.Inf
    for i in range(num_init):
        q = np.random.rand(D, 1)
        q = q/np.linalg.norm(q)
        while True:
            b = mysign(q.T @ X)
            q = (X @ b.T) / np.linalg.norm(X @ b.T)
            bnew = mysign(q.T @ X)
            if np.linalg.norm(bnew @ b.T) == N:
                break
        if l1norm(q.T@X) > metopt:
            qopt = q
            metopt = l1norm(q.T@X)
    return np.squeeze(qopt)

def l1pca_greedy(X, K):
    D = X.shape[0]
    N = X.shape[1]

    Q = np.zeros([D, K])
    # 1st PC
    Q[:, 0] = L1pca_1(X, 50)
    proj_null = X
    # Next PCs
    for i in range(1, K):
        proj_null = proj_null - Q[:, i - 1, None] @ Q[:, i - 1, None].T @ X
        Q[:, i] = L1pca_1(proj_null,10)
    return Q