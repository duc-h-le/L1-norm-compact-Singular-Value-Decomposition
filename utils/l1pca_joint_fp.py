"""
This code implements the suboptimal fixed-point alternating joint algorithm for L1-PCA in
Nie, Feiping, Heng Huang, Chris Ding, Dijun Luo, and Hua Wang. "Robust principal component analysis with non-greedy ℓ1-norm maximization." In Twenty-Second International Joint Conference on Artificial Intelligence. 2011.
"""

import sys
import numpy as np

sys.path.insert(1, '../lib')
from numpy.linalg import svd
from utils import *

# def rand_orthonormal(D, N):
#     U, _, V = svd(np.random.randn(D, N), full_matrices=False)
#     Y = U @ V
#     return Y

def l1pca_joint_fp(X, rank_r, iter):
    D = X.shape[0]
    N = X.shape[1]

    max_metric = 0

    for ii in range(iter):
        U = rand_orthonormal(D, rank_r)
        error = np.Inf
        metric = sum(sum(abs(U.T @ X)))
        iter_count = 1
        while True:
            B = np.sign(X.T @ U)
            U1, _, V1 = svd(X @ B, full_matrices=False)
            U = U1 @ V1
            if np.abs(metric - sum(sum(abs(U.T @ X)))) < metric / 10000 or iter_count > 500:
                if metric > max_metric:
                    Uf = U
                    max_metric = metric
                    idx = ii
                break
            else:
                metric = sum(sum(abs(U.T @ X)))
                iter_count = iter_count + 1
    return Uf
