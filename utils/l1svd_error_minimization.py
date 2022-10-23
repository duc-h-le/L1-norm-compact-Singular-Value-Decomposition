"""
This algorithm implements the L1-norm error minimization low-rank estimation algorithm in
Ke, Qifa, and Takeo Kanade. "Robust l1 norm factorization in the presence of outliers and missing data by alternative convex programming." In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), vol. 1, pp. 739-746. IEEE, 2005.
This code minimizes ||X-QR||_1 like in the cited paper, then take svd(QR) to extend to robust SVD.
"""

import sys
import numpy as np
from numpy.linalg import svd
import cvxpy as cp
sys.path.insert(1, '../lib')

from utils import *

def l1norm(X):
    return sum(sum(abs(X)))

def l1svd_error_minimization(X, K, max_iter, convg_ratio, num_init):
    D = X.shape[0]
    N = X.shape[1]

    metric_opt = np.Inf

    for ii in range(num_init):
        Q = np.random.randn(D, K)
        R = np.random.randn(N, K)
        metric_list = []
        metric = l1norm(X - Q@R.T)

        iter = 0

        while True:
            R = cp.Variable((N, K))
            cost = cp.sum(cp.abs(Q @ R.T - X))
            constraints = []
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(verbose=False)
            R = R.value

            Q = cp.Variable((D, K))
            cost = cp.sum(cp.abs(Q @ R.T - X))
            constraints = []
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(verbose=False)
            Q = Q.value

            if abs(metric - l1norm(X - Q@R.T)) < convg_ratio * metric or iter > max_iter - 1:
                metric = l1norm(X - Q@R.T)
                metric_list = np.append(metric_list, metric)
                break
            else:
                iter = iter + 1
                metric = l1norm(X - Q @ R.T)
                metric_list = np.append(metric_list, metric)
                # print(iter)
        # print(metric_list)

        if metric < metric_opt:
            metric_opt = metric
            Qf = np.copy(Q)
            Rf = np.copy(R)
            idx_opt = ii

    # print('idx_opt: ', idx_opt)
    U, S, V = svd(Qf@Rf.T, full_matrices = False)

    U = U[:,range(K)]
    S = S[range(K)]
    V = V[range(K),:].T

    idx_S = np.array(np.argsort(-1 * abs(S)))
    Sf = S[idx_S]
    Uf = U[:, idx_S]
    Vf = V[:, idx_S]
    # Correct the sign of singular values and vectors
    sgn_S = np.sign(Sf)
    Uf = Uf @ np.diag(sgn_S)
    Sf = Sf @ np.diag(sgn_S)
    Xlow = Uf @ np.diag(Sf) @ Vf.T

    return Uf, Sf, Vf, Xlow