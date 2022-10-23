"""
This code implements the Robust Principal Component Analysis algorithm in
Candès, Emmanuel J., Xiaodong Li, Yi Ma, and John Wright. "Robust principal component analysis?." Journal of the ACM (JACM) 58, no. 3 (2011): 1-37.
The proposed optimization problem is solved by cvxpy.
"""

import cvxpy as cp
import numpy as np
from numpy.linalg import svd
def rpca(X, K, lamb):
    D = X.shape[0]
    N = X.shape[1]
    L = cp.Variable((D, N))
    cost = cp.norm(L, "nuc") + lamb * cp.sum(cp.abs(X - L))
    constraints = []
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(verbose=False)
    L = L.value
    U, S, V = svd(L, full_matrices=False)
    UK = U[:, range(K)]
    SK = S[range(K)]
    VK = V[range(K), :].T
    XK = UK @ np.diag(SK) @ VK.T
    return UK, SK, VK, XK