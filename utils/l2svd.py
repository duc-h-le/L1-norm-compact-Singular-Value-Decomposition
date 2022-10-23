import numpy as np
from numpy.linalg import svd
def l2svd(X, K):
    D = X.shape[0]
    N = X.shape[1]
    U, S, V = svd(X, full_matrices=False)
    UK = U[:, range(K)]
    SK = S[range(K)]
    VK = V[range(K), :].T
    XK = UK @ np.diag(SK) @ VK.T
    return UK, SK, VK, XK