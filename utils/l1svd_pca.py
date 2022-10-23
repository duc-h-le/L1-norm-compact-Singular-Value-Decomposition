import sys
sys.path.insert(1,'../lib')
from numpy.linalg import svd
from utils import *

def l1svd_pca(X, rank_r, Qorth):
    D = X.shape[0]
    N = X.shape[1]
    U, S, V = svd(Qorth@Qorth.T@X, full_matrices = False)
    Xlow = Qorth@Qorth.T@X
    Udiag = U[:,range(rank_r)]
    Sdiag = S[range(rank_r)]
    Vtdiag = V[range(rank_r),:].T
    return Udiag, Sdiag, Vtdiag, Xlow