import sys
import numpy as np
import random

sys.path.insert(1, '../lib')
from numpy.linalg import svd

def rand_orthonormal(D, N):
    U, S, V = svd(np.random.randn(D, N), full_matrices=False)
    if D > N:
        Y = U
    if D <= N:
        Y = V
    return Y
