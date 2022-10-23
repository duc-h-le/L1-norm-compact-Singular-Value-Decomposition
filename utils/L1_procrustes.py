for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Gradient descent approach to solve L1-procrustes problem

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from numpy.linalg import svd

sys.path.insert(1, '../lib')

from utils import *

def L1_procrustes (A, S, V_int, epsn, gamma, max_iter, convg_ratio):
    N = A.shape[0]
    K = A.shape[1]
    V = V_int
    grad = np.copy(V)
    iter = 0
    metric = sum(sum(abs(A - V@np.diag(S))))
    metric_list = [metric]
    while True:
        for i in range(N):
            for j in range(K):
                delta = np.zeros((N,K))
                delta[i,j] = epsn
                grad[i,j] = (sum(sum(abs(A - (V+delta)@np.diag(S)))) - (sum(sum(abs(A - V@np.diag(S))))))/epsn
                V = V - gamma*grad
        V = procrustes(V)
        if iter > max_iter or abs(sum(sum(abs(A - V@np.diag(S)))) - metric) < convg_ratio*metric:
            break
        else:
            metric = sum(sum(abs(A - V@np.diag(S))))
            metric_list = np.append(metric_list, metric)
            iter = iter + 1
    # print('met_list', metric_list)
    return V