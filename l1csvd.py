"""
Author: Duc H. Le (Rochester Institute of Technology, email: dhl3772@rit.edu)
This is the source code implementing the L1-cSVD algorithm. Refer to example.py to run L1-cSVD.

Input:
    X: Data matrix (size DxN).
    K: Number of rank to approximate to.
    max_iter: Maximum number of iterations to prevent algorithm being stuck.
    convg_ratio: Relative change of optimization metric ||U.T@X-S@V.T||_1 when convergence is said to happen.
    num_init: Number of random initializations for V
Output:
    Uf: L1-Left singular vectors (orthonormal DxK)
    Sf: L1-Singular values (Kx1 vector)
    Vf: L1-Right singular vectors (orthonormal NxK)
    Xlow = Uf@diag(Sf)@Vf.T: robust low-rank approximation of X.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from utils import *
import time

def l1norm(X):
    return sum(sum(abs(X)))

def l1csvd(X, K, max_iter, convg_ratio, num_init, verbose):
    # Get matrix dimensions
    t0 = time.time()
    D = X.shape[0]
    N = X.shape[1]

    U = l1pca_greedy(X,D) # Use greedy algorithm to find L1-PCs (recommended). Default number of initializations = 10
    t1 = time.time()

    # Start L1-reorthogonalization of U.T@X
    A = X.T @ U
    metric_opt = np.Inf
    idx_opt = 0 # index of initialization with optimal result
    iter_list = [] # Number of iterations per initialization
    metricf_list = [] # Final metric per initialization

    for ii in range(num_init): # go through multiple initializations
        # Initialize S (singular values) and V (right singular vectors)
        V = rand_orthonormal(N, D)
        S = np.zeros((D))
        metric = l1norm(X.T @ U - V @ np.diag(S)) # L1 reorthogonalization optimization metric
        metric_list = [metric] # metric at every iteration
        iter = 0
        while True:
            for i in range(D):
                S[i] = nearest_L1_vector(A[:, i], V[:, i]) # Fix V, find S
            V = procrustes(A @ np.linalg.inv(np.diag(S)) ) # Fix S, find V
            # Check for convergence
            if abs(metric - l1norm(X.T @ U - V @ np.diag(S))) < convg_ratio * metric or iter > max_iter - 1:
                metric = l1norm(X.T @ U - V @ np.diag(S))
                metric_list = np.append(metric_list, metric)
                break
            else: # end algorithm after max number of iterations to prevent stuck
                metric = l1norm(X.T @ U - V @ np.diag(S))
                iter = iter + 1
                metric_list = np.append(metric_list, metric)
        metricf_list = np.append(metricf_list, metric)
        iter_list=np.append(iter_list, iter)

        # Plot metric by iteration
        if verbose:
            plt.figure(0)
            plt.plot(metric_list/l1norm(U.T@X), marker = 'o', linestyle = '-')
            plt.grid(visible=True)
            plt.xlabel('Iteration')
            plt.ylabel('$||\mathbf{U}^T\mathbf{X - \Sigma V}^T||_{1,1}/||\mathbf{U}^T\mathbf{X}||_{1,1}$')
        # Check if current S and V are more optimal
        if metric < metric_opt:
            metric_opt = metric
            Uf = np.copy(U)
            Sf = np.copy(S)
            Vf = np.copy(V)
            idx_opt = ii # index of initialization with optimal result

    # Sort U, S, V in descending SVs order
    idx_S = np.array(np.argsort(-1 * abs(Sf)))
    Uf, Sf, Vf = Uf[:, idx_S], Sf[idx_S], Vf[:, idx_S]
    # Correct the sign of singular values and vectors
    sgn_S = np.sign(Sf)
    Uf = Uf @ np.diag(sgn_S)
    Sf = Sf @ np.diag(sgn_S)
    Uf, Sf, Vf = Uf[:, range(K)], Sf[range(K)], Vf[:, range(K)] # Truncate to rank K
    Xlow = Uf @ np.diag(Sf) @ Vf.T # K-rank matrix approximated from L1-cSVD
    t2 = time.time()
    if verbose:
        print('~~~ Performance of L1-cSVD ~~~')
        print('Total runtime: ', t2-t0, 's')
        print('Time to run L1-PCA: ', t1-t0, 's')
        print('Time to run L1-reorthogonalization in L1-cSVD: ', t2-t1, 's')
        print('Average number of iterations: ', np.mean(iter_list))
        print('Max optimization metric: ', np.max(metricf_list))
        print('Min (optimal) optimization metric: ', np.min(metricf_list))
        print('~~~~~~~~~~')
    return Uf, Sf, Vf, Xlow, idx_opt