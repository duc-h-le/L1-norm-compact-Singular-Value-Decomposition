"""
Author: Duc H. Le (Rochester Institute of Technology, email: dhl3772@rit.edu)
Description: This code is an example of how the L1-cSVD is run. You can generated data corrupted with outliers.
Then, the singular values (SVs) estimated from corrupted data using SVD, L1-PCA, L1-error minimization, RPCA and the proposed L1-cSVD are displayed and compared to clean SVs.
Max number of iteration, convergence ratio and number of initializations are tunable.
Set verbose to True to see algorithmic performance of L1-cSVD.
"""
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from l1csvd import l1csvd
from utils import *

def main():
    np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

    # Tunable parameters to generate clean and corrupted data
    OSR = 5 # Outlier-to-signal ratio (dB)
    SNR = 10 # Sinal-to-noise ratio (dB)
    D = 8 # number of dimensions
    N = 50 # number of data points
    K = 4 # number of rank to approximate to
    K_o = K # dimension of subspace of outlier
    P_o = 0.04 # ratio of data points corrupted

    # Generate random subspace of data
    U0, _, _ = svd(np.random.randn(D, K), full_matrices=False)
    # Generate random subspace of outlier
    U_o, _, _ = svd(np.random.randn(D, K_o), full_matrices=False)

    # Generate clean (Xc) and corrupted (X) data from the above subspaces
    Xc, X = X_generate_SV(D, N, K, U0, K_o, U_o, P_o, SNR, OSR, False)

    # Calculate clean (L2) singular values
    U2, S2, V2, Xlow = l2svd(Xc, K)
    # print('Clean data matrix: ', Xc)
    # print('Corrupted data matrix: ', X)
    print('Singular values of clean data (ground truth): \n', S2)
    print('--------------------------------------------------------')
    print('PCA (conventional L2-norm)')
    U, S, V, Xlow= l2svd(X, K)
    print('Estimated SVs from corrupted data:\n ', S)
    print('--------------------------------------------------------')
    print('L1-PCA')
    # Uncomment and comment to choose which L1-PCA algorithm to find robust PCs
    # U1 = l1pca_greedy(X, K)
    # U1 = l1pca_joint_fp(X, K, 10)
    U1, _, _ = l1pca_bitflip(X, K, 3, False)
    U, S, V, Xlow = l1svd_pca(X, K, U1)
    print('Estimated SVs from corrupted data:\n ', S)
    print('--------------------------------------------------------')
    print('L1-error minimization')
    U, S, V, Xlow = l1svd_error_minimization(X, K, 10, 0.001, 1)
    print('Estimated SVs from corrupted data:\n ', S)
    print('--------------------------------------------------------')
    print('RPCA')
    U, S, V, Xlow = rpca(X, K, 1/np.sqrt(N))
    print('Estimated SVs from corrupted data:\n ', S)
    print('--------------------------------------------------------')
    print('L1-cSVD (Proposed algorithm)')
    U, S, V, Xlow, idx_opt = l1csvd(X, K, max_iter=20, convg_ratio=0.0001, num_init=5, verbose=True)
    print('Estimated SVs from corrupted data using proposed L1-cSVD:\n ', S)
    print('--------------------------------------------------------')

    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Keyboardfloaterrupt:
        pass