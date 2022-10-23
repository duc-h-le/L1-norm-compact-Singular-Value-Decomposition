"""
This code implements the suboptimal bit-flip joint algorithm for L1-PCA in
Markopoulos, Panos P., Sandipan Kundu, Shubham Chamadia, and Dimitris A. Pados. "Efficient L1-norm principal-component analysis via bit flipping." IEEE Transactions on Signal Processing 65, no. 16 (2017): 4252-4264.
"""

import sys
import time
import numpy as np
sys.path.insert(1,'../lib')
from numpy.linalg import svd

def l1pca_bitflip(X, K, num_init, print_flag):
    # Parameters
    toler =10e-8;

    # Get the dimentions of the matrix.
    dataset_matrix_size = X.shape	
    D = dataset_matrix_size[0]	# Row dimension.
    N = dataset_matrix_size[1]	# Column dimension.

    # Initialize the matrix with the SVD.
    dummy, S_x, V_x = svd(X , full_matrices = False)	# Hint: The singular values are in vector form.
    if D < N:
        V_x = V_x.T
    
    Y = np.diag(S_x)@V_x.T # X_t is Y = S*V'
    # Initialize the required matrices and vectors.
    Bprop = np.ones((N,K),dtype=float)
    nucnormmax = 0
    iterations = np.zeros((1,num_init),dtype=float)
    #vmaxlist = []
    # For each initialization do.
    for ll in range(0, num_init):

        start_time = time.time()	# Start measuring execution time.

        z = X.T @ np.random.randn(D,1)	# Random initialized vector.
        if ll<1:	# In the first initialization, initialize the B matrix to sign of the product of the first 
            # right singular vector of the input matrix with an all-ones matrix.
            z = np.zeros((N,1),dtype=float)
            z = V_x[:,0]
            z = z.reshape(N,1)
        v = z@np.ones((1,K), dtype=float)
        B = np.sign(v)	# Get a binary vector containing the signs of the elements of v.
        #print('Bint', B)
        iterations = []
        iter_ = 0
        
        while True:
            iter_ = iter_ + 1

            flag = False

            # Calculate all the possible binary vectors and all posible bit flips.
            L = list(range(N*K))
            a = np.zeros((N,K))
            
            nucnorm = np.linalg.norm(Y@B, 'nuc')
            
            for x in L:
                l = x//N
                m = x-N*l
                elK = np.zeros(K)
                elK[l] = 1
                a[m,l] = np.linalg.norm(Y@B - 2*B[m,l]*(Y[:,m,None]@ [elK]), 'nuc')
            nucnorm_flip = np.max(a)
            n,k = np.unravel_index(np.nanargmax(a, axis=None), a.shape)
            
            if nucnorm_flip > nucnorm:
                B[n,k] = -B[n,k]
                #print('B_flipped',B)
                L.remove(k*N+n) #sus
                #print('L', L)
            elif nucnorm_flip <= nucnorm + toler and len(L)<N*K:
                #print('Bunflipped',B)
                L = list(range(N*K))
            else:
                break
        U, dummy, V = svd(X@B, full_matrices=False)
        Utemp = U[:,0:K]
        Vtemp = V[:,0:K]
        Q = Utemp@Vtemp.T
        
        nucnorm = sum(sum(abs(Q.T@X)))
        
        # Find the maximum nuclear norm across all initializations.
        if nucnorm > nucnormmax:
            nucnormmax = nucnorm
            Bprop = B
            #print('Bprop',Bprop)
            Qprop = Q
            vmax = nucnorm
        iterations = np.append(iterations, iter_)

    end_time = time.time()	# End of execution timestamp.
    timelapse = (end_time - start_time)	# Calculate the time elapsed.

    convergence_iter = np.mean(iterations, dtype=float) # Calculate the mean iterations per initialization.
    
    if print_flag:
        print("------------------------------------------")
        print("Avg. iterations/initialization: ", (convergence_iter))
        print("Time elapsed (sec):", (timelapse))
        print("Metric value:", vmax)
        print("------------------------------------------")

    return Qprop, Bprop, vmax