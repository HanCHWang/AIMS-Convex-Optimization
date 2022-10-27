# Solution for ALGORITHMS, Q2
# Author: Han Wang
# Date: 26/10/2022
# Can we use primal-dual method for this problem???????
import cvxpy
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import cvxpy as cp

# Generate random signal
n = 1000
n_nnz = 10
hat_x = np.zeros(n)
nnz_idx = np.random.randint(0,n,n_nnz)
hat_x[nnz_idx] = np.random.randn(n_nnz)

# Samples
n_samples = 100
A = np.random.randn(n_samples,n)
y = A @ hat_x

# Constants for projection gradient descent
gamma = 0.1 # gradient descent step size
tolerance = 1e-3 # terminate condition
max_iterations = 1000 # maximal iterations
x = np.random.rand(n)

# First define a function to calculate the Jacobian of \phi(Ax-y)
def Jacobian(A,y,x):
    tmp = np.zeros(n_nnz)
    for i in range(n_samples):
        if np.linalg.norm(A[i,:]*x-y[i],1) <= 1:
            tmp = tmp + 2*(A[i,:]*x-y[i])
        elif A[i,:]*x-y[i] >= 0:
            tmp = tmp + 2*A[i,:]
        else:
            tmp = tmp + -2*A[i,:]
    return tmp 
        

# Lets start iterations!
for i in range(max_iterations):
    gradient = Jacobian(A,y,x)
    x = x-gamma*gradient # gradient descent

print(x)