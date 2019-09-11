#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 26 Apr 19
"""

import numpy as np
import numpy.random as nr
import cvxpy as cp

def train_val_split(n, p):
    """
    Returns indices for a training set and validation set.
    
    inputs:
    n (integer) - number of samples; must be positive
    
    p (float) - fraction of samples for validation; must be between 0 and 1
    """
    
    if not isinstance(n, int) or n < 1:
        raise TypeError('Number of samples must be a positive integer.')
    if not isinstance(p, float) or p < 0 or p > 1:
        raise TypeError('Validation percentage must be between 0 and 1.')
    
    val_idx = nr.choice(n, int(n*p), replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    return list(train_idx), list(val_idx)

def gram(X, ip):
    """
    Computes gram matrix G_ij = ip(X_i, X_j).
    
    inputs:
    elements (X) - list of elements that can be acted on by ip
    inner product (ip) - function with two arguments ip(x1, x_2) that returns a scalar
    
    outputs:
    gram matrix (G) - G_ij = ip(x[i], x[j])
    """
    n = len(X)
    G = np.zeros([n, n])
    triu_index = np.triu_indices(n)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        G[i, j] = ip(X[i], X[j])
    tril_index = np.tril_indices(n, k=-1)
    G[tril_index] = G.T[tril_index]
    return G

def inner_prod(a, b):
    """
    Implements a Frobenius inner product which respects depthwise stacks of
    matrices. It will broadcast if a or b has depth.
    
    inputs:
    matrix (a) - [r x] n x m tensor
    matrix (b) - [r x] n x m tensor
    
    outputs:
    inner product (<a, b>) - [r tensor] scalar
    """
    if len(a.shape) == 3 or len(b.shape) == 3:
        return np.sum(np.multiply(a, b), axis=(1,2))
    else:
        return np.sum(np.multiply(a, b))

def ar_toep_op(x, model_ord):
    """
    Constructs the autoregressive toeplitz operator that appears in the 
    maximum likelihood estimator for the autoregessive coefficients by 
    conditioning on the first p observations, ie 
    p(x_n,...,x_p+1|x_p,...,x_1;A) = (2(n-p))**(-1) \|Y - X A\|_F**2, where
    Y and A are matrices of stacked observations and coefficients respectively.
    
    inputs:
    ar process observation (x) - n x d tensor
    model_ord (p) - scalar
    
    outputs:
    ar_toep - (n-p) x (p*d) tensor
    x_trunc - (n-p) x d tensor
    """
    sample_len, signal_dim = np.shape(x)
    ar_toep = np.zeros([sample_len-model_ord, model_ord*signal_dim])
    # Construct autoregressive Toeplitz operator; reverse order of 
    # observations to achieve convolution effect
    ar_toep[0, :] = np.ndarray.flatten(x[model_ord-1::-1, :])
    for t in np.arange(1, sample_len-model_ord):
        ar_toep[t, :] = np.ndarray.flatten(x[t+model_ord-1:t-1:-1, :])
    return ar_toep, x[model_ord:, :]

def stack_ar_coef(A):
    """
    Stack autoregressive coefficients (A[s])_s as [ A[1] ... A[p] ]^T.
    
    inputs:
    ar coef (A) - p x d x d tensor
    
    outputs:
    ar_coef (A) - (p*d) x d tensor
    """
    model_ord, signal_dim, _  = A.shape
    return np.reshape(np.moveaxis(A, 1, 2), [model_ord*signal_dim, signal_dim])

def unstack_ar_coef(A):
    """
    Unstack autoregressive coefficients (A[s])_s to [[A[1]], ..., [A[p]]].
    
    inputs:
    ar coef (A) - (p*d) x d tensor
    
    outputs:
    ar coef (A) - p x d x d tensor
    """
    
    m, n = A.shape
    model_ord = int(m/n)
    return np.stack(np.split(A.T, model_ord, axis=1), axis=0)

def dict_distance(A, B, p=2):
    n = len(A)
    m = len(B)
    if m != n:
        raise ValueError('Dimension mismatch')
    D = np.zeros([n, n])
    triu_index = np.triu_indices(n)
    for (i, j) in zip(triu_index[0], triu_index[1]):
            D[i, j] = np.minimum(np.sum(np.power(A[i]-B[j], p))**(1/p), 
                                np.sum(np.power(A[i]+B[j], p))**(1/p))
    tril_index = np.tril_indices(n, k=-1)
    D[tril_index] = D.T[tril_index]
    W = cp.Variable(shape=(n, n))
    obj = cp.Minimize(cp.sum(cp.multiply(D, W)))
    con = [W >= 0, cp.sum(W, axis=0) == 1, cp.sum(W, axis=1) == 1]
    prob = cp.Problem(obj, con)
    prob.solve()
    return prob.value, D[W.value>1e-3], W.value