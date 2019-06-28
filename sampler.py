#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 29 Apr 19
"""

import numpy as np
import numpy.random as nr
import scipy.linalg as sl
from utility import gram, inner_prod, ar_toep_op, stack_ar_coef
    
def check_almm_condition(x, D, C):
    """Computes and returns the condition number of the linear operators 
    that participate in the gradient-based method, i.e.,
    G_ij = and
    H_ij[k] = <D_i, (X_k^TX_k) D_j>_F 
    
    inputs:
    ar process (x) - n x m x d tensor
    ar coefficients (D) - r x p x d x d tensor
    coefficients (C) - n x r tensor
    
    outputs:
    condition number (k1) - scalar
    condition number (k2) - n array
    """
    n, m, d = x.shape
    r, p, _, _ = D.shape
    X = np.zeros([n, m-p, p*d])
    for i in range(n):
        X[i, :, :], _ = ar_toep_op(x[i, :, :], p)
    XtX = m**(-1) * np.matmul(np.moveaxis(X, 1, 2), X)
    G1 = np.zeros([p*d*r, p*d*r])
    for i in range(n):
        G1 += np.kron(np.outer(C[i, :], C[i, :]) / n, XtX[i])
    _, s1, _ = sl.svd(G1)
    D_stack = np.zeros([r, p*d, d])
    for j in range(r):
        D_stack[j] = stack_ar_coef(D[j])
    G2 = np.zeros([n, r, r])
    s2 = np.zeros([n, r])
    for i in range(n):
        G2[i] = gram(D_stack, lambda x, y : inner_prod(x, np.dot(XtX[i], y)))
        _, s2[i], _ = sl.svd(G2[i])
    return np.max(s1) / np.min(s1), np.max(s2, axis=1) / np.min(s2, axis=1)

def isstable(A):
    """
    Form companion matrix (C) of polynomial p(z) = z*A[1] + ... + z^p*A[p] and
    check that det(I-zC)=0 only for |z|>1; if so, return True and otherwise
    False.
    
    inputs:
    ar coefficientst (A) - p x d x d tensor
    
    outputs:
    isstable - True/False
    """
    p, d, _ = A.shape
    C = np.eye(p*d, k=-d)
    C[:d, :] = np.concatenate(list(A), axis=1)
    v = sl.eigvals(C, overwrite_a=True)
    return np.all(np.abs(v**(-1)) > 1)

def ar_sample(sample_len, signal_dim, noise_var, ar_coef):
    """
    Generates a random sample of an autoregressive process (x[t])_t=1,...,n
    according to x[t] = A[1] x[t-1] + ... + A[p] x[t-p] + n[t]. Accounts for
    mixing time.
    
    inputs:
    sample_len (n) - scalar
    signal_dim (d) - scalar
    noise_var (v) - scalar
    ar_coef (A) - p x d x d tensor
    
    outputs:
    sample of autoregessive process (x) - n x d tensor
    """
    model_ord = np.shape(ar_coef)[0]
    # Generate more samples than necessary to allow for mixing of the process
    samples = 2*sample_len + model_ord
    x = np.zeros([samples, signal_dim, 1])
    # Generate samples by autoregressive recurrence relation
    x[:model_ord, :, :] = nr.randn(model_ord, signal_dim, 1)
    x[model_ord, :, :] = np.sum(np.matmul(ar_coef, x[model_ord-1::-1, :, :]), axis=0) + noise_var * nr.randn(1, signal_dim, 1)
    for t in np.arange(model_ord+1, samples):
        x[t, :] = np.sum(np.matmul(ar_coef, x[t-1:t-model_ord-1:-1, :, :]), axis=0) + noise_var * nr.randn(1, signal_dim, 1)
    return np.squeeze(x[-sample_len:, :])

def coef_sample(model_ord, signal_dim, sample_len, coef_type=None):
    """
    Generates coefficients for an autoregressive process (A[s])_s as in
    x[t] = A[1] x[t-1] + ... + A[p] x[t-p] + n[t].
    
    inputs:
    model_ord (p) - integer
    signal_dim (d) - integer
    sample len (m) - integer
    coef_type - string {None, 'sparse', 'lag_sparse'}
    
    outputs:
    ar_coef (A) - p x d x d tensor
    """
    ar_coef = nr.randn(model_ord, signal_dim, signal_dim)
    if coef_type == 'sparse':
        ar_coef[ar_coef < np.percentile(ar_coef, 90)] = 0
    elif coef_type == 'lag_sparse':
        ar_coef[list(nr.randint(model_ord, size=(int(0.9*model_ord),), dtype=np.int)), :, :] = 0
    elif coef_type is not None:
        raise ValueError(coef_type+" is not a valid coefficient type, i.e. sparse or lag_sparse.")
    # Enforce unit Frobenius norm
    return ar_coef / sl.norm(ar_coef[:])

def almm_sample(num_samples, sample_len, signal_dim, num_processes, model_ord, 
                coef_support, coef_type=None):
    """
    Generates iid autoregressive linear mixture model samples according to
    x_i[t] = \sum_j c_ij A_j[1] x_i[t-1] + ... + \sum_j c_ij A_j[p] x_i[t-m] + n[t]
    where (c_ij)_j has support of size coef_support and i=1,...,num_processes.
    
    inputs:
    num_samples (n) - integer
    sample_len (m) - integer
    signal_dim (d) - integer
    num_processes (r) - integer
    model_ord (p) - integer
    coef_support (s) - integer {1,...,r}
    coef_type - string {None, 'sparse', 'lag_sparse'}        
    
    outputs:
    sample of autoregessive process (X) - m x n x d tensor
    coefficients (C) - m x r tensor
    ar_coef (A) - r x p x d x d tensor
    """
    A = np.zeros([num_processes, model_ord, signal_dim, signal_dim])
    for i in range(num_processes):
        A[i, :, :, :] = coef_sample(model_ord, signal_dim, sample_len, 
                                    coef_type=coef_type)
    C = np.zeros([num_samples, num_processes])
    for i in range(num_samples):
        support = list(nr.choice(num_processes, size=coef_support, replace=False))
        C[i, support] = signal_dim**(-1/2) * nr.randn(coef_support)
        while not isstable(np.tensordot(C[i, :], A, axes=1)):
            C[i, support]  = signal_dim**(-1/2) * nr.randn(coef_support)
    X = np.zeros([num_samples, sample_len, signal_dim])
    for i in range(num_samples):
        X[i, :, :] = ar_sample(sample_len, signal_dim, signal_dim**(-1/2),
                               np.tensordot(C[i, :], A, axes=1))
    return X, C, A