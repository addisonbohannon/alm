#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 29 Apr 2019
"""

import numpy as np
import numpy.random as nr
import scipy.linalg as sl
from utility import isstable

def autoregressive_sample(sample_len, signal_dim, noise_var, ar_coeffs):
    """
    Generates a random sample of an autoregressive process (x[t])_t=1,...,n
    according to x[t] = A[1] x[t-1] + ... + A[p] x[t-p] + n[t]. Accounts for
    mixing time.
    
    inputs:
    sample_len (n) - scalar
    signal_dim (d) - scalar
    noise_var (v) - scalar
    ar_coeffs (A) - p x d x d tensor
    
    outputs:
    sample of autoregessive process (x) - n x d tensor
    """
    model_ord = np.shape(ar_coeffs)[0]
    # Generate more samples than necessary to allow for mixing of the process
    samples = 2*sample_len + model_ord
    x = np.zeros([samples, signal_dim, 1])
    # Generate samples by autoregressive recurrence relation
    x[:model_ord, :, :] = nr.randn(model_ord, signal_dim, 1)
    x[model_ord, :, :] = np.sum(np.matmul(ar_coeffs, x[model_ord-1::-1, :, :]), axis=0) + noise_var * nr.randn(1, signal_dim, 1)
    for t in np.arange(model_ord+1, samples):
        x[t, :] = np.sum(np.matmul(ar_coeffs, x[t-1:t-model_ord-1:-1, :, :]), axis=0) + noise_var * nr.randn(1, signal_dim, 1)
    return np.squeeze(x[-sample_len:, :])

def ar_coeffs_sample(model_ord, signal_dim, sample_len, coef_type=None):
    """
    Generates coefficients for a stable autoregressive process (A[s])_s as in
    x[t] = A[1] x[t-1] + ... + A[p] x[t-p] + n[t].
    
    inputs:
    model_ord (p) - integer
    signal_dim (d) - integer
    sample len (m) - integer
    coef_type - string {None, 'sparse', 'lag_sparse'}
    
    outputs:
    ar_coeffs (A) - p x d x d tensor
    """
    ar_coeffs = nr.randn(model_ord, signal_dim, signal_dim)
    if coef_type == 'sparse':
        ar_coeffs[ar_coeffs < np.percentile(ar_coeffs, 90)] = 0
    elif coef_type == 'lag_sparse':
        ar_coeffs[list(nr.randint(model_ord, size=(int(0.9*model_ord),), dtype=np.int)), :, :] = 0
    elif coef_type is not None:
        raise ValueError(coef_type+" is not a valid coefficient type, i.e. sparse or lag_sparse.")
    # Enforce unit Frobenius norm
    return ar_coeffs / sl.norm(ar_coeffs[:])

def almm_iid_sample(num_samples, sample_len, signal_dim, num_processes, 
                    model_ord, coef_support, coef_type=None):
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
    ar_coeffs (A) - r x p x d x d tensor
    """
    A = np.zeros([num_processes, model_ord, signal_dim, signal_dim])
    for i in range(num_processes):
        A[i, :, :, :] = ar_coeffs_sample(model_ord, signal_dim, sample_len, 
                                         coef_type=coef_type)
    C = np.zeros([num_samples, num_processes])
    for i in range(num_samples):
        support = list(nr.choice(num_processes, size=coef_support, replace=False))
        C[i, support] = signal_dim**(-1/2) * nr.randn(coef_support)
        while not isstable(np.tensordot(C[i, :], A, axes=1)):
            C[i, support]  = signal_dim**(-1/2) * nr.randn(coef_support)
    X = np.zeros([num_samples, sample_len, signal_dim])
    for i in range(num_samples):
        X[i, :, :] = autoregressive_sample(sample_len, signal_dim, 
                                           signal_dim**(-1/2),
                                           np.tensordot(C[i, :], A, axes=1))
    return X, C, A