#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 26 Apr 2019
"""

import numpy as np
import numpy.random as nr
import scipy.linalg as sl
import scipy.fftpack as sf

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
    samples = 2*sample_len + model_ord
    x = np.zeros([samples, signal_dim, 1])
    x[:model_ord, :, :] = nr.randn(model_ord, signal_dim, 1)
    x[model_ord, :, :] = np.sum(np.matmul(ar_coeffs, x[model_ord-1::-1, :, :]), axis=0) + noise_var * nr.randn(1, signal_dim, 1)
    for t in np.arange(model_ord+1, samples):
        x[t, :] = np.sum(np.matmul(ar_coeffs, x[t-1:t-model_ord-1:-1, :, :]), axis=0) + noise_var * nr.randn(1, signal_dim, 1)
    return np.squeeze(x[-sample_len:, :])

def ar_coeffs_sample(model_ord, signal_dim, sample_len):
    """
    Generates coefficients for a stable autoregressive process (A[s])_s as in
    x[t] = A[1] x[t-1] + ... + A[p] x[t-p] + n[t].
    
    inputs:
    model_ord (p) - scalar
    signal_dim (d) - scalar
    sample_len (n) - scalar
    
    outputs:
    ar_coeffs (A) - p x d x d tensor
    """
    ar_coeffs = nr.rand(model_ord, signal_dim, signal_dim)
    ar_coeffs /= np.max(sl.norm(sf.fft(ar_coeffs, n=sample_len, axis=0), ord=2, axis=0))
    return ar_coeffs

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
    """
    sample_len, signal_dim = np.shape(x)
    ar_toep = np.zeros([sample_len-model_ord, model_ord*signal_dim])
    ar_toep[0, :] = np.ndarray.flatten(x[model_ord-1::-1, :])
    for t in np.arange(1, sample_len-model_ord):
        ar_toep[t, :] = np.ndarray.flatten(x[t+model_ord-1:t-1:-1, :])
    return ar_toep

p = 2
d = 5
A = nr.rand(p, d, d)
nu = d**(-1/2)
n = 10

A = ar_coeffs_sample(p, d, n)
Y = autoregressive_sample(n, d, nu, A)
X = ar_toep_op(Y, p)