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
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit

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
    # Enforce stability of the process
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

def fit_ar_coeffs(x, model_ord, penalty=None, mu=0, cond=1e-4, X=None, Y=None):
    """
    Fits an autoregressive model to an observation x according to the relation
    x[t] = A[1] x[t-1] + ... + A[p] x[t-p] + n[t] using an ordinary least
    squares estimator.
    
    inputs:
    ar process observation (x) - n x d tensor
    model_ord (p) - integer
    penalty (\|.\|_p) - 'l1' or 'l2' (optional)
    mu - scalar (optional)
    cond - scalar (optional)
    X - (n-p) x (p*d) tensor (optional)
    Y - (n-p) x d tensor (optional)
    
    outputs:
    ar_coeffs (A) - p x d x d tensor
    """
    _, d = x.shape
    if X is None or Y is None:
        X, Y = ar_toep_op(x, model_ord)
    if penalty is None:
        A, _, _, _ = sl.lstsq(X, Y, cond=cond)
        A = np.stack(np.split(A.T, model_ord, axis=1), axis=0)
    elif penalty == 'l1':
        if mu == 0:
            A = sl.lstsq(X, Y, cond=cond)
        lm = Lasso(alpha=mu, fit_intercept=False)
        lm.fit(X, Y)
        A = np.stack(np.split(lm.coef_, model_ord, axis=1), axis=0)
    elif penalty == 'l2':
        if mu == 0:
            A = sl.lstsq(X, Y, cond=cond)
        lm = Ridge(alpha=mu, fit_intercept=False)
        lm.fit(X, Y)
        A = np.stack(np.split(lm.coef_, model_ord, axis=1), axis=0)
    else:
        raise ValueError(penalty+' is not a valid penalty argument, i.e. None, l1, or l2.')
    return A

def ar_evaluate(X, A, Y, score_fcn='likelihood'):
    """
    Evaluate the fit of autoregressive coefficients A with stacked
    observations Y and autoregressive Toeplitz operator X.
    
    inputs:
    ar Toeplitz operator (X) - n x (p*d) tensor
    ar coefficients (A) - (p*d) x d tensor
    observations (Y) - n x d tensor
    score_fcn - string {'likelihood', 'aic', 'bic'}
    
    outputs:
    predicted observations (Y_pred) - n x d tensor
    fit score - scalar
    """
    aic = lambda L, k : 2 * k + 2 * L
    bic = lambda L, n, k : np.log(n) * k + 2 * L
    model_ord, signal_dim, _ = A.shape
    A = np.reshape(np.moveaxis(A, 1, 2), [model_ord*signal_dim, signal_dim])
    Y_pred = np.dot(X, A)
    log_likelihood = 0.5 * sl.norm(Y-Y_pred, ord='fro')**2
    if score_fcn == 'likelihood':
        score = log_likelihood
    elif score_fcn == 'aic':
        k = np.prod(np.shape(A))
        score = aic(log_likelihood, k)
    elif score_fcn == 'bic':
        n, _ = Y.shape
        k = np.prod(np.shape(A))
        score = bic(log_likelihood, n, k)
    else:
        raise ValueError(score_fcn + ' is not a valid score function, i.e. likelihood, aic, bic.')
    return Y_pred, score
    

def fit_ar_coeffs_CV(x, k=4, train_size=0.75, model_ord_list=None, 
                     penalty=None, mu_list=None, cv_score_fcn='likelihood', 
                     val_score_fcn='likelihood'):
    """
    Uses k-fold cross validation to select model order and penalty
    parameter for fitting an autoregressive model. Then, fits the 
    autoregressive model for the optimal parameters.
    
    inputs:
    ar process observation (x) - n x d tensor
    k - integer (optional)
    train_pct - scalar (0, 1) (optional)
    model_ord_list (p_1, ..., p_m) - list of integers (optional)
    penalty (\|.\|_p) - 'l1' or 'l2' (optional)
    mu_list - list of scalars (optional)
    score_fcn - string {'likelihood', 'aic', 'bic'} (optional)
    
    outputs:
    ar_coeffs (A) - p x d x d tensor
    model_ord (p) - integer
    mu - scalar (only if penalty is not None)
    val_score - scalar
    """
    n, d = x.shape
    if model_ord_list is None:
        model_ord_list = np.arange(1, 11)
    if mu_list is None and penalty is not None:
        mu_list = np.logspace(-3, 3, num=10)
    x_train, _ = train_test_split(x, shuffle=False, train_size=train_size)
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=k)
    if penalty is None:
        cv_score = np.zeros([len(model_ord_list), k])
        for i, model_ord in enumerate(model_ord_list):
            for j, (train_index, test_index) in enumerate(tscv.split(x_train)):
                A = fit_ar_coeffs(x_train[train_index, :], model_ord)
                X, Y = ar_toep_op(x[test_index, :], model_ord)
                _, cv_score[i, j] = ar_evaluate(X, A, Y, score_fcn=cv_score_fcn)
        model_ord = model_ord_list[np.argmin(np.mean(cv_score, axis=1))]
        A = fit_ar_coeffs(x_train, model_ord)
        X, Y = ar_toep_op(x, model_ord)
        _, X_test, _, Y_test = train_test_split(X, Y, shuffle=False, train_size=train_size)
        _, val_score = ar_evaluate(X_test, A, Y_test, score_fcn=val_score_fcn)
        return A, model_ord, val_score
    else:
        cv_score = np.zeros([len(model_ord_list), len(mu_list), k])
        for i, model_ord in enumerate(model_ord_list):
            for j, mu in enumerate(mu_list):
                for l, (train_index, test_index) in enumerate(tscv.split(x_train)):
                    A = fit_ar_coeffs(x[train_index, :], model_ord, penalty=penalty, mu=mu)
                    X, Y = ar_toep_op(x[test_index, :], model_ord)
                    _, cv_score[i, j, l] = ar_evaluate(X, A, Y, score_fcn=cv_score_fcn)
        i, j = np.unravel_index(np.argmin(np.mean(cv_score, axis=2)), np.array(np.shape(cv_score)[:-1]))
        model_ord = model_ord_list[i]
        mu = mu_list[j]
        A = fit_ar_coeffs(x_train, model_ord, penalty=penalty, mu=mu)
        X, Y = ar_toep_op(x, model_ord)
        _, X_test, _, Y_test = train_test_split(X, Y, shuffle=False, train_size=train_size)
        _, val_score = ar_evaluate(X_test, A, Y_test, score_fcn=val_score_fcn)
        return A, model_ord, mu, val_score