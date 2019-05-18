#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 26 Apr 2019
"""

import numpy as np
import scipy.linalg as sl
import scipy.fftpack as sf
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import cvxpy as cp

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
    
def ar_coeff_fft(A, n=None):
    """
    Implements an FFT on the coefficients of an AR process, i.e. 
    A^(w) = \sum_s=1,...,p exp(2\pi i w t) A[t]; note that it zero-pads the 
    zero lag and transforms on the dimension of the sample length.
    
    inputs:
    ar coefficients (A) - p x d x d tensor
    sample length (n) - integer (optional)
    
    outputs:
    ar frequency coefficients (A^) - n x d x d tensor
    """
    p, d, _ = A.shape
    if n is not None and n <= p:
        raise ValueError("n must be greater than the model order")
    A = np.concatenate((np.zeros([1, d, d]), A), axis=0)
    if n is not None:
        A_hat = sf.rfft(A, n=n, axis=0)
    else:
        A_hat = sf.rfft(A, axis=0)
    return A_hat

def ar_coeff_ifft(A_hat, p=None):
    """
    Implments an inverse FFT on the frequency coefficients of an AR process, 
    i.e. A[s] = \int exp(-2\pi i w s) A^(w) dw; note that it will return only 
    (A[1], ..., A[p]) components.
    
    inputs:
    ar frequency coefficients (A^) - n x d x d tensor
    model order (p) - integer (optional)
    
    outputs:
    ar coefficients (A) - p x d x d tensor
    """
    n, _, _ = A_hat.shape
    if p is not None and p > n:
        raise ValueError("model order (p) must be less than n")
    if p is not None:
        A = sf.irfft(A_hat, axis=0)[1:(p+1)]
    else:
        A = sf.irfft(A_hat, axis=0)[1:]
    return A

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

def stack_ar_coeffs(A):
    """
    Stack autoregressive coefficients (A[s])_s as [ A[1] ... A[p] ]^T.
    
    inputs:
    ar coeffs (A) - p x d x d tensor
    
    outputs:
    ar_coeffs (A) - (p*d) x d tensor
    """
    model_ord, signal_dim, _  = A.shape
    return np.reshape(np.moveaxis(A, 1, 2), [model_ord*signal_dim, signal_dim])

def unstack_ar_coeffs(A):
    """
    Unstack autoregressive coefficients (A[s])_s to [[A[1]], ..., [A[p]]].
    
    inputs:
    ar coeffs (A) - (p*d) x d tensor
    
    outputs:
    ar coeffs (A) - p x d x d tensor
    """
    
    m, n = A.shape
    model_ord = int(m/n)
    return np.stack(np.split(A.T, model_ord, axis=1), axis=0)

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
                _, cv_score[i, j] = ar_evaluate(X, A, Y, 
                                                score_fcn=cv_score_fcn)
        model_ord = model_ord_list[np.argmin(np.mean(cv_score, axis=1))]
        A = fit_ar_coeffs(x_train, model_ord)
        X, Y = ar_toep_op(x, model_ord)
        _, X_test, _, Y_test = train_test_split(X, Y, shuffle=False, 
                                                train_size=train_size)
        _, val_score = ar_evaluate(X_test, A, Y_test, score_fcn=val_score_fcn)
        return A, model_ord, val_score
    else:
        cv_score = np.zeros([len(model_ord_list), len(mu_list), k])
        for i, model_ord in enumerate(model_ord_list):
            for j, mu in enumerate(mu_list):
                for l, (train_index, test_index) in enumerate(tscv.split(x_train)):
                    A = fit_ar_coeffs(x[train_index, :], model_ord, 
                                      penalty=penalty, mu=mu)
                    X, Y = ar_toep_op(x[test_index, :], model_ord)
                    _, cv_score[i, j, l] = ar_evaluate(X, A, Y, 
                                                       score_fcn=cv_score_fcn)
        i, j = np.unravel_index(np.argmin(np.mean(cv_score, axis=2)), 
                                np.array(np.shape(cv_score)[:-1]))
        model_ord = model_ord_list[i]
        mu = mu_list[j]
        A = fit_ar_coeffs(x_train, model_ord, penalty=penalty, mu=mu)
        X, Y = ar_toep_op(x, model_ord)
        _, X_test, _, Y_test = train_test_split(X, Y, shuffle=False, 
                                                train_size=train_size)
        _, val_score = ar_evaluate(X_test, A, Y_test, score_fcn=val_score_fcn)
        return A, model_ord, mu, val_score

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

def dictionary_distance(A, B, p=2):
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