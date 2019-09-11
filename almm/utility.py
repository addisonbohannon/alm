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

#def fit_ar_coef(x, model_ord, penalty=None, mu=0, cond=1e-4, X=None, Y=None):
#    """
#    Fits an autoregressive model to an observation x according to the relation
#    x[t] = A[1] x[t-1] + ... + A[p] x[t-p] + n[t] using an ordinary least
#    squares estimator.
#    
#    inputs:
#    ar process observation (x) - n x d tensor
#    model_ord (p) - integer
#    penalty (\|.\|_p) - 'l1' or 'l2' (optional)
#    mu - scalar (optional)
#    cond - scalar (optional)
#    X - (n-p) x (p*d) tensor (optional)
#    Y - (n-p) x d tensor (optional)
#    
#    outputs:
#    ar_coef (A) - p x d x d tensor
#    """
#    _, d = x.shape
#    if X is None or Y is None:
#        X, Y = ar_toep_op(x, model_ord)
#    if penalty is None:
#        A, _, _, _ = sl.lstsq(X, Y, cond=cond)
#        A = np.stack(np.split(A.T, model_ord, axis=1), axis=0)
#    elif penalty == 'l1':
#        if mu == 0:
#            A = sl.lstsq(X, Y, cond=cond)
#        lm = Lasso(alpha=mu, fit_intercept=False)
#        lm.fit(X, Y)
#        A = np.stack(np.split(lm.coef_, model_ord, axis=1), axis=0)
#    elif penalty == 'l2':
#        if mu == 0:
#            A = sl.lstsq(X, Y, cond=cond)
#        lm = Ridge(alpha=mu, fit_intercept=False)
#        lm.fit(X, Y)
#        A = np.stack(np.split(lm.coef_, model_ord, axis=1), axis=0)
#    else:
#        raise ValueError(penalty+' is not a valid penalty argument, i.e. None, l1, or l2.')
#    return A

#def ar_evaluate(X, A, Y, score_fcn='likelihood'):
#    """
#    Evaluate the fit of autoregressive coefficients A with stacked
#    observations Y and autoregressive Toeplitz operator X.
#    
#    inputs:
#    ar Toeplitz operator (X) - n x (p*d) tensor
#    ar coefficients (A) - (p*d) x d tensor
#    observations (Y) - n x d tensor
#    score_fcn - string {'likelihood', 'aic', 'bic'}
#    
#    outputs:
#    predicted observations (Y_pred) - n x d tensor
#    fit score - scalar
#    """
#    aic = lambda L, k : 2 * k + 2 * L
#    bic = lambda L, n, k : np.log(n) * k + 2 * L
#    model_ord, signal_dim, _ = A.shape
#    A = np.reshape(np.moveaxis(A, 1, 2), [model_ord*signal_dim, signal_dim])
#    Y_pred = np.dot(X, A)
#    log_likelihood = 0.5 * sl.norm(Y-Y_pred, ord='fro')**2
#    if score_fcn == 'likelihood':
#        score = log_likelihood
#    elif score_fcn == 'aic':
#        k = np.prod(np.shape(A))
#        score = aic(log_likelihood, k)
#    elif score_fcn == 'bic':
#        n, _ = Y.shape
#        k = np.prod(np.shape(A))
#        score = bic(log_likelihood, n, k)
#    else:
#        raise ValueError(score_fcn + ' is not a valid score function, i.e. likelihood, aic, bic.')
#    return Y_pred, score

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