#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 28 Apr 2019
"""

import numpy as np
import scipy.linalg as sl
import scipy.spatial.distance as sd
from sklearn.linear_model import orthogonal_mp_gram, lars_path

def ip(a, b):
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

def coeff_update_omp_sklearn(Y, X, D, s):
    """
    Implements orthogonal matching pursuit (c.f. Tropp 2006) for the least
    squares problem \|Y-\sum_j c_j X D_j\|_F**2 for a fixed support size of s.
    Uses orthogonal_mp_gram from sklearn as a subroutine.
    
    inputs:
    observations (Y) - n x d tensor
    observation Toeplitz matrix (X) - n x (p*d) tensor
    dictionary (D) - r x (p*d) x d tensor
    support size (s) - integer {1,...,r}
    
    outputs:
    coefficients (c) - r tensor
    """
    # precompute reused quantities
    r, _, _ = D.shape
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)
    XD_gram = []
    indices = np.triu_indices(r, k=1)
    [XD_gram.append(ip(D[i, :, :], np.dot(XTX, D[j, :, :]))) for (i, j) in zip(indices[0], indices[1])]
    XD_gram = np.array(XD_gram)
    XD2 = np.sum(np.square(np.matmul(X, D)), axis=(1,2))
    DTXTXD = sd.squareform(XD_gram) + np.diagflat(XD2)
    YTXD = [ip(XTY, D[j, :, :]) for j in range(r)]
    # call sklearn subroutine
    c = orthogonal_mp_gram(DTXTXD, YTXD, n_nonzero_coefs=s)
    return c

def coeff_update_omp(Y, X, D, s):
    """
    Implements orthogonal matching pursuit (c.f. Tropp 2006) for the least
    squares problem \|Y-\sum_j c_j X D_j\|_F**2 for a fixed support size of s.
    
    inputs:
    observations (Y) - n x d tensor
    observation Toeplitz matrix (X) - n x (p*d) tensor
    dictionary (D) - r x (p*d) x d tensor
    support size (s) - integer {1,...,r}
    
    outputs:
    coefficients (c) - r tensor
    """
    # precompute reused quantities
    r, _, _ = D.shape
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)
    XD_gram = []
    indices = np.triu_indices(r, k=1)
    [XD_gram.append(ip(D[i, :, :], np.dot(XTX, D[j, :, :]))) for (i, j) in zip(indices[0], indices[1])]
    XD_gram = np.array(XD_gram)
    XD2 = np.sum(np.square(np.matmul(X, D)), axis=(1,2))
    DTXTXD = sd.squareform(XD_gram) + np.diagflat(XD2)
    # orthogonal matching pursuit algorithm
    c = np.zeros([r])
    R = Y
    J = []
    for i in np.arange(s):
        XTR = np.dot(X.T, R)
        Jc = np.setdiff1d(np.arange(r), J)
        # j in Jc that addition of (c_j, D_j) minimizes the residual
#        J.append(Jc[np.argmax([ np.abs(ip(XTR, D[j, :, :]) / XD2[j]) for j in Jc ])])
        J.append(Jc[np.argmin([ np.sum(np.square( R - ip(XTR, D[j, :, :]) / XD2[j] * np.dot(X, D[j, :, :]) )) for j in Jc ])])
        # find optimal (c_j)_J that minimizes the residual
        J_index = [j in J for j in range(r)]
        c[J_index] = sl.solve(DTXTXD[J_index, :][:, J_index], ip(XTY, D[J_index, :, :]), assume_a='pos')
        # update residual
        R = Y - np.tensordot(c[J_index], np.matmul(X, D[J_index, :, :]), axes=1)
    return c

def coeff_update_lasso_sklearn(Y, X, D, s):
    """
    Implements a LASSO solver for \|Y-\sum_j c_j X D_j\|_F**2 + u \sum_j |c_j|
    along the entire u-path. Selects the coefficients along the path with
    support size s. Uses lars_path from sklearn as a subroutine with
    precomputed Gram and Xy matrices.
    
    inputs:
    observations (Y) - n x d tensor
    observation Toeplitz matrix (X) - n x (p*d) tensor
    dictionary (D) - r x (p*d) x d tensor
    support size (s) - integer {1,...,r}
    
    outputs:
    coefficients (c) - r tensor
    """
    # precompute reused quantities
    r, _, _ = D.shape
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, Y)
    XD_gram = []
    indices = np.triu_indices(r, k=1)
    [XD_gram.append(ip(D[i, :, :], np.dot(XTX, D[j, :, :]))) for (i, j) in zip(indices[0], indices[1])]
    XD_gram = np.array(XD_gram)
    XD2 = np.sum(np.square(np.matmul(X, D)), axis=(1,2))
    DTXTXD = sd.squareform(XD_gram) + np.diagflat(XD2)
    YTXD = np.array([ip(XTY, D[j, :, :]) for j in range(r)])
    # call sklearn subroutine
    _, _, coefs = lars_path(X, Y, Xy=YTXD, Gram=DTXTXD, method='lasso')
    return coefs[:, s]