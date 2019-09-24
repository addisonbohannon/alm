#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 27 Jun 19
"""

from timeit import default_timer as timer
import numpy as np
import numpy.random as nr
import scipy.linalg as sl
from almm.utility import gram, inner_prod

def shrink(x, t):
    """
    Implements the proximal operator of the l1-norm (shrink operator).
    
    inputs:
    x (tensor) - argument to apply soft thresholding
    t (scalar) - step size of proximal operator
    
    outputs:
    x (tensor) - soft thresholded argument
    """
    
    return np.sign(x) * np.maximum(np.abs(x)-t, 0)

def threshold(x, t):
    """
    Implements the proximal operator of the l0-norm (threshold operator).
    
    inputs:
    x (tensor) - argument to apply hard thresholding
    t (scalar) - step size of proximal operator
    
    outputs:
    x (tensor) - hard thresholded argument
    
    """
    
    x[x**2 < 2*t] = 0
    return x
    
def proj(z):
    """
    Projects onto the l2-ball: argmin_x (1/2)*\|x-z\|_2^2 s.t. \|x\|_2 = 1.
    
    inputs:
    dictionary (p*d x d tensor) - unnormalized input
    
    outputs:
    dictionary (p*d x d tensor) - normalized output
    """
    
    return z / sl.norm(z[:])

def penalized_ls_gram(G, C, prox, mu, max_iter=1e3, tol=1e-4):
    """
    Implements an ADMM solver for the penalized least squares problem,
    argmin_x 1/2 * \|Ax-b\|^2 + mu * \|x\|_p, using the precomputed gram
    matrix and covariance, i.e. G=A^T.A and C=A^T.b. Implementation based on
    Boyd, et al, Foundations and Trends in Machine Learning, 2011.
    
    inputs:
    G (m x m array) - data matrix
    C (m array) - observations
    mu (scalar) - penalty parameter
    p (0 or 1) - p-norm penalty; must be 0 or 1
    max_iter (int) - maximum iterations of algorithm; must be positive integer
    tol (scalar) - tolerance for terminating algorithm
    
    outputs:
    x (m array) - parameters
    """
    
    m = len(C)
    # initialize variables for solver
    Z = np.zeros_like(C)
    U = np.zeros_like(Z)
    r = []
    s = []
    # precompute values for admm loop
    p = 1e-4
    G_factor = sl.cho_factor(G + np.eye(m))
    # admm solver
    for step in np.arange(int(max_iter)):
        # update X with ridge regression
        X = sl.cho_solve(G_factor, C - U + p * Z)
        # update Z
        Z_upd = prox(X + (1/p) * U, mu / p)
        # update dual residual
        s.append(p * sl.norm(Z_upd-Z))
        Z = Z_upd
        # update Lagrange multiplier, U
        U += p * (X - Z)
        # update primal residual
        r.append(sl.norm(X-Z))
        if (r[step] <= tol*np.maximum(sl.norm(X), sl.norm(Z)) and 
            s[step] <= tol*sl.norm(U)):
            break
        if r[step] > 4 * s[step]:
            p *= 2
            G_factor = sl.cho_factor(G + p*np.eye(m))
        elif s[step] > 4 * r[step]:
            p /= 2
            G_factor = sl.cho_factor(G + p*np.eye(m))
    return X
        
def fit_coefs(XtX, XtY, D, mu, coef_penalty_type):
    """
    Fit coefficients for a known dictionary of autoregressive atoms and
    penalty parameter.
    
    inputs:
    XtX (list of arrays) - sample autocorrelation of observations
    
    XtY (list of arrays) - sample autocorrelation of observations
    
    D (list arrays) - estimates of autoregressive atoms
    
    mu (scalar) - penalty parameter
    
    coef_penalty_type (string) - coefficient penalty of objective; {None, l0, l1}
    
    outputs:
    C (array) - estimate of coefficients; length of XtX x length of D
    
    likelihood (scalar) - negative log likelihood of estimates
    """
    
    if coef_penalty_type is None:
        solve = lambda a, b : sl.solve(a, b, assume_a='pos')
    elif coef_penalty_type == 'l0':
        solve = lambda a, b : penalized_ls_gram(a, b, threshold, mu)
    elif coef_penalty_type == 'l1':
        solve = lambda a, b : penalized_ls_gram(a, b, shrink, mu)
    else:
        raise ValueError('coef_penalty_type not a valid type, i.e. None, l0, or l1')
    
    # Fit coefficients with iterative algorithm
    return np.array([solve(gram(D, lambda x, y : inner_prod(x, np.dot(XtX_i, y))), inner_prod(XtY_i, D))
                     for XtX_i, XtY_i in zip(XtX, XtY)])

def solver_two_stage(XtX, XtY, p, r, mu, coef_penalty_type, D_0=None, max_iter=1e2, step_size=1e-3, tol=1e-6,
                     return_path=False, verbose=False):
    """
    Two-stage algorithm for solve the almm model. It begins by fitting autoregressive coefficients to each observation.
    Then, it attempts to solve the sparse coding problem using alternating minimization.
    
    inputs:
    XtX (n x p*d x p*d array) - sample autocorrelation
    
    XtY (n x p*d x d array) - sample autocorrelation
    
    p (integer) - model order
    
    r (integer) - dictionary atoms
    
    mu (float) - penalty parameter
    
    coef_penalty_type (string) - coefficient penalty of objective; 
    {None, l0, l1}
        
    D_0 (r x p*d * d array) - intial dictionary estimate (optional)
    
    maximum iterations (integer) - Maximum number of iterations for 
    algorithm
        
    step size (scalar) - Factor by which to extend the Lipschitz-based 
    step size; must be less than 1
    
    tolerance (float) - Tolerance to terminate iterative algorithm; must
    be positive
    
    return_path (boolean) - whether or not to return the path of
    dictionary and coefficient estimates
    
    verbose (boolean) - whether or not to print progress during execution; 
    used for debugging
    
    outputs:
    D ([k x] r x p*d x d array) - dictionary estimate [if return_path=True]
    
    C ([k x] n x r array) - coefficient estimate [if return_path=True]
    
    residual D (k array) - residuals of dictionary update
    
    residual C (k array) - residuals of coefficient update
    
    stopping condition ({maximum iteration, relative tolerance}) -
    condition that terminated the iterative algorithm
    """

    start = timer()

    n = len(XtY)
    _, d = XtY[0].shape

    # Initialize dictionary randomly; enforce unit norm
    if D_0 is None:
        D = nr.randn(r, p*d, d)
        for j in range(r):
            D[j] = proj(D[j])
    else: D = D_0

    # Initialize coefficients
    C = fit_coefs(XtX, XtY, D, mu, coef_penalty_type)

    # Initialize estimate path
    if return_path:
        D_path = [np.copy(D)]
        C_path = [np.copy(C)]
    wall_time = [timer()-start]

    # Estimate autoregressive coefficients for each observation
    A = [sl.solve(XtX_i, XtY_i, assume_a='pos') for XtX_i, XtY_i in zip(XtX, XtY)]

    # Decompose autoregressive components into sparse dictionary atoms with alternating minimization algorithm
    stop_condition = 'maximum iteration'
    residual_D = []
    residual_C = []
    for step in range(max_iter):

        # Update dictionary estimate
        temp = np.copy(D)
        ccXtX = {}
        triu_index = np.triu_indices(r)
        for (i, j) in zip(triu_index[0], triu_index[1]):
            ccXtX[(i, j)] = np.tensordot(C[:, i] * C[:, j], XtX, axes=1)
        for j in range(r):
            Aj = ccXtX[(j, j)]
            bj = np.tensordot(C[:, j], np.matmul(XtX, A), axes=1)
            for l in np.setdiff1d(np.arange(r), [j]):
                bj -= np.dot(ccXtX[tuple(sorted((j, l)))], D[l])
            D[j] = proj(sl.solve(Aj, bj, assume_a='pos'))
        delta_D = D - temp

        # Update coefficient estimate
        temp = np.copy(C)
        C = fit_coefs(XtX, np.matmul(XtX, A), D, mu, coef_penalty_type)
        delta_C = C - temp

        # Add current estimates to path
        if return_path:
            D_path.append(np.reshape(np.copy(D), [r, p*d, d]))
            C_path.append(np.copy(C))

        # Compute residuals
        """( (1/r) \sum_j \|dD_j\|^2 / (p*d^2) )^(1/2)"""
        residual_D.append(sl.norm(delta_D[:]) / (r**(1/2) * p**(1/2) * d))
        """( (1/n) \sum_i (\|dC_i\|/beta_i)^2 / r )^(1/2)"""
        residual_C.append(sl.norm(delta_C[:]) / (n**(1/2) * r**(1/2)))

        # Compute wall time
        wall_time.append(timer()-start)

        # Check stopping condition
        if ( step > 0 and residual_D[-1] < tol * residual_D[0]
            and residual_C[-1] < tol * residual_C[0] ):
            stop_condition = 'relative tolerance'
            break

    if verbose:
        end = timer()
        duration = end - start
        print('*Solver: Two-stage')
        print('*Stopping condition: ' + stop_condition)
        print('*Iterations: ' + str(step))
        print('*Duration: ' + str(duration) + 's')

    if return_path:
        return D_path, C_path, residual_D, residual_C, stop_condition, wall_time
    else:
        return np.reshape(D, [r, p*d, d]), C, residual_C, residual_D, stop_condition, wall_time

    
def solver_alt_min(XtX, XtY, p, r, mu, coef_penalty_type, D_0=None, 
                   max_iter=1e2, step_size=1e-3, tol=1e-6, return_path=False, 
                   verbose=False):
    """
    Alternating minimization algorithm for ALMM solver. Alternates between 
    minimizing the following function with respect to (D_j)_j and (C_ij)_ij:
    1/2 * \sum_i \| Y_i - \sum_j c_ij D_j X_i \|_F^2 + mu * \|C\|_p.
    
    inputs:
    XtX (n x p*d x p*d array) - sample autocorrelation
    
    XtY (n x p*d x d array) - sample autocorrelation
    
    p (integer) - model order
    
    r (integer) - dictionary atoms
    
    mu (float) - penalty parameter
    
    coef_penalty_type (string) - coefficient penalty of objective; 
    {None, l0, l1}
        
    D_0 (r x p*d * d array) - intial dictionary estimate (optional)
    
    maximum iterations (integer) - Maximum number of iterations for 
    algorithm
        
    step size (scalar) - Factor by which to divide the Lipschitz-based 
    step size
    
    tolerance (float) - Tolerance to terminate iterative algorithm; must
    be positive
    
    return_path (boolean) - whether or not to return the path of
    dictionary and coefficient estimates
    
    verbose (boolean) - whether or not to print progress during execution; 
    used for debugging
    
    outputs:
    D ([k x] r x p*d x d array) - dictionary estimate [if return_path=True]
    
    C ([k x] n x r array) - coefficient estimate [if return_path=True]
    
    residual D (k array) - residuals of dictionary update
    
    residual C (k array) - residuals of coefficient update
    
    stopping condition ({maximum iteration, relative tolerance}) -
    condition that terminated the iterative algorithm
    """
    
    start = timer()
        
    n = len(XtY)
    _, d = XtY[0].shape
        
    # Initialize dictionary randomly; enforce unit norm
    if D_0 is None:
        D = nr.randn(r, p*d, d)
        for j in range(r):
            D[j] = proj(D[j])
    else: D = D_0
        
    # Initialize coefficients
    C = fit_coefs(XtX, XtY, D, mu, coef_penalty_type)
    
    # Initialize estimate path
    if return_path:
        D_path = [np.copy(D)]
        C_path = [np.copy(C)]
    
    # Begin alternating minimization algorithm
    stop_condition = 'maximum iteration'
    residual_D = []
    residual_C = []
    wall_time = [timer()-start]
    for step in range(max_iter):
        
        # Update dictionary estimate
        temp = np.copy(D)
        ccXtX = {}
        triu_index = np.triu_indices(r)
        for (i, j) in zip(triu_index[0], triu_index[1]):
            ccXtX[(i, j)] = np.tensordot(C[:, i]*C[:, j], XtX, axes=1)
        for j in range(r):
            Aj = ccXtX[(j, j)]
            bj = np.tensordot(C[:, j], XtY, axes=1)
            for l in np.setdiff1d(np.arange(r), [j]):
                bj -= np.dot(ccXtX[tuple(sorted((j, l)))], D[l])
            D[j] = proj(sl.solve(Aj, bj, assume_a='pos'))
        delta_D = D - temp
            
        # Update coefficient estimate
        temp = np.copy(C)
        C = fit_coefs(XtX, XtY, D, mu, coef_penalty_type)
        delta_C = C - temp
        
        # Add current estimates to path
        if return_path:
            D_path.append(np.copy(D))
            C_path.append(np.copy(C))
        
        # Compute residuals
        """( (1/r) \sum_j \|dD_j\|^2 / (p*d^2) )^(1/2)"""
        residual_D.append(sl.norm(delta_D[:]) / (r**(1/2) * p**(1/2) * d))
        """( (1/n) \sum_i (\|dC_i\|/beta_i)^2 / r )^(1/2)"""
        residual_C.append(sl.norm(delta_C[:]) / (n**(1/2) * r**(1/2)))
        
        # Compute wall time
        wall_time.append(timer()-start)
        
        # Check stopping condition
        if ( step > 0 and residual_D[-1] < tol * residual_D[0] 
            and residual_C[-1] < tol * residual_C[0] ):
            stop_condition = 'relative tolerance'
            break
        
    if verbose:
        end = timer()
        duration = end - start
        print('*Solver: Alternating Minimization')
        print('*Stopping condition: ' + stop_condition)
        print('*Iterations: ' + str(step))
        print('*Duration: ' + str(duration) + 's')
    
    if return_path:
        return D_path, C_path, residual_D, residual_C, stop_condition, wall_time
    else:
        return D, C, residual_C, residual_D, stop_condition, wall_time
    
def solver_palm(XtX, XtY, p, r, mu, coef_penalty_type, D_0=None, max_iter=1e3, 
                step_size=1e-1, tol=1e-6, return_path=False, verbose=False):
    """
    Iterative algorithm for ALMM solver. Based on the PALM algorithm
    of Bolte, Sabach, and Teboulle, Math. Program. Ser. A, 2014. Takes
    linearized proximal gradient steps with respect to each dictionary
    atom and coefficients in turn.
    
    inputs:
    XtX (n x p*d x p*d array) - sample autocorrelation
    
    XtY (n x p*d x d array) - sample autocorrelation
    
    p (integer) - model order
    
    r (integer) - dictionary atoms
    
    mu (float) - penalty parameter
    
    coef_penalty_type (string) - coefficient penalty of objective; 
    {None, l0, l1}
        
    D_0 (r x p*d * d array) - intial dictionary estimate (optional)
    
    maximum iterations (integer) - Maximum number of iterations for 
    algorithm
        
    step size (scalar) - Factor by which to extend the Lipschitz-based 
    step size; must be less than 1
    
    tolerance (float) - Tolerance to terminate iterative algorithm; must
    be positive
    
    return_path (boolean) - whether or not to return the path of
    dictionary and coefficient estimates
    
    verbose (boolean) - whether or not to print progress during execution; 
    used for debugging
    
    outputs:
    D ([k x] r x p*d x d array) - dictionary estimate [if return_path=True]
    
    C ([k x] n x r array) - coefficient estimate [if return_path=True]
    
    residual D (k array) - residuals of dictionary update
    
    residual C (k array) - residuals of coefficient update
    
    stopping condition ({maximum iteration, relative tolerance}) -
    condition that terminated the iterative algorithm
    """
    
    start = timer()
    
    # Set proximal function for coefficient
    if coef_penalty_type is None:
        prox_coef = lambda x, t : x
    elif coef_penalty_type == 'l0':
        prox_coef = threshold
    elif coef_penalty_type == 'l1':
        prox_coef = shrink
    else:
        raise ValueError('coef_penalty_type not a valid type, i.e. l0 or l1')
        
    n = len(XtY)
    _, d = XtY[0].shape
        
    # Initialize dictionary randomly; enforce unit norm
    if D_0 is None:
        D = nr.randn(r, p*d, d)
        for j in range(r):
            D[j] = proj(D[j])
    else: D = D_0
        
    # Initialize coefficients
    C = fit_coefs(XtX, XtY, D, mu, coef_penalty_type)
    
    # Initialize estimates of dictionary and coefficients
    if return_path:
        D_path = [np.copy(D)]
        C_path = [np.copy(C)]
        
    # Define gradient functions        
    def grad_D(D, C, j, G=None):
        """
        Computes the gradient of the jth dictionary element for the current 
        values of other dictionary elements and coefficients.
        
        inputs:
        D (r x p*d x d tensor) - dictionary estimate
        
        C (n x r tensor) - coefficient estimate
    
        j ({1,...,r}) - index of the dictionary atom
        
        G (p*d x d tensor) - quantity that is pre-computed in step size 
        calculuation
        
        outputs:
        grad (p*d x d tensor) - gradient of dictionary atom j
        """
        
        if G is None:
            G = np.tensordot(C[:, j]**2 / n, XtX, axes=1)
        grad = - np.tensordot(C[:, j] / n, XtY, axes=1)
        grad += np.dot(G, D[j, :, :])
        for l in np.setdiff1d(np.arange(r), [j]):
            grad += np.dot(np.tensordot(C[:, j]*C[:, l] / n, XtX, axes=1), 
                           D[l, :, :])
        return grad        
        
    def grad_C(D, C, i, G=None):
        """
        Computes the gradient of the ith coefficient vector for the current
        values of the dictionary elements.
        
        inputs:
        D (r x p*d x d tensor) - dictionary estimate
        
        C (n x r tensor) - coefficient estimate
        
        i ({1,...,n}) - index of the observation
        
        G (r x r tensor) - quantity that is pre-computed in step size
        calculation
        
        outputs:
        grad (r array) - gradient of coefficient vector i
        """
        
        if G is None:
            G = gram(D, lambda x, y : inner_prod(x, np.dot(XtX[i], y)))
        return (- inner_prod(XtY[i], D) + np.dot(G, C[i, :].T)) / n
    
    # Begin iterative algorithm
    stop_condition = 'maximum iteration'
    alpha = np.zeros([r])
    beta = np.zeros([n])
    residual_D = []
    residual_C = []
    wall_time = [timer()-start]
    for step in range(max_iter):
        
        # Update dictionary estimate
        temp = np.copy(D)
        for j in range(r):
            
            # compute step size
            Gj = np.tensordot(C[:, j]**2 / n, XtX, axes=1)
            alpha[j] = sl.norm(Gj, ord=2)**(-1) * step_size
            
            # proximal/gradient step
            D[j, :, :] = proj(D[j, :, :] - alpha[j] * grad_D(D, C, j, G=Gj))
        delta_D = D - temp
            
        # Update coefficient estimate
        temp = np.copy(C)
        for i in range(n):
            
            # compute step size
            Gi = gram(D, lambda x, y : inner_prod(x, np.dot(XtX[i], y)))
            beta[i] = n * sl.norm(Gi, ord=2)**(-1) * step_size
            
            # proximal/gradient step
            C[i, :] = prox_coef(C[i, :] - beta[i] * grad_C(D, C, i, G=Gi), 
                                     mu * beta[i] / n)
        delta_C = C - temp
        
        # Add current estimates to path
        if return_path:
            D_path.append(np.copy(D))
            C_path.append(np.copy(C))
        
        # Compute residuals
        """( (1/r) \sum_j (\|dD_j\|/alpha_j)^2 / (p*d^2) )^(1/2)"""
        residual_D.append(sl.norm(sl.norm(delta_D, ord='fro', axis=(1, 2))/alpha) 
                          / (r**(1/2) * p**(1/2) * d))          
        """( (1/n) \sum_i (\|dC_i\|/beta_i)^2 / r )^(1/2)"""
        residual_C.append(sl.norm(sl.norm(delta_C)/beta) / (n**(1/2) * r**(1/2)))
        
        # Compute wall time
        wall_time.append(timer()-start)
        
        # Check stopping condition
        if ( step > 0 and residual_D[-1] < tol * residual_D[0] 
            and residual_C[-1] < tol * residual_C[0] ):
            stop_condition = 'relative tolerance'
            break
        
    if verbose:
        end = timer()
        duration = end - start
        print('*Solver: Proximal Alternating Linearized Minimization')
        print('*Stopping condition: ' + stop_condition)
        print('*Iterations: ' + str(step))
        print('*Duration: ' + str(duration) + 's')
    
    if return_path:
        return D_path, C_path, residual_D, residual_C, stop_condition, wall_time
    else:
        return D, C, residual_C, residual_D, stop_condition, wall_time
    
def likelihood(YtY, XtX, XtY, D, C, mu, coef_penalty_type):
    """
    Computes the negative log likelihood of the current dictionary and 
    coefficient estimates.
    
    inputs:
    YtY (list of arrays) - correlation of observations
    
    XtX (list of arrays) - sample autocorrelation of observations
    
    XtY (list of arrays) - sample autocorrelation of observations
    
    D (list arrays) - estimates of autoregressive atoms
    
    C (array) - estimate of coefficients; length of YtY x length of D
    
    mu (scalar) - penalty parameter
    
    coef_penalty_type (string) - coefficient penalty of objective; 
    {None, l0, l1}
    
    outputs:
    likelihood (scalar) - negative log likelihood of estimates
    """
    
    n = len(XtX)
    r, _, _ = D.shape
    gram_C = [gram(D, lambda x, y : inner_prod(x, np.dot(XtX[i], y))) for i in range(n)]
    likelihood = 0.5 * ( np.mean(YtY) 
                        - 2 * np.sum([np.mean([C[i, j]*inner_prod(XtY[i], D[j]) for i in range(n)]) for j in range(r)])
                        + np.mean(np.matmul(np.expand_dims(C, 1), np.matmul(gram_C, np.expand_dims(C, 2)))) )
    if coef_penalty_type == 'l0':
        likelihood += mu * np.count_nonzero(C[:]) / n
    elif coef_penalty_type == 'l1':
        likelihood += mu * sl.norm(C[:], ord=1) / n
    return likelihood
