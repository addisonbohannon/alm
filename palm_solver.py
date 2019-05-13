#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 1 May 2019
"""

# Import required libraries
import numpy as np
import numpy.random as nr
import scipy.linalg as sl
import scipy.fftpack as sf
from utility import ar_toep_op, stack_ar_coeffs, unstack_ar_coeffs

# Utility functions
def gram(X, ip):
    """
    Computes gram matrix G_ij = ip(X_i, X_j).
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
        
def depthwise_kron(a, b):
    """
    Computes Kronecker product in parallel along depth dimension of 
    tensors. Tensors must share the same depthwise dimension.
    
    inputs:
    tensor a (n x m x l tensor) - first tensor in tensor product
    tensor b (n x q x r tensor) - second tensor in tensor product
    
    outputs:
    tensor K (n x (m*q) x (l*r)) - result of depthwise tensor product
    """
    n, m, l = a.shape
    p, q, r = b.shape
    if n != p:
        raise ValueError('Dimension mismatch in depth of tensors.')
    K = np.zeros([n, m*q, l*r])
    for i in range(n):
        K[i, :, :] = np.kron(a[i, :, :], b[i, :, :])
    return K

def proj_op(D, sample_len):
    """
    Normalizes dictionary atom to have operator norm of one.
    
    inputs:
    dictionary ((p*d) x d tensor) - unnormalized dictionary atom
    
    outputs:
    dictionary ((p*d) x d tensor) - normalized dictionary atom
    """
    return D / np.max(sl.norm(sf.rfft(unstack_ar_coeffs(D), n=sample_len, axis=0), 
                              ord=2, axis=(1, 2)))

def proj_fro(D):
    """
    Normalizes dictionary atom to have Frobenius norm of one.
    
    inputs:
    dictionary ((p*d) x d tensor) - unnormalized dictionary atom
    
    outputs:
    dictionary ((p*d) x d tensor) - normalized dictionary atom
    """
    return D / sl.norm(D, ord='fro')

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

# Solver class
class Almm:
    
    # Class constructor
    def __init__(self, observations, model_order, atoms, penalty_parameter, 
                 dict_penalty_type='fro', coef_penalty_type='l1', alpha=1e-4, 
                 beta=1e-4, max_iter=int(2.5e2), tol=1e-4):
        """
        Class constructor for ALMM solver. Takes as arguments the observations, 
        desired autoregressive model order, number of atoms to fit, the
        sparsity penalty, maximum iterations, and tolerance. Pre-computes 
        re-used quantities and initializes the dictionary and coefficient 
        estimates.
        
        inputs:
        observations (n x m x d tensor) - Observations; first dimension 
        indexes the observation, second dimension indexes the time, and third
        dimension indexes the coordinate
        
        model_order (integer) - Autoregressive model order; must be much less 
        than observation length for reasonable estimation
        
        atoms (integer) - Number of autoregressive components; must be much
        less than number of observations
        
        dict penalty type (string) - Penalty applied to dictionary to enforce
        stability; options include {'fro', 'op'}
        
        coef penalty type (string) - Penalty applied to coefficients to enforce
        sparsity; options include {None, 'l0', 'l1' (default)}
        
        penalty parameter (scalar) - Relative weight of sparsity penalty; must 
        be positive
        
        alpha (scalar) - Step size for dictionary update; must be positive
        
        beta (scalar) - Step size for coefficient update; must be positive
        
        max_iter (integer) - Maximum number of iterations for algorithm
        
        tolerance (float) - Tolerance to terminate iterative algorithm; must
        be positive
        """
        
        # Check arguments
        if isinstance(model_order, int):
            self.p = model_order
        else:
            raise TypeError('Model order must be an integer.')
        if isinstance(atoms, int):
            self.r = atoms
        else:
            raise TypeError('Atoms must be an integer.')
        if len(np.shape(observations)) != 3:
            raise TypeError('Observation dimension invalid. Should be n x m x d.')
        else:
            self.n, self.m, self.d = observations.shape
            self.m -= self.p
            self.Y = np.zeros([self.n, self.m, self.d])
            self.X = np.zeros([self.n, self.m, self.p*self.d])
            for i in range(self.n):
                self.X[i, :, :], self.Y[i, :, :] = ar_toep_op(observations[i, :, :], 
                                                              self.p)
        if dict_penalty_type == 'fro':
            self.dict_penalty_type = dict_penalty_type
            self.prox_dict = proj_fro
        elif dict_penalty_type == 'op':
            self.dict_penalty_type = dict_penalty_type
            self.prox_dict = lambda D : proj_op(D, self.m)
        else:
            raise ValueError(dict_penalty_type+' is not a valid penalty type, i.e. fro or op.')
        if coef_penalty_type is None:
            self.coef_penalty_type = coef_penalty_type
            self.prox_coef = lambda x, t : x
        elif coef_penalty_type == 'l0':
            self.coef_penalty_type = coef_penalty_type
            self.prox_coef = threshold
        elif coef_penalty_type == 'l1':
            self.coef_penalty_type = coef_penalty_type
            self.prox_coef = shrink
        else:
            raise ValueError(coef_penalty_type+' is not a valid penalty type, i.e. None, l0, or l1.')
        if isinstance(penalty_parameter, float) and penalty_parameter > 0:
            self.mu = penalty_parameter
        else:
            raise ValueError('Penalty must be a positive float.')
        if isinstance(alpha, float) and alpha > 0:
            self.alpha = alpha
        else:
            raise ValueError('Alpha must be a positive float.')
        if isinstance(beta, float) and beta > 0:
            self.beta = beta
        else:
            raise ValueError('Beta must be a positive float.')
        if isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            raise TypeError('Max iteration must be an integer.')
        if isinstance(tol, float) and tol > 0:
            self.tol = tol
        else:
            raise ValueError('Tolerance must be a positive float.')
        
        # Pre-compute re-used quantities
        self.XtX = self.m**(-1) * np.matmul(np.moveaxis(self.X, 1, 2), self.X)
        self.XtY = self.m**(-1) * np.matmul(np.moveaxis(self.X, 1, 2), self.Y)
        self.YtY = self.m**(-1) * np.mean(inner_prod(self.Y, self.Y))
        
        # Initialize estimates of dictionary and coefficients
        self.initialize_estimates()
        
        # Fit dictionary and coefficients
        self.residual = np.zeros([self.max_iter])
        self.likelihood = np.zeros_like(self.residual)
        self.fit()
        
        # Remove prox_coef method to allow opening class in Spyder
        del self.prox_coef
        
    def initialize_estimates(self):
        """
        Initializes the dictionary and coefficient estimates.
        """
        
        self.D = nr.randn(self.r, self.p*self.d, self.d)
        for j in range(self.r):
            self.D[j, :, :] = self.prox_dict(self.D[j, :, :])
        self.C = np.zeros([self.n, self.r])
        self.coef_lstsq()
        
    def coef_lstsq(self):
        """
        Fits the coefficients for the current dictionary estimate using an 
        unpenalized least squares estimator.
        """
        
        for i in range(self.n):
            self.C[i, :] = sl.solve(gram(self.D, inner_prod), 
                                    inner_prod(self.XtY[i, :, :], self.D), 
                                    assume_a='pos')
    
    def fit(self):
        """
        Iterative algorithm for ALMM solver. Based on the PALM algorithm
        of Bolte, Sabach, and Teboulle Math. Program. Ser. A, 2014. Takes
        linearized proximal gradient steps with respect to each dictionary
        atom and coefficients in turn.
        """
        
        self.stop_condition = 'maximum iteration'
        for step in range(self.max_iter):
            temp = np.copy(self.D)
            for j in range(self.r):
                self.D[j, :, :] -= self.alpha * self.grad_D(j)
                self.D[j, :, :] = self.prox_dict(self.D[j, :, :])
            delta_D = self.D - temp
            temp = np.copy(self.C)
            for i in range(self.n):
                self.C[i, :] -= self.beta * self.grad_C(i)
                self.C[i, :] = self.prox_coef(self.C[i, :], self.mu*self.beta)
            delta_C = self.C - temp
            self.residual[step] = np.sqrt(np.sum(np.square(delta_D)) + np.sum(np.square(delta_C)))
            self.add_likelihood(step)
            if step > 0 and self.residual[step] < self.tol * self.residual[0]:
                self.stop_condition = 'relative tolerance'
                self.residual = self.residual[:(step+1)]
                self.likelihood = self.likelihood[:(step+1)]
                break
            elif step > 0 and self.residual[step] < self.tol:
                self.stop_condition = 'absolute tolerance'
                self.residual = self.residual[:(step+1)]
                self.likelihood = self.likelihood[:(step+1)]
                break
            
    def grad_D(self, j):
        """
        Computes the gradient of the jth dictionary element for the current 
        values of other dictionary elements and coefficients.
        
        inputs:
        j ({1,...,r}) - index of the dictionary atom
        """
            
        grad = - np.tensordot(self.C[:, j] / self.n, self.XtY, axes=1)
        for l in range(self.r):
            grad += np.dot(np.tensordot(self.C[:, j]*self.C[:, l] / self.n, 
                                     self.XtX, axes=1), self.D[l, :, :])
        return grad
        
    def grad_C(self, i):
        """
        Computes the gradient of the ith coefficient vector for the current
        values of the dictionary elements.
        
        inputs:
        i ({1,...,n}) - index of the observation
        """
        
        return - inner_prod(self.XtY[i, :, :], self.D) + np.dot(gram(self.D, inner_prod), self.C[i, :].T)
    
    def add_likelihood(self, step):
        """
        Computes the likelihood of the current dictionary and coefficient 
        estimate.
        
        inputs:
        step (integer) - current step to which the likelihood will be assigned
        """
        
        self.likelihood[step] += self.YtY
        self.likelihood[step] += np.mean([gram([self.C[i, j]*self.D[j] for j in range(self.r)], lambda d1, d2 : inner_prod(d1, np.matmul(self.XtX[i, :, :], d2))) for i in range(self.n)])
        self.likelihood[step] += 2 * np.sum([np.mean([self.C[i, j]*inner_prod(self.XtY, self.D[j]) for i in range(self.n)]) for j in range(self.r)])
        if self.coef_penalty_type == 'l0':
            self.likelihood[step] += np.count_nonzero(self.C)
        elif self.coef_penalty_type == 'l1':
            self.likelihood[step] += sl.norm(self.C, ord=1)
        
        