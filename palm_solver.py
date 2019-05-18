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
from utility import gram, inner_prod, ar_toep_op

# Utility functions
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
    
def prox_dict(z):
    """
    Implements proximal operator for dictionary: argmin_x (1/2)*\|x-z\|_F^2
    s.t. \|x\|_F = 1.
    
    inputs:
    dictionary (p*d x d tensor) - unnormalized dictionary atom
    
    outputs:
    dictionary (p*d x d tensor) - normalized dictionary atom
    """
    
    return z / sl.norm(z, ord='fro')

# Solver class
class Almm:
    
    # Class constructor
    def __init__(self, observations, model_order, atoms, penalty_parameter, 
                 coef_penalty_type='l1', step_size=10, max_iter=int(2.5e2), 
                 tol=1e-6, return_path=False):
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
        
        coef penalty type (string) - Penalty applied to coefficients to enforce
        sparsity; options include {None, 'l0', 'l1' (default)}
        
        penalty parameter (scalar) - Relative weight of sparsity penalty; must 
        be positive
        
        max_iter (integer) - Maximum number of iterations for algorithm
        
        tolerance (float) - Tolerance to terminate iterative algorithm; must
        be positive
        
        return_path (boolean) - Whether to record the iterative updates of
        the dictionary; memory intensive
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
        if isinstance(float(step_size), float) and step_size > 1:
            self.step_size = float(step_size)
        else:
            raise ValueError('Step size must be a float greater than 1.')
        if isinstance(max_iter, int) and max_iter > 0:
            self.max_iter = max_iter
        else:
            raise TypeError('Max iteration must be a positive integer.')
        if isinstance(tol, float) and tol > 0:
            self.tol = tol
        else:
            raise ValueError('Tolerance must be a positive float.')
        if isinstance(return_path, bool):
            self.return_path = return_path
        else:
            raise TypeError('return_path must be True/False.')
        
        # Pre-compute re-used quantities
        self.XtX = self.m**(-1) * np.matmul(np.moveaxis(self.X, 1, 2), self.X)
        self.XtY = self.m**(-1) * np.matmul(np.moveaxis(self.X, 1, 2), self.Y)
        self.YtY = self.m**(-1) * np.mean(inner_prod(self.Y, self.Y))
        
        # Initialize estimates of dictionary and coefficients
        self.initialize_estimates()
        if self.return_path:
            self.D_path = []
            self.D_path.append(np.copy(self.D))
        
        # Fit dictionary and coefficients
        self.alpha = np.zeros([self.max_iter])
        self.beta = np.zeros_like(self.alpha)
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
            self.D[j, :, :] = prox_dict(self.D[j, :, :])
        self.C = np.zeros([self.n, self.r])
        self.coef_lstsq()
        
    def coef_lstsq(self):
        """
        Fits the coefficients for the current dictionary estimate using an 
        unpenalized least squares estimator.
        """
        
        for i in range(self.n):
            self.C[i, :] = sl.solve(gram(self.D, lambda x, y : inner_prod(x, np.dot(self.XtX[i], y))), 
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
                # TODO: re-use matrix computation in step size and gradient step
                # compute step size
                self.alpha[step] = sl.norm(np.tensordot(self.C[:, j]**2 / self.n, 
                                                        self.XtX, axes=1), 
                                                        ord=2)**(-1) / self.step_size
                # proximal/gradient step
                self.D[j, :, :] = prox_dict(self.D[j, :, :] - self.alpha[step] * self.grad_D(j))
            delta_D = self.D - temp
            temp = np.copy(self.C)
            for i in range(self.n):
                # compute step size
                G = gram(self.D, lambda x, y : inner_prod(x, np.dot(self.XtX[i], y)))
                self.beta[step] = sl.norm(G, ord=2)**(-1) / self.step_size
                # proximal/gradient step
                self.C[i, :] = self.prox_coef(self.C[i, :] - self.beta[step] * self.grad_C(i, G=G), 
                                              self.mu * self.beta[step])
            delta_C = self.C - temp
            self.residual[step] = np.sqrt(np.sum(np.square(delta_D)/self.alpha[step]) 
                                          + np.sum(np.square(delta_C/self.beta[step])))
            self.add_likelihood(step)
            if self.return_path:
                self.D_path.append(np.copy(self.D))
            if step > 0 and self.residual[step] < self.tol * self.residual[0]:
                self.stop_condition = 'relative tolerance'
                self.alpha = self.alpha[:(step+1)]
                self.beta = self.beta[:(step+1)]
                self.residual = self.residual[:(step+1)]
                self.likelihood = self.likelihood[:(step+1)]
                break
            elif step > 0 and self.residual[step] < self.tol:
                self.stop_condition = 'absolute tolerance'
                self.alpha = self.alpha[:(step+1)]
                self.beta = self.beta[:(step+1)]
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
        
        
    def grad_C(self, i, G=None):
        """
        Computes the gradient of the ith coefficient vector for the current
        values of the dictionary elements.
        
        inputs:
        i ({1,...,n}) - index of the observation
        """
        
        if G is None:
            G = gram(self.D, lambda x, y : inner_prod(x, np.dot(self.XtX[i], y)))
        return - inner_prod(self.XtY[i, :, :], self.D) + np.dot(G, self.C[i, :].T)
    
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
            self.likelihood[step] += self.mu * np.count_nonzero(self.C)
        elif self.coef_penalty_type == 'l1':
            self.likelihood[step] += self.mu * sl.norm(self.C, ord=1)
        
        