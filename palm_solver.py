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
                 coef_penalty_type='l1', step_size=10, max_iter=int(2.5e3), 
                 tol=1e-4, return_path=False, likelihood_path=False,
                 starts=5):
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
        
        model order (integer) - Autoregressive model order; must be much less 
        than observation length for reasonable estimation
        
        atoms (integer) - Number of autoregressive components; must be much
        less than number of observations
        
        penalty parameter (scalar) - Relative weight of sparsity penalty; must 
        be positive
        
        coef penalty type (string) - Penalty applied to coefficients to enforce
        sparsity; options include {None, 'l0', 'l1' (default)}
        
        step size (scalar) - Factor by which to divide the Lipschitz-based 
        step size
        
        maximum iterations (integer) - Maximum number of iterations for 
        algorithm
        
        tolerance (float) - Tolerance to terminate iterative algorithm; must
        be positive
        
        return path (boolean) - Whether to record the iterative updates of
        the dictionary; memory intensive
        
        likelihood path (boolean) - Whether to record the likelihood at each
        step; computationally intensive
        
        starts (integer) - How many unique initializations to start
        the algorithm
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
        if isinstance(likelihood_path, bool):
            self.likelihood_path = likelihood_path
            if self.likelihood_path:
                self.return_path = True
        else:
            raise TypeError('likelihood_path must be True/False.')
        if isinstance(starts, int) and starts > 0:
            self.starts = starts
        else:
            raise TypeError('Starts must be a positive integer.')
        
        # Pre-compute re-used quantities
        self.XtX = self.m**(-1) * np.matmul(np.moveaxis(self.X, 1, 2), self.X)
        self.XtY = self.m**(-1) * np.matmul(np.moveaxis(self.X, 1, 2), self.Y)
        self.YtY = self.m**(-1) * np.mean(inner_prod(self.Y, self.Y))
        
        # Fit dictionary and coefficients
        self.D = []
        self.C = []
        self.likelihood = []
        self.stop_condition = []
        for start in range(self.starts):
            Di, Ci, D_residual_i, C_residual_i, stop_condition_i = self.fit()
            self.D.append(Di)
            self.C.append(Ci)
            if self.likelihood_path:
                Li = []
                for Dii, Cii in zip(Di, Ci):
                    Li.append(self.compute_likelihood(Dii, Cii))
                self.likelihood.append(Li)
            else:
                self.likelihood.append(self.compute_likelihood(Di[-1], Ci[-1]))
            self.stop_condition.append(stop_condition_i)
        self.D
        
        # Remove prox_coef method to allow opening class in Spyder
        del self.prox_coef
        
    def initialize_estimates(self):
        """
        Initializes the dictionary and coefficient estimates.
        
        outputs:
        D (r x p*d x d tensor) - initial dictionary estimate
        
        C (n x r tensor) - initial coefficient estimate
        """
        
        def coef_lstsq(D):
            """
            Fits the coefficients for the current dictionary estimate using an 
            unpenalized least squares estimator.
            
            inputs:
            D (r x p*d x d tensor) - initial dictionary estimate
            
            outputs:
            C (n x r tensor) - coefficients which minimize the least squares
            objective
            """
            
            C = np.zeros([self.n, self.r])
            for i in range(self.n):
                C[i, :] = sl.solve(gram(D, lambda x, y : inner_prod(x, np.dot(self.XtX[i], y))), 
                                        inner_prod(self.XtY[i, :, :], D), 
                                        assume_a='pos')
            return C
        
        D = nr.randn(self.r, self.p*self.d, self.d)
        for j in range(self.r):
            D[j, :, :] = prox_dict(D[j, :, :])
        C = coef_lstsq(D)
        return D, C
    
    def compute_likelihood(self, D, C, gram_C=None):
        """
        Computes the log likelihood of the current dictionary and coefficient 
        estimates.
        """
        
        likelihood = 0.5 * self.YtY
        if gram_C is None:
            gram_C = [gram(D, lambda x, y : inner_prod(x, np.dot(self.XtX[i], y))) for i in range(self.n)]
        likelihood += 0.5 * np.mean(np.matmul(np.expand_dims(C, 1), np.matmul(gram_C, np.expand_dims(C, 2))))
        likelihood -= np.sum([np.mean([C[i, j]*inner_prod(self.XtY[i], D[j]) for i in range(self.n)]) for j in range(self.r)])
        if self.coef_penalty_type == 'l0':
            likelihood += self.mu * np.count_nonzero(C[:])
        elif self.coef_penalty_type == 'l1':
            likelihood += self.mu * sl.norm(C[:], ord=1)
        return likelihood
    
    def fit(self):
        """
        Iterative algorithm for ALMM solver. Based on the PALM algorithm
        of Bolte, Sabach, and Teboulle Math. Program. Ser. A, 2014. Takes
        linearized proximal gradient steps with respect to each dictionary
        atom and coefficients in turn.
        
        outputs:
        D path ([k x] r x p*d x d array) - dictionary estimate [if 
        return_path=True]
        
        C path ([k x] n x r array) - coefficient estimate [if 
        return_path=True]
        
        residual D (k array) - residuals of dictionary update
        
        residual C (k array) - residuals of coefficient update
        
        stopping condition ({maximum iteration, relative tolerance}) -
        condition that terminated the iterative algorithm
        """
            
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
                G = np.tensordot(C[:, j]**2 / self.n, self.XtX, axes=1)
            grad = - np.tensordot(C[:, j] / self.n, self.XtY, axes=1)
            grad += np.dot(G, D[j, :, :])
            for l in np.setdiff1d(np.arange(self.r), [j]):
                grad += np.dot(np.tensordot(C[:, j]*C[:, l] / self.n, 
                                            self.XtX, axes=1), D[l, :, :])
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
                G = gram(D, lambda x, y : inner_prod(x, np.dot(self.XtX[i], y)))
            return - inner_prod(self.XtY[i, :, :], D) + np.dot(G, C[i, :].T)
        
        alpha = np.zeros([self.r])
        beta = np.zeros([self.n])
        residual_D = []
        residual_C = []
        
        # Initialize estimates of dictionary and coefficients
        D, C = self.initialize_estimates()
        if self.return_path:
            D_path = []
            D_path.append(np.copy(D))
            C_path = []
            C_path.append(np.copy(C))
        
        # Begin iterative algorithm
        stop_condition = 'maximum iteration'
        for step in range(self.max_iter):
            
            # Update dictionary estimate
            temp = np.copy(D)
            for j in range(self.r):
                
                # compute step size
                Gj = np.tensordot(C[:, j]**2 / self.n, self.XtX, axes=1)
                alpha[j] = sl.norm(Gj, ord=2)**(-1) / self.step_size
                
                # proximal/gradient step
                D[j, :, :] = prox_dict(D[j, :, :] - alpha[j] * grad_D(D, C, j, G=Gj))
            delta_D = D - temp
            
            # Add current dictionary estimate to path
            if self.return_path:
                D_path.append(np.copy(D))
                C_path.append(np.copy(C))
                
            # Update coefficient estimate
            temp = np.copy(C)
            for i in range(self.n):
                
                # compute step size
                Gi = gram(D, lambda x, y : inner_prod(x, np.dot(self.XtX[i], y)))
                beta[i] = sl.norm(Gi, ord=2)**(-1) / self.step_size
                
                # proximal/gradient step
                C[i, :] = self.prox_coef(C[i, :] - beta[i] * grad_C(D, C, i, G=Gi), 
                                         self.mu * beta[i])
            delta_C = C - temp
            
            # Compute residuals
            """( (1/r) \sum_j (\|dD_j\|/alpha_j)^2 / (p*d^2)"""
            residual_D.append(np.mean(sl.norm(delta_D, ord='fro', axis=(1,2))
                              / ( alpha * self.p**(1/2) * self.d )))            
            """(1/n) \sum_i (\|dC_i\|/beta_i)^2 / r )^(1/2)"""
            residual_C.append(np.mean(sl.norm(delta_C, axis=1) 
                              / ( beta * self.r )))     
            
            # Check stopping condition
            if ( step > 0 and residual_D[-1] < self.tol * residual_D[0] 
                and residual_C[-1] < self.tol * residual_C[0] ):
                stop_condition = 'relative tolerance'
                break
        
        if self.return_path:
            return D_path, C_path, residual_D, residual_C, stop_condition
        else:
            return D, C, residual_C, residual_D, stop_condition