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
from sklearn.linear_model import lars_path
from utility import gram, inner_prod, ar_toep_op, train_val_split

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

# Autoregressive observation class
class Observation:
    
    # Class constructor
    def __init__(self, obs):
        """
        Class constructor for ALMM observation. Provides data manipulations 
        and cross-validation backend for the solver.
        
        inputs:
        obs (m x d array) - autoregressive observation
        """
        
        if len(np.shape(obs)) != 2:
            raise TypeError('Observation dimension invalid. Should be m x d.')
        self.x = obs
        self.m, self.d = obs.shape
        
    def Y(self, p):
        """
        Returns stacked observations for a maximum likelihood estimator, i.e.,
        Y = [y[p], ..., y[m]]^T.
        
        inputs:
        p (integer) - model order; must be a positive integer
        
        outputs:
        Y ((m-p) x d array) - stacked observations
        """
        
        if not isinstance(p, int) or p < 1:
            raise TypeError('Model order must be an integer.')
        
        return self.x[p:, :]
    
    def X(self, p):
        """
        Returns autoregressive operator of p-order lag, i.e.,
        [[x^T[p], ..., x^T[1]], ..., [x^T[m-1], ..., x^T[m-p]]]
        
        inputs:
        p (integer) - model order; must be a positive integer
        
        outputs:
        X ((m-p) x p*d array) - autoregressive operator
        """
        
        if not isinstance(p, int) or p < 1:
            raise TypeError('Model order must be an integer.')
        X, _ = ar_toep_op(self.x, p)
        return X
    
    def YtY(self, p):
        """
        Returns sample correlation, i.e., <Y, Y>_F / (m-p)
        
        inputs:
        p (integer) - model order; must be a positive integer
        
        outputs:
        YtY (float) - sample correlation of observation
            
        """
        
        if not isinstance(p, int) or p < 1:
            raise TypeError('Model order must be an integer.')
        _Y = self.Y(p)
        return inner_prod(_Y, _Y) / (self.m - p)
    
    def XtX(self, p):
        """
        Returns the sample autocorrelation of the autoregressive process
        
        inputs:
        p (integer) - model order; must be a positive integer
        
        outputs:
        XtX (p*d x p*d array) - sample autocorrelation
        """
        
        if not isinstance(p, int) or p < 1:
            raise TypeError('Model order must be an integer.')
        _X = self.X(p)
        return np.dot(_X.T, _X) / (self.m - p)
    
    def XtY(self, p):
        """
        Returns sample autocorrelation, i.e., X^T Y
        
        inputs:
        p (integer) - model order; must be a positive integer
        
        outputs:
        XtY (p*d x d array) - sample autocorrelation
        """
        
        if not isinstance(p, int) or p < 1:
            raise TypeError('Model order must be an integer.')
        _X = self.X(p)
        _Y = self.Y(p)
        return np.dot(_X.T, _Y) / (self.m - p)

# Solver class
class Almm:
    
    # Class constructor
    def __init__(self, coef_penalty_type='l1', step_size=10, tol=1e-4, 
                 max_iter=int(2.5e3)):
        """
        Class constructor for ALMM solver.
        
        inputs:
        coef penalty type (string) - Penalty applied to coefficients to enforce
        sparsity; options include {None, 'l0', 'l1' (default)}
        
        step size (scalar) - Factor by which to divide the Lipschitz-based 
        step size
        
        tolerance (float) - Tolerance to terminate iterative algorithm; must
        be positive
        
        maximum iterations (integer) - Maximum number of iterations for 
        algorithm
        """
        
        # Check arguments
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
        if isinstance(float(step_size), float) and step_size > 1:
            self.step_size = float(step_size)
        else:
            raise ValueError('Step size must be a float greater than 1.')
        if isinstance(tol, float) and tol > 0:
            self.tol = tol
        else:
            raise ValueError('Tolerance must be a positive float.')
        if isinstance(max_iter, int) and max_iter > 0:
            self.max_iter = max_iter
        else:
            raise TypeError('Max iteration must be a positive integer.')
        
    def fit(self, obs, p, r, mu, return_path=False):
        """
        Fit the autoregressive linear mixture model to observations.
        
        inputs:
        obs (list) - list of observations; should be m x d arrays
        
        p (integer) - model order; must be a positive integer less than the 
        observation length
        
        r (integer) - dictionary atoms; must be a positive integer
        
        mu (float) - penalty parameter; must be a positive float
        
        starts (integer) - unique initializations of the solver; must be a
        positive integer
        
        return_path (boolean) - whether or not to return the path of
        dictionary and coefficient estimates
        
        outputs:
        D ([k x] r x p*d x d array) - dictionary estimate [if return_path=True]
        
        C ([k x] n x r array) - coefficient estimate [if return_path=True]
        
        L (scalar) - negative log likelihood of estimates
        """
        
        if not isinstance(r, int) or r < 1:
            raise TypeError('Atoms (r) must be a positive integer.')
        if not isinstance(p, int) or p < 1:
            raise TypeError('Model order (p) must be a positive integer.')
        if not isinstance(mu, float) and mu < 0:
            raise ValueError('Penalty parameter (mu) must be a positive float.')
        if not isinstance(return_path, bool):
            raise TypeError('Return path must be a boolean.')
        
        YtY = []
        XtX = []
        XtY = []
        for obs_i in obs:
            ob = Observation(obs_i)
            YtY.append(ob.YtY(p))
            XtX.append(ob.XtX(p))
            XtY.append(ob.XtY(p))
        
        D, C, res_D, res_C, stop_con = self._fit(np.array(XtX), np.array(XtY), 
                                                 p, r, mu, 
                                                 return_path=return_path)
        L = self.likelihood(np.array(YtY), np.array(XtX), np.array(XtY), 
                            D[-1], C[-1], mu)
        return D, C, L
            
    def fit_k(self, obs, p, r, mu, k=5, val_pct=0.25, return_path=False, 
              return_all=False):
        """
        Fit ALMM model to observations using multiple (k-) starts to address 
        the nonconvexity of the objective.
        
        inputs:
        obs (list) - list of observations; shold be m x d arrays
        
        p (integer) - model order; must be a positive integer less than the 
        observation length
        
        r (integer) - dictionary atoms; must be a positive integer
        
        mu (float) - penalty parameter; must be a positive float
        
        k (integer) - unique initializations of the solver; must be a
        positive integer
        
        val_pct (float) = percentage of observations to use for validation;
        must be between 0 and 1
        
        return_path (boolean) - whether or not to return the path of
        dictionary estimates; will not return path of coefficient estimates
        
        return_all (boolean) - whether to return all dictionary and
        coefficient estimates or that of maximum likelihood
        
        outputs:
        D (r x p*d x d array) - dictionary estimate
        
        C (n x r array) - coefficient estimate
        
        Lv (scalar) - negative log likelihood of estimates during validation
        """
        
        if not isinstance(r, int) or r < 1:
            raise TypeError('Atoms (r) must be a positive integer.')
        if not isinstance(p, int) or p < 1:
            raise TypeError('Model order (p) must be a positive integer.')
        if not isinstance(mu, float) and mu < 0:
            raise ValueError('Penalty parameter (mu) must be a positive float.')
        if not isinstance(k, int) or k < 1:
            raise TypeError('Starts (k) must be a positive integer.')
        if not isinstance(val_pct, float) or val_pct < 0 or val_pct > 1:
            raise TypeError('Validation percentage must be between 0 and 1.')
        if not isinstance(return_path, bool):
            raise TypeError('Return path must be a boolean.')
        if not isinstance(return_all, bool):
            raise TypeError('Return all must be a boolean.')
        
        
        YtY = []
        XtX = []
        XtY = []
        for obs_i in obs:
            ob = Observation(obs_i)
            YtY.append(ob.YtY(p))
            XtX.append(ob.XtX(p))
            XtY.append(ob.XtY(p))
        d = ob.d
        
        def lasso(Xy, XX):
            _, _, b = lars_path(np.zeros([1, p*d]), np.zeros([1]), Xy=Xy, 
                                Gram=XX, method='lasso', copy_Gram=False, 
                                alpha_min=mu, return_path=False)
            return b
        
        # split observations into training and validation
        train_idx, val_idx = train_val_split(len(obs), val_pct)
        YtY_val = [YtY[i] for i in val_idx]
        XtX_train = [XtX[i] for i in train_idx]
        XtX_val = [XtX[i] for i in val_idx]
        XtY_train = [XtY[i] for i in train_idx]
        XtY_val = [XtY[i] for i in val_idx]
        
        # Fit dictionary to training observations with unique initialization
        D = []
        C_train = []
        for _ in range(k):
            D_s, C_s, _, _, _ = self._fit(np.array(XtX_train), 
                                        np.array(XtY_train), p, r, mu, 
                                        return_path=return_path)
            D.append(D_s)
            C_train.append(C_s)
            
        if return_path:
            # Fit coefficients with LASSO estimator
            C_val = [np.array([lasso(inner_prod(XtY_i, D_s[-1]), 
                                     gram(D_s[-1], lambda x, y : inner_prod(x, np.dot(XtX_i, y)))) 
                                     for XtX_i, XtY_i in zip(XtX_val, XtY_val)]) 
                                     for D_s in D]
    
            # Calculate negative log likelihood of estimates
            Lv = [self.likelihood(YtY_val, XtX_val, XtY_val, D_s[-1], C_s, mu) 
                  for D_s, C_s in zip(D, C_val)]
            
            # Merge coefficient lists
            # TODO: Make this a list comprehension
            C = []
            for Cts, Cvs in zip(C_train, C_val):
                C_s = [i for i in zip(train_idx, list(Cts[-1]))]
                C_s.extend([i for i in zip(val_idx, list(Cvs))])
                C_s.sort()
                Cs = np.array([c for _, c in C_s])
                C.append(Cs)
        else:
            # Fit coefficients with LASSO estimator
            C_val = [np.array([lasso(inner_prod(XtY_i, D_s), 
                                     gram(D_s, lambda x, y : inner_prod(x, np.dot(XtX_i, y)))) 
                                     for XtX_i, XtY_i in zip(XtX_val, XtY_val)]) 
                                     for D_s in D]
    
            # Calculate negative log likelihood of estimates
            Lv = [self.likelihood(YtY_val, XtX_val, XtY_val, D_s, C_s, mu) 
                  for D_s, C_s in zip(D, C_val)]
            
            # Merge coefficient lists
            # TODO: Make this a list comprehension
            C = []
            for Cts, Cvs in zip(C_train, C_val):
                C_s = [i for i in zip(train_idx, list(Cts))]
                C_s.extend([i for i in zip(val_idx, list(Cvs))])
                C_s.sort()
                Cs = np.array([c for _, c in C_s])
                C.append(Cs)
            
        if return_all:
            return D, C, Lv
        else:
            opt = np.argmin(Lv)
            return D[opt], C[opt], Lv[opt]
        
    
    def _fit(self, XtX, XtY, p, r, mu, return_path=False):
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
        
        return_path (boolean) - whether or not to return the path of
        dictionary and coefficient estimates
        
        outputs:
        D ([k x] r x p*d x d array) - dictionary estimate [if return_path=True]
        
        C ([k x] n x r array) - coefficient estimate [if return_path=True]
        
        residual D (k array) - residuals of dictionary update
        
        residual C (k array) - residuals of coefficient update
        
        stopping condition ({maximum iteration, relative tolerance}) -
        condition that terminated the iterative algorithm
        """
        
        n, _, d = XtY.shape
        
        def initialize_estimates():
            """
            Initializes the dictionary and coefficient estimates.
            
            outputs:
            D (r x p*d x d tensor) - initial dictionary estimate
            
            C (n x r tensor) - initial coefficient estimate
            """
            
            # Initialize dictionary randomly; enforce unit norm
            D = nr.randn(r, p*d, d)
            for j in range(r):
                D[j, :, :] = prox_dict(D[j, :, :])
            # Initialize coefficients with unpenalized least squares
            C = np.zeros([n, r])
            for i in range(n):
                C[i, :] = sl.solve(gram(D, lambda x, y : inner_prod(x, np.dot(XtX[i], y))), 
                                        inner_prod(XtY[i, :, :], D), 
                                        assume_a='pos')
            return D, C
            
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
                grad += np.dot(np.tensordot(C[:, j]*C[:, l] / n, 
                                            XtX, axes=1), D[l, :, :])
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
            return - inner_prod(XtY[i, :, :], D) + np.dot(G, C[i, :].T)        
        
        # Initialize estimates of dictionary and coefficients
        D, C = initialize_estimates()
        if return_path:
            D_path = []
            D_path.append(np.copy(D))
            C_path = []
            C_path.append(np.copy(C))
        
        # Begin iterative algorithm
        stop_condition = 'maximum iteration'
        alpha = np.zeros([r])
        beta = np.zeros([n])
        residual_D = []
        residual_C = []
        for step in range(self.max_iter):
            
            # Update dictionary estimate
            temp = np.copy(D)
            for j in range(r):
                
                # compute step size
                Gj = np.tensordot(C[:, j]**2 / n, XtX, axes=1)
                alpha[j] = sl.norm(Gj, ord=2)**(-1) / self.step_size
                
                # proximal/gradient step
                D[j, :, :] = prox_dict(D[j, :, :] - alpha[j] * grad_D(D, C, j, G=Gj))
            delta_D = D - temp
            
            # Add current dictionary estimate to path
            if return_path:
                D_path.append(np.copy(D))
                C_path.append(np.copy(C))
                
            # Update coefficient estimate
            temp = np.copy(C)
            for i in range(n):
                
                # compute step size
                Gi = gram(D, lambda x, y : inner_prod(x, np.dot(XtX[i], y)))
                beta[i] = sl.norm(Gi, ord=2)**(-1) / self.step_size
                
                # proximal/gradient step
                C[i, :] = self.prox_coef(C[i, :] - beta[i] * grad_C(D, C, i, G=Gi), 
                                         mu * beta[i])
            delta_C = C - temp
            
            # Compute residuals
            """( (1/r) \sum_j (\|dD_j\|/alpha_j)^2 / (p*d^2)"""
            residual_D.append(np.mean(sl.norm(delta_D, ord='fro', axis=(1,2))
                              / ( alpha * p**(1/2) * d )))            
            """(1/n) \sum_i (\|dC_i\|/beta_i)^2 / r )^(1/2)"""
            residual_C.append(np.mean(sl.norm(delta_C, axis=1) 
                              / ( beta * r )))     
            
            # Check stopping condition
            if ( step > 0 and residual_D[-1] < self.tol * residual_D[0] 
                and residual_C[-1] < self.tol * residual_C[0] ):
                stop_condition = 'relative tolerance'
                break
        
        if return_path:
            return D_path, C_path, residual_D, residual_C, stop_condition
        else:
            return D, C, residual_C, residual_D, stop_condition
    
    def likelihood(self, YtY, XtX, XtY, D, C, mu):
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
        
        outputs:
        likelihood (scalar) - negative log likelihood of estimates
        """
        
        n = len(XtX)
        r, _, _ = D.shape
        gram_C = [gram(D, lambda x, y : inner_prod(x, np.dot(XtX[i], y))) for i in range(n)]
        likelihood = 0.5 * ( np.mean(YtY) 
                            - 2 * np.sum([np.mean([C[i, j]*inner_prod(XtY[i], D[j]) for i in range(n)]) for j in range(r)])
                            + np.mean(np.matmul(np.expand_dims(C, 1), np.matmul(gram_C, np.expand_dims(C, 2))))  )
        if self.coef_penalty_type == 'l0':
            likelihood += mu * np.count_nonzero(C[:])
        elif self.coef_penalty_type == 'l1':
            likelihood += mu * sl.norm(C[:], ord=1)
        return likelihood