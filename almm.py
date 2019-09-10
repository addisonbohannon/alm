#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 1 May 19
"""

# Import required libraries
from itertools import product
import numpy as np
import scipy.linalg as sl
from almm.utility import train_val_split
from almm.solver import fit_coefs, likelihood
from almm.timeseries import Timeseries

# Solver class
class Almm:
    
    # Class constructor
    def __init__(self, coef_penalty_type='l1', step_size=1e-1, tol=1e-4, 
                 max_iter=int(2.5e3), solver='palm', verbose=False):
        """
        Class constructor for ALMM solver.
        
        inputs:
        coef penalty type (string) - Penalty applied to coefficients to enforce
        sparsity; options include {None, 'l0', 'l1' (default)}
        
        step size (scalar) - Factor by which to extend the Lipschitz-based 
        step size; must be less than 1
        
        tolerance (float) - Tolerance to terminate iterative algorithm; must
        be positive
        
        maximum iterations (integer) - Maximum number of iterations for 
        algorithm
        
        solver (string) - Algorithm used to fit dictionary and coefficients,
        i.e. palm or alt_min
        
        verbose (boolean) - Whether to print progress during execution; used 
        for debugging
        """
        
        # Check arguments
        if coef_penalty_type is None:
            self.coef_penalty_type = coef_penalty_type
        elif coef_penalty_type == 'l0':
            self.coef_penalty_type = coef_penalty_type
        elif coef_penalty_type == 'l1':
            self.coef_penalty_type = coef_penalty_type
        else:
            raise ValueError(coef_penalty_type+' is not a valid penalty type, i.e. None, l0, or l1.')
        if isinstance(float(step_size), float) and step_size < 1:
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
        if solver == 'palm':
            from almm.solver import solver_palm
            self._fit = solver_palm
        elif solver == 'alt_min':
            from almm.solver import solver_alt_min
            self._fit = solver_alt_min
        else:
            raise ValueError('Solver is not a valid option: palm, alt_min.')
        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise TypeError('Verbose must be a boolean.')
        
    def fit(self, ts, p, r, mu, k=5, D_0=None, return_path=False, return_all=False):
        """
        Fit the autoregressive linear mixture model to observations.
        
        inputs:
        ts (list) - list of observations; should be m x d arrays
        
        p (integer) - model order; must be a positive integer less than the 
        observation length
        
        r (integer) - dictionary atoms; must be a positive integer
        
        mu (float) - penalty parameter; must be a positive float

        k (integer) - unique initializations of the solver; must be a
        positive integer
        
        D_0 (r x p*d * d array) - initial dictionary estimate (optional)
        
        starts (integer) - unique initializations of the solver; must be a
        positive integer
        
        return_path (boolean) - whether or not to return the path of
        dictionary and coefficient estimates

        return_all (boolean) - whether to return all dictionary and
        coefficient estimates or that of maximum likelihood
        
        outputs:
        D ([k x] r x p*d x d array) - dictionary estimate [if return_path=True]
        
        C ([k x] n x r array) - coefficient estimate [if return_path=True]
        
        L (list) - negative log likelihood of estimates

        T (list) - wall time of execution
        """
        
        if not isinstance(r, int) or r < 1:
            raise TypeError('Atoms (r) must be a positive integer.')
        if not isinstance(p, int) or p < 1:
            raise TypeError('Model order (p) must be a positive integer.')
        if not isinstance(mu, float) and mu < 0:
            raise ValueError('Penalty parameter (mu) must be a positive float.')
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer.')
        if D_0 is not None and k == 1:
            _, d = ts[0].shape
            if np.shape(D_0) != (r, p*d, d):
                raise ValueError('Initial dictionary estimate must be of shape [r, p*d, d].')
            elif np.any(sl.norm(D_0, axis=(1, 2), ord='fro') != 1):
                D_0 = np.array([D_i / sl.norm(D_i, ord='fro') for D_i in D_0])
                print('Initial dictionary estimate scaled to unit norm.')
        if not isinstance(return_path, bool):
            raise TypeError('Return path must be a boolean.')
        if not isinstance(return_all, bool):
            raise TypeError('Return all must be a boolean.')
        
        if self.verbose:
            print('-Formatting data...', end=" ", flush=True)
        YtY = []
        XtX = []
        XtY = []
        for ts_i in ts:
            ts_i = Timeseries(ts_i)
            YtY.append(ts_i.YtY(p))
            XtX.append(ts_i.XtX(p))
            XtY.append(ts_i.XtY(p))
        if self.verbose:
            print('Complete.')
            
        if self.verbose:
            print('-Fitting model...')
        D = []
        C = []
        T = []
        for ki in range(k):
            if self.verbose:
                print('--Start: ' + str(ki))
            D_i, C_i, res_D, res_C, stop_con, T_i = self._fit(XtX, XtY, p, r, mu, self.coef_penalty_type,  D_0=D_0,
                                                              max_iter=self.max_iter, step_size=self.step_size,
                                                              tol=self.tol, return_path=return_path,
                                                              verbose=self.verbose)
            D.append(D_i)
            C.append(C_i)
            T.append(T_i)
        if self.verbose:
            print('--Complete.')
        
        if self.verbose:
            print('-Computing likelihood...', end=" ", flush=True)
        L = []
        for D_i, C_i in zip(D, C):
            if return_path:
                L_i = [likelihood(YtY, XtX, XtY, Dis, Cis, mu, self.coef_penalty_type) for Dis, Cis in zip(D_i, C_i)]
            else:
                L_i = likelihood(YtY, XtX, XtY, D[-1], C[-1], mu, self.coef_penalty_type)
            L.append(L_i)
        if self.verbose:
            print('Complete.')

        if return_all:
            if (k > 1):
                return D, C, L, T
            else:
                return D[0], C[0], L[0], T[0]
        else:
            opt = 0
            for i, L_i in enumerate(L):
                if L_i[-1] < L_i[-1]:
                    opt = i
            return D[opt], C[opt], L[opt], T[opt]
        
    def fit_cv(self, ts, p=None, r=None, mu=None, k=5, val_pct=0.25, 
               return_path=False, return_all=False):
        """
        Fit ALMM model to observations for various values of model order,
        number of dictionary atoms, and penalty parameter. For each unique
        tuple of parameters, multiple (k-) models are fit.
        
        inputs:
        ts (list) - list of observations; should be m x d arrays
        
        p (list or integer) - model order; must be a positive integer or list
        of positive integers
        
        r (list or integer) - dictionary atoms; must be a positive integer or
        list of positive integers
        
        mu (list or float) - penalty parameter; must be a positive float or
        list of positive floats
        
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
        
        if isinstance(p, int):
            if p > 0:
                p = [p]
            else:
                raise ValueError('Model order must be a positive integer or list of positive integers.')
        elif isinstance(p, list):
            if not all([isinstance(i, int) and i > 0 for i in p]):
                raise ValueError('Model order must be a positive integer or list of positive integers.')
        elif p is None:
            p = [1]
        else:
            raise TypeError('Model order must be a positive integer or list of positive integers.')
        if isinstance(r, int):
            if r > 0:
                r = [r]
            else:
                raise ValueError('Dictionary atoms must be a positive integer or list of positive integers.')
        elif isinstance(r, list):
            if not all([isinstance(i, int) and i > 0 for i in r]):
                raise ValueError('Dictionary atoms must be a positive integer or list of positive integers.')
        elif r is None:
            r = [1]
        else:
            raise TypeError('Dictionary atoms must be a positive integer or list of positive integers.')
        if isinstance(mu, float):
            if mu >= 0:
                mu = [mu]
            else:
                raise ValueError('Penalty parameter must be a non-negative float or list of non-negative floats.')
        elif isinstance(mu, list):
            if not all([isinstance(i, float) and i >= 0 for i in mu]):
                raise ValueError('Penalty parameter must be a non-negative float or list of non-negative floats.')
        elif mu is None:
            mu = [0]
        else:
            raise TypeError('Penalty parameter must be a non-negative float or list of non-negative floats.')
        if not isinstance(k, int) or k < 1:
            raise TypeError('Starts (k) must be a positive integer.')
            
        if self.verbose:
            print('-Formatting and splitting data...', end=" ", flush=True)
        ts = [Timeseries(ts_i) for ts_i in ts]        
        train_idx, val_idx = train_val_split(len(ts), val_pct)
        ts_train = [ts[i] for i in train_idx]
        ts_val = [ts[i] for i in val_idx]
        if self.verbose:
            print('Complete.')
        
        if self.verbose:
            print('-Fitting model to training data...')
        D = []
        C = []
        Lv = []
        params = product(p, r, mu)
        for (p_i, r_i, mu_i) in params:
            print('--Parameters: p=' + str(p_i) + ', r=' + str(r_i) + ', mu=' + str(mu_i))
            # Fit dictionary to training observations for each set of parameters
            D_s, Cts, _ = self._fit_k(ts_train, p_i, r_i, mu_i, k=k,
                                      val_pct=val_pct, return_path=return_path, 
                                      return_all=False)
            D.append(D_s)
            print('-Complete.')
            
            # Prepare validation observations
            YtY_val = [ob.YtY(p_i) for ob in ts_val]
            XtX_val = [ob.XtX(p_i) for ob in ts_val]
            XtY_val = [ob.XtY(p_i) for ob in ts_val]
            
            # Fit coefficients to validation observation and compute negative 
            # log likelihood
            if self.verbose:
                print('-Fitting coefficients to validation data...', end=" ", 
                      flush=True)
            if return_path:
                Cvs = fit_coefs(XtX_val, XtY_val, D_s[-1], mu_i, 
                                self.coef_penalty_type)
                L_s = likelihood(YtY_val, XtX_val, XtY_val, D_s[-1], Cvs, mu_i, 
                                 self.coef_penalty_type)
            else:
                Cvs = fit_coefs(XtX_val, XtY_val, D_s, mu_i,
                                self.coef_penalty_type)
                L_s = likelihood(YtY_val, XtX_val, XtY_val, D_s, Cvs, mu_i, 
                                 self.coef_penalty_type)
            Lv.append(L_s)
            if self.verbose:
                print('Complete.')
            
            # Merge coefficient lists
            # TODO: Make this a list comprehension
            if self.verbose:
                print('--Merging training and validation coefficients...', 
                      end=" ", flush=True)
            C_s = [i for i in zip(train_idx, list(Cts))]
            C_s.extend([i for i in zip(val_idx, list(Cvs))])
            C_s.sort()
            Cs = np.array([c for _, c in C_s])
            C.append(Cs)
            if self.verbose:
                print('Complete.')

        params = [i for i in product(p, r, mu)]
        if return_all:
            return D, C, Lv, params
        else:
            opt = np.argmin(Lv)
            return D[opt], C[opt], Lv[opt], params[opt]
