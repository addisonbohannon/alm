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
        
    def fit(self, ts, p, r, mu, return_path=False):
        """
        Fit the autoregressive linear mixture model to observations.
        
        inputs:
        ts (list) - list of observations; should be m x d arrays
        
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
        D, C, res_D, res_C, stop_con = self._fit(XtX, XtY, p, r, mu, 
                                                 self.coef_penalty_type, 
                                                 max_iter=self.max_iter, 
                                                 step_size=self.step_size, 
                                                 tol=self.tol, 
                                                 return_path=return_path,
                                                 verbose=self.verbose)
        if self.verbose:
            print('-Complete.')
        
        if self.verbose:
            print('-Computing likelihood...', end=" ", flush=True)
        if return_path:
            L = [likelihood(YtY, XtX, XtY, Di, Ci, mu, self.coef_penalty_type) 
                 for Di, Ci in zip(D, C)]
        else:
            L = likelihood(YtY, XtX, XtY, D[-1], C[-1], mu, 
                           self.coef_penalty_type)
        if self.verbose:
            print('Complete.')
            
        return D, C, L
            
    def fit_k(self, ts, p, r, mu, k=5, val_pct=0.25, return_path=False, 
              return_all=False):
        """
        Fit ALMM model to observations using multiple (k-) starts to address 
        the nonconvexity of the objective.
        
        inputs:
        ts (list) - list of observations; shold be m x d arrays
        
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
            
        ts = [Timeseries(ts_i) for ts_i in ts]
        
        return self._fit_k(ts, p, r, mu, k=k, val_pct=val_pct,
                           return_path=return_path, return_all=return_all)
        
    def fit_cv(self, ts, p=None, r=None, mu=None, k=5, val_pct=0.25, 
               return_path=False, return_all=False):
        """
        Fit ALMM model to observations for various values of model order,
        number of dictionary atoms, and penalty parameter. For each unique
        tuple of parameters, multiple (k-) models are fit.
        
        inputs:
        ts (list) - list of observations; shold be m x d arrays
        
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
        
    def _fit_k(self, ts, p, r, mu, k=5, val_pct=0.25, return_path=False, 
               return_all=False):
        """
        Fit ALMM model to observations using multiple (k-) starts to address 
        the nonconvexity of the objective. Internal function to the solver
        class so it requires no explicit error handling.
        
        inputs:
        ts (list) - list of timeseries objects
        
        p (integer) - model order; must be a positive integer less than the 
        timeseries length
        
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
        
        if self.verbose:
            print('--Formatting and splitting data...', end=" ", flush=True)
        YtY = [ts_i.YtY(p) for ts_i in ts]
        XtX = [ts_i.XtX(p) for ts_i in ts]
        XtY = [ts_i.XtY(p) for ts_i in ts]
        
        # split observations into training and validation
        train_idx, val_idx = train_val_split(len(ts), val_pct)
        YtY_val = [YtY[i] for i in val_idx]
        XtX_train = [XtX[i] for i in train_idx]
        XtX_val = [XtX[i] for i in val_idx]
        XtY_train = [XtY[i] for i in train_idx]
        XtY_val = [XtY[i] for i in val_idx]
        if self.verbose:
            print('Complete.')
        
        # Fit dictionary to training observations with unique initialization
        if self.verbose:
            print('--Fitting model to training data...')
        D = []
        C_train = []
        for ki in range(k):
            if self.verbose:
                print('--Start: ' + str(ki))
            D_s, C_s, _, _, _ = self._fit(XtX_train, XtY_train, p, r, mu, 
                                          self.coef_penalty_type,
                                          max_iter=self.max_iter,
                                          step_size=self.step_size, 
                                          tol=self.tol, 
                                          return_path=return_path,
                                          verbose=self.verbose)
            D.append(D_s)
            if return_path:
                C_train.append(C_s[-1])
            else:
                C_train.append(C_s)
        if self.verbose:
            print('--Complete.')
            
        # Fit coefficients to validation observation and compute negative log 
        # likelihood
        if self.verbose:
            print('--Fitting coefficients to validation data...', end=" ", 
                  flush=True)
        C_val = []
        Lv = []
        for D_s in D:
            if return_path:
                C_s = fit_coefs(XtX_val, XtY_val, D_s[-1], mu, 
                                self.coef_penalty_type)
                L_s = likelihood(YtY_val, XtX_val, XtY_val, D_s[-1], C_s, mu, 
                                 self.coef_penalty_type)
            else:
                C_s = fit_coefs(XtX_val, XtY_val, D_s, mu, 
                                self.coef_penalty_type)
                L_s = likelihood(YtY_val, XtX_val, XtY_val, D_s, C_s, mu, 
                                 self.coef_penalty_type)
            C_val.append(C_s)
            Lv.append(L_s)
        if self.verbose:
            print('Complete.')
            
        # Merge coefficient lists
        # TODO: Make this a list comprehension
        if self.verbose:
            print('--Merging training and validation coefficients...', end=" ", 
                  flush=True)
        C = []
        for Cts, Cvs in zip(C_train, C_val):
            C_s = [i for i in zip(train_idx, list(Cts))]
            C_s.extend([i for i in zip(val_idx, list(Cvs))])
            C_s.sort()
            Cs = np.array([c for _, c in C_s])
            C.append(Cs)
        if self.verbose:
            print('Complete.')
            
        if return_all:
            return D, C, Lv
        else:
            opt = np.argmin(Lv)
            return D[opt], C[opt], Lv[opt]
