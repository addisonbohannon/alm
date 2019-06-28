#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 27 Jun 19
"""

import numpy as np
from utility import inner_prod, ar_toep_op

# Timeseries observation class
class Timeseries:
    
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