#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM Copyright (C) 2019  Addison Bohannon
"""

import numpy as np
import scipy.linalg as sl
from alm.utility import circulant_matrix, initialize_autoregressive_components
from experiments.utility import load_isruc_data, save_results

SUBJ = 8
"""The typical orders for multichannel EEG datasets vary in range from 3 to 9 (Kaminski and Liang 2005)"""
# {4, 6, 8, 10, 12}
MODEL_ORDER = range(4, 14, 2)
# {2, 6, 10, 14, 18}
NUM_COMPS = range(2, 20, 4)
PENALTY_PARAM = 1e-1
NUM_STARTS = 1
MAX_ITER = int(5e0)
TOL = 1e-3

def expectation_maximization(X, Y, num_comps, max_iter, tol):
    
    # Initialize algorithm
    num_obs, obs_len, signal_dim = Y.shape
    _, _, model_ord_by_signal_dim = X.shape
    model_ord = int(model_ord_by_signal_dim/signal_dim)
    tau = np.zeros([num_obs, obs_len, num_comps])
    mixing_coef = np.random.rand(num_obs, num_comps)
    mixing_coef = mixing_coef / np.sum(mixing_coef, axis=1, keepdims=True)
    ar_coef = initialize_autoregressive_components(num_comps, model_ord, signal_dim)
    residual = []
    end_cond = 'maximum iteration'

    # Expectation-Maximization (EM) Algorithm
    for iteration in range(max_iter):
    
        # Expectation
        for j in range(num_comps):
            tau[:, :, j] = np.expand_dims(mixing_coef[:, j], axis=1) * np.exp(-0.5 * sl.norm(Y-np.matmul(X, ar_coef[j]), axis=2))
        tau = tau / np.sum(tau, axis=2, keepdims=True)
        
        # Maximization
        mixing_coef_prev = np.copy(mixing_coef)
        ar_coef_prev = np.copy(ar_coef)
        mixing_coef = np.mean(tau, axis=1)
        for j in range(num_comps):
            tauX = np.expand_dims(tau[:, :, j], axis=2) * X
            ar_coef[j] = sl.solve(np.tensordot(tauX, X, axes=((0, 1), (0, 1))), 
                                  np.tensordot(tauX, Y, axes=((0, 1), (0, 1))), assume_a='pos')
            
        residual.append(np.array([sl.norm(ar_coef-ar_coef_prev), sl.norm(mixing_coef-mixing_coef_prev)]))
            
        # Check convergence
        if iteration > 0 and np.all(residual[iteration] < tol*residual[0]):
            end_cond = 'relative tolerance'
            break
        
    # Compute weighted negative log likelihood (Font, et al., 2007)
    for j in range(num_comps):
        tau[:, :, j] = np.expand_dims(mixing_coef[:, j], axis=1) * np.exp(-0.5 * sl.norm(Y-np.matmul(X, ar_coef[j]), axis=2))
    w_nll = - np.sum(np.log(np.sum(tau, axis=2)))
    
    return ar_coef, mixing_coef, w_nll, residual, end_cond

def fit(obs, model_ord, num_comps, num_starts, max_iter, tol):
    """
    Fit MVAR model to observation using EM algorithm (Fong, et al., 2007).
    :param obs: list of obs_len x signal_dim numpy array
    :param model_ord: positive integer
    :param num_comps: positive integer
    :param num_starts: positive integer
    :return ar_comps: [list of] num_comps x model_ord*signal_dim x signal_dim numpy array
    :return mixing_coef: [list of] num_observations x num_comps numpy array  
    """

    # Organize observations
    num_obs, obs_len, signal_dim = obs.shape
    obs_len -= model_ord
    X, Y = np.zeros([num_obs, obs_len, model_ord*signal_dim]), np.zeros([num_obs, obs_len, signal_dim])
    for i in range(num_obs):
        X[i], Y[i] = circulant_matrix(obs[i], model_ord)
        
    # Fit MVAR model
    ar_comps, mixing_coef, w_nll = np.zeros([num_starts, num_comps, model_ord*signal_dim, signal_dim]), \
        np.zeros([num_starts, num_obs, num_comps]), np.zeros([num_starts])
    for start in range(num_starts):
        print('Start: ' + str(start))
        ar_comps[start], mixing_coef[start], w_nll[start], _, _ = expectation_maximization(X, Y, num_comps, max_iter, tol)
        
    opt = 0
    for start in range(1, num_starts):
        if w_nll[start] < w_nll[opt]:
            opt = start
            
    return ar_comps[opt], mixing_coef[opt], w_nll[opt]



data, _ = load_isruc_data(SUBJ)
num_obs, obs_len, sig_dim = data.shape
nll = np.zeros([len(MODEL_ORDER), len(NUM_COMPS)])
num_params = np.zeros_like(nll)
num_params_wcoef = np.zeros_like(nll)
for i, model_ord in enumerate(MODEL_ORDER):
    for j, num_comps in enumerate(NUM_COMPS):
        num_params[i, j] = model_ord * num_comps * sig_dim**2
        num_params_wcoef[i, j] = num_params[i, j] + num_obs * num_comps 
        ar_comps, mixing_coef, nll[i, j] = fit(data, model_ord, num_comps, NUM_STARTS, MAX_ITER, TOL)
""" aic: -2 * log(L) + 2 * k """
aic = 2 * (num_obs * obs_len * nll + num_params)
aic_wcoef = 2 * (num_obs * obs_len * nll + num_params_wcoef)
""" bic: -2 * log(L) + log(n) * k """
bic = 2 * num_obs * obs_len * nll + (np.log(num_obs * obs_len)) * num_params
bic_wcoef = 2 * num_obs * obs_len * nll + (np.log(num_obs * obs_len)) * num_params_wcoef
save_results([nll, num_params, num_params_wcoef, aic, aic_wcoef, bic, bic_wcoef], 'model_selection-mvar.pickle')
