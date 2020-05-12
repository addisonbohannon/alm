#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM Copyright (C) 2019  Addison Bohannon
"""

import numpy as np
import scipy.linalg as sl
from alm.utility import circulant_matrix, initialize_autoregressive_components
from experiments.utility import load_isruc_data, save_results

MAX_ITER = int(1e3)
TOL = 1e-3
NUM_STARTS = 5
SUBJS = [8]
MODEL_ORD = 4
NUM_COMPS= 10

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
    
    return ar_coef, mixing_coef, residual, end_cond

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
    ar_comps, mixing_coef = np.zeros([num_starts, num_comps, model_ord*signal_dim, signal_dim]), \
        np.zeros([num_starts, num_obs, num_comps])
    for start in range(num_starts):
        ar_comps[start], mixing_coef[start], _, _ = expectation_maximization(X, Y, num_comps, max_iter, tol)
        
    return np.squeeze(ar_comps), np.squeeze(mixing_coef)

for subj in SUBJS:
    print(subj)
    data, labels = load_isruc_data(subj)
    ar_comps, mixing_coef = fit(data, MODEL_ORD, NUM_COMPS, NUM_STARTS, MAX_ITER, TOL)
    save_results([ar_comps, mixing_coef, labels], 'S' + str(subj) + '-mvar.pickle')
