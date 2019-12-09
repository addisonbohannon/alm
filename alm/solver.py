#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM: model fitting for instantaneous mixtures of network processes
Copyright (C) 2019  Addison Bohannon

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see
<http://www.gnu.org/licenses/>.
"""

from random import shuffle
import numpy as np
import scipy.linalg as sl
from alm.utility import component_corr_matrix, component_gram_matrix, coef_gram_matrix, coef_corr_matrix


EPS = 1e-8


def shrink(x, t):
    """
    Implements the proximal operator of the l1-norm (shrink operator)
    :param x: float
    :param t: float
    :return x: float
    """
    
    return np.sign(x) * np.maximum(np.abs(x)-t, 0)


def threshold(x, t):
    """
    Implements the proximal operator of the l0-norm (threshold operator).
    :param x: float
    :param t: float
    :return x: float
    """
    
    x[x**2 < 2*t] = 0

    return x

    
def project(z):
    """
    Projects onto the l2/F-sphere
    :param z: float
    :return z: float
    """
    
    return z / sl.norm(z[:])


def component_update_altmin(XtX, XtY, current_comps, current_coef, step_size):
    """
    Update all autoregressive components for fixed mixing coefficients
    :param XtX: num_observations x model_ord*signal_dim x model_ord*signal_dim numpy array
    :param XtY: num_observations x model_ord*signal_dim x signal_dim numpy array
    :param current_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :param current_coef: num_observations x num_comps numpy array
    :param step_size: float
    :return new_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :return change_in_component: num_comps x model_ord*signal_dim x signal_dim numpy array
    """

    num_comps, model_ord_by_signal_dim, signal_dim = current_comps.shape
    model_ord = int(model_ord_by_signal_dim/signal_dim)
    coef_gram = coef_gram_matrix(XtX, current_coef)
    a = np.zeros([num_comps * model_ord * signal_dim, num_comps * model_ord * signal_dim])
    for (i, j), coef_gram_ij in coef_gram.items():
        a[i * model_ord * signal_dim:(i + 1) * (model_ord * signal_dim),
          (j * model_ord * signal_dim):(j + 1) * (model_ord * signal_dim)] = coef_gram_ij
        if i != j:
            a[j * model_ord * signal_dim:(j + 1) * (model_ord * signal_dim),
              (i * model_ord * signal_dim):(i + 1) * (model_ord * signal_dim)] = coef_gram_ij.T
    coef_corr = coef_corr_matrix(XtY, current_coef)
    b = np.zeros([num_comps * model_ord * signal_dim, signal_dim])
    for j in range(num_comps):
        b[(j * model_ord * signal_dim):(j + 1) * model_ord * signal_dim, :] = coef_corr[j]
    new_comps = sl.solve(a + EPS * np.eye(num_comps * model_ord * signal_dim), b, assume_a='pos')
    new_comps = np.array([project(new_comps[(j * model_ord * signal_dim):(j + 1) * model_ord * signal_dim, :])
                              for j in range(num_comps)])

    return new_comps, new_comps - current_comps


def component_update_bcd(XtX, XtY, current_comps, current_coef, step_size):
    """
    Update all autoregressive components sequentially (random) for fixed mixing coefficients
    :param XtX: num_observations x model_ord*signal_dim x model_ord*signal_dim numpy array
    :param XtY: num_observations x model_ord*signal_dim x signal_dim numpy array
    :param current_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :param current_coef: num_observations x num_comps numpy array
    :param step_size: float
    :return new_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :return change_in_component: num_comps x model_ord*signal_dim x signal_dim numpy array
    """
    
    def randomly(seq):
        shuffled = list(seq)
        shuffle(shuffled)
        return iter(shuffled)

    num_comps, model_ord_by_signal_dim, signal_dim = current_comps.shape
    model_ord = int(model_ord_by_signal_dim / signal_dim)
    new_comps = np.copy(current_comps)
    coef_gram = coef_gram_matrix(XtX, current_coef)
    coef_corr = coef_corr_matrix(XtY, current_coef)
    for j in randomly(range(num_comps)):
        a = coef_gram[(j, j)]
        b = coef_corr[j]
        for l in np.setdiff1d(np.arange(num_comps), [j]):
            b -= np.dot(coef_gram[tuple(sorted((j, l)))], new_comps[l])
        new_comps[j] = project(sl.solve(a + EPS * np.eye(model_ord * signal_dim), b, assume_a='pos'))

    return new_comps, new_comps - current_comps


def component_update_palm(XtX, XtY, current_comps, current_coef, step_size):
    """
    Update all autoregressive components for fixed mixing coefficients
    :param XtX: num_observations x model_ord*signal_dim x model_ord*signal_dim numpy array
    :param XtY: num_observations x model_ord*signal_dim x signal_dim numpy array
    :param current_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :param current_coef: num_observations x num_comps numpy array
    :param step_size: float
    :return new_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :return change_in_component: num_comps x model_ord*signal_dim x signal_dim numpy array
    """

    num_comps, _, _ = current_comps.shape
    new_comps = np.copy(current_comps)
    coef_gram = coef_gram_matrix(XtX, current_coef)
    coef_corr = coef_corr_matrix(XtY, current_coef)

    def gradient(component_index):
        grad = - coef_corr[component_index]
        for l in range(num_comps):
            grad += np.dot(coef_gram[tuple(sorted((component_index, l)))], new_comps[l])
        return grad

    alpha = np.zeros([num_comps])
    for j in range(num_comps):
        alpha[j] = step_size / sl.norm(coef_gram[(j, j)], ord=2)
        new_comps[j] = project(current_comps[j] - alpha[j] * gradient(j))

    return new_comps, new_comps - current_comps

        
def coef_update(XtX, XtY, current_comps, current_coef, penalty_param, coef_penalty_type, step_size):
    """
    Fit mixing coefficients for fixed autoregressive components
    :param XtX: num_observations x model_ord*signal_dim x model_ord*signal_dim numpy array
    :param XtY: num_observations x model_ord*signal_dim x signal_dim numpy array
    :param current_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :param current_coef: num_observations x num_comps numpy array
    :param penalty_param: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :param step_size: float
    :return new_coef: num_observations x num_comps numpy array
    :return change_in_coef: num_observations x num_comps numpy array
    """
    
    if coef_penalty_type is None:
        def solve(a, b): return sl.solve(a, b, assume_a='pos')
    elif coef_penalty_type == 'l0':
        def solve(a, b): return penalized_ls_gram(a, b, threshold, penalty_param)
    elif coef_penalty_type == 'l1':
        def solve(a, b): return penalized_ls_gram(a, b, shrink, penalty_param)
    else:
        raise TypeError('Coefficient penalty type must be None, l0, or l1.')

    comp_gram = component_gram_matrix(XtX, current_comps)
    comp_corr = component_corr_matrix(XtY, current_comps)
    new_coef = np.array([solve(gram_i, corr_i) for gram_i, corr_i in zip(comp_gram, comp_corr)])

    return new_coef, new_coef - current_coef


def coef_update_palm(XtX, XtY, current_comps, current_coef, penalty_param, coef_penalty_type, step_size):
    """
    Implement one linearized proximal gradient step with respect to the mixing coefficients for fixed autoregressive
    components
    :param XtX: num_obs x model_ord*signal_dim x model_ord*signal_dim numpy array
    :param XtY: num_obs x model_ord*signal_dim x signal_dim numpy array
    :param current_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :param current_coef: num_obs x num_comps numpy array
    :param penalty_param: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :param step_size: float
    :return new_coef: num_obs x num_comps numpy array
    :return change_in_coef: num_obs x num_comps numpy array
    """

    if coef_penalty_type is None:
        def proximal_function(x, t): return x
    elif coef_penalty_type == 'l0':
        def proximal_function(x, t): return threshold(x, t)
    elif coef_penalty_type == 'l1':
        def proximal_function(x, t): return shrink(x, t)
    else:
        raise TypeError('Coefficient penalty type must be None, l0, or l1.')

    num_obs = len(XtX)
    comp_gram = component_gram_matrix(XtX, current_comps)
    comp_corr = component_corr_matrix(XtY, current_comps)
    gradient = (- comp_corr + np.squeeze(np.matmul(comp_gram, np.expand_dims(current_coef, 2)))) / num_obs
    beta = (num_obs * step_size) / np.array([sl.norm(comp_gram, ord=2) for comp_gram in comp_gram])
    new_coef = proximal_function(current_coef - np.multiply(np.expand_dims(beta, 1), gradient),
                                 np.expand_dims(beta * (penalty_param / num_obs), 1))

    return new_coef, new_coef - current_coef


def penalized_ls_gram(gram_matrix, cov, prox_fcn, penalty_param, max_iter=1e3, tol=1e-4):
    """
    Implements iterative solver for penalized least squares using precomputed Gram matrix
    :param gram_matrix: m x m numpy array
    :param cov: m x n numpy array
    :param prox_fcn: function
    :param penalty_param: float
    :param max_iter: integer
    :param tol: float
    :return m x n numpy array
    """

    def update_lagrange_parameter(penalty_param, pri_res, dual_res):
        if pri_res > 4 * dual_res:
            penalty_param *= 2
        else:
            penalty_param /= 2

        return penalty_param

    m = cov.shape[0]
    pri_var_1 = np.zeros_like(cov)
    pri_var_2 = np.zeros_like(cov)
    dual_var = np.zeros_like(pri_var_2)
    primal_residual = []
    dual_residual = []
    lagrange_param = 1e-4
    gram_factor = sl.cho_factor(gram_matrix + np.eye(m))
    for step in np.arange(int(max_iter)):
        pri_var_1 = sl.cho_solve(gram_factor, cov - dual_var + lagrange_param * pri_var_2)
        primal_variable_2_update = prox_fcn(pri_var_1 + (1 / lagrange_param) * dual_var, penalty_param / lagrange_param)
        dual_residual.append(lagrange_param * sl.norm(primal_variable_2_update-pri_var_2))
        pri_var_2 = primal_variable_2_update
        dual_var += lagrange_param * (pri_var_1 - pri_var_2)
        primal_residual.append(sl.norm(pri_var_1-pri_var_2))
        if (primal_residual[step] <= tol*np.maximum(sl.norm(pri_var_1), sl.norm(pri_var_2)) and
                dual_residual[step] <= tol*sl.norm(dual_var)):
            break
        lagrange_param = update_lagrange_parameter(lagrange_param, primal_residual[step], dual_residual[step])
        gram_factor = sl.cho_factor(gram_matrix + lagrange_param * np.eye(m))

    return pri_var_1

    
def negative_log_likelihood(YtY, XtX, XtY, ar_comps, mixing_coef, penalty_param, coef_penalty_type):
    """
    Computes the negative log likelihood for ALMM
    :param YtY: list
    :param XtX: num_obs x model_ord*signal_dim x model_ord*signal_dim numpy array
    :param XtY: num_obs x model_ord*signal_dim x signal_dim numpy array
    :param ar_comps: numpy array
    :param mixing_coef: numpy array
    :param penalty_param: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :return nll: float
    """
    
    num_obs = len(XtX)
    comp_gram = component_gram_matrix(XtX, ar_comps)
    comp_corr = component_corr_matrix(XtY, ar_comps)
    nll = 0.5 * (np.mean(YtY) - 2 * np.mean([np.dot(mixing_coef[i, :], comp_corr[i]) for i in range(num_obs)])
                 + np.mean(np.matmul(np.expand_dims(mixing_coef, 1), np.matmul(comp_gram,
                                                                               np.expand_dims(mixing_coef, 2)))))
    if coef_penalty_type == 'l0':
        nll += penalty_param * np.count_nonzero(mixing_coef[:]) / num_obs
    elif coef_penalty_type == 'l1':
        nll += penalty_param * sl.norm(mixing_coef[:], ord=1) / num_obs

    return nll
