#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def component_update_altmin(XtX, XtY, current_component, current_coef, step_size):
    """
    Update all autoregressive components for fixed mixing coefficients
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param current_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param current_coef: num_observations x num_components numpy array
    :param step_size: float
    :return new_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :return change_in_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    """

    num_components, model_order_by_signal_dimension, signal_dimension = current_component.shape
    model_order = int(model_order_by_signal_dimension/signal_dimension)
    coef_gram = coef_gram_matrix(XtX, current_coef)
    a = np.zeros([num_components * model_order * signal_dimension, num_components * model_order * signal_dimension])
    for (i, j), coef_gram_ij in coef_gram.items():
        a[i * model_order * signal_dimension:(i + 1) * (model_order * signal_dimension),
          (j * model_order * signal_dimension):(j + 1) * (model_order * signal_dimension)] = coef_gram_ij
        if i != j:
            a[j * model_order * signal_dimension:(j + 1) * (model_order * signal_dimension),
              (i * model_order * signal_dimension):(i + 1) * (model_order * signal_dimension)] = coef_gram_ij.T
    coef_corr = coef_corr_matrix(XtY, current_coef)
    b = np.zeros([num_components * model_order * signal_dimension, signal_dimension])
    for j in range(num_components):
        b[(j * model_order * signal_dimension):(j + 1) * model_order * signal_dimension, :] = coef_corr[j]
    new_component = sl.solve(a + EPS * np.eye(num_components * model_order * signal_dimension), b, assume_a='pos')
    new_component = np.array([project(new_component[(j * model_order * signal_dimension):
                                                    (j + 1) * model_order * signal_dimension, :])
                              for j in range(num_components)])

    return new_component, new_component - current_component


def component_update_bcd(XtX, XtY, current_component, current_coef, step_size):
    """
    Update all autoregressive components sequentially (random) for fixed mixing coefficients
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param current_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param current_coef: num_observations x num_components numpy array
    :param step_size: float
    :return new_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :return change_in_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    """
    
    def randomly(seq):
        shuffled = list(seq)
        shuffle(shuffled)
        return iter(shuffled)

    num_components, model_order_by_signal_dimension, signal_dimension = current_component.shape
    model_order = int(model_order_by_signal_dimension / signal_dimension)
    new_component = np.copy(current_component)
    coef_gram = coef_gram_matrix(XtX, current_coef)
    coef_corr = coef_corr_matrix(XtY, current_coef)
    for j in randomly(range(num_components)):
        a = coef_gram[(j, j)]
        b = coef_corr[j]
        for l in np.setdiff1d(np.arange(num_components), [j]):
            b -= np.dot(coef_gram[tuple(sorted((j, l)))], new_component[l])
        new_component[j] = project(sl.solve(a + EPS * np.eye(model_order * signal_dimension), b, assume_a='pos'))

    return new_component, new_component - current_component


def component_update_palm(XtX, XtY, current_component, current_coef, step_size):
    """
    Update all autoregressive components for fixed mixing coefficients
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param current_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param current_coef: num_observations x num_components numpy array
    :param step_size: float
    :return new_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :return change_in_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    """

    num_components, _, _ = current_component.shape
    new_component = np.copy(current_component)
    coef_gram = coef_gram_matrix(XtX, current_coef)
    coef_corr = coef_corr_matrix(XtY, current_coef)

    def gradient(component_index):
        grad = - coef_corr[component_index]
        for l in range(num_components):
            grad += np.dot(coef_gram[tuple(sorted((component_index, l)))], new_component[l])
        return grad

    alpha = np.zeros([num_components])
    for j in range(num_components):
        alpha[j] = step_size / sl.norm(coef_gram[(j, j)], ord=2)
        new_component[j] = project(current_component[j] - alpha[j] * gradient(j))

    return new_component, new_component - current_component

        
def coef_update(XtX, XtY, current_component, current_coef, penalty_parameter, coef_penalty_type, step_size):
    """
    Fit mixing coefficients for fixed autoregressive components
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param current_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param current_coef: num_observations x num_components numpy array
    :param penalty_parameter: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :param step_size: float
    :return new_coef: num_observations x num_components numpy array
    :return change_in_coef: num_observations x num_components numpy array
    """
    
    if coef_penalty_type is None:
        def solve(a, b): return sl.solve(a, b, assume_a='pos')
    elif coef_penalty_type == 'l0':
        def solve(a, b): return penalized_ls_gram(a, b, threshold, penalty_parameter)
    elif coef_penalty_type == 'l1':
        def solve(a, b): return penalized_ls_gram(a, b, shrink, penalty_parameter)
    else:
        raise TypeError('Coefficient penalty type must be None, l0, or l1.')

    gram = component_gram_matrix(XtX, current_component)
    corr = [component_corr_matrix(XtY_i, current_component) for XtY_i in XtY]
    new_coef = np.array([solve(gram_i, corr_i) for gram_i, corr_i in zip(gram, corr)])

    return new_coef, new_coef - current_coef


def coef_update_palm(XtX, XtY, current_component, current_coef, penalty_parameter, coef_penalty_type, step_size):
    """
    Implement one linearized proximal gradient step with respect to the mixing coefficients for fixed autoregressive
    components
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param current_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param current_coef: num_observations x num_components numpy array
    :param penalty_parameter: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :param step_size: float
    :return new_coef: num_observations x num_components numpy array
    :return change_in_coef: num_observations x num_components numpy array
    """

    if coef_penalty_type is None:
        def proximal_function(x, t): return x
    elif coef_penalty_type == 'l0':
        def proximal_function(x, t): return threshold(x, t)
    elif coef_penalty_type == 'l1':
        def proximal_function(x, t): return shrink(x, t)
    else:
        raise TypeError('Coefficient penalty type must be None, l0, or l1.')

    num_observations = len(XtX)
    component_gram = component_gram_matrix(XtX, current_component)
    component_corr = [component_corr_matrix(XtY_i, current_component) for XtY_i in XtY]

    def gradient(observation_index):
        return (- component_corr[observation_index] + np.dot(component_gram[observation_index],
                                                             current_coef[observation_index, :].T)) / num_observations

    new_coef = np.copy(current_coef)
    beta = map(lambda x: num_observations * step_size / x, [sl.norm(comp_gram, ord=2) for comp_gram in component_gram])
    for i, beta_i in enumerate(beta):
        new_coef[i, :] = proximal_function(new_coef[i, :] - beta_i * gradient(i),
                                           penalty_parameter * beta_i / num_observations)

    return new_coef, new_coef - current_coef


def penalized_ls_gram(gram_matrix, covariance, proximal_function, penalty_parameter, max_iter=1e3, tol=1e-4):
    """
    Implements iterative solver for penalized least squares using precomputed Gram matrix
    :param gram_matrix: m x m numpy array
    :param covariance: m x n numpy array
    :param proximal_function: function
    :param penalty_parameter: float
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

    m = covariance.shape[0]
    primal_variable_1 = np.zeros_like(covariance)
    primal_variable_2 = np.zeros_like(covariance)
    dual_variable = np.zeros_like(primal_variable_2)
    primal_residual = []
    dual_residual = []
    lagrange_parameter = 1e-4
    gram_factor = sl.cho_factor(gram_matrix + np.eye(m))
    for step in np.arange(int(max_iter)):
        primal_variable_1 = sl.cho_solve(gram_factor,
                                         covariance - dual_variable + lagrange_parameter * primal_variable_2)
        primal_variable_2_update = proximal_function(primal_variable_1 + (1 / lagrange_parameter) * dual_variable,
                                                     penalty_parameter / lagrange_parameter)
        dual_residual.append(lagrange_parameter * sl.norm(primal_variable_2_update-primal_variable_2))
        primal_variable_2 = primal_variable_2_update
        dual_variable += lagrange_parameter * (primal_variable_1 - primal_variable_2)
        primal_residual.append(sl.norm(primal_variable_1-primal_variable_2))
        if (primal_residual[step] <= tol*np.maximum(sl.norm(primal_variable_1), sl.norm(primal_variable_2)) and
                dual_residual[step] <= tol*sl.norm(dual_variable)):
            break
        lagrange_parameter = update_lagrange_parameter(lagrange_parameter, primal_residual[step], dual_residual[step])
        gram_factor = sl.cho_factor(gram_matrix + lagrange_parameter * np.eye(m))

    return primal_variable_1

    
def negative_log_likelihood(YtY, XtX, XtY, component, coef, penalty_parameter, coef_penalty_type):
    """
    Computes the negative log likelihood for ALMM
    :param YtY: list
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param component: numpy array
    :param coef: numpy array
    :param penalty_parameter: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :return nll: float
    """
    
    num_observations = len(XtX)
    num_components, _, _ = component.shape
    gram = component_gram_matrix(XtX, component)
    corr = [component_corr_matrix(XtY_i, component) for XtY_i in XtY]
    nll = 0.5 * (np.mean(YtY) - 2 * np.mean([np.dot(coef[i, :], corr[i]) for i in range(num_observations)])
                 + np.mean(np.matmul(np.expand_dims(coef, 1), np.matmul(gram, np.expand_dims(coef, 2)))))
    if coef_penalty_type == 'l0':
        nll += penalty_parameter * np.count_nonzero(coef[:]) / num_observations
    elif coef_penalty_type == 'l1':
        nll += penalty_parameter * sl.norm(coef[:], ord=1) / num_observations

    return nll
