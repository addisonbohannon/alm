#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 27 Jun 19
"""

from timeit import default_timer as timer
import numpy as np
import numpy.random as nr
import scipy.linalg as sl
from almm.utility import gram_matrix, inner_product


def shrink(x, t):
    """
    Implements the proximal operator of the l1-norm (shrink operator)
    :param x: float
    :param t: float
    :return: x: float
    """
    
    return np.sign(x) * np.maximum(np.abs(x)-t, 0)


def threshold(x, t):
    """
    Implements the proximal operator of the l0-norm (threshold operator).
    :param x: float
    :param t: float
    :return: x: float
    """
    
    x[x**2 < 2*t] = 0

    return x

    
def project(z):
    """
    Projects onto the l2-ball
    :param z: float
    :return: z: float
    """
    
    return z / sl.norm(z[:])


def penalized_ls_gram(gram_matrix, covariance, proximal_function, penalty_parameter, max_iter=1e3, tol=1e-4):
    """
    Implements iterative solver for penalized least squares using precomputed Gram matrix
    :param gram_matrix: m x m array
    :param covariance: m x n array
    :param proximal_function: function
    :param penalty_parameter: float
    :param max_iter: integer
    :param tol: float
    :return: m x n array
    """

    def update_lagrange_parameter(penalty_param, pri_res, dual_res):
        """
        Implement adaptive penalty parameter in ADMM formulation
        :param penalty_param: float
        :param pri_res: float
        :param dual_res: float
        :return:
        """
        if pri_res > 4 * dual_res:
            penalty_param *= 2
        else:
            penalty_param /= 2

        return penalty_param
    
    m = len(covariance)
    primal_variable_2 = np.zeros_like(covariance)
    dual_variable = np.zeros_like(primal_variable_2)
    primal_residual = []
    dual_residual = []
    lagrange_parameter = 1e-4
    gram_factor = sl.cho_factor(gram_matrix + np.eye(m))
    for step in np.arange(int(max_iter)):
        primal_variable_1 = sl.cho_solve(gram_factor, covariance - dual_variable + lagrange_parameter * primal_variable_2)
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

        
def update_coef(XtX, XtY, component, penalty_parameter, coef_penalty_type):
    """
    Fit coefficients for fixed autoregressive components
    :param XtX: list
    :param XtY: list
    :param component: numpy array
    :param penalty_parameter: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :return: coef: numpy array
    """
    
    if coef_penalty_type is None:
        solve = lambda a, b: sl.solve(a, b, assume_a='pos')
    elif coef_penalty_type == 'l0':
        solve = lambda a, b: penalized_ls_gram(a, b, threshold, penalty_parameter)
    elif coef_penalty_type == 'l1':
        solve = lambda a, b: penalized_ls_gram(a, b, shrink, penalty_parameter)

    return np.array([solve(gram_matrix(component, lambda component_1, component_2: inner_product(component_1, np.dot(XtX_i, component_2))),
                           inner_product(XtY_i, component)) for XtX_i, XtY_i in zip(XtX, XtY)])


def solver_altmin(XtX, XtY, model_order, num_components, penalty_parameter, coef_penalty_type, component, max_iter=int(2.5e3),
                  step_size=1e-3, tol=1e-6, return_path=False, verbose=False):
    """
    Alternating minimization algorithm for ALMM solver.

    inputs:
    XtX (n x p*d x p*d array) - sample autocorrelation

    XtY (n x p*d x d array) - sample autocorrelation

    p (integer) - model order

    r (integer) - dictionary atoms

    mu (float) - penalty parameter

    coef_penalty_type (string) - coefficient penalty of objective;
    {None, l0, l1}

    D_0 (r x p*d * d array) - intial dictionary estimate (optional)

    maximum iterations (integer) - Maximum number of iterations for
    algorithm

    step size (scalar) - Factor by which to divide the Lipschitz-based
    step size

    tolerance (float) - Tolerance to terminate iterative algorithm; must
    be positive

    return_path (boolean) - whether or not to return the path of
    dictionary and coefficient estimates

    verbose (boolean) - whether or not to print progress during execution;
    used for debugging

    outputs:
    D ([k x] r x p*d x d array) - dictionary estimate [if return_path=True]

    C ([k x] n x r array) - coefficient estimate [if return_path=True]

    residual D (k array) - residuals of dictionary update

    residual C (k array) - residuals of coefficient update

    stopping condition ({maximum iteration, relative tolerance}) -
    condition that terminated the iterative algorithm
    """

    def compute_component_residual(component_diff):
        return sl.norm(component_diff[:]) / (num_components ** (1 / 2) * model_order ** (1 / 2) * signal_dimension)

    def compute_coef_residual(coef_diff):
        return sl.norm(coef_diff[:]) / (num_observations ** (1/2) * num_components ** (1 / 2))

    def stopping_condition(current_step, residual_1, residual_2):
        if current_step == 0:
            return False
        elif residual_1[-1] >= tol * residual_1[0]:
            return False
        elif residual_2[-1] >= tol * residual_2[0]:
            return False
        else:
            return True

    start_time = timer()
    num_observations = len(XtY)
    _, signal_dimension = XtY[0].shape
    mixing_coef = update_coef(XtX, XtY, component, penalty_parameter, coef_penalty_type)
    elapsed_time = [timer()-start_time]
    stop_condition = 'maximum iteration'
    if return_path:
        component_path, coef_path = [np.copy(component)], [np.copy(mixing_coef)]
    component_residual, coef_residual = [], []
    for step in range(max_iter):
        component, component_change = update_component()
        mixing_coef, coef_change = update_coef(XtX, XtY, component, penalty_parameter, coef_penalty_type)
        if return_path:
            component_path.append(np.copy(component))
            coef_path.append(np.copy(mixing_coef))
        component_residual.append(compute_component_residual(component_change))
        coef_residual.append(compute_coef_residual(coef_change))
        elapsed_time.append(timer()-start_time)
        if stopping_condition(step, component_residual, coef_residual):
            stop_condition = 'relative tolerance'
            break
    if verbose:
        duration = timer() - start_time
        print('*Solver: ' + solver)
        print('*Stopping condition: ' + stop_condition)
        print('*Iterations: ' + str(step))
        print('*Duration: ' + str(duration) + 's')

    if return_path:
        return component_path, coef_path, component_residual, coef_residual, stop_condition, elapsed_time
    else:
        return component, mixing_coef, coef_residual, component_residual, stop_condition, elapsed_time


def component_update_altmin(XtX, XtY, component_current, mixing_coef):
    num_components, model_order_by_signal_dimension, signal_dimension = component_current.shape
    model_order = int(model_order_by_signal_dimension/signal_dimension)
    ccXtX = {}
    triu_index = np.triu_indices(num_components)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        ccXtX[(i, j)] = np.tensordot(mixing_coef[:, i]*mixing_coef[:, j], XtX, axes=1)
    A = np.zeros([num_components * model_order * signal_dimension, num_components * model_order * signal_dimension])
    for (i, j) in zip(triu_index[0], triu_index[1]):
        A[i * model_order * signal_dimension:(i + 1) * (model_order * signal_dimension), (j * model_order * signal_dimension):(j + 1) * (model_order * signal_dimension)] = ccXtX[(i, j)]
    tril_index = np.tril_indices(num_components, k=-1)
    A[tril_index] = A.T[tril_index]
    b = np.zeros([num_components * model_order * signal_dimension, signal_dimension])
    for j in range(num_components):
        b[(j * model_order * signal_dimension):(j + 1) * model_order * signal_dimension, :] = np.tensordot(mixing_coef[:, j], XtY, axes=1)
    new_component = sl.solve(A, b, assume_a='pos')
    new_component = np.array([project(new_component[(j * model_order * signal_dimension):(j + 1) * model_order * signal_dimension, :]) for j in range(num_components)])

    return new_component, new_component - component_current


def component_update_bcd(XtX, XtY, current_component, mixing_coef):
    num_components, _, _ = current_component.shape
    new_component = np.zeros_like(current_component)
    ccXtX = {}
    triu_index = np.triu_indices(num_components)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        ccXtX[(i, j)] = np.tensordot(mixing_coef[:, i] * mixing_coef[:, j], XtX, axes=1)
    for j in range(num_components):
        Aj = ccXtX[(j, j)]
        bj = np.tensordot(mixing_coef[:, j], XtY, axes=1)
        for l in np.setdiff1d(np.arange(num_components), [j]):
            bj -= np.dot(ccXtX[tuple(sorted((j, l)))], D[l])
        new_component[j] = project(sl.solve(Aj, bj, assume_a='pos'))

    return new_component, new_component - current_component


def component_update_palm(XtX, XtY, current_component, mixing_coef, step_size):
    num_observations = len(XtX)
    num_components, _, _ = current_component.shape

    def gradient(j, G=None):
        if G is None:
            G = np.tensordot(mixing_coef[:, j] ** 2 / num_observations, XtX, axes=1)
        grad = - np.tensordot(mixing_coef[:, j] / num_observations, XtY, axes=1)
        grad += np.dot(G, current_component[j, :, :])
        for l in np.setdiff1d(np.arange(num_components), [j]):
            grad += np.dot(np.tensordot(mixing_coef[:, j] * mixing_coef[:, l] / num_observations, XtX, axes=1),
                           current_component[l, :, :])
        return grad

    new_component = np.zeros_like(current_component)
    alpha = np.zeros([num_components])
    for j in range(num_components):
        Gj = np.tensordot(mixing_coef[:, j] ** 2 / num_observations, XtX, axes=1)
        alpha[j] = sl.norm(Gj, ord=2) ** (-1) * step_size
        new_component[j, :, :] = project(current_component[j, :, :] - alpha[j] * gradient(j, G=Gj))

    return new_component, new_component - current_component


def coef_update_palm(XtX, XtY, current_component, current_coef, step_size, proximal_function, penalty_parameter):
    num_observations = len(XtX)

    def gradient(i, G=None):
        if G is None:
            G = gram_matrix(current_component, lambda x, y: inner_product(x, np.dot(XtX[i], y)))
        return (- inner_product(XtY[i], current_component) + np.dot(G, current_coef[i, :].T)) / num_observations

    new_coef = np.zeros_like(current_coef)
    beta = np.zeros([num_observations])
    for i in range(num_observations):
        Gi = gram_matrix(current_component, lambda x, y: inner_product(x, np.dot(XtX[i], y)))
        beta[i] = num_observations * sl.norm(Gi, ord=2) ** (-1) * step_size

        # proximal/gradient step
        new_coef[i, :] = proximal_function(new_coef[i, :] - beta[i] * gradient(i, G=Gi),
                                           penalty_parameter * beta[i] / num_observations)

    return new_coef, new_coef - current_coef

    
def negative_log_likelihood(YtY, XtX, XtY, component, coef, penalty_parameter, coef_penalty_type):
    """
    Computes the negative log likelihood for ALMM
    :param YtY: list
    :param XtX: list
    :param XtY: list
    :param component: numpy array
    :param coef: numpy array
    :param penalty_parameter: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :return: nll: float
    """
    
    num_observations = len(XtX)
    num_components, _, _ = component.shape
    gram_C = [gram_matrix(component, lambda x, y: inner_product(x, np.dot(XtX_i, y))) for XtX_i in XtX]
    nll = 0.5 * (np.mean(YtY)
                 - 2 * np.sum([np.mean([coef[i, j] * inner_product(XtY[i], component[j]) for i in range(num_observations)]) for j in range(num_components)])
                 + np.mean(np.matmul(np.expand_dims(coef, 1), np.matmul(gram_C, np.expand_dims(coef, 2)))))
    if coef_penalty_type == 'l0':
        nll += penalty_parameter * np.count_nonzero(coef[:]) / num_observations
    elif coef_penalty_type == 'l1':
        nll += penalty_parameter * sl.norm(coef[:], ord=1) / num_observations

    return nll
