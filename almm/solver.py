#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from timeit import default_timer as timer
import numpy as np
import scipy.linalg as sl
from almm.utility import component_corr_matrix, component_gram_matrix, coef_gram_matrix, coef_corr_matrix


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


def component_update_altmin(XtX, XtY, current_component, current_coef):
    """
    Update all autoregressive components for fixed mixing coefficients
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param current_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param current_coef: num_observations x num_components numpy array
    :return: new_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :return: change_in_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    """

    num_observations = len(XtX)
    num_components, model_order_by_signal_dimension, signal_dimension = current_component.shape
    model_order = int(model_order_by_signal_dimension/signal_dimension)
    coef_gram = coef_gram_matrix(XtX, current_coef)
    A = np.zeros([num_components * model_order * signal_dimension, num_components * model_order * signal_dimension])
    for (i, j), coef_gram_ij in coef_gram:
        A[i * model_order * signal_dimension:(i + 1) * (model_order * signal_dimension),
          (j * model_order * signal_dimension):(j + 1) * (model_order * signal_dimension)] = coef_gram_ij
    tril_index = np.tril_indices(num_components, k=-1)
    A[tril_index] = A.T[tril_index]
    coef_corr = coef_corr_matrix(XtY, current_coef)
    b = np.zeros([num_components * model_order * signal_dimension, signal_dimension])
    for j in range(num_components):
        b[(j * model_order * signal_dimension):(j + 1) * model_order * signal_dimension, :] = coef_corr[j]
    new_component = sl.solve(A, b, assume_a='pos')
    new_component = np.array([project(new_component[(j * model_order * signal_dimension):(j + 1) * model_order * signal_dimension, :])
                              for j in range(num_components)])

    return new_component, new_component - current_component


def component_update_bcd(XtX, XtY, current_component, current_coef):
    """
    Update all autoregressive components for fixed mixing coefficients
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param current_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param current_coef: num_observations x num_components numpy array
    :return: new_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :return: change_in_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    """

    num_components, _, _ = current_component.shape
    new_component = np.zeros_like(current_component)
    coef_gram = coef_gram_matrix(XtX, current_coef)
    coef_corr = coef_corr_matrix(XtY, current_coef)
    for j in range(num_components):
        Aj = coef_gram[(j, j)]
        bj = coef_corr[j]
        for l in np.setdiff1d(np.arange(num_components), [j]):
            bj -= np.dot(coef_gram[tuple(sorted((j, l)))], current_component[l])
        new_component[j] = project(sl.solve(Aj, bj, assume_a='pos'))

    return new_component, new_component - current_component


def component_update_palm(XtX, XtY, current_component, current_coef, step_size):
    """
    Update all autoregressive components for fixed mixing coefficients
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param current_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param current_coef: num_observations x num_components numpy array
    :param step_size: float
    :return: new_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :return: change_in_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    """

    num_observations = len(XtX)
    num_components, _, _ = current_component.shape
    coef_gram = coef_gram_matrix(XtX, current_coef)
    coef_corr = coef_corr_matrix(XtY, current_coef)

    def gradient(component_index):
        grad = - coef_corr[component_index]
        for l in range(num_components):
            grad += np.dot(coef_gram[tuple(sorted((component_index, l)))], current_component[l])
        return grad

    new_component = np.zeros_like(current_component)
    alpha = np.zeros([num_components])
    for j in range(num_components):
        alpha[j] = sl.norm(coef_gram[(j, j)], ord=2) ** (-1) * step_size
        new_component[j] = project(current_component[j] - alpha[j] * gradient(j))

    return new_component, new_component - current_component

        
def coef_update(XtX, XtY, current_component, current_coef, penalty_parameter, coef_penalty_type):
    """
    Fit mixing coefficients for fixed autoregressive components
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param current_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param current_coef: num_observations x num_components numpy array
    :param penalty_parameter: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :return: new_coef: num_observations x num_components numpy array
    :return: change_in_coef: num_observations x num_components numpy array
    """
    
    if coef_penalty_type is None:
        solve = lambda a, b: sl.solve(a, b, assume_a='pos')
    elif coef_penalty_type == 'l0':
        solve = lambda a, b: penalized_ls_gram(a, b, threshold, penalty_parameter)
    elif coef_penalty_type == 'l1':
        solve = lambda a, b: penalized_ls_gram(a, b, shrink, penalty_parameter)
    else:
        raise TypeError('Coefficient penalty type must be None, l0, or l1.')

    new_coef = np.array([solve(component_gram_matrix([XtX_i], current_component),
                               component_corr_matrix(XtY_i, current_component)) for XtX_i, XtY_i in zip(XtX, XtY)])

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
    :return: new_coef: num_observations x num_components numpy array
    :return: change_in_coef: num_observations x num_components numpy array
    """

    if coef_penalty_type is None:
        proximal_function = lambda x, t: x
    elif coef_penalty_type == 'l0':
        proximal_function = lambda x, t: threshold(x, t)
    elif coef_penalty_type == 'l1':
        proximal_function = lambda x, t: shrink(x, t)
    else:
        raise TypeError('Coefficient penalty type must be None, l0, or l1.')

    num_observations = len(XtX)
    component_gram = component_gram_matrix(XtX, current_component)

    def gradient(observation_index):
        return (- component_corr_matrix(XtY[observation_index], current_component)
                + np.dot(component_gram[observation_index], current_coef[observation_index, :].T)) / num_observations

    new_coef = np.zeros_like(current_coef)
    beta = np.zeros([num_observations])
    for i in range(num_observations):
        beta[i] = num_observations * sl.norm(component_gram[i], ord=2) ** (-1) * step_size
        new_coef[i, :] = proximal_function(new_coef[i, :] - beta[i] * gradient(i),
                                           penalty_parameter * beta[i] / num_observations)

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
    :return: m x n numpy array
    """

    def update_lagrange_parameter(penalty_param, pri_res, dual_res):
        if pri_res > 4 * dual_res:
            penalty_param *= 2
        else:
            penalty_param /= 2

        return penalty_param

    m, n = covariance.shape
    primal_variable_1 = np.zeros([m, n])
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


def fit_almm(XtX, XtY, model_order, num_components, penalty_parameter, coef_penalty_type, initial_component,
             solver='bcd', max_iter=int(2.5e3), step_size=1e-3, tol=1e-6, return_path=False, verbose=False):
    """
    Fit ALMM according to algorithm specified by solver
    :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param model_order: positive integer
    :param num_components: positive integer
    :param penalty_parameter: float
    :param coef_penalty_type: None, 'l0', or 'l1'
    :param initial_component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :param solver: 'bcd', 'altmin', or 'palm'
    :param max_iter: positive integer
    :param step_size: float; (0, 1)
    :param tol: float; (0, 1)
    :param return_path: boolean
    :param verbose: boolean
    :return: component: [list of] num_components x model_order*signal_dimension x signal_dimension numpy array
    :return: mixing_coef: [list of] num_observations x num_components numpy array
    :return: coef_residual: list of float
    :return: component_residual: list of float
    :return: stop_condition: string
    :return: elapsed_time: list of float
    """

    if solver == 'bcd':
        update_component = lambda current_component, current_coef: component_update_bcd(XtX, XtY, current_component,
                                                                                        current_coef)
        update_coef = lambda current_component, current_coef: coef_update(XtX, XtY, current_component, current_coef,
                                                                          penalty_parameter, coef_penalty_type)
    elif solver == 'altmin':
        update_component = lambda current_component, current_coef: component_update_altmin(XtX, XtY, current_component,
                                                                                           current_coef)
        update_coef = lambda current_component, current_coef: coef_update(XtX, XtY, current_component, current_coef,
                                                                          penalty_parameter, coef_penalty_type)
    elif solver == 'palm':
        update_component = lambda current_component, current_coef: component_update_palm(XtX, XtY, current_component,
                                                                                         current_coef, step_size)
        update_coef = lambda current_component, current_coef: coef_update_palm(XtX, XtY, current_component,
                                                                               current_coef, penalty_parameter,
                                                                               coef_penalty_type, step_size)
    else:
        raise TypeError('Solver must be bcd, altmin, or palm.')

    def compute_component_residual(component_diff):
        return sl.norm(component_diff[:]) / (num_components ** (1/2) * model_order ** (1/2) * signal_dimension)

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
    component = np.copy(initial_component)
    mixing_coef = np.zeros([num_observations, num_components])
    mixing_coef, _ = coef_update(XtX, XtY, initial_component, mixing_coef, penalty_parameter, coef_penalty_type)
    elapsed_time = [timer()-start_time]
    stop_condition = 'maximum iteration'
    if return_path:
        component_path, coef_path = [np.copy(component)], [np.copy(mixing_coef)]
    component_residual, coef_residual = [], []
    for step in range(max_iter):
        component, component_change = update_component(component, mixing_coef)
        mixing_coef, coef_change = update_coef(component, mixing_coef)
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
        print('*Solver: ' + solver)
        print('*Stopping condition: ' + stop_condition)
        print('*Iterations: ' + str(step))
        print('*Duration: ' + str(elapsed_time[-1]) + 's')

    if return_path:
        return component_path, coef_path, component_residual, coef_residual, stop_condition, elapsed_time
    else:
        return component, mixing_coef, coef_residual, component_residual, stop_condition, elapsed_time

    
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
    :return: nll: float
    """
    
    num_observations = len(XtX)
    num_components, _, _ = component.shape
    gram = component_gram_matrix(XtX, component)
    nll = 0.5 * (np.mean(np.sum(YtY**2, axis=(1, 2)))
                 - 2 * np.mean([np.dot(coef[i, :], component_corr_matrix(XtY[i], component))
                                for i in range(num_observations)])
                 + np.mean(np.matmul(np.expand_dims(coef, 1), np.matmul(gram, np.expand_dims(coef, 2)))))
    if coef_penalty_type == 'l0':
        nll += penalty_parameter * np.count_nonzero(coef[:]) / num_observations
    elif coef_penalty_type == 'l1':
        nll += penalty_parameter * sl.norm(coef[:], ord=1) / num_observations

    return nll
