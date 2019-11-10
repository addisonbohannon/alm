#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as nr
import scipy.linalg as sl


def train_val_split(samples, val_pct):
    """
    Returns the indices of a random training-experiments split
    :param samples: positive integer
    :param val_pct: float (0, 1)
    :return train_idx, val_idx: lists
    """
    
    val_idx = nr.choice(samples, int(samples * val_pct), replace=False)
    train_idx = np.setdiff1d(np.arange(samples), val_idx)

    return list(train_idx), list(val_idx)


def gram_matrix(data, local_inner_product):
    """
    Computes the gram matrix for a list of elements for a given inner product
    :param data: list
    :param local_inner_product: symmetric function of two arguments
    :return g: len(x) x len(x) numpy array
    """

    n = len(data)
    g = np.zeros([n, n])
    triu_index = np.triu_indices(n)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        g[i, j] = local_inner_product(data[i], data[j])
    tril_index = np.tril_indices(n, k=-1)
    g[tril_index] = g.T[tril_index]

    return g


def broadcast_inner_product(matrix_1, matrix_2):
    """
    Returns the Frobenius inner product with broadcasting
    :param matrix_1: numpy array
    :param matrix_2: numpy array
    :return ab: numpy array
    """

    return np.sum(np.multiply(matrix_1, matrix_2), axis=(1, 2))


def circulant_matrix(observation, model_order):
    """
    Returns the circulant observation matrix
    :param observation: sample_length x signal_dimension numpy array
    :param model_order: positive integer
    :return circulant_observation, stacked_observation: (sample_length-model_order) x (model_order*signal_dimension),
    (sample_length-model_order) x signal_dimension
    """

    sample_length, signal_dimension = np.shape(observation)
    circulant_observation = np.zeros([sample_length - model_order, model_order * signal_dimension])
    # Reverse order of observations to achieve convolution effect
    circulant_observation[0, :] = np.ndarray.flatten(observation[model_order - 1::-1, :])
    for t in np.arange(1, sample_length - model_order):
        circulant_observation[t, :] = np.ndarray.flatten(observation[t + model_order - 1:t - 1:-1, :])

    return circulant_observation, observation[model_order:, :]


def package_observations(observations, model_order):
    """
    Compute the sample sufficient statistics (autocovariance)
    :param observations: num_observations x observation_length x signal_dimension numpy array
    :param model_order: positive integer
    :return YtY: num_observations numpy array
    :return XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :return XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    """

    observation_length = observations.shape[1]
    Y = observations[:, model_order:, :]
    X = np.array([circulant_matrix(observation, model_order)[0] for observation in observations])
    YtY = broadcast_inner_product(Y, Y) / (observation_length - model_order)
    XtY = np.matmul(np.moveaxis(X, 1, 2), Y) / (observation_length - model_order)
    XtX = np.matmul(np.moveaxis(X, 1, 2), X) / (observation_length - model_order)

    return YtY, XtY, XtX


def stack_coef(coef):
    """
    Returns the stacked coefficients of an autoregressive model
    :param coef: model_order x signal_dimension x signal_dimension numpy array
    :return stacked_coef: (model_order*signal_dimension) x signal_dimension numpy array
    """

    model_order, signal_dimension, _ = coef.shape

    return np.reshape(np.moveaxis(coef, 1, 2), [model_order * signal_dimension, signal_dimension])


def unstack_coef(coef):
    """
    Returns the unstacked coefficients of an autoregressive model
    :param coef: (model_order*signal_dimension) x signal_dimension numpy array
    :return unstacked_coef: model_order x signal_dimension x signal_dimension numpy array
    """
    
    model_order_by_signal_dimension, signal_dimension = coef.shape
    model_order = int(model_order_by_signal_dimension/signal_dimension)

    return np.stack(np.split(coef.T, model_order, axis=1), axis=0)


def initialize_components(num_components, model_order, signal_dimension, stacked=True):
    """
    Initialize random components for ALMM
    :param num_components: integer
    :param model_order: integer
    :param signal_dimension: integer
    :param stacked: boolean
    :return initial_component: num_components x model_order x signal_dimension x signal_dimension numpy array
    """

    if stacked:
        component = nr.randn(num_components, model_order*signal_dimension, signal_dimension)
    else:
        component = nr.randn(num_components, model_order, signal_dimension, signal_dimension)

    return np.array([component_j/sl.norm(component_j[:]) for component_j in component])


def component_gram_matrix(autocorrelation, component):
    """
    Computes component Gram matrix with respect to sample autocorrelation
    :param autocorrelation: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :return component_gram_matrix: num_observations x num_components x num_components numpy array
    """

    num_observations = autocorrelation.shape[0]
    num_components = component.shape[0]
    gram_matrix = np.zeros([num_observations, num_components, num_components])
    for j in range(num_components):
        tmp = np.matmul(autocorrelation, component[j])
        for k in range(j, num_components):
            gram_matrix[:, k, j] = broadcast_inner_product(component[k], tmp)
            if j != k:
                gram_matrix[:, j, k] = gram_matrix[:, k, j]

    return gram_matrix

#    return np.array([gram_matrix(component, lambda comp_1, comp_2: np.sum(np.multiply(comp_1, np.dot(XtX_i, comp_2))))
#            for XtX_i in autocorrelation])


def component_corr_matrix(correlation, component):
    """
    Computes component correlation matrix
    :param correlation: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param component: num_components x model_order*signal_dimension x signal_dimension numpy array
    :return component_corr_matrix: num_observations x num_components numpy array
    """

    return np.tensordot(correlation, np.moveaxis(component, 0, -1), axes=2)


def coef_gram_matrix(autocorrelation, coef):
    """
    Computes the coefficient gram matrix with respect to sample autocovariance
    :param autocorrelation: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
    :param coef: num_observations x num_components numpy array
    :return coef_gram: dictionary of model_order*signal_dimension x model_order*signal_dimension numpy arrays indexed
    by upper triangular indices
    """

    num_observations, num_components = coef.shape
    coef_gram = {}
    triu_index = np.triu_indices(num_components)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        coef_gram[(i, j)] = np.tensordot(coef[:, i] * coef[:, j], autocorrelation, axes=1) / num_observations

    return coef_gram


def coef_corr_matrix(correlation, coef):
    """
    Computes coefficient correlation matrix
    :param correlation: num_observations x model_order*signal_dimension x signal_dimension numpy array
    :param coef: num_observations x num_components numpy array
    :return coef_corr: num_components x model_order*signal_dimension x signal_dimension numpy array
    """

    num_observations = correlation.shape[0]

    return np.tensordot(coef.T, correlation, axes=1) / num_observations
