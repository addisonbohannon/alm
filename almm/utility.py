#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as nr


def train_val_split(samples, val_pct):
    """
    Returns the indices of a random training-validation split
    :param samples: positive integer
    :param val_pct: float (0, 1)
    :return: train_idx, val_idx: lists
    """
    
    if not isinstance(samples, int) or samples < 1:
        raise TypeError('Number of samples must be a positive integer.')
    if not isinstance(val_pct, float) or val_pct < 0 or val_pct > 1:
        raise TypeError('Validation percentage must be between 0 and 1.')
    
    val_idx = nr.choice(samples, int(samples * val_pct), replace=False)
    train_idx = np.setdiff1d(np.arange(samples), val_idx)

    return list(train_idx), list(val_idx)


def gram_matrix(x, inner_product):
    """
    Computes the gram matrix for a list of elements for a given inner product
    :param x: list
    :param inner_product: symmetric function of two arguments
    :return: g: len(x) x len(x) numpy array
    """

    n = len(x)
    g = np.zeros([n, n])
    triu_index = np.triu_indices(n)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        g[i, j] = inner_product(x[i], x[j])
    tril_index = np.tril_indices(n, k=-1)
    g[tril_index] = g.T[tril_index]

    return g


def inner_product(a, b):
    """
    Returns the Frobenius inner product
    :param a: numpy array
    :param b: numpy array
    :return: ab: numpy array
    """

    if len(a.shape) == 3 or len(b.shape) == 3:
        return np.sum(np.multiply(a, b), axis=(1, 2))
    else:
        return np.sum(np.multiply(a, b))


def circulant_matrix(observation, model_order):
    """
    Returns the circulant observation matrix
    :param observation: sample_length x signal_dimension numpy array
    :param model_order: positive integer
    :return: circulant_observation, stacked_observation: (sample_length-model_order) x (model_order*signal_dimension),
    (sample_length-model_order) x signal_dimension
    """

    if not isinstance(model_order, int) or model_order < 1:
        raise TypeError('Model order must be a positive integer.')

    sample_length, signal_dimension = np.shape(observation)
    circulant_observation = np.zeros([sample_length - model_order, model_order * signal_dimension])
    # Reverse order of observations to achieve convolution effect
    circulant_observation[0, :] = np.ndarray.flatten(observation[model_order - 1::-1, :])
    for t in np.arange(1, sample_length - model_order):
        circulant_observation[t, :] = np.ndarray.flatten(observation[t + model_order - 1:t - 1:-1, :])

    return circulant_observation, observation[model_order:, :]


def stack_coef(coef):
    """
    Returns the stacked coefficients of an autoregressive model
    :param coef: model_order x signal_dimension x signal_dimension numpy array
    :return: stacked_coef: (model_order*signal_dimension) x signal_dimension numpy array
    """

    model_order, signal_dimension, _ = coef.shape

    return np.reshape(np.moveaxis(coef, 1, 2), [model_order * signal_dimension, signal_dimension])


def unstack_coef(coef):
    """
    Returns the unstacked coefficients of an autoregressive model
    :param coef: (model_order*signal_dimension) x signal_dimension numpy array
    :return: unstacked_coef: model_order x signal_dimension x signal_dimension numpy array
    """
    
    model_order_by_signal_dimension, signal_dimension = coef.shape
    model_order = int(model_order_by_signal_dimension/signal_dimension)

    return np.stack(np.split(coef.T, model_order, axis=1), axis=0)