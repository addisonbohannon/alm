#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as nr
import scipy.linalg as sl
from almm.timeseries import Timeseries
from almm.utility import stack_coef, initialize_components, coef_gram_matrix, component_gram_matrix

MIXING_FACTOR = 4
MAX_ITER = int(1e2)


def check_almm_condition(observation, component, mixing_coef):
    """
    Computes the condition numbers that show-up in the various iterative procedures
    :param observation: number_observations x observation_length x signal_dimension numpy array
    :param component: number_components x model_order x signal_dimension x signal_dimension numpy array
    :param mixing_coef: number_observations x number_components numpy array
    :return component_condition, coef_condition: float, float
    """

    number_observations, _, signal_dimension = observation.shape
    number_components, model_order, _, _ = component.shape

    XtX = []
    for x_i in observation:
        x_i = Timeseries(x_i)
        XtX.append(x_i.XtX(model_order))
    coef_gram_list =  coef_gram_matrix(XtX, mixing_coef)
    component_matrix = np.zeros([model_order*signal_dimension*number_components,
                                 model_order*signal_dimension*number_components])
    for (i, j), coef_gram in coef_gram_list:
        component_matrix[(i*model_order*signal_dimension):((i+1)*model_order*signal_dimension),
                         (j*model_order*signal_dimension):(j+1)*model_order*signal_dimension] = coef_gram
        if i != j:
            component_matrix[(j * model_order * signal_dimension):((j + 1) * model_order * signal_dimension),
                             (i * model_order * signal_dimension):(i + 1) * model_order * signal_dimension] = coef_gram
    component_singvals = sl.svdvals(component_matrix)
    component = [stack_coef(component_j) for component_j in component]
    component_gram = component_gram_matrix(XtX, component)
    coef_singvals = np.zeros([number_observations, number_components])
    for i in range(number_observations):
        coef_singvals[i] = sl.svdvals(component_gram[i])

    return np.max(component_singvals) / np.min(component_singvals), \
           np.max(coef_singvals, axis=1) / np.min(coef_singvals, axis=1)


def isstable(autoregressive_coef):
    """
    Form companion matrix (C) of polynomial p(z) = z*A[1] + ... + z^p*A[p] and
    check that det(I-zC)=0 only for |z|>1; if so, return True and otherwise
    False.
    :param autoregressive_coef: model_order x signal_dimension x signal_dimension numpy array
    :return stability: boolean
    """

    model_order, signal_dimension, _ = autoregressive_coef.shape
    companion_matrix = np.eye(model_order*signal_dimension, k=-signal_dimension)
    companion_matrix[:signal_dimension, :] = np.concatenate(list(autoregressive_coef), axis=1)
    eigvals = sl.eigvals(companion_matrix, overwrite_a=True)

    return np.all(np.abs(eigvals) < 1)


def autoregressive_sample(observation_length, signal_dimension, noise_variance, autoregressive_coef):
    """
    Generates a random sample of an autoregressive process
    :param observation_length: positive integer
    :param signal_dimension: positive integer
    :param noise_variance: positive float
    :param autoregressive_coef: model_order x signal_dimension x signal_dimension
    :return observation: observation_length x signal_dimension numpy array
    """

    model_order, _, _ = autoregressive_coef.shape
    # Generate more samples than necessary to allow for mixing of the process
    observation_length_with_mixing = MIXING_FACTOR * observation_length + model_order
    observation = np.zeros([observation_length_with_mixing, signal_dimension, 1])
    observation[:model_order, :, :] = nr.randn(model_order, signal_dimension, 1)
    observation[model_order, :, :] = (np.sum(np.matmul(autoregressive_coef, observation[model_order - 1::-1, :, :]),
                                             axis=0) + noise_variance * nr.randn(1, signal_dimension, 1))
    for t in np.arange(model_order+1, observation_length_with_mixing):
        observation[t, :] = (np.sum(np.matmul(autoregressive_coef, observation[t - 1:t - model_order - 1:-1, :, :]),
                                    axis=0) + noise_variance * nr.randn(1, signal_dimension, 1))

    return np.squeeze(observation[-observation_length:, :])


def almm_sample(number_observations, observation_length, signal_dimension, number_components, model_order, coef_support,
                coef_condition=None, component_condition=None):
    """
    Generates random samples according to the ALMM
    :param number_observations: positive integer
    :param observation_length: positive integer
    :param signal_dimension: positive integer
    :param number_components: positive integer
    :param model_order: positive integer
    :param coef_support: positive integer less than number_components
    :param coef_condition: positive float
    :param component_condition: positive float
    :return observation: number_observations x observation_length x signal_dimension numpy array
    :return mixing_coef: number_observations x number_components numpy array
    :return components: number_components x model_order x signal_dimension x signal_dimension numpy array
    """
    
    for step in range(MAX_ITER):
        components = initialize_components(number_components, model_order, signal_dimension, stacked=False)
        mixing_coef = np.zeros([number_observations, number_components])
        for i in range(number_observations):
            support = list(nr.choice(number_components, size=coef_support,
                                     replace=False))
            mixing_coef[i, support] = signal_dimension ** (-1 / 2) * nr.randn(coef_support)
            while not isstable(np.tensordot(mixing_coef[i, :], components, axes=1)):
                mixing_coef[i, support] = signal_dimension ** (-1 / 2) * nr.randn(coef_support)
        observation = np.zeros([number_observations, observation_length, signal_dimension])
        for i in range(number_observations):
            observation[i, :, :] = autoregressive_sample(observation_length, signal_dimension,
                                                         signal_dimension ** (-1 / 2),
                                                         np.tensordot(mixing_coef[i, :], components, axes=1))
        k1, k2 = check_almm_condition(observation, components, mixing_coef)
        if k1 < coef_condition and np.all(k2 < component_condition):
            break

    return observation, mixing_coef, components
