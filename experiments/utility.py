#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import pickle
import numpy as np
import cvxpy as cp


DATA_PATH = '/home/addison/Python/almm/results/'


def component_distance(component_1, component_2, p=2):
    """
    :param component_1: list of numpy arrays
    :param component_2: list of numpy arrays
    :param p: positive float
    :return distance: float
    :return component-wise distance: numpy array
    :return permutation: numpy array
    """

    n = len(component_1)
    m = len(component_2)

    if m != n:
        raise ValueError('Components must be of same length')
    if p < 0:
        raise ValueError('p must be a positive value.')

    d = np.zeros([n, n])
    triu_index = np.triu_indices(n)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        d[i, j] = np.minimum(np.sum(np.power(component_1[i] - component_2[j], p)) ** (1 / p),
                             np.sum(np.power(component_1[i] + component_2[j], p)) ** (1 / p))
    tril_index = np.tril_indices(n, k=-1)
    d[tril_index] = d.T[tril_index]
    w = cp.Variable(shape=(n, n))
    obj = cp.Minimize(cp.sum(cp.multiply(d, w)))
    con = [w >= 0, cp.sum(w, axis=0) == 1, cp.sum(w, axis=1) == 1]
    prob = cp.Problem(obj, con)
    prob.solve()

    return prob.value, d[w.value > 1e-3], w.value


def load_individual_results(subj_id, start=None):
    """
    Load results from fitted ALM model
    :param subj_id: integer, [1, 11]
    :param start: integer, [0, 4]
    :return components: list of num_components x model_order*signal_dim x signal_dim numpy arrays
    :return mixing_coef: list of num_obs x num_components numpy arrays
    :return labels: list of integers
    """

    if not isinstance(subj_id, int) or not (0 < subj_id <= 11):
        raise ValueError('Subject ID must be between 0 and 9.')
    if start is not None and (not isinstance(start, int) or not (0 <= start < 5)):
        raise ValueError('Start must be between 0 and 4.')

    with open(join(DATA_PATH, 'individual/subj_' + str(subj_id) + '_results.pickle'), 'rb') as f:
        components, mixing_coef, labels = pickle.load(f)
    if start is not None:
        components = components[start]
        mixing_coef = mixing_coef[start]

    return components, mixing_coef, labels


def load_group_results(model_order=12, num_components=10, penalty_parameter=0.1):
    """
    Load results from fitted group ALM model
    :param model_order: integer, {12, 16, 20}
    :param num_components: integer, {10, 15, 20}
    :param penalty_parameter: float, {0.1, 1}
    :return components: list of num_components x model_order*signal_dim x signal_dim numpy arrays
    :return mixing_coef: list of num_obs x num_components numpy arrays
    :return labels: list of integers
    """

    if model_order not in [12, 16, 20]:
        raise ValueError('Model order must be 12, 16, or 20.')
    if num_components not in [10, 15, 20]:
        raise ValueError('Number of components must be 10, 15, or 20.')
    if penalty_parameter == 1 and not (model_order == 20 or num_components == 20):
        raise ValueError('Penalty parameter must be 0.1, or 0.1 or 1 for p=20 and r=20.')

    with open(join(DATA_PATH, 'group/p' + str(model_order) + '_r' + str(num_components) + '_mu' + str(penalty_parameter) + '.pickle'), 'rb') as f:
        components, mixing_coef, labels = pickle.load(f)

    return components, mixing_coef, labels
