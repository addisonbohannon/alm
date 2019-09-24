#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp


def component_distance(component_1, component_2, p=2):
    """
    :param component_1: list of numpy arrays
    :param component_2: list of numpy arrays
    :param p: positive float
    :return: distance, component-wise distance, permutation: float, numpy array, numpy array
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
