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


def gram_matrix(data, inner_prod):
    """
    Computes the gram matrix for a list of elements for a given inner product
    :param data: list
    :param inner_prod: symmetric function of two arguments
    :return g: len(x) x len(x) numpy array
    """

    n = len(data)
    g = np.zeros([n, n])
    triu_index = np.triu_indices(n)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        g[i, j] = inner_prod(data[i], data[j])
    tril_index = np.tril_indices(n, k=-1)
    g[tril_index] = g.T[tril_index]

    return g


def broadcast_inner_product(mat_1, mat_2):
    """
    Returns the Frobenius inner product with broadcasting
    :param mat_1: [k x] m x n numpy array
    :param mat_2: [k x] m x n numpy array
    :return ab: [k] x 1 numpy array
    """

    return np.sum(np.multiply(mat_1, mat_2), axis=(1, 2))


def circulant_matrix(obs, model_ord):
    """
    Returns the circulant obs matrix
    :param obs: obs_len x signal_dim numpy array
    :param model_ord: positive integer
    :return circ_obs, stacked_observation: (obs_len-model_ord) x (model_ord*signal_dim),
    (obs_len-model_ord) x signal_dim
    """

    obs_len, signal_dim = np.shape(obs)
    circ_obs = np.zeros([obs_len - model_ord, model_ord * signal_dim])
    # Reverse order of obs to achieve convolution effect
    circ_obs[0, :] = np.ndarray.flatten(obs[model_ord - 1::-1, :])
    for t in np.arange(1, obs_len - model_ord):
        circ_obs[t, :] = np.ndarray.flatten(obs[t + model_ord - 1:t - 1:-1, :])

    return circ_obs, obs[model_ord:, :]


def package_observations(obs, model_ord):
    """
    Compute the sample sufficient statistics (autocovariance)
    :param obs: num_observations x obs_len x signal_dim numpy array
    :param model_ord: positive integer
    :return YtY: num_observations numpy array
    :return XtY: num_observations x model_ord*signal_dim x signal_dim numpy array
    :return XtX: num_observations x model_ord*signal_dim x model_ord*signal_dim numpy array
    """

    obs_len = obs.shape[1]
    Y = obs[:, model_ord:, :]
    X = np.array([circulant_matrix(obs_i, model_ord)[0] for obs_i in obs])
    YtY = broadcast_inner_product(Y, Y) / (obs_len - model_ord)
    XtY = np.matmul(np.moveaxis(X, 1, 2), Y) / (obs_len - model_ord)
    XtX = np.matmul(np.moveaxis(X, 1, 2), X) / (obs_len - model_ord)

    return YtY, XtY, XtX


def stack_ar_coef(ar_coef):
    """
    Returns the stacked coefficients of an autoregressive model
    :param ar_coef: model_ord x signal_dim x signal_dim numpy array
    :return stacked_coef: (model_ord*signal_dim) x signal_dim numpy array
    """

    model_ord, signal_dim, _ = ar_coef.shape

    return np.reshape(np.moveaxis(ar_coef, 1, 2), [model_ord * signal_dim, signal_dim])


def unstack_ar_coef(ar_coef):
    """
    Returns the unstacked coefficients of an autoregressive model
    :param ar_coef: (model_ord*signal_dim) x signal_dim numpy array
    :return unstacked_coef: model_ord x signal_dim x signal_dim numpy array
    """
    
    model_ord_by_signal_dim, signal_dim = ar_coef.shape
    model_ord = int(model_ord_by_signal_dim/signal_dim)

    return np.stack(np.split(ar_coef.T, model_ord, axis=1), axis=0)


def initialize_autoregressive_components(num_comps, model_ord, signal_dim, stacked=True):
    """
    Initialize random components for ALMM
    :param num_comps: integer
    :param model_ord: integer
    :param signal_dim: integer
    :param stacked: boolean
    :return initial_comps: num_comps x model_ord x signal_dim x signal_dim numpy array
    """

    if stacked:
        ar_comps = nr.randn(num_comps, model_ord * signal_dim, signal_dim)
    else:
        ar_comps = nr.randn(num_comps, model_ord, signal_dim, signal_dim)

    return np.array([ar_comp/sl.norm(ar_comp[:]) for ar_comp in ar_comps])


def component_gram_matrix(autocov, ar_comps):
    """
    Computes ar_comps Gram matrix with respect to sample autocov
    :param autocov: num_obs x model_ord*signal_dim x model_ord*signal_dim numpy array
    :param ar_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :return component_gram_matrix: num_obs x num_comps x num_comps numpy array
    """

    num_obs = autocov.shape[0]
    num_comps = ar_comps.shape[0]
    comp_gram = np.zeros([num_obs, num_comps, num_comps])
    for j in range(num_comps):
        tmp = np.matmul(autocov, ar_comps[j])
        for k in range(j, num_comps):
            comp_gram[:, k, j] = broadcast_inner_product(ar_comps[k], tmp)
            if j != k:
                comp_gram[:, j, k] = comp_gram[:, k, j]

    return comp_gram

#    return np.array([gram_matrix(ar_comps, lambda ar_comp_1, comp_2: np.sum(np.multiply(ar_comp_1, np.dot(XtX_i, comp_2))))
#            for XtX_i in autocov])


def component_corr_matrix(autocov, ar_comps):
    """
    Computes ar_comps autocov matrix
    :param autocov: num_observations x model_ord*signal_dim x signal_dim numpy array
    :param ar_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
    :return component_corr_matrix: num_observations x num_comps numpy array
    """

    return np.tensordot(autocov, np.moveaxis(ar_comps, 0, -1), axes=2)


def coef_gram_matrix(autocov, mixing_coef):
    """
    Computes the coefficient gram matrix with respect to sample autocovariance
    :param autocov: num_obs x model_ord*signal_dim x model_ord*signal_dim numpy array
    :param mixing_coef: num_obs x num_comps numpy array
    :return coef_gram: dictionary of model_ord*signal_dim x model_ord*signal_dim numpy arrays indexed
    by upper triangular indices
    """

    num_obs, num_comps = mixing_coef.shape
    coef_gram = {}
    triu_index = np.triu_indices(num_comps)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        coef_gram[(i, j)] = np.tensordot(mixing_coef[:, i] * mixing_coef[:, j], autocov, axes=1) / num_obs

    return coef_gram


def coef_corr_matrix(autocov, mixing_coef):
    """
    Computes coefficient autocov matrix
    :param autocov: num_obs x model_ord*signal_dim x signal_dim numpy array
    :param mixing_coef: num_obs x num_comps numpy array
    :return coef_corr: num_comps x model_ord*signal_dim x signal_dim numpy array
    """

    num_obs = autocov.shape[0]

    return np.tensordot(mixing_coef.T, autocov, axes=1) / num_obs
