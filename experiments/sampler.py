#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as nr
import scipy.linalg as sl
from alm.utility import stack_ar_coef, initialize_autoregressive_components, coef_gram_matrix, component_gram_matrix, \
    package_observations

MIXING_TIME = 1000
MAX_ITER = int(1e2)


def check_alm_condition(obs, ar_comps, mixing_coef):
    """
    Computes the condition numbers that show-up in the various iterative procedures
    :param obs: num_obs x obs_len x signal_dim numpy array
    :param ar_comps: num_comps x model_ord x signal_dim x signal_dim numpy array
    :param mixing_coef: num_obs x num_comps numpy array
    :return comp_cond, coef_cond: float, float
    """

    num_obs, _, signal_dim = obs.shape
    num_comps, model_ord, _, _ = ar_comps.shape

    _, _, XtX = package_observations(obs, model_ord)
    coef_gram = coef_gram_matrix(XtX, mixing_coef)
    comp_cond_mat = np.zeros([model_ord*signal_dim*num_comps, model_ord*signal_dim*num_comps])
    for (i, j), coef_gram in coef_gram.items():
        comp_cond_mat[(i*model_ord*signal_dim):((i+1)*model_ord*signal_dim),
                      (j*model_ord*signal_dim):(j+1)*model_ord*signal_dim] = coef_gram
        if i != j:
            comp_cond_mat[(j * model_ord * signal_dim):((j + 1) * model_ord * signal_dim),
                          (i * model_ord * signal_dim):(i + 1) * model_ord * signal_dim] = coef_gram
    comp_singvals = sl.svdvals(comp_cond_mat)
    ar_comps = np.array([stack_ar_coef(component_j) for component_j in ar_comps])
    comp_gram = component_gram_matrix(XtX, ar_comps)
    coef_singvals = [sl.svdvals(comp_gram_i) for comp_gram_i in comp_gram]

    return np.max(comp_singvals) / np.min(comp_singvals), np.max(coef_singvals, axis=1) / np.min(coef_singvals, axis=1)


def isstable(ar_coef):
    """
    Form companion matrix (C) of polynomial p(z) = z*A[1] + ... + z^p*A[p] and
    check that det(I-zC)=0 only for |z|>1; if so, return True and otherwise
    False.
    :param ar_coef: model_ord x signal_dim x signal_dim numpy array
    :return stability: boolean
    """

    model_ord, signal_dim, _ = ar_coef.shape
    comp_mat = np.eye(model_ord*signal_dim, k=-signal_dim)
    comp_mat[:signal_dim, :] = np.concatenate(list(ar_coef), axis=1)
    eigvals = sl.eigvals(comp_mat, overwrite_a=True)

    return np.all(np.abs(eigvals) < 1)


def autoregressive_sample(obs_len, signal_dim, noise_var, ar_coef):
    """
    Generates a random sample of an autoregressive process
    :param obs_len: positive integer
    :param signal_dim: positive integer
    :param noise_var: positive float
    :param ar_coef: model_ord x signal_dim x signal_dim
    :return obs: obs_len x signal_dim numpy array
    """

    model_ord, _, _ = ar_coef.shape
    # Generate more samples than necessary to allow for mixing of the process
    obs_len_wmix = MIXING_TIME + obs_len + model_ord
    obs = np.zeros([obs_len_wmix, signal_dim, 1])
    obs[:model_ord, :, :] = nr.randn(model_ord, signal_dim, 1)
    obs[model_ord, :, :] = (np.sum(np.matmul(ar_coef, obs[model_ord - 1::-1, :, :]), axis=0)
                            + noise_var * nr.randn(1, signal_dim, 1))
    for t in np.arange(model_ord+1, obs_len_wmix):
        obs[t, :] = (np.sum(np.matmul(ar_coef, obs[t - 1:t - model_ord - 1:-1, :, :]), axis=0)
                     + noise_var * nr.randn(1, signal_dim, 1))

    return np.squeeze(obs[-obs_len:, :])


def alm_sample(num_obs, obs_len, signal_dim, num_comps, model_ord, coef_supp, coef_cond=None, comp_cond=None):
    """
    Generates random samples according to the ALM
    :param num_obs: positive integer
    :param obs_len: positive integer
    :param signal_dim: positive integer
    :param num_comps: positive integer
    :param model_ord: positive integer
    :param coef_supp: positive integer less than num_comps
    :param coef_cond: positive float
    :param comp_cond: positive float
    :return obs: number_observations x obs_len x signal_dim numpy array
    :return mixing_coef: number_observations x num_comps numpy array
    :return ar_comps: num_comps x model_ord x signal_dim x signal_dim numpy array
    """
    
    nr.seed()
    for step in range(MAX_ITER):
        ar_comps = initialize_autoregressive_components(num_comps, model_ord, signal_dim, stacked=False)
        mixing_coef = np.zeros([num_obs, num_comps])
        for i in range(num_obs):
            supp = list(nr.choice(num_comps, size=coef_supp, replace=False))
            mixing_coef[i, supp] = num_comps ** (-1 / 2) * nr.randn(coef_supp)
            while not isstable(np.tensordot(mixing_coef[i, :], ar_comps, axes=1)):
                mixing_coef[i, supp] = num_comps ** (-1 / 2) * nr.randn(coef_supp)
        obs = np.zeros([num_obs, obs_len, signal_dim])
        for i in range(num_obs):
            obs[i, :, :] = autoregressive_sample(obs_len, signal_dim, signal_dim ** (-1 / 2),
                                                         np.tensordot(mixing_coef[i, :], ar_comps, axes=1))
        if coef_cond is not None and comp_cond is not None:
            k1, k2 = check_alm_condition(obs, ar_comps, mixing_coef)
            if k1 < coef_cond and np.all(k2 < comp_cond):
                break
        else:
            break

    return obs, mixing_coef, ar_comps
