#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import pickle
import numpy as np
import scipy.linalg as sl
import scipy.fftpack as sf
import cvxpy as cp

DATA_PATH = '/home/addison/Python/almm/results/'


def ar_comp_dist(ar_comp_1, ar_comp_2, p=2):
    """
    :param ar_comp_1: list of numpy arrays
    :param ar_comp_2: list of numpy arrays
    :param p: positive float
    :return distance: float
    :return component-wise distance: numpy array
    :return permutation: numpy array
    """

    n = len(ar_comp_1)
    m = len(ar_comp_2)

    if m != n:
        raise ValueError('Components must be of same length')
    if p < 0:
        raise ValueError('p must be a positive value.')

    d = np.zeros([n, n])
    triu_index = np.triu_indices(n)
    for (i, j) in zip(triu_index[0], triu_index[1]):
        d[i, j] = np.minimum(np.sum(np.power(ar_comp_1[i] - ar_comp_2[j], p)) ** (1 / p),
                             np.sum(np.power(ar_comp_1[i] + ar_comp_2[j], p)) ** (1 / p))
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
    :return ar_comps: list of num_comps x model_ord*signal_dim x signal_dim numpy arrays
    :return mixing_coef: list of num_obs x num_comps numpy arrays
    :return labels: list of integers
    """

    if not np.issubdtype(type(subj_id), np.int) or not (0 < subj_id <= 11):
        raise ValueError('Subject ID must be between 0 and 9.')
    if start is not None and (not np.issubdtype(type(start), np.int) or not (0 <= start < 5)):
        raise ValueError('Start must be between 0 and 4.')

    with open(join(DATA_PATH, 'individual/subj_' + str(subj_id) + '_results.pickle'), 'rb') as f:
        ar_comps, mixing_coef, labels = pickle.load(f)
    if subj_id == 11:
        ar_comps, mixing_coef = [ar_comps], [mixing_coef]
    if start is not None:
        ar_comps = ar_comps[start]
        mixing_coef = mixing_coef[start]

    return ar_comps, mixing_coef, labels


def load_group_results(model_ord=12, num_comps=10, penalty_param=0.1):
    """
    Load results from fitted group ALM model
    :param model_ord: integer, {12, 16, 20}
    :param num_comps: integer, {10, 15, 20}
    :param penalty_param: float, {0.1, 1}
    :return ar_comps: list of num_comps x model_ord*signal_dim x signal_dim numpy arrays
    :return mixing_coef: list of num_obs x num_comps numpy arrays
    :return labels: list of integers
    """

    if model_ord not in [12, 16, 20]:
        raise ValueError('Model order must be 12, 16, or 20.')
    if num_comps not in [10, 15, 20]:
        raise ValueError('Number of ar_comps must be 10, 15, or 20.')
    if penalty_param == 1 and not (model_ord == 20 or num_comps == 20):
        raise ValueError('Penalty parameter must be 0.1, or 0.1 or 1 for p=20 and r=20.')

    with open(join(DATA_PATH, 'group/p' + str(model_ord) + '_r' + str(num_comps) + '_mu' + str(penalty_param) + '.pickle'), 'rb') as f:
        ar_comps, mixing_coef, labels = pickle.load(f)

    return ar_comps, mixing_coef, labels


def periodogram_from_filter(filter_coef, srate, fft_len=None):
    """
    Generate periodogram from filter coefficients
    :param filter_coef: model_ord x signal_dim x signal_dim numpy array
    :param srate: positive integer
    :param fft_len: positive integer
    :return periodogram: fft_len/2 numpy array
    :return freqs:  fft_len/2 numpy array
    """

    if not (len(filter_coef.shape) == 3):
        raise ValueError('Filter coefficients must be model order x signal dimension x signal dimension numpy array.')
    model_ord, signal_dim1, signal_dim2 = filter_coef.shape
    if signal_dim1 != signal_dim2:
        raise ValueError('Filter coefficients must be signal dimension by signal dimension.')
    del signal_dim1, signal_dim2
    if not np.issubdtype(type(srate), np.int) or not srate > 0:
        raise ValueError('Samping rate must be a positive integer.')
    if fft_len is None:
        fft_len = len(filter_coef) + 1
    elif not np.issubdtype(type(fft_len), np.int):
        raise TypeError('Periodogram length must be an integer.')

    trans_fcn, freqs = transfer_function_from_filter(filter_coef, srate, fft_len=fft_len)
    periodogram = np.square(sl.norm(trans_fcn, axis=(1, 2), ord=2))

    return periodogram, freqs


def transfer_function_from_filter(filter_coef, srate, fft_len=None):
    """
    Compute transfer function from filter coefficients
    :param filter_coef: model_ord x signal_dim x signal_dim numpy array
    :param fft_len: positive integer
    :return transfer_function: fft_len/2 x signal_dim x signal_dim numpy array
    """

    if not (len(filter_coef.shape) == 3):
        raise ValueError('Filter coefficients must be model order x signal dimension x signal dimension numpy array.')
    model_ord, signal_dim1, signal_dim2 = filter_coef.shape
    if signal_dim1 != signal_dim2:
        raise ValueError('Filter coefficients must be signal dimension by signal dimension.')
    else:
        signal_dim = signal_dim1
        del signal_dim2
    if fft_len is None:
        fft_len = len(filter_coef) + 1
    elif not np.issubdtype(type(fft_len), np.int):
        raise TypeError('Periodogram length must be an integer.')

    def frequency(k): return k * srate / fft_len

    return_len = int(fft_len / 2)
    filter_coef = np.concatenate((np.expand_dims(np.eye(signal_dim), 0), -filter_coef), axis=0)

    return np.array([sl.inv(H) for H in sf.fft(filter_coef, n=fft_len, axis=0)[:return_len]]), \
        frequency(np.arange(return_len))
