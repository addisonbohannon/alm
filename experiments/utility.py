#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import pickle
import numpy as np
import scipy.linalg as sl
import scipy.fftpack as sf
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

    if not np.issubdtype(type(subj_id), np.int) or not (0 < subj_id <= 11):
        raise ValueError('Subject ID must be between 0 and 9.')
    if start is not None and (not np.issubdtype(type(start), np.int) or not (0 <= start < 5)):
        raise ValueError('Start must be between 0 and 4.')

    with open(join(DATA_PATH, 'individual/subj_' + str(subj_id) + '_results.pickle'), 'rb') as f:
        components, mixing_coef, labels = pickle.load(f)
    if subj_id == 11:
        components, mixing_coef = [components], [mixing_coef]
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


def periodogram_from_filter(filter_coef, sampling_rate, fft_len=None):
    """
    Generate periodogram from filter coefficients
    :param filter_coef: model_order x signal_dim x signal_dim numpy array
    :param sampling_rate: positive integer
    :param fft_len: positive integer
    :return periodogram: fft_len/2 numpy array
    :return frequencies:  fft_len/2 numpy array
    """

    if not (len(filter_coef.shape) == 3):
        raise ValueError('Filter coefficients must be model order x signal dimension x signal dimension numpy array.')
    model_order, signal_dim1, signal_dim2 = filter_coef.shape
    if signal_dim1 != signal_dim2:
        raise ValueError('Filter coefficients must be signal dimension by signal dimension.')
    del signal_dim1, signal_dim2
    if not np.issubdtype(type(sampling_rate), np.int) or not sampling_rate > 0:
        raise ValueError('Samping rate must be a positive integer.')
    if fft_len is None:
        fft_len = len(filter_coef) + 1
    elif not np.issubdtype(type(fft_len), np.int):
        raise TypeError('Periodogram length must be an integer.')

    transfer_function, frequencies = transfer_function_from_filter(filter_coef, sampling_rate, fft_len=fft_len)
    periodogram = np.square(sl.norm(transfer_function, axis=(1, 2), ord=2))

    return periodogram, frequencies


def transfer_function_from_filter(filter_coef, sampling_rate, fft_len=None):
    """
    Compute transfer function from filter coefficients
    :param filter_coef: model_order x signal_dim x signal_dim numpy array
    :param fft_len: positive integer
    :return transfer_function: fft_len/2 x signal_dim x signal_dim numpy array
    """

    if not (len(filter_coef.shape) == 3):
        raise ValueError('Filter coefficients must be model order x signal dimension x signal dimension numpy array.')
    model_order, signal_dim1, signal_dim2 = filter_coef.shape
    if signal_dim1 != signal_dim2:
        raise ValueError('Filter coefficients must be signal dimension by signal dimension.')
    else:
        signal_dim = signal_dim1
        del signal_dim2
    if fft_len is None:
        fft_len = len(filter_coef) + 1
    elif not np.issubdtype(type(fft_len), np.int):
        raise TypeError('Periodogram length must be an integer.')

    def frequency(k): return k * sampling_rate / fft_len

    return_len = int(fft_len / 2)
    filter_coef = np.concatenate((np.expand_dims(np.eye(signal_dim), 0), -filter_coef), axis=0)

    return np.array([sl.inv(H) for H in sf.fft(filter_coef, n=fft_len, axis=0)[:return_len]]), \
        frequency(np.arange(return_len))
