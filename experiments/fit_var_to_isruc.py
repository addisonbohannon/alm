#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM Copyright (C) 2019  Addison Bohannon
"""

import numpy as np
import scipy.linalg as sl
from alm.utility import package_observations
from experiments.utility import load_isruc_data, save_results

SUBJS = [8]
MODEL_ORD = 4

def fit(obs, model_ord):
    """
    Fit VAR model using least squares estimator.
    :param obs: list of obs_len x signal_dim numpy array
    :param model_ord: positive integer
    :return ar_coefs: [list of] num_comps x model_ord*signal_dim x signal_dim numpy array 
    """

    # Organize observations
    _, XtY, XtX = package_observations(obs, model_ord)
        
    # Fit VAR model
    ar_coefs = np.zeros_like(XtY)
    for i, (XtXi, XtYi) in enumerate(zip(XtX, XtY)):
        ar_coefs[i] = sl.solve(XtXi, XtYi, assume_a='pos')
        
    return ar_coefs

for subj in SUBJS:
    print(subj)
    data, labels = load_isruc_data(subj)
    ar_comps = fit(data, MODEL_ORD)
    save_results([ar_comps, labels], 'S' + str(subj) + '-var.pickle')
