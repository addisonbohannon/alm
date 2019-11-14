#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from alm.alm import Alm
from experiments.utility import load_isruc_data

SUBJ = 8
"""The typical orders for multichannel EEG datasets vary in range from 3 to 9 (Kaminski and Liang 2005)"""
MODEL_ORDER = range(4, 10, 2) # {4, 6, 8}
NUM_COMPS = range(6, 12, 2) # {6, 8, 10}
PENALTY_PARAM = 1e-1
NUM_STARTS = 5

data, _ = load_isruc_data(SUBJ)
num_obs, obs_len, sig_dim = data.shape
nll = np.zeros([len(MODEL_ORDER), len(NUM_COMPS)])
num_params = np.zeros_like(nll)
for i, model_ord in enumerate(MODEL_ORDER):
    for j, num_comps in enumerate(NUM_COMPS):
        num_params = model_ord * num_comps
        alm_model = Alm(tol=1e-3, solver='palm', verbose=False)
        _, _, nll[i, j], _ = alm_model.fit(data, model_ord, num_comps, PENALTY_PARAM, num_starts=NUM_STARTS)
""" aic: -2 * log(L) + 2 * k """
aic = 2 * (num_obs * obs_len * nll + num_params)
""" bic: -2 * log(L) + sqrt(n) * k """
bic = 2 * num_obs * obs_len * nll + (np.log(num_obs * obs_len)) * num_params
