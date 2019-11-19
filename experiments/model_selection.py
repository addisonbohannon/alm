#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import pickle
import numpy as np
from alm.alm import Alm
from experiments.utility import load_isruc_data

SUBJ = 8
"""The typical orders for multichannel EEG datasets vary in range from 3 to 9 (Kaminski and Liang 2005)"""
# {4, 6, 8, 10, 12}
MODEL_ORDER = range(4, 14, 2)
# {2, 6, 10, 14, 18}
NUM_COMPS = range(2, 20, 4)
PENALTY_PARAM = 1e-1
NUM_STARTS = 5
RESULTS_PATH = '/home/addison/Python/almm/results/'

data, _ = load_isruc_data(SUBJ)
num_obs, obs_len, sig_dim = data.shape
nll = np.zeros([len(MODEL_ORDER), len(NUM_COMPS)])
num_params = np.zeros_like(nll)
num_params_wcoef = np.zeros_like(nll)
for i, model_ord in enumerate(MODEL_ORDER):
    for j, num_comps in enumerate(NUM_COMPS):
        num_params[i, j] = model_ord * num_comps * sig_dim**2
        num_params_wcoef[i, j] = num_params[i, j] + num_obs * num_comps 
        alm_model = Alm(tol=1e-3, solver='palm', verbose=False)
        _, _, nll[i, j], _ = alm_model.fit(data, model_ord, num_comps, PENALTY_PARAM, num_starts=NUM_STARTS)
""" aic: -2 * log(L) + 2 * k """
aic = 2 * (num_obs * obs_len * nll + num_params)
aic_wcoef = 2 * (num_obs * obs_len * nll + num_params_wcoef)
""" bic: -2 * log(L) + log(n) * k """
bic = 2 * num_obs * obs_len * nll + (np.log(num_obs * obs_len)) * num_params
bic_wcoef = 2 * num_obs * obs_len * nll + (np.log(num_obs * obs_len)) * num_params_wcoef

with open(join(RESULTS_PATH, 'model_selection.pickle'), 'wb') as file:
    pickle.dump([nll, num_params, num_params_wcoef, aic, aic_wcoef, bic, bic_wcoef], file)
