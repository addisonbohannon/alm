#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM Copyright (C) 2019  Addison Bohannon
"""

import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
from alm.utility import initialize_autoregressive_components, package_observations, unstack_ar_coef
from experiments.sampler import autoregressive_sample
from experiments.utility import load_results, save_results

NUM_OBS = [2**i for i in range(4, 11)]
OBS_LEN = [2**i for i in range(4, 11)]
SIGNAL_DIM = 5
NUM_COMPONENTS = 1
COEF_SUPPORT = 1
MODEL_ORDER = 2
NUM_STARTS = 10
PENALTY_PARAM = 1e-2
NUM_ITERATIONS = 10

error = np.zeros([NUM_ITERATIONS, len(NUM_OBS), len(OBS_LEN)])
#nll = np.zeros_like(error)
for iteration in range(NUM_ITERATIONS):
    D = initialize_autoregressive_components(max(NUM_OBS), MODEL_ORDER, SIGNAL_DIM)
    x = np.zeros([max(NUM_OBS), max(OBS_LEN), SIGNAL_DIM])
    for obs in range(max(NUM_OBS)):
        x[obs] = autoregressive_sample(max(OBS_LEN), SIGNAL_DIM, SIGNAL_DIM ** (-1 / 2), unstack_ar_coef(D[obs]))
    for i, n_i in enumerate(NUM_OBS):
        for j, m_i in enumerate(OBS_LEN):
            _, XtY, XtX = package_observations(x[:(n_i - 1), :(m_i - 1), :], MODEL_ORDER)
            D_ls = [sl.solve(XtX_i, XtY_i, assume_a='pos') for XtX_i, XtY_i in zip(XtX, XtY)]
#            nll[iteration, :, i, j] = np.array(L_palm)
            error[iteration, i, j] = sl.norm(D[:(n_i-1)] - D_ls) / np.sqrt(n_i)

###################
# save results
###################
#save_results(error, 'n_vs_m-var.pickle')

###################
# load results
###################
_, error = load_results('n_vs_m.pickle')
error_var = load_results('n_vs_m-var.pickle')

vmax = np.maximum(np.max(error), np.max(error_var))

fig, axs = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.55)
images = []
for i in range(2):
    axs[i].set_xlabel('Len. of real.', fontsize=12)
    axs[i].set_xticks(np.arange(len(OBS_LEN)+1))
    axs[i].set_xticklabels(OBS_LEN, fontsize=8)
    axs[i].set_ylabel('Num. of real.', fontsize=12)
    axs[i].set_yticks(np.arange(len(NUM_OBS)+1))
    axs[i].set_yticklabels(NUM_OBS, fontsize=12)
images.append(axs[0].imshow(np.mean(np.min(error, axis=1), axis=0), origin='lower', vmin=0, vmax=vmax, cmap=plt.cm.Blues))
images.append(axs[1].imshow(np.mean(error_var, axis=0), origin='lower', vmin=0, vmax=vmax, cmap=plt.cm.Blues))
fig.colorbar(images[-1], ax=axs, fraction=0.024, pad=0.04)
