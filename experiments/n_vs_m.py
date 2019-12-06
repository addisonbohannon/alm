#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.alm import Alm
from alm.utility import unstack_ar_coef, initialize_autoregressive_components
from experiments.sampler import alm_sample
from experiments.utility import ar_comp_dist, load_results, save_results

NUM_OBS = [2**i for i in range(4, 11)]
OBS_LEN = [2**i for i in range(4, 11)]
SIGNAL_DIM = 5
NUM_COMPONENTS = 10
MODEL_ORDER = 2
COEF_SUPPORT = 3
NUM_STARTS = 10
PENALTY_PARAM = 1e-2
NUM_ITERATIONS = 10

nll = np.zeros([NUM_ITERATIONS, NUM_STARTS, len(NUM_OBS), len(OBS_LEN)])
error = np.zeros_like(nll)
for iteration in range(NUM_ITERATIONS):
    x, _, D = alm_sample(max(NUM_OBS), max(OBS_LEN), SIGNAL_DIM, NUM_COMPONENTS, MODEL_ORDER, COEF_SUPPORT,
                         coef_cond=1e2, comp_cond=1e2)
    D_0 = [initialize_autoregressive_components(NUM_COMPONENTS, MODEL_ORDER, SIGNAL_DIM) for _ in range(NUM_STARTS)]
    for i, n_i in enumerate(NUM_OBS):
        for j, m_i in enumerate(OBS_LEN):
            alm_model = Alm(solver='palm')
            D_palm, _, L_palm, _ = alm_model.fit(x[:(n_i - 1), :(m_i - 1), :], MODEL_ORDER, NUM_COMPONENTS,
                                                 PENALTY_PARAM, num_starts=NUM_STARTS, initial_comps=D_0,
                                                 return_all=True)
            nll[iteration, :, i, j] = np.array(L_palm)
            error_palm = []
            for D_k in D_palm:
                D_pred = [unstack_ar_coef(Dj) for Dj in D_k]
                d_loss, _, _ = ar_comp_dist(D, D_pred)
                error_palm.append(d_loss)
            error[iteration, :, i, j] = np.array(error_palm)

###################
# save results
###################
save_results([nll, error], 'n_vs_m.pickle')

###################
# load results
###################
# nll, error = load_results('n_vs_m.pickle')

fig, axs = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.55)
images = []
for i in range(2):
    axs[i].set_xlabel('Obs. length', fontsize=12)
    axs[i].set_xticks(np.arange(len(OBS_LEN)+1))
    axs[i].set_xticklabels(OBS_LEN, fontsize=8)
    axs[i].set_ylabel('Num. of obs.', fontsize=12)
    axs[i].set_yticks(np.arange(len(NUM_OBS)+1))
    axs[i].set_yticklabels(NUM_OBS, fontsize=12)
axs[0].set_title('Avg. min. error', fontsize=12)
axs[1].set_title('Avg. std. error', fontsize=12)
images.append(axs[0].imshow(np.mean(np.min(error, axis=1), axis=0), origin='lower', vmin=0, cmap=plt.cm.Blues))
fig.colorbar(images[-1], ax=axs[0], fraction=0.046, pad=0.04)
images.append(axs[1].imshow(np.mean(np.std(error, axis=1), axis=0), origin='lower', vmin=0, cmap=plt.cm.Blues))
fig.colorbar(images[-1], ax=axs[1], fraction=0.046, pad=0.04)
