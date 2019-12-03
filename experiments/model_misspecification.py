#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from alm.alm import Alm
from experiments.sampler import alm_sample

NUM_OBS = 1000
OBS_LEN = 10000
SIGNAL_DIM = 5
COEF_SUPP = 2
NUM_STARTS = 1
PENALTY_PARAM = 1e-2
NUM_SAMPLES = 10
RESULTS_PATH = "/home/addison/Python/almm/results"
MODEL_ORDER = np.arange(2, 17, 2)
NUM_COMPONENTS = np.arange(2, 17, 2)

nll_model_ord = np.zeros([NUM_SAMPLES, len(MODEL_ORDER), len(MODEL_ORDER)])
for sample in range(NUM_SAMPLES):
    for j, model_order_gen in enumerate(MODEL_ORDER):
        data, _, D = alm_sample(NUM_OBS, OBS_LEN, SIGNAL_DIM, NUM_COMPONENTS[4], model_order_gen, COEF_SUPP,
                                coef_cond=1e2, comp_cond=1e2)
        for i, model_order_fit in enumerate(MODEL_ORDER):
            print('Generative model order: ' + str(model_order_gen) + ', Fitted model order: ' + str(model_order_fit))
            alm_model = Alm(solver='palm', tol=1e-3)
            _, _, nll_model_ord[sample, i, j], _ = alm_model.fit(data, model_order_fit, NUM_COMPONENTS, PENALTY_PARAM,
                                                       num_starts=NUM_STARTS)

nll_num_comps = np.zeros([NUM_SAMPLES, len(NUM_COMPONENTS), len(NUM_COMPONENTS)])
for sample in range(NUM_SAMPLES):
    for j, num_comps_gen in enumerate(NUM_COMPONENTS):
        data, _, D = alm_sample(NUM_OBS, OBS_LEN, SIGNAL_DIM, num_comps_gen, MODEL_ORDER[0], COEF_SUPP, coef_cond=1e2,
                                comp_cond=1e2)
        for i, num_comps_fit in enumerate(NUM_COMPONENTS):
            print('Generative number of components: ' + str(num_comps_gen) + ', Fitted number of components: '
                  + str(num_comps_fit))
            alm_model = Alm(solver='palm', tol=1e-3)
            _, _, nll_num_comps[sample, i, j], _ = alm_model.fit(data, MODEL_ORDER, num_comps_fit, PENALTY_PARAM,
                                                       num_starts=NUM_STARTS)

###################
# save results
###################
#with open(join(RESULTS_PATH, "model_misspec.pickle"), 'wb') as f:
#    pickle.dump([nll_model_ord, nll_num_comps], f)

###################
# load results
###################
with open(join(RESULTS_PATH, "model_misspec.pickle"), 'rb') as f:
   nll_model_ord, nll_num_comps = pickle.load(f)

fig, axs = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.55)
image = []
axs[0].set_xlabel('Generative model order', fontsize=12)
axs[0].set_xticks(np.arange(len(MODEL_ORDER)+1))
axs[0].set_xticklabels(MODEL_ORDER, fontsize=12)
axs[0].set_ylabel('Fitted model order', fontsize=12)
axs[0].set_yticks(np.arange(len(MODEL_ORDER)+1))
axs[0].set_yticklabels(MODEL_ORDER, fontsize=12)
image.append(axs[0].imshow(np.mean(nll_model_ord, axis=0), origin='lower', cmap=plt.cm.Blues))
fig.colorbar(image[-1], ax=axs[0], fraction=0.046, pad=0.04)

axs[1].set_xlabel('Generative num. of comps.', fontsize=12)
axs[1].set_xticks(np.arange(len(NUM_COMPONENTS)+1))
axs[1].set_xticklabels(NUM_COMPONENTS, fontsize=12)
axs[1].set_ylabel('Fitted num. of comps.', fontsize=12)
axs[1].set_yticks(np.arange(len(NUM_COMPONENTS)+1))
axs[1].set_yticklabels(NUM_COMPONENTS, fontsize=12)
image.append(axs[1].imshow(np.mean(nll_num_comps, axis=0), origin='lower', cmap=plt.cm.Blues))
fig.colorbar(image[-1], ax=axs[1], fraction=0.046, pad=0.04)
