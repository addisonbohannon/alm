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
NUM_COMPONENTS = np.arange(2, 17, 2)
MODEL_ORDER = 2
COEF_SUPP = 2
NUM_STARTS = 1
PENALTY_PARAM = 1e-2
NUM_SAMPLES = 10
SAVE_PATH = '/home/addison/Python/almm/results'

nll = np.zeros([NUM_SAMPLES, len(NUM_COMPONENTS), len(NUM_COMPONENTS)])
for sample in range(NUM_SAMPLES):
    for j, num_comps_gen in enumerate(NUM_COMPONENTS):
        data, _, D = alm_sample(NUM_OBS, OBS_LEN, SIGNAL_DIM, num_comps_gen, MODEL_ORDER, COEF_SUPP, coef_cond=1e2,
                                comp_cond=1e2)
        for i, num_comps_fit in enumerate(NUM_COMPONENTS):
            print('Generative number of components: ' + str(num_comps_gen) + ', Fitted number of components: '
                  + str(num_comps_fit))
            alm_model = Alm(solver='palm', tol=1e-3)
            _, _, nll[sample, i, j], _ = alm_model.fit(data, MODEL_ORDER, num_comps_fit, PENALTY_PARAM,
                                                       num_starts=NUM_STARTS)
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Generative number of components')
    # ax.set_xticks(np.arange(len(NUM_COMPONENTS)+1))
    # ax.set_xticklabels(NUM_COMPONENTS)
    # ax.set_ylabel('Fitted number of components')
    # ax.set_yticks(np.arange(len(NUM_COMPONENTS)+1))
    # ax.set_yticklabels(NUM_COMPONENTS)
    # image = ax.imshow(np.mean(nll, axis=0), origin='lower', cmap=plt.cm.Blues)
    # fig.colorbar(image, fraction=0.046, pad=0.04)
    # plt.savefig(join(SAVE_PATH, 'num_components_misspecification2.eps'))
    with open(join(SAVE_PATH, 'num_components_misspecification2.pickle'), 'wb') as f:
        pickle.dump(nll, f)
