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
NUM_COMPONENTS = 10
MODEL_ORDER = np.arange(2, 17, 2)
COEF_SUPP = 3
NUM_STARTS = 1
PENALTY_PARAM = 1e-2
NUM_SAMPLES = 10
SAVE_PATH = '/home/addison/Python/almm/results'

nll = np.zeros([NUM_SAMPLES, len(MODEL_ORDER), len(MODEL_ORDER)])
for sample in range(NUM_SAMPLES):
    for j, model_order_gen in enumerate(MODEL_ORDER):
        data, _, D = alm_sample(NUM_OBS, OBS_LEN, SIGNAL_DIM, NUM_COMPONENTS, model_order_gen, COEF_SUPP,
                                coef_condition=1e2, component_condition=1e2)
        for i, model_order_fit in enumerate(MODEL_ORDER):
            print('Generative model order: ' + str(model_order_gen) + ', Fitted model order: ' + str(model_order_fit))
            alm_model = Alm(solver='palm', tol=1e-3)
            _, _, nll[sample, i, j], _ = alm_model.fit(data, model_order_fit, NUM_COMPONENTS, PENALTY_PARAM,
                                                       num_starts=NUM_STARTS)
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Generative model order')
    # ax.set_xticks(np.arange(len(NUM_COMPONENTS)+1))
    # ax.set_xticklabels(MODEL_ORDER)
    # ax.set_ylabel('Fitted model order')
    # ax.set_yticks(np.arange(len(NUM_COMPONENTS)+1))
    # ax.set_yticklabels(MODEL_ORDER)
    # image = ax.imshow(np.mean(nll, axis=0), origin='lower', cmap=plt.cm.Blues)
    # fig.colorbar(image, fraction=0.046, pad=0.04)
    # plt.savefig(join(SAVE_PATH, 'model_order_misspecification2.eps'))
    with open(join(SAVE_PATH, 'model_order_misspecification2.pickle'), 'wb') as f:
        pickle.dump(nll, f)
