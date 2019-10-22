#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.alm import Alm
from experiments.sampler import alm_sample

NUM_OBS = 100
OBS_LEN = 1000
SIGNAL_DIM = 5
NUM_COMPONENTS = 10
MODEL_ORDER = [1, 2, 4, 8]
COEF_SUPP = 3
NUM_STARTS = 1
PENALTY_PARAM = 1e-2

nll = np.zeros([len(MODEL_ORDER), len(MODEL_ORDER)])
error = np.zeros_like(nll)
for j, model_order_gen in enumerate(MODEL_ORDER):
    data, _, D = alm_sample(NUM_OBS, OBS_LEN, SIGNAL_DIM, NUM_COMPONENTS, model_order_gen, COEF_SUPP,
                            coef_condition=1e2, component_condition=1e2)
    for i, model_order_fit in enumerate(MODEL_ORDER):
        print('Generative model order: ' + str(model_order_gen) + ', Fitted model order: ' + str(model_order_fit))
        alm_model = Alm(solver='palm', tol=1e-3)
        _, _, nll[i, j], _ = alm_model.fit(data, model_order_fit, NUM_COMPONENTS, PENALTY_PARAM, num_starts=NUM_STARTS)
        
fig, ax = plt.subplots()
ax.set_xlabel('Generative model order')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(MODEL_ORDER)
ax.set_ylabel('Fitted model order')
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(MODEL_ORDER)
image = ax.imshow(nll, origin='lower', cmap=plt.cm.Blues)
fig.colorbar(image, fraction=0.046, pad=0.04)
