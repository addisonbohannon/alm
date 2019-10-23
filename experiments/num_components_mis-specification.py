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
NUM_COMPONENTS = [5, 10, 15, 20]
MODEL_ORDER = 2
COEF_SUPP = 3
NUM_STARTS = 1
PENALTY_PARAM = 1e-2
SAVE_PATH = '/home/addison/Python/almm/results'

nll = np.zeros([len(NUM_COMPONENTS), len(NUM_COMPONENTS)])
error = np.zeros_like(nll)
for j, num_comps_gen in enumerate(NUM_COMPONENTS):
    data, _, D = alm_sample(NUM_OBS, OBS_LEN, SIGNAL_DIM, num_comps_gen, MODEL_ORDER, COEF_SUPP, coef_condition=1e2,
                            component_condition=1e2)
    for i, num_comps_fit in enumerate(NUM_COMPONENTS):
        print('Generative number of components: ' + str(num_comps_gen) + ', Fitted number of components: '
              + str(num_comps_fit))
        alm_model = Alm(solver='palm', tol=1e-3)
        _, _, nll[i, j], _ = alm_model.fit(data, MODEL_ORDER, num_comps_fit, PENALTY_PARAM, num_starts=NUM_STARTS)
        
fig, ax = plt.subplots()
ax.set_xlabel('Generative number of components')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(NUM_COMPONENTS)
ax.set_ylabel('Fitted number of components')
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(NUM_COMPONENTS)
image = ax.imshow(nll, origin='lower', cmap=plt.cm.Blues)
fig.colorbar(image, fraction=0.046, pad=0.04)
plt.savefig(join(SAVE_PATH, 'num_components_misspecification.eps'))
with open(join(SAVE_PATH, 'num_components_misspecification.pickle'), 'wb') as f:
    pickle.dump(nll, f)
