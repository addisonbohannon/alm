#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import getpid
from os.path import join
from datetime import datetime as dt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from alm.alm import Alm
from alm.utility import unstack_coef, initialize_components
from experiments.sampler import alm_sample
from experiments.utility import component_distance

NUM_OBS = [2**i for i in range(4, 11)]
OBS_LEN = [2**i for i in range(4, 11)]
SIGNAL_DIM = 5
NUM_COMPONENTS = 10
MODEL_ORDER = 2
COEF_SUPPORT = 3
NUM_STARTS = 10
PENALTY_PARAM = 1e-2
NUM_ITERATIONS = 10
VMIN = 0
VMAX = 14
SAVE_PATH = "/home/addison/Python/almm/results"

nll = np.zeros([NUM_ITERATIONS, len(NUM_OBS), len(OBS_LEN)])
error = np.zeros_like(nll)
for iteration in range(NUM_ITERATIONS):
    x, _, D = alm_sample(max(NUM_OBS), max(OBS_LEN), SIGNAL_DIM, NUM_COMPONENTS, MODEL_ORDER, COEF_SUPPORT,
                         coef_condition=1e2, component_condition=1e2)
    D_0 = [initialize_components(NUM_COMPONENTS, MODEL_ORDER, SIGNAL_DIM) for _ in range(NUM_STARTS)]
    for i, n_i in enumerate(NUM_OBS):
        for j, m_i in enumerate(OBS_LEN):
            alm_model = Alm(solver='palm')
            D_palm, _, L_palm, _ = alm_model.fit(x[:(n_i - 1), :(m_i - 1), :], MODEL_ORDER, NUM_COMPONENTS,
                                                 PENALTY_PARAM, num_starts=NUM_STARTS, initial_component=D_0,
                                                 return_all=True)
            nll[iteration, i, j] = min(L_palm)
            error_palm = []
            for D_k in D_palm:
                D_pred = [unstack_coef(Dj) for Dj in D_k]
                d_loss, _, _ = component_distance(D, D_pred)
                error_palm.append(d_loss)
            error[iteration, i, j] = min(error_palm)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Observation length')
    ax.set_xticks(np.arange(len(OBS_LEN)+1))
    ax.set_xticklabels(OBS_LEN)
    ax.set_ylabel('Number of observations')
    ax.set_yticks(np.arange(len(NUM_OBS)+1))
    ax.set_yticklabels(NUM_OBS)
    image = ax.imshow(error, origin='lower', vmin=VMIN, vmax=VMAX, cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax, fraction=0.012, pad=0.04)
    plt.savefig(join(SAVE_PATH, "n_vs_m-" + dt.now().strftime("%y%b%d_%H%M") + '-' + str(getpid()) + ".svg"))
    with open(join(SAVE_PATH, "n_vs_m-" + dt.now().strftime("%y%b%d_%H%M") + '-' + str(getpid()) + ".pickle"), 'wb') as f:
        pickle.dump([nll, error], f)
