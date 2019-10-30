#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from alm.alm import Alm
from alm.utility import unstack_coef, initialize_components
from experiments.sampler import alm_sample
from experiments.utility import component_distance

NUM_OBS = 1000
OBS_LEN = 10000
SIG_DIM = 5
NUM_COMPS = 10
MODEL_ORD = 2
SPARSITY = 3
NUM_STARTS = 5
PENALTY_PARAM = 1e-2
path = "/home/addison/Python/almm/results"
colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']

x, C, D = alm_sample(NUM_OBS, OBS_LEN, SIG_DIM, NUM_COMPS, MODEL_ORD, SPARSITY, coef_condition=1e1,
                     component_condition=1e1)
D_0 = [initialize_components(NUM_COMPS, MODEL_ORD, SIG_DIM) for _ in range(NUM_STARTS)]
alm = Alm(solver='palm', verbose=True)
D_palm, C_palm, palm_likelihood, _ = alm.fit(x, MODEL_ORD, NUM_COMPS, PENALTY_PARAM, num_starts=NUM_STARTS,
                                             initial_component=D_0, return_path=True, return_all=True)
fig, axs = plt.subplots(1, 2)
axs[0].set_xlabel('Iteration')
axs[1].set_xlabel('Iteration')
axs[0].set_ylabel('Negative Log Likelihood')
axs[1].set_ylabel('Component Error')
for likelihood, color in zip(palm_likelihood, colors):
    plt_palm0, = axs[0].plot(likelihood, color=color)
palm_error = []
for i, Di in enumerate(D_palm):
    loss = []
    for SPARSITY, Dis in enumerate(Di):
        Dis_pred = np.zeros([NUM_COMPS, MODEL_ORD, SIG_DIM, SIG_DIM])
        for j in range(NUM_COMPS):
            Dis_pred[j] = unstack_coef(Dis[j])
        d_loss, _, _ = component_distance(D, Dis_pred)
        loss.append(d_loss)
    palm_error.append(loss)
    plt_palm1, = axs[1].plot(loss, color=colors[i])
plt.savefig(join(path, "performance.svg"))
with open(join(path, "performance.pickle"), 'wb') as f:
    pickle.dump([palm_likelihood, palm_error], f)
