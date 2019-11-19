#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from alm.alm import Alm
from alm.utility import unstack_ar_coef, initialize_autoregressive_components
from experiments.sampler import alm_sample
from experiments.utility import ar_comp_dist

NUM_OBS = 1000
OBS_LEN = 10000
SIG_DIM = 5
NUM_COMPS = 10
MODEL_ORD = 2
SPARSITY = 3
NUM_STARTS = 5
PENALTY_PARAM = 1e-2
RESULTS_PATH = "/home/addison/Python/almm/results"
colors = ['#ffffbf', '#fdae61', '#d7191c', '#abdda4', '#2b83ba']

x, C, D = alm_sample(NUM_OBS, OBS_LEN, SIG_DIM, NUM_COMPS, MODEL_ORD, SPARSITY, coef_cond=1e1,
                     comp_cond=1e1)
D_0 = [initialize_autoregressive_components(NUM_COMPS, MODEL_ORD, SIG_DIM) for _ in range(NUM_STARTS)]
alm = Alm(solver='palm', verbose=True)
D_palm, C_palm, palm_likelihood, _ = alm.fit(x, MODEL_ORD, NUM_COMPS, PENALTY_PARAM, num_starts=NUM_STARTS,
                                             initial_comps=D_0, return_path=True, return_all=True)
palm_error = []
for i, Di in enumerate(D_palm):
    loss = []
    for Dis in Di:
        Dis_pred = np.zeros([NUM_COMPS, MODEL_ORD, SIG_DIM, SIG_DIM])
        for j in range(NUM_COMPS):
            Dis_pred[j] = unstack_ar_coef(Dis[j])
        d_loss, _, _ = ar_comp_dist(D, Dis_pred)
        loss.append(d_loss)
    palm_error.append(loss)

###################
# save results
###################
# with open(join(RESULTS_PATH, "performance.pickle"), 'wb') as f:
#    pickle.dump([palm_likelihood, palm_error], f)

###################
# load results
###################
# palm_likelihood, palm_error = pickle.load(open(RESULTS_PATH + 'performance.pickle', 'rb'))

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(8.5, 5.5)
axs[0].set_xlabel('Iteration', fontsize=12)
axs[1].set_xlabel('Iteration', fontsize=12)
axs[0].set_ylabel('Negative Log Likelihood', fontsize=12)
axs[1].set_ylabel('Component Error', fontsize=12)
for likelihood, color in zip(palm_likelihood, colors):
    plt_palm0, = axs[0].plot(likelihood, color=color, zorder=4, linewidth=3.0)
for error, color in zip(palm_error, colors):
    plt_palm0, = axs[1].plot(error, color=color, zorder=4, linewidth=3.0)
axs[0].set_facecolor("#f2f3f4")
axs[0].grid(b=True, which='major', linestyle="-", linewidth=1.5, color="#ffffff", zorder=3)
axs[0].grid(b=True, which='minor', linewidth=0.75, color="#ffffff", zorder=3)
axs[1].set_facecolor("#f2f3f4")
axs[1].grid(b=True, which='major', linestyle="-", linewidth=1.5, color="#ffffff", zorder=3)
axs[1].grid(b=True, which='minor', linewidth=0.75, color="#ffffff", zorder=3)
