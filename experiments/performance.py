#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
from datetime import datetime as dt
from timeit import default_timer as timer
import pickle
import numpy as np
import matplotlib.pyplot as plt
from alm.alm import Alm
from alm.utility import unstack_coef, initialize_components
from experiments.sampler import alm_sample
from experiments.utility import component_distance

n = 1000
m = 10000
d = 5
r = 10
p = 2
s = 3
k = 3
mu = 1e-2
path = "/home/addison/Python/almm/results"
colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']

x, C, D = alm_sample(n, m, d, r, p, s, coef_condition=1e1, component_condition=1e1)
D_0 = [initialize_components(r, p, d) for _ in range(k)]
alm = Alm(solver='altmin', verbose=True)
D_palm, C_palm, palm_likelihood, _ = alm.fit(x, p, r, mu, num_starts=k, initial_component=D_0, return_path=True,
                                             return_all=True)
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
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_coef(Dis[j])
        d_loss, _, _ = component_distance(D, Dis_pred)
        loss.append(d_loss)
    palm_error.append(loss)
    plt_palm1, = axs[1].plot(loss, color=colors[i])
plt.savefig(join(path, "performance.svg"))
with open(join(path, "performance.pickle"), 'wb') as f:
    pickle.dump([palm_likelihood, palm_error], f)
