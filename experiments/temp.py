#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.almm import Almm
from alm.utility import unstack_coef
from experiments.sampler import almm_sample
from experiments.utility import component_distance

n = 200
m = 800
d = 5
r = 10
p = 2
s = 3
k = 5
mu = 1e-2

x, C, D = almm_sample(n, m, d, r, p, s, coef_condition=1e1, component_condition=1e1)
almm_model = Almm(max_iter=25, solver='bcd', verbose=True)
D_pred, C_pred, L, T = almm_model.fit(x, p, r, mu, num_starts=k, return_path=True, return_all=True, compute_likelihood_path=False)
fig, axs = plt.subplots(2, 1)
axs[1].set_xlabel('Iteration')
axs[0].set_ylabel('Dictionary Error')
axs[1].set_ylabel('Likelihood')
loss=[]
for i, Di in enumerate(D_pred):
    loss = []
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_coef(Dis[j])
        d_loss, _, _ = component_distance(D, Dis_pred)
        loss.append(d_loss)
    axs[0].plot(loss, 'r-')
for Li in L:
    axs[1].plot(Li, 'r-')
    