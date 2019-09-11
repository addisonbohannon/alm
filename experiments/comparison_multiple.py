#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 2 Jul 19
"""

from os.path import join
from datetime import datetime as dt
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from almm.almm import Almm
from almm.sampler import almm_sample
from almm.utility import unstack_ar_coef, dict_distance

n = 1000
m = 10000
d = 5
r = 10
p = 2
s = 3

# Generate almm sample
print('Generating ALMM sample...', end=" ", flush=True)
t1 = timer()
x, C, D = almm_sample(n, m, d, r, p, s, coef_cond=1e2, dict_cond=1e2)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')

# Fit model with proximal solver
print('Fitting ALMM model...')
t1 = timer()
almm_model = Almm(tol=1e-3, verbose=True)
D_palm, C_palm, palm_likelihood, _ = almm_model.fit(x, p, r, k=5, mu=1e-2, 
                                                    return_path=True, 
                                                    return_all=True)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')
        
# Fit model with alternating solver
print('Fitting ALMM model...')
t3 = timer()
almm_model = Almm(tol=1e-3, solver='alt_min', verbose=True)
D_altmin, C_altmin, altmin_likelihood, _ = almm_model.fit(x, p, r, k=5, 
                                                          mu=1e-2, 
                                                          return_path=True, 
                                                          return_all=True)
t4 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t4-t3) + 's')
        
# Fit model with two-stage solver
print('Fitting ALMM model...')
t5 = timer()
almm_model = Almm(tol=1e-6, solver='two_stage', verbose=True)
D_two, C_two, two_likelihood, two_time = almm_model.fit(x, p, r, k=5, mu=1e-2, 
                                                        return_path=True, 
                                                        return_all=True)
t6 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t6-t5) + 's')

# Compute dictionary error
print('Computing dictionary error...', end=" ", flush=True)
fig, axs = plt.subplots(2, 1)
axs[1].set_xlabel('Iteration')
axs[0].set_ylabel('Dictionary Error')
axs[1].set_ylabel('Likelihood')
# Proximal error
loss=[]
for i, Di in enumerate(D_palm):
    loss = []
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_ar_coef(Dis[j])
        d_loss, _, _ = dict_distance(D, Dis_pred)
        loss.append(d_loss)
    plt_palm0, = axs[0].plot(loss, 'b-')
# Alternating error
loss=[]
for i, Di in enumerate(D_altmin):
    loss = []
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_ar_coef(Dis[j])
        d_loss, _, _ = dict_distance(D, Dis_pred)
        loss.append(d_loss)
    plt_altmin0, = axs[0].plot(loss, 'r-')
# Two-stage error
loss=[]
for i, Di in enumerate(D_two):
    loss = []
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_ar_coef(Dis[j])
        d_loss, _, _ = dict_distance(D, Dis_pred)
        loss.append(d_loss)
    plt_two0, = axs[0].plot(loss, 'g-')
axs[0].legend((plt_palm0, plt_altmin0, plt_two0), ('Proximal', 'Alternating', 'Two-stage'))
for likelihood in palm_likelihood:
    plt_palm1, = axs[1].plot(likelihood, 'b-')
for likelihood in altmin_likelihood:
    plt_altmin1, = axs[1].plot(likelihood, 'r-')
for likelihood in two_likelihood:
    plt_two1, = axs[1].plot(likelihood, 'g-')
axs[1].legend((plt_palm1, plt_altmin1, plt_two1), ('Proximal', 'Alternating', 'Two-stage'))
print('Complete.')

path = "/home/addison/Python/almm/results"
plt.savefig(join(path, "comparison_multiple-"+dt.now().strftime("%y%b%d_%H%M")+".svg"))
