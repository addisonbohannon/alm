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
import numpy.random as nr
import scipy.linalg as sl
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

# Initialize dictionary estimate
D_0 = nr.randn(r, p*d, d)
D_0 = np.array([D_i / sl.norm(D_i, ord='fro') for D_i in D_0])

# Fit model with proximal solver
print('Fitting ALMM model...')
t1 = timer()
almm_model = Almm(tol=1e-3, verbose=True)
D_palm, C_palm, palm_likelihood, palm_time = almm_model.fit(x, p, r, k=1,
                                                            mu=1e-2, D_0=D_0,
                                                            return_path=True,
                                                            return_all=True)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')
        
# Fit model with alternating solver
print('Fitting ALMM model...')
t3 = timer()
almm_model = Almm(tol=1e-3, solver='alt_min', verbose=True)
D_altmin, C_altmin, altmin_likelihood, altmin_time = almm_model.fit(x, p, r, 
                                                                    k=1, 
                                                                    mu=1e-2,
                                                                    D_0=D_0,
                                                                    return_path=True,
                                                                    eturn_all=True)
t4 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t4-t3) + 's')
        
# Fit model with two-stage solver
print('Fitting ALMM model...')
t5 = timer()
almm_model = Almm(tol=1e-6, solver='two_stage', verbose=True)
D_two, C_two, two_likelihood, two_time = almm_model.fit(x, p, r, k=1, mu=1e-2,
                                                        D_0=D_0,
                                                        return_path=True,
                                                        return_all=True)
t6 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t6-t5) + 's')

# Compute dictionary error
print('Computing dictionary error...', end=" ", flush=True)
fig, axs = plt.subplots(2, 2)
axs[1, 0].set_xlabel('Iteration')
axs[1, 1].set_xlabel('Wall Time')
axs[0, 0].set_ylabel('Dictionary Error')
axs[1, 0].set_ylabel('Likelihood')
# Proximal error
loss=[]
for s, Dis in enumerate(D_palm):
    Dis_pred = np.zeros([r, p, d, d])
    for j in range(r):
        Dis_pred[j] = unstack_ar_coef(Dis[j])
    d_loss, _, _ = dict_distance(D, Dis_pred)
    loss.append(d_loss)
plt_palm00, = axs[0, 0].plot(loss, 'b-')
plt_palm01, = axs[0, 1].plot(palm_time, loss, 'b-')
# Alternating error
loss=[]
for s, Dis in enumerate(D_altmin):
    Dis_pred = np.zeros([r, p, d, d])
    for j in range(r):
        Dis_pred[j] = unstack_ar_coef(Dis[j])
    d_loss, _, _ = dict_distance(D, Dis_pred)
    loss.append(d_loss)
plt_altmin00, = axs[0, 0].plot(loss, 'r-')
plt_altmin01, = axs[0, 1].plot(altmin_time, loss, 'r-')
# Two-stage error
loss=[]
for s, Dis in enumerate(D_two):
    Dis_pred = np.zeros([r, p, d, d])
    for j in range(r):
        Dis_pred[j] = unstack_ar_coef(Dis[j])
    d_loss, _, _ = dict_distance(D, Dis_pred)
    loss.append(d_loss)
plt_two00, = axs[0, 0].plot(loss, 'g-')
plt_two01, = axs[0, 1].plot(two_time, loss, 'g-')
axs[0, 0].legend((plt_palm00, plt_altmin00, plt_two00), ('Proximal', 'Alternating', 'Two-stage'))
axs[0, 1].legend((plt_palm01, plt_altmin01, plt_two01), ('Proximal', 'Alternating', 'Two-stage'))
plt_palm10, = axs[1, 0].plot(palm_likelihood, 'b-')
plt_altmin10, = axs[1, 0].plot(altmin_likelihood, 'r-')
plt_two10, = axs[1, 0].plot(two_likelihood, 'g-')
plt_palm11, = axs[1, 1].plot(palm_time, palm_likelihood, 'b-')
plt_altmin11, = axs[1, 1].plot(altmin_time, altmin_likelihood, 'r-')
plt_two11, = axs[1, 1].plot(two_time, two_likelihood, 'g-')
axs[1, 0].legend((plt_palm10, plt_altmin10, plt_two10), ('Proximal', 'Alternating', 'Two-stage'))
axs[1, 1].legend((plt_palm11, plt_altmin11, plt_two11), ('Proximal', 'Alternating', 'Two-stage'))
print('Complete.')

path = "/home/addison/Python/almm/results"
plt.savefig(join(path, "comparison_single-"+dt.now().strftime("%y%b%d_%H%M")+".svg"))
