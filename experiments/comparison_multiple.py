#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
from datetime import datetime as dt
from timeit import default_timer as timer
import pickle
import numpy as np
import matplotlib.pyplot as plt
from almm.almm import Almm
from almm.utility import unstack_coef
from experiments.sampler import almm_sample
from experiments.utility import component_distance

n = 1000
m = 10000
d = 5
r = 10
p = 2
s = 3
k = 10
mu = 1e-2

# Generate almm sample
print('Generating ALMM sample...', end=" ", flush=True)
t1 = timer()
x, C, D = almm_sample(n, m, d, r, p, s, coef_condition=1e2, component_condition=1e2)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')

# Fit model with alternating minimization solver
print('Fitting ALMM model...')
t1 = timer()
almm_model = Almm(tol=1e-3, solver='altmin', verbose=True)
D_altmin, C_altmin, altmin_likelihood, altmin_time = almm_model.fit(x, p, r, mu, num_starts=k, return_path=True,
                                                                    return_all=True)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')
        
# Fit model with block coordinate descent solver
print('Fitting ALMM model...')
t3 = timer()
almm_model = Almm(tol=1e-3, solver='bcd', verbose=True)
D_bcd, C_bcd, bcd_likelihood, bcd_time = almm_model.fit(x, p, r, mu, num_starts=k, return_path=True, return_all=True)
t4 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t4-t3) + 's')
        
# Fit model with proximal alternating linearized minimization solver
print('Fitting ALMM model...')
t5 = timer()
almm_model = Almm(tol=1e-3, solver='palm', verbose=True)
D_palm, C_palm, palm_likelihood, palm_time = almm_model.fit(x, p, r, mu, num_starts=k, return_path=True,
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
            Dis_pred[j] = unstack_coef(Dis[j])
        d_loss, _, _ = component_distance(D, Dis_pred)
        loss.append(d_loss)
    plt_palm0, = axs[0].plot(loss, 'b-')
# Alternating error
loss=[]
for i, Di in enumerate(D_altmin):
    loss = []
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_coef(Dis[j])
        d_loss, _, _ = component_distance(D, Dis_pred)
        loss.append(d_loss)
    plt_altmin0, = axs[0].plot(loss, 'r-')
# Block descent error
loss=[]
for i, Di in enumerate(D_bcd):
    loss = []
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_coef(Dis[j])
        d_loss, _, _ = component_distance(D, Dis_pred)
        loss.append(d_loss)
    plt_bcd0, = axs[0].plot(loss, 'g-')
axs[0].legend((plt_palm0, plt_altmin0, plt_bcd0), ('PALM', 'AltMin', 'BCD'))
for likelihood in palm_likelihood:
    plt_palm1, = axs[1].plot(likelihood, 'b-')
for likelihood in altmin_likelihood:
    plt_altmin1, = axs[1].plot(likelihood, 'r-')
for likelihood in bcd_likelihood:
    plt_bcd1, = axs[1].plot(likelihood, 'g-')
axs[1].legend((plt_palm1, plt_altmin1, plt_bcd1), ('PALM', 'AltMin', 'BCD'))
print('Complete.')

path = "/home/addison/Python/almm/results"
plt.savefig(join(path, "comparison_multiple-"+dt.now().strftime("%y%b%d_%H%M")+".svg"))
with open(join(path, "comparison_single-"+dt.now().strftime("%y%b%d_%H%M")+".pickle"), 'wb') as f:
    pickle.dump([D_palm, D_altmin, D_bcd, palm_likelihood, altmin_likelihood, bcd_likelihood, palm_time, altmin_time, bcd_time], f)