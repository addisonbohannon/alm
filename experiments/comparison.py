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

# Generate alm sample
print('Generating ALMM sample...', end=" ", flush=True)
t1 = timer()
x, C, D = alm_sample(n, m, d, r, p, s, coef_condition=1e1, component_condition=1e1)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')

# Initialize dictionary estimates
D_0 = [initialize_components(r, p, d) for _ in range(k)]

# Fit model with alternating minimization solver
print('Fitting ALMM with alternating minimization...')
t1 = timer()
alm_altmin = Alm(solver='altmin', verbose=True)
D_altmin, C_altmin, altmin_likelihood, _ = alm_altmin.fit(x, p, r, mu, num_starts=k, initial_component=D_0,
                                                          return_path=True, return_all=True)
t2 = timer()
print('Elapsed time: ' + str(t2-t1) + 's')
        
# Fit model with block coordinate descent solver
print('Fitting ALMM with block coordinate descent...')
t3 = timer()
alm_bcd = Alm(solver='bcd', verbose=True)
D_bcd, C_bcd, bcd_likelihood, _ = alm_bcd.fit(x, p, r, mu, num_starts=k, initial_component=D_0, return_path=True,
                                              return_all=True)
t4 = timer()
print('Elapsed time: ' + str(t4-t3) + 's')
        
# Fit model with proximal alternating linearized minimization solver
print('Fitting ALMM with proximal alternating linearized minimization...')
t5 = timer()
alm_palm = Alm(solver='palm', verbose=True)
D_palm, C_palm, palm_likelihood, _ = alm_palm.fit(x, p, r, mu, num_starts=k, initial_component=D_0, return_path=True,
                                                  return_all=True)
t6 = timer()
print('Elapsed time: ' + str(t6-t5) + 's')

# Display results
print('Displaying results...', end=" ", flush=True)
fig, axs = plt.subplots(1, 2)
axs[0].set_xlabel('Iteration')
axs[1].set_xlabel('Iteration')
axs[0].set_ylabel('Negative Log Likelihood')
axs[1].set_ylabel('Component Error')
# Plot PALM results
for likelihood in palm_likelihood:
    plt_palm0, = axs[0].plot(likelihood, 'g-')
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
    plt_palm1, = axs[1].plot(loss, 'g-')
# Plot AltMin results
for likelihood in altmin_likelihood:
    plt_altmin0, = axs[0].plot(likelihood, color=(0.5, 0, 0.5), linestyle='dashed')
altmin_error = []
for i, Di in enumerate(D_altmin):
    loss = []
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_coef(Dis[j])
        d_loss, _, _ = component_distance(D, Dis_pred)
        loss.append(d_loss)
    altmin_error.append(loss)
    plt_altmin1, = axs[1].plot(loss, color=(0.5, 0, 0.5), linestyle='dashed')
# Plot BCD results
for likelihood in bcd_likelihood:
    plt_bcd0, = axs[0].plot(likelihood, 'b:')
bcd_error = []
for i, Di in enumerate(D_bcd):
    loss = []
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_coef(Dis[j])
        d_loss, _, _ = component_distance(D, Dis_pred)
        loss.append(d_loss)
    bcd_error.append(loss)
    plt_bcd1, = axs[1].plot(loss, 'b:')
axs[0].legend((plt_palm0, plt_altmin0, plt_bcd0), ('PALM', 'AltMin', 'BCD'))
axs[1].legend((plt_palm1, plt_altmin1, plt_bcd1), ('PALM', 'AltMin', 'BCD'))
print('Complete.')
print('Saving results...', end=' ', flush=True)
path = "/home/addison/Python/almm/results"
plt.savefig(join(path, "comparison-"+dt.now().strftime("%y%b%d_%H%M")+".svg"))
with open(join(path, "comparison-"+dt.now().strftime("%y%b%d_%H%M")+".pickle"), 'wb') as f:
    pickle.dump([palm_likelihood, altmin_likelihood, bcd_likelihood, palm_error, altmin_error, bcd_error], f)
print('Complete.')
