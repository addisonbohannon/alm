#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
from datetime import datetime as dt
from timeit import default_timer as timer
import pickle
import numpy as np
import matplotlib.pyplot as plt
from almm.almm import Almm
from almm.utility import unstack_coef, initialize_components
from validation.sampler import almm_sample
from validation.utility import component_distance

n = 200
m = 1000
d = 5
r = 10
p = 2
s = 3
k = 5
mu = 1e-1

# Generate almm sample
print('Generating ALMM sample...', end=" ", flush=True)
t1 = timer()
x, C, D = almm_sample(n, m, d, r, p, s, coef_condition=1e1, component_condition=1e1)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')

# Initialize dictionary estimates
D_0 = [initialize_components(r, p, d) for _ in range(k)]

# Fit model with alternating minimization solver
#print('Fitting ALMM with alternating minimization...')
#t1 = timer()
#almm_altmin = Almm(solver='altmin', verbose=True)
#D_altmin, C_altmin, altmin_likelihood, altmin_time = almm_altmin.fit(x, p, r, mu, num_starts=k, initial_component=D_0,
#                                                                     return_path=True, return_all=True)
#t2 = timer()
#print('Elapsed time: ' + str(t2-t1) + 's')
        
# Fit model with block coordinate descent solver
print('Fitting ALMM with block coordinate descent...')
t3 = timer()
almm_bcd = Almm(solver='bcd', verbose=True)
D_bcd, C_bcd, bcd_likelihood, bcd_time = almm_bcd.fit(x, p, r, mu, num_starts=k, initial_component=D_0,
                                                      return_path=True, return_all=True)
t4 = timer()
print('Elapsed time: ' + str(t4-t3) + 's')
        
# Fit model with proximal alternating linearized minimization solver
#print('Fitting ALMM with proximal alternating linearized minimization...')
#t5 = timer()
#almm_palm = Almm(solver='palm', verbose=True)
#D_palm, C_palm, palm_likelihood, palm_time = almm_palm.fit(x, p, r, mu, num_starts=k, initial_component=D_0,
#                                                           return_path=True, return_all=True)
#t6 = timer()
#print('Elapsed time: ' + str(t6-t5) + 's')

# Display results
print('Displaying results...', end=" ", flush=True)
fig, axs = plt.subplots(2, 2)
axs[1, 0].set_xlabel('Iteration')
axs[1, 1].set_xlabel('Wall Time')
axs[0, 0].set_ylabel('Dictionary Error')
axs[1, 0].set_ylabel('Likelihood')
# Proximal alternating linearized minimization error
#error_palm = []
#for i, (Di, time) in enumerate(zip(D_palm, palm_time)):
#    loss = []
#    for s, Dis in enumerate(Di):
#        Dis_pred = np.zeros([r, p, d, d])
#        for j in range(r):
#            Dis_pred[j] = unstack_coef(Dis[j])
#        d_loss, _, _ = component_distance(D, Dis_pred)
#        loss.append(d_loss)
#    error_palm.append(loss)
#    plt_palm00, = axs[0, 0].plot(loss, 'b-')
#    plt_palm01, = axs[0, 1].plot(time, loss, 'b-')
# Alternating minimization error
#error_altmin = []
#for i, (Di, time) in enumerate(zip(D_altmin, altmin_time)):
#    loss = []
#    for s, Dis in enumerate(Di):
#        Dis_pred = np.zeros([r, p, d, d])
#        for j in range(r):
#            Dis_pred[j] = unstack_coef(Dis[j])
#        d_loss, _, _ = component_distance(D, Dis_pred)
#        loss.append(d_loss)
#    error_altmin.append(loss)
#    plt_altmin00, = axs[0, 0].plot(loss, 'r-')
#    plt_altmin01, = axs[0, 1].plot(time, loss, 'r-')
# Block coordinate descent error
error_bcd = []
for i, (Di, time) in enumerate(zip(D_bcd, bcd_time)):
    loss = []
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_coef(Dis[j])
        d_loss, _, _ = component_distance(D, Dis_pred)
        loss.append(d_loss)
    error_bcd.append(loss)
    plt_bcd00, = axs[0, 0].plot(loss, 'g-')
    plt_bcd01, = axs[0, 1].plot(time, loss, 'g-')
#axs[0, 0].legend((plt_palm00, plt_altmin00, plt_bcd00), ('PALM', 'AltMin', 'BCD'))
#axs[0, 1].legend((plt_palm01, plt_altmin01, plt_bcd01), ('PALM', 'AltMin', 'BCD'))
#for likelihood, time in zip(palm_likelihood, palm_time):
#    plt_palm10, = axs[1, 0].plot(likelihood, 'b-')
#    plt_palm11, = axs[1, 1].plot(time, likelihood, 'b-')
#for likelihood, time in zip(altmin_likelihood, altmin_time):
#    plt_altmin10, = axs[1, 0].plot(likelihood, 'r-')
#    plt_altmin11, = axs[1, 1].plot(time, likelihood, 'r-')
for likelihood, time in zip(bcd_likelihood, bcd_time):
    plt_bcd10, = axs[1, 0].plot(likelihood, 'g-')
    plt_bcd11, = axs[1, 1].plot(time, likelihood, 'g-')
#axs[1, 0].legend((plt_palm10, plt_altmin10, plt_bcd10), ('PALM', 'AltMin', 'BCD'))
#axs[1, 1].legend((plt_palm11, plt_altmin11, plt_bcd11), ('PALM', 'AltMin', 'BCD'))
#print('Complete.')
#print('Saving results...', end=' ', flush=True)
#path = "/home/addison/Python/almm/results"
#plt.savefig(join(path, "comparison-"+dt.now().strftime("%y%b%d_%H%M")+".svg"))
#with open(join(path, "comparison-"+dt.now().strftime("%y%b%d_%H%M")+".pickle"), 'wb') as f:
#    pickle.dump([D_palm, D_altmin, D_bcd, palm_likelihood, altmin_likelihood, bcd_likelihood, palm_time, altmin_time,
#                 bcd_time], f)
#print('Complete.')
