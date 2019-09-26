#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
from datetime import datetime as dt
from timeit import default_timer as timer
import pickle
import numpy as np
import numpy.random as nr
import scipy.linalg as sl
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
k = 1
mu = 1e-2

# Generate almm sample
print('Generating ALMM sample...', end=" ", flush=True)
t1 = timer()
x, C, D = almm_sample(n, m, d, r, p, s, coef_condition=1e2, component_condition=1e2)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')

# Initialize dictionary estimate
D_0 = nr.randn(r, p*d, d)
D_0 = np.array([D_i / sl.norm(D_i, ord='fro') for D_i in D_0])

# Fit model with alternating minimization solver
print('Fitting ALMM model...')
t1 = timer()
almm_model = Almm(max_iter=100, solver='altmin', verbose=True)
D_altmin, C_altmin, altmin_likelihood, altmin_time = almm_model.fit(x, p, r, mu, num_starts=k, initial_component=D_0,
                                                                    return_path=True, return_all=True)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')
        
# Fit model with block coordinate descent solver
print('Fitting ALMM model...')
t3 = timer()
almm_model = Almm(max_iter=10, solver='bcd', verbose=True)
D_bcd, C_bcd, bcd_likelihood, bcd_time = almm_model.fit(x, p, r, mu, num_starts=k, initial_component=D_0,
                                                        return_path=True, return_all=True)
t4 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t4-t3) + 's')
        
# Fit model with proximal alternating linearized minimization solver
print('Fitting ALMM model...')
t5 = timer()
almm_model = Almm(max_iter=10, solver='palm', verbose=True)
D_palm, C_palm, palm_likelihood, palm_time = almm_model.fit(x, p, r, mu, num_starts=k, initial_component=D_0,
                                                            return_path=True, return_all=True)
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
# Proximal alternating linearized minimization error
loss=[]
for s, Dis in enumerate(D_palm):
    Dis_pred = np.zeros([r, p, d, d])
    for j in range(r):
        Dis_pred[j] = unstack_coef(Dis[j])
    d_loss, _, _ = component_distance(D, Dis_pred)
    loss.append(d_loss)
plt_palm00, = axs[0, 0].plot(loss, 'b-')
plt_palm01, = axs[0, 1].plot(palm_time, loss, 'b-')
# Alternating minimization error
loss=[]
for s, Dis in enumerate(D_altmin):
    Dis_pred = np.zeros([r, p, d, d])
    for j in range(r):
        Dis_pred[j] = unstack_coef(Dis[j])
    d_loss, _, _ = component_distance(D, Dis_pred)
    loss.append(d_loss)
plt_altmin00, = axs[0, 0].plot(loss, 'r-')
plt_altmin01, = axs[0, 1].plot(altmin_time, loss, 'r-')
# Block coordinate descent error
loss=[]
for s, Dis in enumerate(D_bcd):
    Dis_pred = np.zeros([r, p, d, d])
    for j in range(r):
        Dis_pred[j] = unstack_coef(Dis[j])
    d_loss, _, _ = component_distance(D, Dis_pred)
    loss.append(d_loss)
plt_bcd00, = axs[0, 0].plot(loss, 'g-')
plt_bcd01, = axs[0, 1].plot(bcd_time, loss, 'g-')
axs[0, 0].legend((plt_palm00, plt_altmin00, plt_bcd00), ('PALM', 'AltMin', 'BCD'))
axs[0, 1].legend((plt_palm01, plt_altmin01, plt_bcd01), ('PALM', 'AltMin', 'BCD'))
plt_palm10, = axs[1, 0].plot(palm_likelihood, 'b-')
plt_altmin10, = axs[1, 0].plot(altmin_likelihood, 'r-')
plt_bcd10, = axs[1, 0].plot(bcd_likelihood, 'g-')
plt_palm11, = axs[1, 1].plot(palm_time, palm_likelihood, 'b-')
plt_altmin11, = axs[1, 1].plot(altmin_time, altmin_likelihood, 'r-')
plt_bcd11, = axs[1, 1].plot(bcd_time, bcd_likelihood, 'g-')
axs[1, 0].legend((plt_palm10, plt_altmin10, plt_bcd10), ('PALM', 'AltMin', 'BCD'))
axs[1, 1].legend((plt_palm11, plt_altmin11, plt_bcd11), ('PALM', 'AltMin', 'BCD'))
print('Complete.')

path = "/home/addison/Python/almm/results"
plt.savefig(join(path, "comparison_single-"+dt.now().strftime("%y%b%d_%H%M")+".svg"))
#with open(join(path, "comparison_single-"+dt.now().strftime("%y%b%d_%H%M")+".pickle"), 'wb') as f:
#    pickle.dump([D_palm, D_altmin, D_bcd, palm_likelihood, altmin_likelihood, bcd_likelihood, palm_time, altmin_time, bcd_time], f)