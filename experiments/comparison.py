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

n = 100
m = 800
d = 5
r = 10
p = 2
s = 3

# Generate almm sample
print('Generating ALMM sample...', end=" ", flush=True)
t1 = timer()
x, C, D = almm_sample(n, m, d, r, p, s, coef_cond=1e1, dict_cond=1e1)
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
D_palm, C_palm, palm_likelihood = almm_model.fit(x, p, r, mu=1e-2, D_0=D_0,
                                                 return_path=True)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')
        
# Fit model with alternating solver
print('Fitting ALMM model...')
t3 = timer()
almm_model = Almm(tol=1e-3, solver='alt_min', verbose=True)
D_altmin, C_altmin, altmin_likelihood = almm_model.fit(x, p, r, mu=1e-2, 
                                                       D_0=D_0, 
                                                       return_path=True)
t4 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t4-t3) + 's')

# Compute dictionary error
print('Computing dictionary error...', end=" ", flush=True)
fig, axs = plt.subplots(2, 1)
axs[1].set_xlabel('Iteration')
axs[0].set_ylabel('Dictionary Error')
axs[1].set_ylabel('Likelihood')
# Proximal error
loss=[]
for s, Dis in enumerate(D_palm):
    Dis_pred = np.zeros([r, p, d, d])
    for j in range(r):
        Dis_pred[j] = unstack_ar_coef(Dis[j])
    d_loss, _, _ = dict_distance(D, Dis_pred)
    loss.append(d_loss)
plt_palm, = axs[0].plot(loss, 'b-')
# Alternating error
loss=[]
for s, Dis in enumerate(D_altmin):
    Dis_pred = np.zeros([r, p, d, d])
    for j in range(r):
        Dis_pred[j] = unstack_ar_coef(Dis[j])
    d_loss, _, _ = dict_distance(D, Dis_pred)
    loss.append(d_loss)
plt_altmin, = axs[0].plot(loss, 'r-')
axs[0].legend((plt_palm, plt_altmin), ('Proximal', 'Alternating'))
plt_palm_2, = axs[1].plot(palm_likelihood, 'b-')
plt_altmin_2, = axs[1].plot(altmin_likelihood, 'r-')
axs[1].legend((plt_palm, plt_altmin), ('Proximal', 'Alternating'))
print('Complete.')

#path = "/home/addison/Python/almm/results"
#plt.savefig(join(path, "comparison-"+dt.now().strftime("%y%b%d_%H%M")+".svg"))
