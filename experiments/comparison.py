#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 2 Jul 19
"""

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from almm import Almm
from sampler import almm_sample
from utility import unstack_ar_coef, dict_distance

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

# Fit model with proximal solver
print('Fitting ALMM model...')
t1 = timer()
almm_model = Almm(tol=1e-3, verbose=True)
D_palm, C_palm, likelihood = almm_model.fit_k(x, p, r, mu=1e-1, 
                                              return_path=True, 
                                              return_all=True)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')
        
# Fit model with alternating solver
print('Fitting ALMM model...')
t3 = timer()
almm_model = Almm(tol=1e-3, max_iter=int(1e2), solver='alt_min', verbose=True)
D_altmin, C_altmin, likelihood = almm_model.fit_k(x, p, r, mu=1e-1, 
                                                  return_path=True, 
                                                  return_all=True)
t4 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t4-t3) + 's')

# Compute dictionary error
print('Computing dictionary error...', end=" ", flush=True)
fig, axs = plt.subplots()
axs.set_xlabel('Iteration')
axs.set_ylabel('Dictionary Error')
for i, (Dp_i, Da_i) in enumerate(zip(D_palm, D_altmin)):
    # Proximal error
    loss=[]
    for s, Dis in enumerate(Dp_i):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_ar_coef(Dis[j])
        d_loss, _, _ = dict_distance(D, Dis_pred)
        loss.append(d_loss)
    plt_palm, = axs.plot(loss, 'b-')
    # Alternating error
    loss=[]
    for s, Dis in enumerate(Da_i):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_ar_coef(Dis[j])
        d_loss, _, _ = dict_distance(D, Dis_pred)
        loss.append(d_loss)
    plt_altmin, = axs.plot(loss, 'r-')
axs.legend((plt_palm, plt_altmin), ('Proximal', 'Alternating'))
print('Complete.')