#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 1 Jul 19
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
        
# Implement solver with multiple runs
print('Fitting ALMM model...')
t1 = timer()
almm_model = Almm(tol=1e-3, verbose=True)
D_pred, C_pred, L = almm_model.fit_k(x, p, r, mu=1e-1, return_path=True, 
                                     return_all=True)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')

# Computing dictionary error
print('Computing dictionary error...', end=" ", flush=True)
fig, axs = plt.subplots()
axs.set_xlabel('Iteration')
axs.set_ylabel('Dictionary Error')
for i, Di in enumerate(D_pred):
    loss=[]
    for s, Dis in enumerate(Di):
        Dis_pred = np.zeros([r, p, d, d])
        for j in range(r):
            Dis_pred[j] = unstack_ar_coef(Dis[j])
        d_loss, _, _ = dict_distance(D, Dis_pred)
        loss.append(d_loss)
    axs.plot(loss)
print('Complete.')