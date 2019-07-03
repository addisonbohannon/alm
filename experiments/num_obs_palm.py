#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 3 Jul 19
"""

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from almm import Almm
from sampler import almm_sample
from utility import unstack_ar_coef, dict_distance

n = 400
dn = 100
m = 1000
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
        
# Implement solver for increasing number of observations
D_pred = []
C_pred = []
L = []
for n_i in range(0, n, dn):
    print('Fitting ALMM model for n=' + str(n_i+dn) + '...')
    t1 = timer()
    almm_model = Almm(tol=1e-3, verbose=True)
    Di_pred, Ci_pred, Li = almm_model.fit_k(x[:(n_i+dn-1)], p, r, mu=1e-2, 
                                            return_path=True, return_all=True)
    t2 = timer()
    D_pred.append(Di_pred)
    C_pred.append(Ci_pred)
    L.append(Li)
    print('Complete.', end=" ", flush=True)
    print('Elapsed time: ' + str(t2-t1) + 's')

# Computing dictionary error
print('Computing dictionary error...', end=" ", flush=True)
fig, axs = plt.subplots()
axs.set_xlabel('Iteration')
axs.set_ylabel('Dictionary Error')
for clr, Di in zip(['r-', 'b-', 'g-', 'y-'], D_pred):
    for Dis in Di:
        loss=[]
        for Disj in Dis:
            Disj_pred = np.zeros([r, p, d, d])
            for j in range(r):
                Disj_pred[j] = unstack_ar_coef(Disj[j])
            d_loss, _, _ = dict_distance(D, Disj_pred)
            loss.append(d_loss)
        axs.plot(loss, clr)
print('Complete.')