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
        
# Implement solver with cross-validation
mu_list = [1e-5, 5e-5, 1e-4, 5e-4]
print('Fitting ALMM model...')
t1 = timer()
almm_model = Almm(tol=1e-3, solver='alt_min',  verbose=True)
D_pred, C_pred, likelihood, params = almm_model.fit_cv(x, p, r, mu=mu_list, 
                                                       k=5, return_path=True, 
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
axs.legend(mu_list)
print('Complete.')

path = "/home/addison/Python/almm/results"
plt.savefig(join(path, "cv_mu_altmin-"+dt.now().strftime("%y%b%d_%H%M")+".svg"))
