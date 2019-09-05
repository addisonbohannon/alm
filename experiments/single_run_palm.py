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
scale = 1 / r
mu = d**(1/2) / (n * m * scale)

# Generate almm sample
print('Generating ALMM sample...', end=" ", flush=True)
t1 = timer()
x, C, D = almm_sample(n, m, d, r, p, scale, coef_cond=1e2, dict_cond=1e2)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')
        
# Implement solver
print('Fitting ALMM model...')
t1 = timer()
almm_model = Almm(tol=1e-3, verbose=True)
D_pred, C_pred, likelihood = almm_model.fit(x, p, r, mu=mu, return_path=True)
t2 = timer()
print('Complete.', end=" ", flush=True)
print('Elapsed time: ' + str(t2-t1) + 's')

# Compute dictionary loss
print('Computing dictionary error...', end=" ", flush=True)
loss = []
for i, Di in enumerate(D_pred):
    Di_pred = np.zeros([r, p, d, d])
    for j in range(r):
        Di_pred[j] = unstack_ar_coef(Di[j])
    d_loss, _, _ = dict_distance(D, Di_pred)
    loss.append(d_loss)
print('Complete.')

# Plot results
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
axs[0].plot(loss)
axs[0].set_ylabel('Dictionary Error')
axs[1].plot(likelihood)
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Negative Log Likelihood')

#path = "/home/addison/Python/almm/results"
#plt.savefig(join(path, "single_run_palm-"+dt.now().strftime("%y%b%d_%H%M")+".svg"))
