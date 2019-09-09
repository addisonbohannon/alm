#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Addison Bohannon
Project: Autoregressive Linear Mixture Model (ALMM)
Date: 3 Jul 19
"""

from os.path import join
from datetime import datetime as dt
from timeit import default_timer as timer
from itertools import product
import pickle
import numpy as np
import matplotlib.pyplot as plt
from almm.almm import Almm
from almm.sampler import almm_sample
from almm.utility import unstack_ar_coef, dict_distance

n = 400
dn = 100
m = 1000
dm = 200
d = 5
r = 10
p = 2
s = 3
N = 5

palm_dict_loss = []
palm_likelihood = []
altmin_dict_loss = []
altmin_likelihood = []
for i in range(N):
    # Generate almm samples
    print('Generating ALMM samples...', end=" ", flush=True)
    t1 = timer()
    x, _, D = almm_sample(n, m, d, r, p, s, coef_cond=1e1, dict_cond=1e1)
    t2 = timer()
    print('Complete.', end=" ", flush=True)
    print('Elapsed time: ' + str(t2-t1) + 's')
        
    # Implement solver for increasing number of observations
    D_palm = []
    L_palm = []
    D_altmin = []
    L_altmin = []
    for (n_i, m_i) in product(range(0, n, dn), range(0, m, dm)):
        print(n_i, m_i)
        # PALM
        print('Fitting ALMM model for n=' + str(n_i+dn) + '...')
        t1 = timer()
        almm_model = Almm(tol=1e-3)
        Di_palm, _, Li_palm = almm_model.fit_k(x[:(n_i+dn-1), :(m_i+dm-1), :], 
                                               p, r, mu=1e-2, return_all=True)
        t2 = timer()
        L_palm.append(Li_palm)        
        loss_palm = []
        for Dis_palm in Di_palm:
            D_pred = [unstack_ar_coef(Dj) for Dj in Dis_palm]
            d_loss, _, _ = dict_distance(D, D_pred)
            loss_palm.append(d_loss)
        D_palm.append(loss_palm)
        print('Complete.', end=" ", flush=True)
        print('Elapsed time: ' + str(t2-t1) + 's')
        
        # AltMin
        print('Fitting ALMM model for n=' + str(n_i+dn) + '...')
        t3 = timer()
        almm_model = Almm(tol=1e-3, solver='alt_min')
        Di_altmin, _, Li_altmin = almm_model.fit_k(x[:(n_i+dn-1), :(m_i+dm-1), :], 
                                                   p, r, mu=1e-2, 
                                                   return_all=True)
        t4 = timer()
        L_altmin.append(Li_altmin)
        loss_altmin = []
        for Dis_altmin in Di_altmin:
            D_pred = [unstack_ar_coef(Dj) for Dj in Dis_altmin]
            d_loss, _, _ = dict_distance(D, D_pred)
            loss_altmin.append(d_loss)
        D_altmin.append(loss_altmin)
        print('Complete.', end=" ", flush=True)
        print('Elapsed time: ' + str(t4-t3) + 's')
        
    palm_dict_loss.append(D_palm)
    palm_likelihood.append(L_palm)
    altmin_dict_loss.append(D_altmin)
    altmin_likelihood.append(L_altmin)
        
path = "/home/addison/Python/almm/results"
with open(join(path, "n_vs_m-"+dt.now().strftime("%y%b%d_%H%M")+".svg"), 'wb') as f:
    pickle.dump([palm_dict_loss, palm_likelihood, altmin_dict_loss, 
                 altmin_likelihood], f)