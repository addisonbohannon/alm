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
from almm.almm import Almm
from almm.sampler import almm_sample
from almm.utility import unstack_ar_coef, dict_distance

n = 1000
dn = 200
m = 10000
dm = 2000
d = 5
r = 10
p = 2
s = 3
N = 10
k = 10
mu = 1e-2

palm_dict_loss, palm_likelihood = [], []
altmin_dict_loss, altmin_likelihood = [], []
bcd_dict_loss, bcd_likelihood = [], []
for i in range(N):
    # Generate almm samples
    print('Generating ALMM samples...', end=" ", flush=True)
    t1 = timer()
    x, _, D = almm_sample(n, m, d, r, p, s, coef_cond=1e2, dict_cond=1e2)
    t2 = timer()
    print('Complete.', end=" ", flush=True)
    print('Elapsed time: ' + str(t2-t1) + 's')
        
    # Implement solver for increasing number of observations
    D_palm, L_palm = [], []
    D_altmin, L_altmin = [], []
    D_bcd, L_bcd = [], []
    for (n_i, m_i) in product(range(0, n, dn), range(0, m, dm)):
        print(n_i+dn, m_i+dm)
        
        # PALM
        print('Fitting PALM...')
        t1 = timer()
        almm_model = Almm(tol=1e-3)
        Di_palm, _, Li_palm, _ = almm_model.fit(x[:(n_i+dn-1), :(m_i+dm-1), :], 
                                                p, r, mu=mu, k=k, 
                                                return_all=True)
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
        print('Fitting AltMin...')
        t3 = timer()
        almm_model = Almm(tol=1e-3, solver='altmin')
        Di_altmin, _, Li_altmin, _ = almm_model.fit(x[:(n_i+dn-1), :(m_i+dm-1), :], 
                                                    p, r, mu=mu, k=k, 
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
        
        # BCD
        print('Fitting BCD...')
        t5 = timer()
        almm_model = Almm(tol=1e-3, solver='bcd')
        Di_bcd, _, Li_bcd, _ = almm_model.fit(x[:(n_i+dn-1), :(m_i+dm-1), :], 
                                              p, r, mu=mu, k=k, 
                                              return_all=True)
        t6 = timer()
        L_bcd.append(Li_bcd)
        loss_bcd = []
        for Dis_bcd in Di_bcd:
            D_pred = [unstack_ar_coef(Dj) for Dj in Dis_bcd]
            d_loss, _, _ = dict_distance(D, D_pred)
            loss_bcd.append(d_loss)
        D_bcd.append(loss_bcd)
        print('Complete.', end=" ", flush=True)
        print('Elapsed time: ' + str(t6-t5) + 's')

    palm_dict_loss.append(D_palm)
    palm_likelihood.append(L_palm)
    altmin_dict_loss.append(D_altmin)
    altmin_likelihood.append(L_altmin)
    bcd_dict_loss.append(D_bcd)
    bcd_likelihood.append(L_bcd)
        
path = "/home/addison/Python/almm/results"
with open(join(path, "n_vs_m-"+dt.now().strftime("%y%b%d_%H%M")+".pickle"), 'wb') as f:
    pickle.dump([D_palm, D_altmin, D_bcd, L_palm, L_altmin, L_bcd], f)
