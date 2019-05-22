#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:21:22 2019

@author: addison
"""

import numpy as np
from utility import (unstack_ar_coeffs, dictionary_distance, 
                     check_almm_condition)
from sampler import almm_iid_sample
import matplotlib.pyplot as plt
from palm_solver import Almm

n = 100
m = 800
d = 5
r = 10
p = 2
s = 3

# Generate almm sample
x, C, D = almm_iid_sample(n, m, d, r, p, s)
flag = True
while flag:
    k1, k2 = check_almm_condition(x, D, C)
    if k1 < 1e1 and np.all(k2 < 1e1):
        flag = False
    else:
        print('fail')
        x, C, D = almm_iid_sample(n, m, d, r, p, s)
        
# Fit almm model
almm_model = Almm(x, p, r, 5e-2, tol=1e-3, coef_penalty_type='l1', 
                  return_path=True, likelihood_path=True)

for D_path in almm_model.D:
    loss = []
    for Di in D_path:
        D_pred = np.zeros([r, p, d, d])
        for j in range(r):
            D_pred[j] = unstack_ar_coeffs(Di[j])
        d_loss, _, _ = dictionary_distance(D, D_pred)
        loss.append(d_loss)
    ax = plt.plot(loss)
    
loss = []
likelihood = []
for D_path, likelihood_path in zip(almm_model.D, almm_model.likelihood):
    D_pred = np.zeros([r, p, d, d])
    for j in range(r):
        D_pred[j] = unstack_ar_coeffs(D_path[-1][j])
    d_loss, _, _ = dictionary_distance(D, D_pred)
    loss.append(d_loss)
    likelihood.append(likelihood_path[-1])