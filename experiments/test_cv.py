#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:21:22 2019

@author: addison
"""

import numpy as np
import matplotlib.pyplot as plt
from palm_solver import Almm
from utility import (unstack_ar_coeffs, dictionary_distance, 
                     check_almm_condition)
from sampler import almm_iid_sample

if __name__ == '__main__':

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
    p_list = [1, 2]
    r_list = [5, 10]
    mu_list = [1e-2, 5e-2, 1e-1]
    almm_model = Almm(tol=1e-3, coef_penalty_type='l1')
    D_pred, C_pred, likelihood = almm_model.fit_cv(x, p=p_list, r=r_list,
                                                   mu=mu_list, k=2,
                                                   return_path=False, 
                                                   return_all=False)
    
#    for Ds in D_pred:  
#        loss = []
#        for Di in Ds:
#            Di_pred = np.zeros([r, p, d, d])
#            for j in range(r):
#                Di_pred[j] = unstack_ar_coeffs(Di[j])
#            d_loss, _, _ = dictionary_distance(D, Di_pred)
#            loss.append(d_loss)
#        ax = plt.plot(loss)