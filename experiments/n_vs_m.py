#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import getpid
from os.path import join
from datetime import datetime as dt
from itertools import product
from multiprocessing import Pool
import pickle
import numpy.random as nr
from alm.alm import Alm
from alm.utility import unstack_coef, initialize_components
from experiments.sampler import alm_sample
from experiments.utility import component_distance

n = [10, 100, 1000]
m = [10, 100, 1000, 10000]
d = 5
r = 10
p = 2
s = 3
N = 10
k = 10
mu = 1e-2
NUM_PROCESSES = 3


def n_vs_m(experiment_id):
    # Set seed
    nr.seed()
    # Generate alm samples
    x, _, D = alm_sample(max(n), max(m), d, r, p, s, coef_condition=1e2, component_condition=1e2)
    # Initialize variables
    palm_component_error, palm_likelihood = [], []
    altmin_component_error, altmin_likelihood = [], []
    bcd_component_error, bcd_likelihood = [], []
    # Generate initial autoregressive component estimate
    D_0 = [initialize_components(r, p, d) for _ in range(k)]
    # Implement solver for varying number of observations and length of observations
    for (n_i, m_i) in product(n, m):
        # PALM
        alm_model = Alm(solver='palm')
        Di_palm, _, Li_palm, _ = alm_model.fit(x[:(n_i-1), :(m_i-1), :], p, r, mu, num_starts=k,
                                                initial_component=D_0, return_all=True)
        palm_likelihood.append(Li_palm)
        loss_palm = []
        for Dis_palm in Di_palm:
            D_pred = [unstack_coef(Dj) for Dj in Dis_palm]
            d_loss, _, _ = component_distance(D, D_pred)
            loss_palm.append(d_loss)
        palm_component_error.append(loss_palm)
        # AltMin
        alm_model = Alm(solver='altmin')
        Di_altmin, _, Li_altmin, _ = alm_model.fit(x[:(n_i-1), :(m_i-1), :], p, r, mu, num_starts=k,
                                                    initial_component=D_0, return_all=True)
        altmin_likelihood.append(Li_altmin)
        loss_altmin = []
        for Dis_altmin in Di_altmin:
            D_pred = [unstack_coef(Dj) for Dj in Dis_altmin]
            d_loss, _, _ = component_distance(D, D_pred)
            loss_altmin.append(d_loss)
        altmin_component_error.append(loss_altmin)
        # BCD
        alm_model = Alm(solver='bcd')
        Di_bcd, _, Li_bcd, _ = alm_model.fit(x[:(n_i-1), :(m_i-1), :], p, r, mu, num_starts=k,
                                              initial_component=D_0, return_all=True)
        bcd_likelihood.append(Li_bcd)
        loss_bcd = []
        for Dis_bcd in Di_bcd:
            D_pred = [unstack_coef(Dj) for Dj in Dis_bcd]
            d_loss, _, _ = component_distance(D, D_pred)
            loss_bcd.append(d_loss)
        bcd_component_error.append(loss_bcd)
    # Save results
    path = "/home/addison/Python/almm/results"
    with open(join(path, "n_vs_m-log-" + dt.now().strftime("%y%b%d_%H%M") + '-' + str(getpid()) + ".pickle"), 'wb') as f:
        pickle.dump([palm_component_error, altmin_component_error, bcd_component_error, palm_likelihood,
                     altmin_likelihood, bcd_likelihood], f)

    return True

with Pool() as pool:
    pool.map(n_vs_m, range(NUM_PROCESSES))
