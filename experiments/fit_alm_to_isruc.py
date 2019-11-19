#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import pickle
from alm.alm import Alm
from experiments.utility import load_isruc_data

MODEL_ORD = 4
NUM_COMPS = 10
SUBJECTS = range(1, 11)
PENALTY_PARAM = 1e-1
NUM_STARTS = 5
RESULTS_PATH = '/home/addison/Python/almm/results/individual'

for subj in SUBJECTS:
    print(subj)
    data, labels = load_isruc_data(subj)
    alm_model = Alm(tol=1e-3, solver='palm')
    ar_comps, mixing_coef, nll, _ = alm_model.fit(data, MODEL_ORD, NUM_COMPS, PENALTY_PARAM, num_starts=NUM_STARTS,
                                                  return_all=True)
    with open(join(RESULTS_PATH, 'S' + str(subj) + '.pickle'), 'wb') as file:
        pickle.dump([ar_comps, mixing_coef, nll, labels], file)
