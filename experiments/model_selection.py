#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from alm.alm import Alm
from experiments.utility import load_isruc_data

SUBJ = 8
MODEL_ORDER = 12
NUM_COMPONENTS = 6
PENALTY_PARAM = 1e-1
NUM_STARTS = 1
DATA_DIR = '/home/addison/Python/almm/data/'

# Load data
data, _ = load_isruc_data(SUBJ)
# Fit ALM model
alm_model = Alm(tol=1e-3, solver='palm', verbose=False)
D, C, nll, _ = alm_model.fit(data, MODEL_ORDER, NUM_COMPONENTS, PENALTY_PARAM, num_starts=NUM_STARTS, return_path=True)
