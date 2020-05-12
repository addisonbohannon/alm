#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM Copyright (C) 2019  Addison Bohannon
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from experiments.utility import load_results, save_results

SUBJS = [8]
CV_SPLITS = 5

score = []
for subj in SUBJS:
    ar_comps, labels = load_results('S' + str(subj) + '-var.pickle')
    var_features = np.reshape(ar_comps, [len(ar_comps), -1])
    sklr = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced')
    cv_score = cross_val_score(sklr, var_features, labels, cv=CV_SPLITS)
    score.append(np.mean(cv_score))
    
###################
# save results
###################
save_results(score, 'discrimination-var.pickle')

###################
# load results
###################
# score = load_results('discrimination-var.pickle')


