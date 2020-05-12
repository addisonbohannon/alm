#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM Copyright (C) 2019  Addison Bohannon
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from experiments.utility import load_results, save_results

SUBJS = [8]
INNER_N_SPLITS = 5
OUTER_N_SPLITS = 5
NUM_STARTS = 5

outer_score = np.zeros([len(SUBJS), OUTER_N_SPLITS])
inner_score = np.zeros([NUM_STARTS, INNER_N_SPLITS])

for i, subj in enumerate(SUBJS):
    inner_skcv = StratifiedKFold(n_splits=INNER_N_SPLITS)
    outer_skcv = StratifiedKFold(n_splits=OUTER_N_SPLITS)
    _, subj_coef, subj_labels = load_results('S' + str(subj) + '-mvar.pickle')
    for outer_cv, (outer_train_idx, outer_test_idx) in enumerate(outer_skcv.split(np.zeros_like(subj_labels), subj_labels)):
        outer_train_labels, outer_test_labels = subj_labels[outer_train_idx], subj_labels[outer_test_idx]
        for inner_cv, (inner_train_idx, inner_test_idx) in enumerate(inner_skcv.split(np.zeros_like(outer_train_labels), outer_train_labels)):
            inner_train_labels, inner_test_labels = outer_train_labels[inner_train_idx], outer_train_labels[inner_test_idx]
            for start, subj_coef_per_start in enumerate(subj_coef):
                inner_sklr = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced')
                inner_sklr.fit(subj_coef_per_start[outer_train_idx[inner_train_idx]], inner_train_labels)
                inner_score[start, inner_cv] = inner_sklr.score(subj_coef_per_start[outer_train_idx[inner_test_idx]], inner_test_labels)
        best_start = np.argmax(np.mean(inner_score, axis=1))
        outer_sklr = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced')
        outer_sklr.fit(subj_coef[best_start][outer_train_idx], outer_train_labels)
        outer_score[i, outer_cv] = outer_sklr.score(subj_coef[best_start][outer_test_idx], outer_test_labels)
score = np.mean(outer_score, axis=1)
    
###################
# save results
###################
save_results(score, 'discrimination-mvar.pickle')

###################
# load results
###################
# score = load_results('discrimination-mvar.pickle')


