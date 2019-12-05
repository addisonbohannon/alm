#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from experiments.utility import load_isruc_results, load_results, save_results

score, lr_coef = [], []
skf = StratifiedKFold(n_splits=5)
for subj in range(1, 11):
    _, subj_coef, subj_labels = load_isruc_results(subj)
    subj_score, subj_lr_coef = [], []
    for subj_coef_k in subj_coef:
        subj_score_k, subj_lr_coef_k = [], []
        for train_index, test_index in skf.split(subj_coef_k, subj_labels):
            mixing_coef_train, mixing_coef_test = subj_coef_k[train_index], subj_coef_k[test_index]
            labels_train, labels_test = subj_labels[train_index], subj_labels[test_index]
            sklr = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced')
            sklr.fit(mixing_coef_train, labels_train)
            subj_score_k.append(sklr.score(mixing_coef_test, labels_test))
            subj_lr_coef_k.append(sklr.coef_)
        subj_score.append(np.mean(subj_score_k))
        subj_lr_coef.append(subj_lr_coef_k)
    score.append(subj_score)
    lr_coef.append(subj_lr_coef)
    
###################
# save results
###################
save_results([score, lr_coef], 'discrimination.pickle')

###################
# load results
###################
# score, lr_coef = load_results('discrimination.pickle')
    
max_score = [max(subj_score) for subj_score in score]
avg_score = sum(max_score)/len(max_score)


