#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from experiments.utility import load_results

score, lr_coef = [], []
skf = StratifiedKFold(n_splits=5)
for subj in range(10):
    _, subj_coef, subj_labels = load_results(subj)
    subj_score, subj_lr_coef = [], []
    for train_index, test_index in skf.split(subj_coef, subj_labels):
        mixing_coef_train, mixing_coef_test = subj_coef[train_index], subj_coef[test_index]
        labels_train, labels_test = subj_labels[train_index], subj_labels[test_index]
        sklr = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced')
        sklr.fit(mixing_coef_train, labels_train)
        subj_score.append(sklr.score(mixing_coef_test, labels_test))
        subj_lr_coef.append(sklr.coef_)
    score.append(np.mean(subj_score))
    lr_coef.append(subj_lr_coef)
