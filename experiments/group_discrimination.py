#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from experiments.utility import load_group_results

MODEL_ORDER = [12, 12, 16]
NUM_COMPONENTS = [10, 15, 10]
PENALTY_PARAMETER = 0.1

skf = StratifiedKFold(n_splits=5)
score, lr_coef = [], []
for p_i, r_i in zip(MODEL_ORDER, NUM_COMPONENTS):
    _, coef, labels = load_group_results(model_order=p_i, num_components=r_i, penalty_parameter=PENALTY_PARAMETER)
    labels = np.array(labels)
    score_i, lr_coef_i = [], []
    for train_index, test_index in skf.split(coef, labels):
        mixing_coef_train, mixing_coef_test = coef[list(train_index)], coef[list(test_index)]
        labels_train, labels_test = labels[list(train_index)], labels[list(test_index)]
        sklr = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced')
        sklr.fit(mixing_coef_train, labels_train)
        score_i.append(sklr.score(mixing_coef_test, labels_test))
        lr_coef_i.append(sklr.coef_)
    score.append(np.mean(score_i))
    lr_coef.append(lr_coef_i)
