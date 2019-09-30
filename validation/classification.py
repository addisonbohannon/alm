#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy.random as nr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

NUM_OBS = 100
NUM_COMPONENTS = 10

mixing_coef = nr.randn(NUM_OBS, NUM_COMPONENTS)
labels = nr.choice(['wake', 'N1', 'N2', 'N3', 'REM'], (NUM_OBS,))

skf = StratifiedKFold(n_splits=5)
sklr, score, estimate = [],  [], []
for train_index, test_index in skf.split(mixing_coef, labels):
    mixing_coef_train, mixing_coef_test = mixing_coef[train_index], mixing_coef[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    sklr.append(LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced'))
    sklr[-1].fit(mixing_coef_train, labels_train)
    score.append(sklr[-1].score(mixing_coef_test, labels_test))
    estimate.append(sklr[-1].predict_proba(mixing_coef_train))
