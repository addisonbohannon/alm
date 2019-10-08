#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import numpy as np
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

SUBJECTS = ['subj' + str(subj) for subj in np.setdiff1d(range(1, 11), [9])]
DATA_PATH = '/home/addison/Python/almm/results/application-mu-e-1/'
INFO_PATH = '/home/addison/Python/almm/ISRUC-SLEEP/'
subject_score = []
subject_cm = []
for subject in SUBJECTS:
    data = sio.loadmat(join(DATA_PATH, subject + '_results_palm.mat'))
    mixing_coef = data['C_palm']
    mixing_coef = [mixing_coef_i[-1] for mixing_coef_i in mixing_coef]
    del data
    subj_data = sio.loadmat(join(INFO_PATH, subject + '.mat'))
    labels = np.squeeze(subj_data['Y'])
    del subj_data
    score = []
    cm = []
    for mixing_coef_i in mixing_coef:
        mixing_coef_i[-1]
        skf = StratifiedKFold(n_splits=5)
        score_i = []
        cm_i = []
        for train_index, test_index in skf.split(mixing_coef_i, labels):
            mixing_coef_train, mixing_coef_test = mixing_coef_i[train_index], mixing_coef_i[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            sklr = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced')
            sklr.fit(mixing_coef_train, labels_train)
            score_i.append(sklr.score(mixing_coef_test, labels_test))
            cm_i.append(confusion_matrix(labels_test, sklr.predict(mixing_coef_test)))
        score.append(np.mean(score_i))
        cm.append(np.sum(np.array(cm_i), axis=0))
    subject_score.append(score)
    subject_cm.append(cm)
