#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join
import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import scipy.fftpack as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from almm.utility import unstack_coef, gram_matrix
from validation.utility import component_distance

NUM_COMPONENTS = 10
MODEL_ORDER = 12
SIGNAL_DIM = 6
SAMPLING_RATE = 200
OBS_LENGTH = 100
DELTA_MIN = 0
DELTA_MAX = 4
THETA_MIN = 4
THETA_MAX = 8
ALPHA_MIN = 8
ALPHA_MAX = 13
BETA_MIN = 13
BETA_MAX = 30


def frequency(k): return k * SAMPLING_RATE / OBS_LENGTH


DELTA = [k for k in range(OBS_LENGTH) if DELTA_MIN <= frequency(k) < DELTA_MAX]
THETA = [k for k in range(OBS_LENGTH) if THETA_MIN <= frequency(k) < THETA_MAX]
ALPHA = [k for k in range(OBS_LENGTH) if ALPHA_MIN <= frequency(k) < ALPHA_MAX]
BETA = [k for k in range(OBS_LENGTH) if BETA_MIN <= frequency(k) < BETA_MAX]

class Subject:

    def __init__(self, subj_id, data_path, info_path):
        self.id = subj_id
        data = sio.loadmat(join(data_path, 'subj' + str(subj_id) + '_results_palm.mat'))
        self.components = [D[-1] for D in data['D_palm']]
        self.pairwise_component_error = gram_matrix(self.components, lambda d1, d2: component_distance(d1, d2)[0])
        self.mixing_coef = [C[-1] for C in data['C_palm']]
        del data
        subj_data = sio.loadmat(join(info_path, 'subj' + str(subj_id) + '.mat'))
        self.labels = np.squeeze(subj_data['Y'])
        del subj_data
        self.score, self.lr_coef = [], []
        self.alpha, self.beta, self.delta, self.theta = [], [], [], []
        self.k = None

    def classify(self):
        skf = StratifiedKFold(n_splits=5)
        for mixing_coef_k in self.mixing_coef:
            score_k, sklr_coef = [], []
            for train_index, test_index in skf.split(mixing_coef_k, self.labels):
                mixing_coef_train, mixing_coef_test = mixing_coef_k[train_index], mixing_coef_k[test_index]
                labels_train, labels_test = self.labels[train_index], self.labels[test_index]
                sklr = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced')
                sklr.fit(mixing_coef_train, labels_train)
                score_k.append(sklr.score(mixing_coef_test, labels_test))
                sklr_coef.append(sklr.coef_)
            self.score.append(np.mean(score_k))
            self.lr_coef.append(sklr_coef[np.argmax(score_k)])

    def best_k(self):
        self.k = np.argmax(self.score)
        self.score = self.score[self.k]
        self.lr_coef = self.lr_coef[self.k]
        self.components = self.components[self.k]
        self.mixing_coef = self.mixing_coef[self.k]

    def analyze(self):
#        if self.k is None:
#            raise TypeError('Must run best_k before analyze.')
        N = int(OBS_LENGTH/2)
        for component_k in self.components:
            alpha, beta, delta, theta = [], [], [], []
            for component_j in component_k:
                component_j = unstack_coef(component_j)
                component_j = np.concatenate((np.expand_dims(np.eye(SIGNAL_DIM), 0), -component_j), axis=0)
                fD = sf.fft(component_j, n=OBS_LENGTH, axis=0)
                psd = [max(np.abs(sl.eigvals(fD[k]))**(-2)) for k in range(N)]
                alpha.append(sum([psd[k] for k in ALPHA]))
                beta.append(sum([psd[k] for k in BETA]))
                delta.append(sum([psd[k] for k in DELTA]))
                theta.append(sum([psd[k] for k in THETA]))
            self.alpha.append(alpha)
            self.beta.append(beta)
            self.delta.append(delta)
            self.theta.append(theta)
            
    def spectral_analysis(self):
        if self.alpha is [] or self.beta is [] or self.delta is [] or self.theta is []:
            self.analyze()
        self.power = [np.array(power).T for power in zip(self.delta, self.theta, self.alpha, self.beta)]
            

DATA_PATH = '/home/addison/Python/almm/results/application-mu-e-1/'
INFO_PATH = '/home/addison/Python/almm/ISRUC-SLEEP/'
subj = []
for subject in np.setdiff1d(range(1, 11), [9]):
    subj.append(Subject(subject, DATA_PATH, INFO_PATH))
#    subj[-1].classify()
#    subj[-1].best_k()
    subj[-1].analyze()
    subj[-1].spectral_analysis()
    
#fig, axs = plt.subplots(4, NUM_COMPONENTS)
#images = []
#for i in range(NUM_COMPONENTS):
#    images.append(axs[0, i].imshow(D_delta[i], cmap='cool', vmin=VMIN, vmax=VMAX))
#    images.append(axs[1, i].imshow(D_theta[i], cmap='cool', vmin=VMIN, vmax=VMAX))
#    images.append(axs[2, i].imshow(D_alpha[i], cmap='cool', vmin=VMIN, vmax=VMAX))
#    images.append(axs[3, i].imshow(D_beta[i], cmap='cool', vmin=VMIN, vmax=VMAX))
#    axs[-1, i].set_xlabel('Comp.' + str(i))
#axs[0, 0].set_ylabel('Delta (0-4 Hz)')
#axs[1, 0].set_ylabel('Theta (4-8 Hz)')
#axs[2, 0].set_ylabel('Alpha (8-13 Hz)')
#axs[3, 0].set_ylabel('Beta (13-30 Hz)')
#fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
