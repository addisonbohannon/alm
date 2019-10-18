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
import matplotlib.colors as colors
from alm.utility import unstack_coef, gram_matrix
from experiments.utility import component_distance
from experiments.sampler import autoregressive_sample

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
ANY = [k for k in range(OBS_LENGTH) if DELTA_MIN <= frequency(k) < BETA_MAX]

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
        self.dtf, self.connectivity = [], []
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
        N = int(OBS_LENGTH/2)
        for component_k in self.components:
            dtf, alpha, beta, delta, theta = [], [], [], [], []
            for component_j in component_k:
                component_j = unstack_coef(component_j)
                component_j = np.concatenate((np.expand_dims(np.eye(SIGNAL_DIM), 0), -component_j), axis=0)
                H = [sl.inv(A) for A in sf.fft(component_j, n=OBS_LENGTH, axis=0)]
                dtf.append((np.abs(H) / sl.norm(H, axis=-1, keepdims=True))**2)
                psd = [max(np.abs(sl.eigvals(H[k]))) for k in range(N)]
                alpha.append(np.mean([psd[k] for k in ALPHA]))
                beta.append(np.mean([psd[k] for k in BETA]))
                delta.append(np.mean([psd[k] for k in DELTA]))
                theta.append(np.mean([psd[k] for k in THETA]))
            self.dtf.append(dtf)
            self.alpha.append(alpha)
            self.beta.append(beta)
            self.delta.append(delta)
            self.theta.append(theta)
            
    def spectral_analysis(self):
        if self.alpha is [] or self.beta is [] or self.delta is [] or self.theta is []:
            self.analyze()
        self.power = [np.array(power).T for power in zip(self.delta, self.theta, self.alpha, self.beta)]
        self.connectivity = [[sl.norm([dtf_j[k] for k in ANY], axis=0) for dtf_j in dtf_k] for dtf_k in self.dtf]
            

DATA_PATH = '/home/addison/Python/alm/results/application-mu-e-1/'
INFO_PATH = '/home/addison/Python/alm/ISRUC-SLEEP/'
subj = []
for subject in range(1, 11):
    subj.append(Subject(subject, DATA_PATH, INFO_PATH))
    subj[-1].classify()
#    subj[-1].best_k()
    subj[-1].analyze()
    subj[-1].spectral_analysis()
    
fig, axs = plt.subplots(5, NUM_COMPONENTS)
images = []
for i, connectivity_matrix in enumerate(subj[7].connectivity):
    axs[i, 0].set_ylabel('Start: ' + str(i))
    for j, connectivity in enumerate(connectivity_matrix):
        if i == 1:
            axs[-1, j].set_xlabel('Component: ' + str(j))
        axs[i, j].set_xticks([], [])
        axs[i, j].set_yticks([], [])
        images.append(axs[i, j].imshow(connectivity,
                   norm=colors.LogNorm(vmin=1e-3, vmax=4)))
fig.colorbar(images[0], ax=axs)

#fig, axs = plt.subplots(2, 5)
#images = []
#for i, components in enumerate(subj[7].components[:2]):
#    axs[i, 0].set_ylabel('Start: ' + str(i))
#    for j, component in enumerate(components[:5]):
#        signal = autoregressive_sample(5*SAMPLING_RATE, SIGNAL_DIM, SIGNAL_DIM**(-1/2), unstack_coef(component))
#        if i == 1:
#            axs[-1, j].set_xlabel('Component: ' + str(j))
#        axs[i, j].set_xticks([], [])
#        images.append(axs[i, j].plot(signal + np.arange(0, 12, 2)))
        
fig, axs = plt.subplots(2, 1)
images = []
for j, component in enumerate(subj[7].components[0][:2]):
    signal = autoregressive_sample(10*SAMPLING_RATE, SIGNAL_DIM, SIGNAL_DIM**(-1/2), unstack_coef(component))
    axs[j].set_xticks(np.arange(0, 10*SAMPLING_RATE, SAMPLING_RATE))
    axs[j].set_xticklabels(np.arange(10))
    axs[j].set_ylabel('Component: ' + str(j))
    axs[j].set_yticks(np.arange(0, 18, 3))
    axs[j].set_yticklabels(['FL', 'PL', 'OL', 'FR', 'PR', 'OR'])
    images.append(axs[j].plot(signal + np.arange(0, 18, 3)))
        
fig, axs = plt.subplots(5, 1)
images = []
class_label = ['Awake', 'N1', 'N2', 'N3', 'REM']
for i, label in enumerate(np.unique(subj[7].labels)):
    axs[i].set_ylabel(class_label[i])
    axs[i].set_xticks(np.arange(0, 10*SAMPLING_RATE, SAMPLING_RATE))
    axs[i].set_xticklabels(np.arange(10))
    mixing_coef = np.mean(subj[7].mixing_coef[0][subj[7].labels==label], axis=0)
    ar_coef = unstack_coef(np.tensordot(mixing_coef, subj[7].components[0], axes=1))
    signal = autoregressive_sample(10*SAMPLING_RATE, SIGNAL_DIM, SIGNAL_DIM ** (-1 / 2), ar_coef)
    images.append(axs[i].plot(signal + np.arange(0, 18, 3)))
    
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