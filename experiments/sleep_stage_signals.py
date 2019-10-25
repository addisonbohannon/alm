#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_coef
from experiments.sampler import autoregressive_sample
from experiments.utility import load_individual_results

SIGNAL_DIM = 6
SAMPLING_RATE = 200
SAMPLE_LEN = 2
CLASS_LABEL = ['Awake', 'N1', 'N2', 'N3', 'REM']

fig, axs = plt.subplots(5, 1)
subj_components, subj_mixing_coef, subj_labels = load_individual_results(8, start=0)
images = []
for i, label in enumerate(np.unique(subj_labels)):
    axs[i].set_ylabel(CLASS_LABEL[i])
    axs[i].set_xticks(np.arange(0, SAMPLE_LEN*SAMPLING_RATE+1, SAMPLING_RATE))
    axs[i].set_xticklabels(np.arange(SAMPLE_LEN+1))
    axs[i].set_yticks(np.arange(0, 18, 3))
    # Need to replace these placeholders with real channels
    axs[i].set_yticklabels(['F3', 'C3', 'O1', 'F4', 'C4', 'O2'])
    subj_mixing_coef_i = np.mean(subj_mixing_coef[subj_labels == label], axis=0)
    ar_coef = unstack_coef(np.tensordot(subj_mixing_coef_i, subj_components, axes=1))
    signal = autoregressive_sample(SAMPLE_LEN*SAMPLING_RATE, SIGNAL_DIM, SIGNAL_DIM ** (-1 / 2), ar_coef)
    images.append(axs[i].plot(signal + np.arange(0, 18, 3)))
