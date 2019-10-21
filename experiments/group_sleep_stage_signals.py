#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_coef
from experiments.sampler import autoregressive_sample
from experiments.utility import load_group_results

MODEL_ORDER = 12
NUM_COMPONENTS = 10
PENALTY_PARAMETER = 0.1
SIGNAL_DIM = 6
SAMPLING_RATE = 200
SAMPLE_LEN = 2
CLASS_LABEL = ['Awake', 'N1', 'N2', 'N3', 'REM']

fig, axs = plt.subplots(5, 1)
components, mixing_coef, labels = load_group_results(model_order=MODEL_ORDER, num_components=NUM_COMPONENTS,
                                                     penalty_parameter=PENALTY_PARAMETER)
images = []
for i, label in enumerate(np.unique(labels)):
    axs[i].set_ylabel(CLASS_LABEL[i])
    axs[i].set_xticks(np.arange(0, SAMPLE_LEN*SAMPLING_RATE+1, SAMPLING_RATE))
    axs[i].set_xticklabels(np.arange(SAMPLE_LEN+1))
    axs[i].set_yticks(np.arange(0, 18, 3))
    # Need to replace these placeholders with real channels
    axs[i].set_yticklabels(['F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1'])
    subj_mixing_coef_i = np.mean(mixing_coef[labels == label], axis=0)
    ar_coef = unstack_coef(np.tensordot(subj_mixing_coef_i, components, axes=1))
    signal = autoregressive_sample(2000, SIGNAL_DIM, SIGNAL_DIM ** (-1 / 2), ar_coef)
    signal = signal[:SAMPLE_LEN*SAMPLING_RATE]
    images.append(axs[i].plot(signal + np.arange(0, 18, 3)))
