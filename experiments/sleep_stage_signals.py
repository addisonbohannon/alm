#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_coef
from experiments.sampler import autoregressive_sample
from experiments.utility import load_results

SIGNAL_DIM = 6
SAMPLING_RATE = 200
CLASS_LABEL = ['Awake', 'N1', 'N2', 'N3', 'REM']

fig, axs = plt.subplots(5, 1)
subj_components, subj_mixing_coef, subj_labels = load_results(7, start=0)
images = []
for i, label in enumerate(np.unique(subj_labels)):
    axs[i].set_ylabel(CLASS_LABEL[i])
    axs[i].set_xticks(np.arange(0, 2*SAMPLING_RATE, SAMPLING_RATE))
    axs[i].set_xticklabels(np.arange(10))
    subj_mixing_coef = np.mean(subj_mixing_coef[0][subj_labels == label], axis=0)
    ar_coef = unstack_coef(np.tensordot(subj_mixing_coef, subj_components, axes=1))
    signal = autoregressive_sample(2*SAMPLING_RATE, SIGNAL_DIM, SIGNAL_DIM ** (-1 / 2), ar_coef)
    images.append(axs[i].plot(signal + np.arange(0, 18, 3)))
