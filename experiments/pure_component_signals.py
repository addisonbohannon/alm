#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_coef
from experiments.sampler import autoregressive_sample
from experiments.utility import load_results

SIGNAL_DIM = 6
SAMPLING_RATE = 200

fig, axs = plt.subplots(2, 1)
subj_components, _, _ = load_results(7, start=0)
images = []
for j, component in enumerate(subj_components[:2]):
    signal = autoregressive_sample(2*SAMPLING_RATE, SIGNAL_DIM, SIGNAL_DIM**(-1/2), unstack_coef(component))
    axs[j].set_xticks(np.arange(0, 2*SAMPLING_RATE, SAMPLING_RATE))
    axs[j].set_xticklabels(np.arange(2))
    axs[j].set_ylabel('Component: ' + str(j))
    axs[j].set_yticks(np.arange(0, 18, 3))
    # Need to replace these placeholders with real channels
    axs[j].set_yticklabels(['FL', 'PL', 'OL', 'FR', 'PR', 'OR'])
    images.append(axs[j].plot(signal + np.arange(0, 18, 3)))
