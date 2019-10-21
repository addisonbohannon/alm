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

fig, axs = plt.subplots(1, 3)
components, _, _ = load_group_results(model_order=MODEL_ORDER, num_components=NUM_COMPONENTS,
                                      penalty_parameter=PENALTY_PARAMETER)
images = []
for j, component in enumerate(components[:3]):
    signal = autoregressive_sample(2000, SIGNAL_DIM, SIGNAL_DIM**(-1/2), unstack_coef(component))
    signal = signal[:SAMPLE_LEN*SAMPLING_RATE]
    axs[j].set_xticks(np.arange(0, SAMPLE_LEN*SAMPLING_RATE+1, SAMPLING_RATE))
    axs[j].set_xticklabels(np.arange(SAMPLE_LEN+1))
    axs[j].set_ylabel('Component: ' + str(j))
    axs[j].set_yticks(np.arange(0, 18, 3))
    # Need to replace these placeholders with real channels
    axs[j].set_yticklabels(['F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1'])
    images.append(axs[j].plot(signal + np.arange(0, 18, 3)))
