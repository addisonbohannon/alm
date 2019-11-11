#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_ar_coef
from experiments.utility import load_individual_results, periodogram_from_filter

SIGNAL_DIM = 6
SAMPLING_RATE = 200
FFT_LEN = 100
CLASS_LABEL = ['Awake', 'N1', 'N2', 'N3', 'REM']

subj_components, subj_mixing_coef, subj_labels = load_individual_results(8, start=0)
fig, axs = plt.subplots(5, 1, sharey=True)
images = []
for i, label in enumerate(sorted(np.unique(subj_labels))):
    subj_mixing_coef_i = np.mean(subj_mixing_coef[subj_labels == label], axis=0)
    ar_coef = unstack_ar_coef(np.tensordot(subj_mixing_coef_i, subj_components, axes=1))
    periodogram, freqs = periodogram_from_filter(ar_coef, SAMPLING_RATE, fft_len=FFT_LEN)
    axs[i].set_ylabel(CLASS_LABEL[i])
    axs[i].set_xticks([])
    images.append(axs[i].plot(periodogram))
axs[-1].set_xticks(np.arange(0, len(freqs), 10, dtype=np.int))
axs[-1].set_xticklabels(freqs[::10])
axs[-1].set_xlabel('Frequency (Hz)')
