#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from experiments.sampler import autoregressive_sample
from alm.utility import unstack_coef
from experiments.utility import load_individual_results, periodogram_from_filter

SIGNAL_DIM = 6
SAMPLING_RATE = 200
SAMPLE_LEN = 2
CLASS_LABEL = ['Awake', 'N1', 'N2', 'N3', 'REM']

fig, axs = plt.subplots(5, 2)
subj_components, subj_mixing_coef, subj_labels = load_individual_results(8, start=0)
images = []
for i, label in enumerate(np.unique(subj_labels)):
    axs[i, 0].set_ylabel(CLASS_LABEL[i], fontsize=12)
    axs[i, 0].set_yticks(np.arange(0, 18, 3))
    axs[i, 0].set_xticklabels([])
    # Need to replace these placeholders with real channels
    axs[i, 0].set_yticklabels(['F3', 'C3', 'O1', 'F4', 'C4', 'O2'])
    subj_mixing_coef_i = np.mean(subj_mixing_coef[subj_labels == label], axis=0)
    ar_coef = unstack_coef(np.tensordot(subj_mixing_coef_i, subj_components, axes=1))
    signal = autoregressive_sample(SAMPLE_LEN*SAMPLING_RATE, SIGNAL_DIM, SIGNAL_DIM ** (-1 / 2), ar_coef)
    images.append(axs[i, 0].plot(signal + np.arange(0, 18, 3)))
    axs[i,0].set_facecolor("#f2f3f4")
    axs[i,0].grid(b=True,which='major',linestyle="-",linewidth=1.5, color="#ffffff",zorder=3)
    axs[i,0].grid(b=True,which='minor',linewidth=0.75, color="#ffffff",zorder=3)
axs[-1, 0].set_xlabel('time (s)', fontsize=12)
axs[-1, 0].set_xticks(np.arange(0, SAMPLE_LEN*SAMPLING_RATE+1, SAMPLING_RATE))
axs[-1, 0].set_xticklabels(np.arange(SAMPLE_LEN+1))
axs[0, 0].set_title('Signals', fontsize=14)


SIGNAL_DIM = 6
SAMPLING_RATE = 200
FFT_LEN = 100
CLASS_LABEL = ['Awake', 'N1', 'N2', 'N3', 'REM']

subj_components, subj_mixing_coef, subj_labels = load_individual_results(8, start=0)
images = []
for i, label in enumerate(sorted(np.unique(subj_labels))):
    subj_mixing_coef_i = np.mean(subj_mixing_coef[subj_labels == label], axis=0)
    ar_coef = unstack_coef(np.tensordot(subj_mixing_coef_i, subj_components, axes=1))
    periodogram, freqs = periodogram_from_filter(ar_coef, SAMPLING_RATE, fft_len=FFT_LEN)
    axs[i,1].set_xticklabels([])
    images.append(axs[i,1].plot(periodogram[:20]))
    axs[i,1].set_ylim((0,400))
    axs[i,1].set_facecolor("#f2f3f4")
    axs[i,1].grid(b=True,which='major',linestyle="-",linewidth=1.5, color="#ffffff",zorder=3)
    axs[i,1].grid(b=True,which='minor',linewidth=0.75, color="#ffffff",zorder=3)
axs[-1, 1].set_xticks(np.arange(0, len(freqs[:20]), 5, dtype=np.int))
axs[-1, 1].set_xticklabels(freqs[::5])
axs[-1, 1].set_xlabel('Frequency (Hz)',fontsize=12)
axs[0, 1].set_title('Periodogram',fontsize=14)
