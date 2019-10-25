#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_coef
from experiments.utility import load_individual_results, periodogram_from_filter

SAMPLING_RATE = 200
FFT_LEN = 100

components, _, _ = load_individual_results(11, start=0)
periodogram = [periodogram_from_filter(unstack_coef(component_j), SAMPLING_RATE, fft_len=FFT_LEN)
               for component_j in components]
fig, axs = plt.subplots(len(components), 1, sharey=True)
for ax, (Pxx, freqs) in zip(axs, periodogram):
    ax.plot(Pxx)
    ax.set_xticks([])
axs[-1].set_xticks(np.arange(0, len(freqs), 10, dtype=np.int))
axs[-1].set_xticklabels(freqs[::10])
axs[-1].set_xlabel('Frequency (Hz)')
