#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM Copyright (C) 2019  Addison Bohannon
"""

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_ar_coef
from experiments.utility import load_results, periodogram_from_filter

SAMPLING_RATE = 200
FFT_LEN = 100
colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]

components, _, _ = load_results('S8-mvar.pickle')
periodograms = [[periodogram_from_filter(unstack_ar_coef(component), SAMPLING_RATE, fft_len=FFT_LEN)
                for component in component_k] for component_k in components]
fig, axs = plt.subplots(1, 5)
for i, periodogram in enumerate(periodograms):
    for Pxx, freqs in periodogram:
        axs[i].plot((Pxx[:25]), linewidth=3.0)
        axs[i].set_xticks(np.arange(0, len(freqs[:25]), 5, dtype=np.int))
        axs[i].set_xticklabels(freqs[::5])
        axs[i].set_yscale('log')
        axs[i].set_ylabel('Gain', fontsize=12)
        axs[i].set_xlabel('Frequency (Hz)', fontsize=12)
        axs[i].set_title('Initialization ' + str(i), fontsize=14)
        axs[i].set_facecolor("#f2f3f4")
        axs[i].grid(b=True, which='major', linestyle="-", linewidth=1.5, color="#ffffff", zorder=3)
        axs[i].grid(b=True, which='minor', linewidth=0.75, color="#ffffff", zorder=3)
        axs[i].legend(('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'))
