#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM Copyright (C) 2019  Addison Bohannon
"""

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_ar_coef
from experiments.utility import load_isruc_results, periodogram_from_filter

SAMPLING_RATE = 200
FFT_LEN = 100

components, _, _ = load_isruc_results(8, start=2)
periodogram = [periodogram_from_filter(unstack_ar_coef(component), SAMPLING_RATE, fft_len=FFT_LEN)
               for component in components]
f = plt.figure()
f.hold = True
colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]
for Pxx, freqs in periodogram:
    plt.plot((Pxx[:25]), figure=f, linewidth=3.0)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, len(freqs[:25]), 5, dtype=np.int))
    ax.set_xticklabels(freqs[::5])
    ax.set_ylabel('Gain', fontsize=12)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
#    ax.set_title('Component Periodograms', fontsize=14)
    ax.set_facecolor("#f2f3f4")
    ax.grid(b=True, which='major', linestyle="-", linewidth=1.5, color="#ffffff", zorder=3)
    ax.grid(b=True, which='minor', linewidth=0.75, color="#ffffff", zorder=3)
plt.legend(('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'))
plt.grid('minor')
plt.box('none')
