#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_coef
from experiments.utility import load_individual_results, periodogram_from_filter

SAMPLING_RATE = 200
FFT_LEN = 100

components, _, _ = load_individual_results(8, start=0)
periodogram = [periodogram_from_filter(unstack_coef(component_j), SAMPLING_RATE, fft_len=FFT_LEN)
               for component_j in components]
fig, axs = plt.subplots(len(components), 1, sharey=True)
for ax, (Pxx, freqs) in zip(axs, periodogram):
    ax.plot(Pxx[:20])
    ax.set_xticks([])
    
for i in range(10):
    axs[i].set_ylabel('C'+str(i+1),fontsize=12)
        
axs[-1].set_xticks(np.arange(0, len(freqs[:20]), 5, dtype=np.int))
axs[-1].set_xticklabels(freqs[::5], fontsize=12)
axs[-1].set_xlabel('Frequency (Hz)', fontsize=12)
axs[0].set_title('Component Periodograms', fontsize=15)
plt.yticks(rotation=90)


#plt.savefig('name.png')

# combined plot
f = plt.figure()
f.hold = True
for Pxx in zip(periodogram):
    plt.plot((Pxx[0][0][:25]),figure=f,linewidth=2.0)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, len(freqs[:25]), 5, dtype=np.int))
    ax.set_xticklabels(freqs[::5])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Component Periodograms')
    
plt.legend(('C1','C2','C3','C4','C5','C6','C7','C8','C9','C10'))
plt.grid('major')
plt.box('none')