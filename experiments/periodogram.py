#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import unstack_ar_coef
from experiments.utility import load_individual_results, periodogram_from_filter

SAMPLING_RATE = 200
FFT_LEN = 100

components, _, _ = load_individual_results(8, start=2)
periodogram = [periodogram_from_filter(unstack_ar_coef(component), SAMPLING_RATE, fft_len=FFT_LEN)
               for component in components]
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
colors=["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a"]
cnt = 0
for Pxx, freqs in periodogram:
    plt.plot((Pxx[:25]),figure=f,linewidth=3.0)#,color=colors[cnt])
    cnt=cnt+1
    ax = plt.gca()
    ax.set_xticks(np.arange(0, len(freqs[:25]), 5, dtype=np.int))
    ax.set_xticklabels(freqs[::5])
    ax.set_ylabel('Gain', fontsize=12)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_title('Component Periodograms', fontsize=15)
    ax.set_facecolor("#f2f3f4")
    ax.grid(b=True,which='major',linestyle="-",linewidth=1.5, color="#ffffff",zorder=3)
    ax.grid(b=True,which='minor',linewidth=0.75, color="#ffffff",zorder=3)
plt.legend(('C1','C2','C3','C4','C5','C6','C7','C8','C9','C10'))
plt.grid('minor')
plt.box('none')

Acc=[0.766, 0.691, 0.791, 0.766, 0.765, 0.687, 0.749, 0.857, 0.642, 0.701]
AvgAcc = np.mean(Acc)