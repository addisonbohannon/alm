#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt

VMIN = 0
VMAX = 12

PATH = '/home/addison/Python/almm/results/n_vs_m'
os.chdir(PATH)

files = [file for _, _, file in os.walk(PATH)].pop()
error_palm, error_altmin, error_bcd = [], [], []
data = []
for filename in files:
    if '.pickle' and ('19Oct11' or '19Oct14') in filename:
        print(filename)
        with open(join(PATH, filename), 'rb') as f:
            data = pickle.load(f)
            for i, data_i in enumerate(data[:3]):
                error_i = np.min(np.array(data_i), axis=1)
                if i == 0:
                    error_palm.append(error_i)
                elif i == 1:
                    error_altmin.append(error_i)
                else:
                    error_bcd.append(error_i)
error_palm = np.reshape(np.mean(np.array(error_palm), axis=0), [3, 4])
error_altmin = np.reshape(np.mean(np.array(error_altmin), axis=0), [3, 4])
error_bcd = np.reshape(np.mean(np.array(error_bcd), axis=0), [3, 4])
images = []
fig, axs = plt.subplots(1, 3)
for ax in axs:
    ax.set_xlabel('Observation length')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([10, 100, 1000, 10000])
    ax.set_ylabel('Number of obs')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([10, 100, 1000])
images.append(axs[0].imshow(error_altmin, origin='lower', vmin=VMIN, vmax=VMAX, cmap=plt.cm.Blues))
axs[0].set_title('AltMin')
images.append(axs[1].imshow(error_bcd, origin='lower', vmin=VMIN, vmax=VMAX, cmap=plt.cm.Blues))
axs[1].set_title('BCD')
images.append(axs[2].imshow(error_palm, origin='lower', vmin=VMIN, vmax=VMAX, cmap=plt.cm.Blues))
axs[2].set_title('PALM')
fig.colorbar(images[0], ax=axs, fraction=0.012, pad=0.04)
