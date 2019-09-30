#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as nr
import scipy.fftpack as sf
import matplotlib.pyplot as plt
from almm.utility import unstack_coef

OBS_LENGTH = 1000
NUM_COMPONENTS = 10
MODEL_ORDER = 2
SIGNAL_DIM = 5
SAMPLING_RATE = 256 ** (-1)
frequency = lambda k: k / (OBS_LENGTH * SAMPLING_RATE)

DELTA_MIN = 0
DELTA_MAX = 4
THETA_MIN = 4
THETA_MAX = 8
ALPHA_MIN = 8
ALPHA_MAX = 13
BETA_MIN = 13
BETA_MAX = 30

delta = [k for k in range(OBS_LENGTH) if frequency(k) >= DELTA_MIN and frequency(k) < DELTA_MAX]
theta = [k for k in range(OBS_LENGTH) if frequency(k) >= THETA_MIN and frequency(k) < THETA_MAX]
alpha = [k for k in range(OBS_LENGTH) if frequency(k) >= ALPHA_MIN and frequency(k) < ALPHA_MAX]
beta = [k for k in range(OBS_LENGTH) if frequency(k) >= BETA_MIN and frequency(k) < BETA_MAX]

D = nr.randn(NUM_COMPONENTS, MODEL_ORDER * SIGNAL_DIM, SIGNAL_DIM)
D = [unstack_coef(D_j) for D_j in D]

fD = [sf.fft(D_j, n=OBS_LENGTH, axis=0) for D_j in D]

D_delta = [np.mean(np.power(np.abs(fD_j[delta]), 2), axis=0) for fD_j in fD]
D_theta = [np.mean(np.power(np.abs(fD_j[theta]), 2), axis=0) for fD_j in fD]
D_alpha = [np.mean(np.power(np.abs(fD_j[alpha]), 2), axis=0) for fD_j in fD]
D_beta = [np.mean(np.power(np.abs(fD_j[beta]), 2), axis=0) for fD_j in fD]

VMIN = 0
VMAX = np.maximum(max([np.maximum(np.max(D1), np.max(D2)) for D1, D2 in zip(D_delta, D_theta)]), 
                  max([np.maximum(np.max(D1), np.max(D2)) for D1, D2 in zip(D_alpha, D_beta)]))

fig, axs = plt.subplots(4, NUM_COMPONENTS)
images = []
for i in range(NUM_COMPONENTS):
    images.append(axs[0, i].imshow(D_delta[i], cmap='cool', vmin=VMIN, vmax=VMAX))
    images.append(axs[1, i].imshow(D_theta[i], cmap='cool', vmin=VMIN, vmax=VMAX))
    images.append(axs[2, i].imshow(D_alpha[i], cmap='cool', vmin=VMIN, vmax=VMAX))
    images.append(axs[3, i].imshow(D_beta[i], cmap='cool', vmin=VMIN, vmax=VMAX))
    axs[-1, i].set_xlabel('Comp.' + str(i))
axs[0, 0].set_ylabel('Delta (0-4 Hz)')
axs[1, 0].set_ylabel('Theta (4-8 Hz)')
axs[2, 0].set_ylabel('Alpha (8-13 Hz)')
axs[3, 0].set_ylabel('Beta (13-30 Hz)')
fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
