import numpy as np
import scipy.linalg as sl
import scipy.fftpack as sf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from alm.utility import unstack_coef
from experiments.utility import load_individual_results

SIGNAL_DIM = 6
NUM_COMPONENTS = 10
SAMPLING_RATE = 200
OBS_LENGTH = int(SAMPLING_RATE / 2)
DELTA_MIN = 0
DELTA_MAX = 4
THETA_MIN = 4
THETA_MAX = 8
ALPHA_MIN = 8
ALPHA_MAX = 13
BETA_MIN = 13
BETA_MAX = 30


def frequency(k): return k * SAMPLING_RATE / OBS_LENGTH


DELTA = [k for k in range(OBS_LENGTH) if DELTA_MIN <= frequency(k) < DELTA_MAX]
THETA = [k for k in range(OBS_LENGTH) if THETA_MIN <= frequency(k) < THETA_MAX]
ALPHA = [k for k in range(OBS_LENGTH) if ALPHA_MIN <= frequency(k) < ALPHA_MAX]
BETA = [k for k in range(OBS_LENGTH) if BETA_MIN <= frequency(k) < BETA_MAX]
ANY = [k for k in range(OBS_LENGTH) if DELTA_MIN <= frequency(k) < BETA_MAX]

subj_components, _, _ = load_individual_results(8)
connectivity = []
for start_component in subj_components:
    connectivity_start = []
    for component_j in start_component:
        component_j = unstack_coef(component_j)
        component_j = np.concatenate((np.expand_dims(np.eye(SIGNAL_DIM), 0), -component_j), axis=0)
        H = [sl.inv(A) for A in sf.fft(component_j, n=OBS_LENGTH, axis=0)]
        dtf = np.abs(H) / sl.norm(H, axis=-1, keepdims=True)
        connectivity_start.append(sl.norm([dtf[k] for k in ANY], axis=0))
    connectivity.append(connectivity_start)
fig, axs = plt.subplots(5, NUM_COMPONENTS)
images = []
for i, connectivity_matrix in enumerate(connectivity):
    axs[i, 0].set_ylabel('Start: ' + str(i))
    for j, connectivity_j in enumerate(connectivity_matrix):
        if i == 1:
            axs[-1, j].set_xlabel('Component: ' + str(j))
        axs[i, j].set_xticks([], [])
        axs[i, j].set_yticks([], [])
        images.append(axs[i, j].imshow(connectivity_j, norm=colors.LogNorm(vmin=1e-2, vmax=4), cmap=plt.cm.Blues))
fig.colorbar(images[0], ax=axs, fraction=0.046, pad=0.04)
