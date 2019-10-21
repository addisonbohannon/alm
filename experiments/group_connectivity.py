import numpy as np
import scipy.linalg as sl
import scipy.fftpack as sf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from alm.utility import unstack_coef
from experiments.utility import load_group_results

MODEL_ORDER = 12
PENALTY_PARAMETER = 0.1
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

components, mixing_coef, _ = load_group_results(model_order=MODEL_ORDER, num_components=NUM_COMPONENTS,
                                                penalty_parameter=PENALTY_PARAMETER)
connectivity = []
for component_j in components:
    component_j = unstack_coef(component_j)
    component_j = np.concatenate((np.expand_dims(np.eye(SIGNAL_DIM), 0), -component_j), axis=0)
    H = [sl.inv(A) for A in sf.fft(component_j, n=OBS_LENGTH, axis=0)]
    dtf = np.abs(H) / sl.norm(H, axis=-1, keepdims=True)
    connectivity.append(sl.norm([dtf[k] for k in ANY], axis=0))
fig, axs = plt.subplots(1, NUM_COMPONENTS)
images = []
for j, connectivity_j in enumerate(connectivity):
    axs[j].set_xlabel('Component: ' + str(j))
    axs[j].set_xticks([], [])
    axs[j].set_yticks([], [])
    images.append(axs[j].imshow(connectivity_j, norm=colors.LogNorm(vmin=1e-2, vmax=4), cmap=plt.cm.Blues))
fig.colorbar(images[0], ax=axs, fraction=0.006, pad=0.04)
