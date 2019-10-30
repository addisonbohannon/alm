import numpy as np
import scipy.linalg as sl
import scipy.fftpack as sf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from alm.utility import unstack_coef
from experiments.utility import load_individual_results, transfer_function_from_filter

SIGNAL_DIM = 6
NUM_COMPONENTS = 10
SAMPLING_RATE = 200
FFT_LEN = int(SAMPLING_RATE / 2)
FREQ_MIN = 0
FREQ_MAX = 30

components, _, _ = load_individual_results(8, start=0)
connectivity = []
for component_j in components:
    transfer_function, frequencies = transfer_function_from_filter(unstack_coef(component_j), SAMPLING_RATE, fft_len=FFT_LEN)
    for channel in range(SIGNAL_DIM):
        transfer_function[:, channel, channel] = 0
    dtf = np.abs(transfer_function) / sl.norm(transfer_function, axis=-1, keepdims=True)
    connectivity.append(sl.norm([dtf[k] for k, freq in enumerate(frequencies) if FREQ_MIN <= freq <= FREQ_MAX], axis=0))
fig, axs = plt.subplots(1, NUM_COMPONENTS)
images = []
for j, connectivity_j in enumerate(connectivity):
    axs[j].set_xlabel('Component: ' + str(j))
    axs[j].set_xticks([], [])
    axs[j].set_yticks([], [])
    images.append(axs[j].imshow(connectivity_j, cmap=plt.cm.Blues))
fig.colorbar(images[0], ax=axs, fraction=0.008, pad=0.04)

CHANNELS = ['O2', 'C4', 'F4', 'O1', 'C3', 'F3']
fig, ax = plt.subplots()
ax.set_xticks(np.arange(7))
ax.set_xticklabels(CHANNELS)
ax.set_yticks(np.arange(7))
ax.set_yticklabels(CHANNELS)
image = ax.imshow(connectivity[9], cmap=plt.cm.Blues)
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)