import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
from alm.utility import unstack_ar_coef
from experiments.utility import load_isruc_results, transfer_function_from_filter

SIGNAL_DIM = 6
NUM_COMPONENTS = 10
SAMPLING_RATE = 200
FFT_LEN = int(SAMPLING_RATE / 2)
FREQ_MIN = 0
FREQ_MAX = 30

components, _, _ = load_isruc_results(8, start=2)
connectivity = []
for component_j in components:
    transfer_function, frequencies = transfer_function_from_filter(unstack_ar_coef(component_j), SAMPLING_RATE, fft_len=FFT_LEN)
    for channel in range(SIGNAL_DIM):
        transfer_function[:, channel, channel] = 0
    dtf = np.abs(transfer_function) / sl.norm(transfer_function, axis=-1, keepdims=True)
    connectivity.append(sl.norm([dtf[k] for k, freq in enumerate(frequencies) if FREQ_MIN <= freq <= FREQ_MAX], axis=0))
vmax = max([np.max(connectivity_j) for connectivity_j in connectivity])
fig, axs = plt.subplots(2, int(NUM_COMPONENTS/2))
images = []
row = 0
for j, connectivity_j in enumerate(connectivity):
    if j >= int(NUM_COMPONENTS/2):
        row = 1
    axs[row, np.remainder(j, int(NUM_COMPONENTS/2))].set_xlabel('Component: ' + str(j+1))
    axs[row, np.remainder(j, int(NUM_COMPONENTS/2))].set_xticks(np.arange(6), [])
    axs[row, np.remainder(j, int(NUM_COMPONENTS/2))].set_xticklabels(['F3', 'C3', 'O1', 'F4', 'C4', 'O2'])
    axs[row, np.remainder(j, int(NUM_COMPONENTS/2))].set_yticks(np.arange(6), [])
    axs[row, np.remainder(j, int(NUM_COMPONENTS/2))].set_yticklabels(['F3', 'C3', 'O1', 'F4', 'C4', 'O2'])
    images.append(axs[row, np.remainder(j, int(NUM_COMPONENTS/2))].imshow(connectivity_j, vmin=0, vmax=vmax, cmap=plt.cm.Blues))
fig.colorbar(images[0], ax=axs, fraction=0.030, pad=0.04)