import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from alm.utility import unstack_coef
from experiments.utility import load_group_results, transfer_function_from_filter

MODEL_ORDER = 12
PENALTY_PARAMETER = 0.1
NUM_COMPONENTS = 10
SAMPLING_RATE = 200
FFT_LEN = 100
FREQ_MIN = 0
FREQ_MAX = 30

components, mixing_coef, _ = load_group_results(model_order=MODEL_ORDER, num_components=NUM_COMPONENTS,
                                                penalty_parameter=PENALTY_PARAMETER)
connectivity = []
for component_j in components:
    transfer_function, frequencies = transfer_function_from_filter(unstack_coef(component_j), SAMPLING_RATE,
                                                                   fft_len=FFT_LEN)
    dtf = np.abs(transfer_function) / sl.norm(transfer_function, axis=-1, keepdims=True)
    connectivity.append(sl.norm([dtf[k] for k, freq in enumerate(frequencies) if FREQ_MIN <= freq <= FREQ_MAX], axis=0))
fig, axs = plt.subplots(1, NUM_COMPONENTS)
images = []
for j, connectivity_j in enumerate(connectivity):
    axs[j].set_xlabel('Component: ' + str(j))
    axs[j].set_xticks([], [])
    axs[j].set_yticks([], [])
    images.append(axs[j].imshow(connectivity_j, norm=colors.LogNorm(vmin=1e-2, vmax=4), cmap=plt.cm.Blues))
fig.colorbar(images[0], ax=axs, fraction=0.008, pad=0.04)
