import numpy as np
import numpy.random as nr
import scipy.linalg as sl
import matplotlib.pyplot as plt
from alm.utility import unstack_ar_coef
from experiments.utility import load_isruc_results, transfer_function_from_filter

SIGNAL_DIM = 6
NUM_COMPONENTS = 10
SAMPLING_RATE = 200
FFT_LEN = int(SAMPLING_RATE / 2)
# FREQ_MIN = 13
# FREQ_MAX = 30
ALPHA_MIN = 8
ALPHA_MAX = 13
BETA_MIN = 13
BETA_MAX = 30

def keep_significant_edges(R, p=95, num_samples=100):
    """
    This function will set to zero edges which are below percentile p for
    the randomize_network null model.
    """
    
    dim, _ = R.shape
    R = R.copy()
    randomized_R = np.zeros([num_samples, dim, dim])
    for sample in range(num_samples):
        randomized_R[sample] = randomize_network(R)
    R[R < np.percentile(randomized_R, p, axis=0)] = 0
    
    return R

def randomize_network(R, swaps=None):
    '''
    This function randomizes a directed network, while preserving the in-
    and out-degree distributions. In weighted networks, the function
    preserves the out-strength but not the in-strength distributions. Based on
    Maslov and Sneppen 2002.
    '''
    
    R = R.copy()
    i, j = np.nonzero(R)
    i.setflags(write=True)
    j.setflags(write=True)
    k = len(i)
    if swaps is None:
        swaps = 10*len(i)

    for _ in range(swaps):
        for _ in range(10):
            e1, e2 = nr.randint(k, size=2)
            while e1 == e2:
                e1, e2 = nr.randint(k, size=2)
            a = i[e1]
            b = j[e1]
            c = i[e2]
            d = j[e2]
            if a != c and a != d and b != c and b != d:
                break
        R[a, d], R[a, b] = R[a, b], R[a, d]
        R[c, b], R[c, d] = R[c, d], R[c, b]
    return R

components, _, _ = load_isruc_results(8, start=2)
alpha_connectivity = np.zeros([NUM_COMPONENTS, SIGNAL_DIM, SIGNAL_DIM])
beta_connectivity = np.zeros_like(alpha_connectivity)
for j, component_j in enumerate(components):
    transfer_function, frequencies = transfer_function_from_filter(unstack_ar_coef(component_j), SAMPLING_RATE, fft_len=FFT_LEN)
    # for channel in range(SIGNAL_DIM):
    #     transfer_function[:, channel, channel] = 0
    transfer_function[:, np.arange(SIGNAL_DIM), np.arange(SIGNAL_DIM)] = 0
    dtf = np.abs(transfer_function) / sl.norm(transfer_function, axis=-1, keepdims=True)
    alpha_connectivity[j] = sl.norm([dtf[k] for k, freq in enumerate(frequencies) if ALPHA_MIN <= freq < ALPHA_MAX], axis=0)
    alpha_connectivity[j] = keep_significant_edges(alpha_connectivity[j])
    beta_connectivity[j] = sl.norm([dtf[k] for k, freq in enumerate(frequencies) if BETA_MIN <= freq < BETA_MAX], axis=0)
    beta_connectivity[j] = keep_significant_edges(beta_connectivity[j])
vmax = max(np.max(alpha_connectivity), np.max(beta_connectivity))
fig, axs = plt.subplots(2, NUM_COMPONENTS)
images = []
for j, (alpha_plot, beta_plot) in enumerate(zip(alpha_connectivity, beta_connectivity)):
    for i, (freq_label, plot) in enumerate(zip(['Alpha', 'Beta'], [alpha_plot, beta_plot])):
        if j == 0:
            axs[i, j].set_ylabel(freq_label)
        if i == 1:
            axs[i, j].set_xlabel('Component: ' + str(j+1))
        axs[i, j].set_xticks(np.arange(6), [])
        axs[i, j].set_xticklabels(['F3', 'C3', 'O1', 'F4', 'C4', 'O2'])
        axs[i, j].set_yticks(np.arange(6), [])
        axs[i, j].set_yticklabels(['F3', 'C3', 'O1', 'F4', 'C4', 'O2'])
        images.append(axs[i, j].imshow(plot, vmin=0, vmax=vmax, cmap=plt.cm.Blues))
fig.colorbar(images[0], ax=axs, fraction=0.012, pad=0.012)