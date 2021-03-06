#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALM Copyright (C) 2019  Addison Bohannon
"""

import numpy as np
import matplotlib.pyplot as plt
from experiments.utility import load_results
from MulticoreTSNE import MulticoreTSNE as TSNE
import time

SLEEP_CLASS = ['Awake', 'N1', 'N2', 'N3', 'REM']

ar_coef, labels = load_results('S8-var.pickle')
ar_coef = np.reshape(ar_coef, [len(ar_coef), -1])
num_obs, num_features = ar_coef.shape
avg_ar_coef = np.zeros((len(SLEEP_CLASS), num_features))
class_label = np.zeros(len(SLEEP_CLASS))
for i, label in enumerate(np.unique(labels)):
    class_label[i] = label
    avg_ar_coef[i, :] = np.mean(ar_coef[labels == label], axis=0)
# visualize with different tsne perplexities which roughly equates to the number of neighbors in a cluster. 
perplexities = [10, 50, 100, 200, 500, 1000]
for perplexity in perplexities:
    tsne = TSNE(perplexity=perplexity, n_jobs=40)
    print("Running TSNE Fit...")
    ar_coef_combined = np.concatenate((ar_coef, avg_ar_coef), axis=0)
    labels_combined = np.concatenate((labels, class_label), axis=0)
    start_time = time.time()
    dataCombined = tsne.fit_transform(ar_coef_combined)
    data = dataCombined[:len(ar_coef), :]
    mean_data = dataCombined[len(ar_coef):, :]
    print("Finished TSNE Fit --- {} seconds ---".format(time.time() - start_time))
    # plot
    fig, ax1 = plt.subplots()
    ax1.set_facecolor("#f2f3f4")
    ax1.grid(b=True, which='major', linestyle="-", linewidth=1.5, color="#ffffff", zorder=3)
    ax1.grid(b=True, which='minor', linewidth=0.75, color="#ffffff", zorder=3)
    plt.minorticks_on()
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # colors = ["#8dd3c7","#b3de69","#fb8072","#80b1d3","#fdb462"] # data visualization friendly colors
    colors = ["#d7191c", "#fdae61", "#ffffaf", "#abdda4", "#2b83ba"]  # bw friendly colors
    plot = []
    for color, label, category in zip(colors, np.unique(labels), SLEEP_CLASS):
        ax1.scatter(data[labels == label, 0], data[labels == label, 1], s=6, c=color, alpha=1.0, zorder=4)
        ax1.scatter(mean_data[class_label == label, 0], mean_data[class_label == label, 1], s=40, label=category,
                    c=color, alpha=1.0, zorder=4)
        ax1.scatter(mean_data[class_label == label, 0], mean_data[class_label == label, 1], marker="o", s=235.0,
                    c="#000000", zorder=5)
        ax1.scatter(mean_data[class_label == label, 0], mean_data[class_label == label, 1], marker="o", s=175.0,
                    c=color, zorder=6)
        ax1.scatter(mean_data[class_label == label, 0], mean_data[class_label == label, 1], marker="2", s=125.0,
                    c="#000000", zorder=7)
    legend = ax1.legend(prop={'size': 12}, loc='best', shadow=True, ncol=1)
    frame = legend.get_frame()
    frame.set_facecolor("#f5f5f5")
    frame.set_edgecolor("#000000")
    print("Completed perplexity: {}".format(perplexity))
