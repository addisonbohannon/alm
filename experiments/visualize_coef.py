#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from experiments.utility import load_individual_results

_, mixing_coef, labels = load_individual_results(8, start=0)
pca = PCA(n_components=3)
data = pca.fit_transform(mixing_coef)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['y', 'g', 'b', 'c', 'm']
plot = []
for color, label in zip(colors, np.unique(labels)):
    plot.append(ax.scatter(data[labels==label, 0], data[labels==label, 1], data[labels==label, 2], c=color))
ax.legend(plot, ['Awake', 'N1', 'N2', 'N3', 'REM'])
