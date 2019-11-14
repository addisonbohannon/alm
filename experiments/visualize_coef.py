#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# visualize the mixing coefficients

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from experiments.utility import load_isruc_results
from MulticoreTSNE import MulticoreTSNE as TSNE
import time

# load mixing coefficiencts
_, mixing_coef, labels = load_isruc_results(8, start=0)

# get average mixing components
avg_mixing_coef=np.zeros((5,10))
class_label=np.zeros(5)
for i, label in enumerate(np.unique(labels)):
    class_label[i] = label
    avg_mixing_coef[i,:] = np.mean(mixing_coef[labels == label], axis=0)
    

# visualize coefficients using pca
pca = PCA(n_components=3)
data = pca.fit_transform(mixing_coef)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['y', 'g', 'b', 'c', 'm']
plot = []
for color, label in zip(colors, np.unique(labels)):
    plot.append(ax.scatter(data[labels==label, 0], data[labels==label, 1], data[labels==label, 2], c=color))
ax.legend(plot, ['Awake', 'N1', 'N2', 'N3', 'REM'])


# visualize coefficients using the way cooler and way more impressive t-sne.
categories = ['Awake', 'N1', 'N2', 'N3', 'REM']

# visualize with different tsne perplexities which roughly equates to the number of neighbors in a cluster. 
perplexities = [10, 50, 100, 200, 500, 1000]

# this perplexity value looks the best
perplexities = [100]

for perplexity in perplexities:
    tsne = TSNE(perplexity=perplexity, n_jobs=40)
    print("Running TSNE Fit...")
    mixing_coef_combined = np.concatenate((mixing_coef,avg_mixing_coef),axis=0)
    labels_combined = np.concatenate((labels,class_label),axis=0)
    start_time = time.time()
    dataCombined = tsne.fit_transform(mixing_coef_combined)
    data = dataCombined[:1000,:]
    mean_data = dataCombined[1000:,:]
    print("Finished TSNE Fit --- {} seconds ---".format(time.time() - start_time))
    
    # plot
    fig,ax1 = plt.subplots()
    ax1.set_facecolor("#f2f3f4")
    ax1.grid(b=True,which='major',linestyle="-",linewidth=1.5, color="#ffffff",zorder=3)
    ax1.grid(b=True,which='minor',linewidth=0.75, color="#ffffff",zorder=3)
    plt.minorticks_on()
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    #fig.set_size_inches(16,16)
    # colors that follow the priciples of data visualization
    #colors = ["#8dd3c7","#b3de69","#fb8072","#80b1d3","#fdb462"]
    colors = ["#d7191c","#fdae61","#ffffaf","#abdda4","#2b83ba"] #bw friendly colors
                 
    plot = []
    for color, label, category in zip(colors, np.unique(labels), categories):
        ax1.scatter(data[labels==label,0],data[labels==label,1],s=6, c=color,alpha=1.0,zorder=4)
        ax1.scatter(mean_data[class_label==label,0],mean_data[class_label==label,1],s=40,label=category, c=color, alpha=1.0, zorder=4)
        ax1.scatter(mean_data[class_label==label,0],mean_data[class_label==label,1],marker="o",s=235.0, c="#000000", zorder=5)
        ax1.scatter(mean_data[class_label==label,0],mean_data[class_label==label,1],marker="o",s=175.0, c=color, zorder=6)
        ax1.scatter(mean_data[class_label==label,0],mean_data[class_label==label,1],marker="2",s=125.0, c="#000000", zorder=7)
    legend=ax1.legend(prop={'size':12},loc="upper center",bbox_to_anchor=(0.15,0.97),shadow=True, ncol=1)
    frame = legend.get_frame()
    frame.set_facecolor("#f5f5f5")
    frame.set_edgecolor("#000000")
    # save figure    
    plt.savefig("TSNE_image_perp_{}.eps".format(perplexity),bbox_inches="tight")#,transparent=True)
            
    print("Completed perplexity: {}".format(perplexity))