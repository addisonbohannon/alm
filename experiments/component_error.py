#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import initialize_autoregressive_components
from experiments.utility import ar_comp_dist

samples = 10000
num_components = 10
model_order = 12
signal_dimension = 6

D = [(initialize_autoregressive_components(num_components, model_order, signal_dimension),
      initialize_autoregressive_components(num_components, model_order, signal_dimension))
     for _ in range(samples)]
cdist = [ar_comp_dist(D_i, D_j)[0] for (D_i, D_j) in D]
plt.hist(cdist)
p = np.percentile(cdist, 5)