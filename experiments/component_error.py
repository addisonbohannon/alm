#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from alm.utility import initialize_components
from experiments.utility import component_distance

samples = 10000
num_components = 10
model_order = 12
signal_dimension = 6

D = [(initialize_components(num_components, model_order, signal_dimension), 
      initialize_components(num_components, model_order, signal_dimension))
     for _ in range(samples)]
cdist = [component_distance(D_i, D_j)[0] for (D_i, D_j) in D]
plt.hist(cdist)
p = np.percentile(cdist, 5)