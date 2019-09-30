#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy.random as nr
from sklearn.linear_model import LogisticRegression

NUM_OBS = 100
NUM_COMPONENTS = 10

C = nr.randn(NUM_OBS, NUM_COMPONENTS)
labels = nr.choice(5, (NUM_OBS,))

sklr = LogisticRegression(multi_class='multinomial')
sklr.fit(C, labels)
