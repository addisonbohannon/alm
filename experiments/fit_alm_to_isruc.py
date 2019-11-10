#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import requests
from unrar import rarfile
import mne

# Subject 1

data_dir = '/home/addison/Python/almm/data/'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
# Download sleep data from website
r = requests.get('http://sleeptight.isr.uc.pt/ISRUC_Sleep/ISRUC_Sleep/subgroupIII/1.rar', stream=True)
with open(os.path.join(data_dir, 'download.rar'), 'wb+', buffering=0) as fd:
    for chunk in r.iter_content(chunk_size=128):
        fd.write(chunk)
# Unpack the compressed file 
with rarfile.RarFile(os.path.join(data_dir, 'download.rar')) as rar_ref:
    rar_ref.extract('1/1.rec', path=data_dir)
    rar_ref.extract('1/1_1.txt', path=data_dir)
os.rename(os.path.join(data_dir, '1/1.rec'), os.path.join(data_dir, 'S1.edf'))
os.rename(os.path.join(data_dir, '1/1_1.txt'), os.path.join(data_dir, 'S1.txt'))
os.remove(os.path.join(data_dir, 'download.rar'))
os.rmdir(os.path.join(data_dir, '1'))
# Read raw data file into mne-python
raw = mne.io.read_raw_edf(os.path.join(data_dir, 'S1.edf'), preload=True)
# Select EEG channels
raw.pick([i for i in range(2, 8)])
# Bandpass filter with FIR, Hamming
raw.filter(0.3, 35)
# Epoch data
events = mne.make_fixed_length_events(raw, duration=30)
data = mne.Epochs(raw.pick('eeg'), events, tmin=0, tmax=30, baseline=None)
# Whiten data within channel and epoch
X = data.get_data()
X = (X - np.mean(X, axis=-1, keepdims=True)) / np.std(X, axis=-1, keepdims=True)
X = np.moveaxis(X, 1, 2)
Y = np.loadtxt(os.path.join(data_dir, 'S1.txt'))
Y = Y[ [drop_epoch==[] for drop_epoch in data.drop_log] ]
