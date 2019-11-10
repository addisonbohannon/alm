#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import rename
from os.path import join
from unrar import rarfile
import requests
from mne.io import read_raw_edf

data_dir = '/home/addison/Python/almm/data/'
r = requests.get('http://sleeptight.isr.uc.pt/ISRUC_Sleep/ISRUC_Sleep/subgroupIII/1.rar', stream=True)
with open(join(data_dir, 'download.rar'), 'wb+', buffering=0) as fd:
    for chunk in r.iter_content(chunk_size=128):
        fd.write(chunk)
with rarfile.RarFile(join(data_dir, 'download.rar')) as rar_ref:
    rar_ref.extract('1/1.rec', path=data_dir)
rename(join(data_dir, '1/1.rec'), join(data_dir, 'subj_01.edf'))
raw = read_raw_edf(join(data_dir, 'subj_01.edf'))

