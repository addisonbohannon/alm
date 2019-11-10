#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='alm',
    version='0.0.0',
    packages=['alm', 'experiments'],
    url='https://gitlab.sitcore.net/addison.bohannon/almm',
    license='',
    author='Addison Bohannon',
    author_email='addison.bohannon@gmail.com',
    description='Autoregressive Linear Mixture Model',
    install_requires=['cvxpy', 'mne', 'numpy', 'matplotlib', 'requests', 'scikit-learn', 'scipy', 'unrar'],
    scripts=[]
)
