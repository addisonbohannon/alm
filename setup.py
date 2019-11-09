#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup
# from Cython.Build import cythonize

setup(
    name='alm',
    version='0.0.0',
    packages=['alm', 'experiments'],
    url='https://gitlab.sitcore.net/addison.bohannon/almm',
    license='',
    author='Addison Bohannon',
    author_email='addison.bohannon@gmail.com',
    description='Autoregressive Linear Mixture Model',
    install_requires=['numpy', 'scipy', 'cvxpy', 'matplotlib', 'scikit-learn'],
    #setup_requires=['Cython'],
    scripts=[],
    #ext_modules=cythonize("alm/utility.py", compiler_directives={'language_level': "3"})
)
