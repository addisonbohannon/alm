# Autoregressive Linear Mixture Model (ALMM)

## Introduction

This project offers multiple solvers for estimating the parameters of an Autoregressive Linear Mixture Model (ALMM):

x<sub>i</sub>[t] = &sum;<sub>s</sub> &sum;<sub>j</sub> c<sub>i,j</sub> D<sub>j</sub>[s] x<sub>i</sub>[t-s] + n<sub>i</sub>[t], i&isin;{1,...,n}.

It has implementations for block coordinate descent, alternating minimization, and proximal gradient algorithms.

## Dependencies

- [Scipy](https://www.scipy.org/)
- [Numpy](https://numpy.org/)
- [MATPLOTLIB](https://matplotlib.org/)
- [CVXPY](https://www.cvxpy.org/)

## Installation

In order to use the software, you clone the repository and install the package in an appropriate environment.

```bash
cd my_workspace
git clone https://gitlab.sitcore.net/addison.bohannon/almm.git
conda new -n almm python=3.7 scipy numpy matplotlib
source activate almm
pip install --upgrade setuptools
pip install cvxpy
python setup.py install
```
