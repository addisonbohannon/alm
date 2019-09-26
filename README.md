# Autoregressive Linear Mixture Model (ALMM)

## Introduction

This project offers multiple solvers for estimating the parameters of an Autoregressive Linear Mixture Model (ALMM):

```math
\mathbf{x}[t] = \sum_{s=1}^p \sum_{j=1}^r c_j \mathbf{D}_j[s] \mathbf{x}[t-s] + \mathbf{x}[t],
```

where $`p`$ is the model order and $`r`$ is the number of autoregressive components. The maximum likelihood estimator (MLE) for this model is nonconvex and thus nontrivial to solve. This package has implementations for block coordinate descent, alternating minimization, and proximal gradient algorithms.

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
