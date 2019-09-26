# Autoregressive Linear Mixture Model (ALMM)

## Introduction

This project offers multiple solvers for estimating the parameters of an Autoregressive Linear Mixture Model (ALMM):

```math
\mathbf{x}_i[t] = \sum_{s=1}^m \sum_{j=1}^r c_{i, j} \mathbf{D}_j[s] \mathbf{x}_i[t-s] + \mathbf{x}_i[t],
```

where $`i\in\{1,\ldots,n\}`$. The maximum likelihood estimator (MLE) for this model is nonconvex and thus nontrivial to solve. This package has implementations for block coordinate descent, alternating minimization, and proximal gradient algorithms.

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
