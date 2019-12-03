# Introduction

This package includes utility functions and scripts to re-create the experiments from the validation and application sections of the paper.

## Utility

`utility.py` contains all of the supporting functions for the validation and application.

## Sampling

In order to test the algorithms, `sampler.py` generates data according to the $`ALM(p, r)`$ model. For example, if I
wanted to generate $`100`$ observations in $`\mathbb{R}^5`$ of length $`1000`$ according to $`ALM(2, 20)`$ such that
each observation comprised a mixture of $`4`$ components, I could execute the following:

```python
from alm.sampler import alm_sample
n, m, d, r, p, s = 100, 1000, 5, 2, 20, 4
data, mixing_coef, alm_component = alm_sample(n, m, d, r, p, s)
```

Additionally, there are parameters for `coef_condition` and `component_condition` which default to `None`. If values
are passed to these parameters, then `alm_sample` will generate samples until the samples satisfy the given condition.

## Use

Currently, all paths are hard-coded and need to be updated by the user to reflect the local workspace.

# Validation

The validation results can be generated using the following three scripts:

- `performance.py`
- `n_vs_m.py`
- `model_misspecification.py`

# Application

The application results can be generated using the following scripts:

- `discrimination.py`
- `visualize_coef.py`
- `periodogram.py`
- `sleep_stage_signals_and_periodogram.py`

In addition, `fit_alm_to_isruc.py` will fit the ALM model to the ISRUC data.
