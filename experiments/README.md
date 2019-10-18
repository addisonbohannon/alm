## Introduction

These scripts re-create the experiments used in the paper. The package includes several scripts:

- `comparison.py`
- `n_vs_m.py`
- `n_vs_m-analysis.py`

It also includes modules `sampler.py` for generating data according to the $`ALM(p, r)`$ model.

## Comparison

This experiment compares the performance of alternating minimization (AltMin), block coordinate descent (BCD), and 
proximal alternating linearized minimization (PALM) on a common problem. It used multiple starts on account of the 
non-convexity. All algorithms use common initializations for each start. The script can be run as follows:

```
cd my_workspace/almm
python validation/comparison.py
```

## Number of observations versus observation length

This experiment compares the performance of AltMin, BCD, and PALM for varying values of number of observations and 
observation length. It is currently set-up to use multiprocessing to take advantage of multi-CPU machines. It can be run
as follows:

```
cd my_workspace/almm
python validation/n_vs_m.py
```

## Sampling

In order to test the algorithms, `sampler.py` generates data according to the $`ALMM(p, r)`$ model. For example, if I
wanted to generate $`100`$ observations in $`\mathbb{R}^5`$ of length $`1000`$ according to $`ALMM(2, 20)`$ such that
each observation comprised a mixture of $`4`$ components, I could execute the following:

```python
from alm.sampler import almm_sample
n, m, d, r, p, s = 100, 1000, 5, 2, 20, 4
data, mixing_coef, almm_component = almm_sample(n, m, d, r, p, s)
```

Additionally, there are parameters for `coef_condition` and `component_condition` which default to `None`. If values
are passed to these parameters, then `almm_sample` will generate samples until the samples satisfy the given condition.
