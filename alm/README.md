## Autogregressive Linear Mixture (ALM) Model

```python
class alm.alm.Almm(coef_penalty_type='l1', step_size=1e-1, tol=1e-4, max_iter=int(2.5e3), solver='bcd', verbose=False)
```

The class for fitting the $`ALM(p, r)`$ model. The maximum *a posteriori* estimator for the model is given by the following objective:

```math
\operatorname{arg\,min} \frac{1}{n}\sum_{i=1}^n \frac{1}{2m} \left\lVert \mathbf{Y}_i - \sum_{j=1}^r c_{i, j} \mathbf{X}_i \mathbf{D}_j \right\rVert_F^2 + \frac{1}{n}\sum_{i=1}^n \mu \left\lVert \mathbf{c}_i \right\rVert_1 ~\text{s.t.}~ \left\lVert\mathbf{D}_j\right\rVert_F=1, j=1,\ldots,r
```

where we are intersted in the mixing coefficients $`\left(\mathbf{c}_i\right)_{i=1,\ldots,n}`$ and the autoregressive components $`\left(\mathbf{D}_j\right)_{j=1,\ldots,r}`$. 

| Paramater | Values | Description |
| :--- | :---: | :--- |
| coef_penalty_type | None, 'l0', 'l1' | Penalty for the mixing coefficients |
| step_size | float | Step-size used by PALM algorithm; must be in $`(0, 1)`$ |
| tol | float | Tolerance used for terminating algorithm; must be positive |
| max_iter | integer | Maximum number of iterations for which the algorithm will run; must be positive |
| solver | 'altmin', 'bcd', 'palm' | Algorithm which will be used to minimize the objective |
| verbose | boolean | Whether or not to print reports from solver during run-time |

| Attributes | Values | Description |
| :--- | :---: | :--- |
| coef_penalty_type | None, 'l0', 'l1' | Penalty for the mixing coefficients |
| step_size | float | Step-size used by PALM algorithm; must be in $`(0, 1)`$ |
| tol | float | Tolerance used for terminating algorithm; must be positive |
| max_iter | integer | Maximum number of iterations for which the algorithm will run; must be positive |
| solver | 'altmin', 'bcd', 'palm' | Algorithm which will be used to minimize the objective |
| verbose | boolean | Whether or not to print reports from solver during run-time |
| ar_comps | list |  Nested list of autoregressive component estimates, `(num_comps, model_ord*signal_dim, signal_dim)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| mixing_coef | list | Nested list of mixing coefficient estimates, `(num_obs, num_comps)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| solver_time | list | Nested list of wall time for algorithm, `float`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| nll | list | Nested list of negative likelihood values, `float`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| residual | list | Nested list of residuals from algorithm, `tuple(float, float)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration |
| stop_condition | list | List of stopping conditions from algorithm, `str`, indexed by initialization |

| Methods | Description |
| :--- | :--- |
| `fit` | Fit model with desired algorithm |

```python
fit(observation, model_ord, num_comps, penalty_param, num_starts=5, initial_comps=None, 
    return_path=False, return_all=False)
```

This method will implement the desired algorithm to fit the $`ALM(p, r)`$ model to observations.

| Paramater | Values | Description |
| :--- | :---: | :--- |
| observation | numpy array | Observations with which to fit model; `(n_obs, obs_len, signal_dim)` |
| model_ord | integer | Model order of ALM model |
| num_comps | integer | Number of autoregressive components of ALM model |
| penalty_param | float | Value with which to weight the mixing coefficient penalty |
| num_starts | integer | Number of unique initializations for algorithm |
| initial_comps | list | Initial estimate of autoregressive components, `(num_comps, model_ord*signal_dim, signal_dim)`; `len(initial_comps)=num_starts` |
| return_path | boolean | Whether to return estimates from each iteration |
| return_all | boolean | Whether to return the result of all `num_starts` unique initializations or that of maximum likelihood |

| Returns | Values | Description |
| :--- | :---: | :--- |
| ar_comps | list |  Nested list of autoregressive component estimates, `(num_comps, model_ord*signal_dim, signal_dim)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| mixing_coef | list | Nested list of mixing coefficient estimates, `(num_obs, num_comps)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| nll | list | Nested list of negative likelihood values, `float`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True`, unless over-ruled by `compute_likelihood_path=False` |
| solver_time | list | Nested list of wall time for algorithm, `float`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |