## Autogregressive Linear Mixture Model (ALMM)

```python
class almm.almm.Almm(coef_penalty_type='l1', step_size=1e-1, tol=1e-4, max_iter=int(2.5e3), solver='bcd', verbose=False)
```

The class for fitting the $`ALMM(p, r)`$ model. The solvers minimize the following objective:

```math
\frac{1}{n}\sum_{i=1}^n \frac{1}{2m} \left\lVert \mathbf{Y}_i - \sum_{j=1}^r c_{i, j} \mathbf{X}_i \mathbf{D}_j \right\rVert_F^2 + \frac{1}{n}\sum_{i=1}^n \mu \left\lVert \mathbf{c}_i \right\rVert_1
```

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
| component | list |  Nested list of autoregressive component estimates, `(num_components, model_order*obs_dim, obs_dim)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| mixing_coef | list | Nested list of mixing coefficient estimates, `(num_obs, num_components)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| solver_time | list | Nested list of wall time for algorithm, `float`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| nll | list | Nested list of negative likelihood values, `float`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| residual | list | Nested list of residuals from algorithm, `tuple(float, float)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration |
| stop_condition | list | List of stopping conditions from algorithm, 'str', indexed by initialization |

| Methods | Description |
| :--- | :--- |
| `fit` | Fit model with desired algorithm |

```python
fit(observation, model_order, num_components, penalty_parameter, num_starts=5, initial_component=None, 
    return_path=False, return_all=False)
```

This method will implement the desired algorithm to fit the $`ALMM(p, r)`$ model to observations.

| Paramater | Values | Description |
| :--- | :---: | :--- |
| observation | numpy array | Observations with which to fit model; `(n_obs, obs_len, obs_dim)` |
| model_order | integer | Model order of ALMM model |
| num_components | integer | Number of autoregressive components of ALMM model |
| penalty_parameter | float | Value with which to weight the mixing coefficient penalty |
| num_starts | integer | Number of unique initializations for algorithm |
| initial_components | list | Initial estimate of autoregressive components, `(num_components, model_order*obs_dim, obs_dim)`; `len(initial_components)=num_starts` |
| return_path | boolean | Whether to return estimates from each iteration |
| return_all | boolean | Whether to return the result of all `num_starts` unique initializations or that of maximum likelihood |


| Returns | Values | Description |
| :--- | :---: | :--- |
| component | list |  Nested list of autoregressive component estimates, `(num_components, model_order*obs_dim, obs_dim)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| mixing_coef | list | Nested list of mixing coefficient estimates, `(num_obs, num_components)`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| nll | list | Nested list of negative likelihood values, `float`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |
| solver_time | list | Nested list of wall time for algorithm, `float`; outer list indexed by initialization, `return_all=True`; inner list indexed by iteration, `return_path=True` |