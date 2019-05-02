# Autoregressive Linear Mixture Model (ALMM)

x<sub>i</sub>[t] = &sum;<sub>s</sub> &sum;<sub>j</sub> c<sub>i,j</sub> A<sub>j</sub>[s] x<sub>i</sub>[t-s] + n<sub>i</sub>[t], i&isin;{1,...,n}

## Dependencies

- Numpy/Scipy
- [CVXPY](https://www.cvxpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
