# Autoregressive Linear Mixture (ALM) Model

This project offers multiple solvers for estimating the parameters of an Autoregressive Linear Mixture (ALM) Model:

```math
\mathbf{x}[t] = \sum_{s=1}^p \sum_{j=1}^r c_j \mathbf{D}_j[s] \mathbf{x}[t-s] + \mathbf{n}[t],
```

where $`p`$ is the model order, $`r`$ is the number of autoregressive components, and $`\mathbf{n}[t]`$ is a Gaussian random variable. 
The model defined by this recurrence relation is denoted $`ALM(p, r)`$. $`\left(c_j\right)_{j=1,\ldots,r}`$ are called the mixing
coefficients, and $`\left(\mathbf{D}_j[s]\right)_{s=1,\ldots,p}`$ is called the $`j`$ autoregressive component. The maximum likelihood 
estimator (MLE) for this model is nonconvex and thus nontrivial to solve. This package has implementations for approximately solving
the maximum *a posteriori* (MAP) estimator using block coordinate descent, alternating minimization, and proximal gradient algorithms.

## Dependencies

- [Scipy](https://www.scipy.org/)
- [Numpy](https://numpy.org/)
- [MATPLOTLIB](https://matplotlib.org/)
- [CVXPY](https://www.cvxpy.org/)
- [Scikit Learn](https://scikit-learn.org/)
- [Requests](https://requests.kennethreitz.org/en/master/)
- [UnRAR](https://github.com/matiasb/python-unrar)
- [MulticoreTSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)

## Installation (Linux)

In order to use the software, we clone the repository and install the package in an appropriate environment. First, we clone the remote directory to a local directory `my_workspace`.

```
cd my_workspace
git clone https://gitlab.sitcore.net/addison.bohannon/almm.git
```

Now, we create an environment with [Anaconda](https://www.anaconda.com/) that installs all of the required packages.

```
conda env create --file environment.yml
conda activate alm
```

We will need to also download the [UnRAR](https://www.rarlab.com/) library binaries, build it, and make the library findable by unrar.

```
cd /usr/lib
sudo wget https://www.rarlab.com/rar/unrarsrc-5.9.2.tar.gz
sudo tar xzvf unrarsrc-5.9.2.tar.gz
sudo rm -r unrarsrc-5.9.2.tar.gz
cd unrar
sudo make lib
sudo make install-lib
echo 'export UNRAR_LIB_PATH=/usr/lib/libunrar.so' >> ~/.bashrc 
```

Now, we can install the `alm` package.

```
python setup.py install
```
