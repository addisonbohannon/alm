from setuptools import setup

setup(
    name='almm',
    version='0.0.0',
    packages=['almm', 'validation'],
    url='https://gitlab.sitcore.net/addison.bohannon/almm',
    license='',
    author='Addison Bohannon',
    author_email='addison.bohannon@gmail.com',
    description='Autoregressive Linear Mixture Model',
    install_requires=['numpy', 'scipy', 'cvxpy', 'matplotlib', 'scikit-learn'],
    scripts=[]
)
