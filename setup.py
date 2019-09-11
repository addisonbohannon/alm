from setuptools import setup

setup(
    name='almm',
    version='0.0.0',
    packages=['almm'],
    url='https://gitlab.sitcore.net/addison.bohannon/almm',
    license='',
    author='Addison Bohannon',
    author_email='addison.bohannon@gmail.com',
    description='Autoregressive Linear Mixture Model',
	scripts=[
			 'experiments/comparison_single.py',
			 'experiments/comparison_multiple.py',
			 'experiments/n_vs_m.py'
			]
)
