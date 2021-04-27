# Kernel density fit framework

This is a general purpose statistical analysis framework that uses an adaptive
bandwidth kernel density estimation technique to construct smooth probability
distribution functions with finite data, which are then used to statistically
extract parameters from a dataset using either binned or unbinned maximum 
likelihood fits.

Where possible, and if available, calculations are accelerated on a GPU.

See [my blog post on kernel density estimation](https://ben.land/post/2021/04/18/kernel-density-estimation-unbinned-likelihood/)
for an overview of the mathematics underpinning this implementation of kernel
density estimation, as well as brief description of how to use this package.

The `examples` directory also contains some toy fits in Jupyter notebooks.

## Work in progress

Currently `kdfit` is pre-alpha. API is subject to change at any time.

## Dependencies

This project depends on features present in Python 3.9+. The included setup.py 
will install all necessary dependencies:

[CuPy](https://cupy.dev/)
[NumPy](https://numpy.org/)
[SciPy](https://www.scipy.org/)
[Uproot4](https://uproot.readthedocs.io/en/latest/) - optional
[h5py](https://www.h5py.org/) - optional

## Installing

First clone this repository:

`git clone https://github.com/BenLand100/kdfit`

And then install with pip:

`pip install --user -e kdfit`

This will create an editable installation that will link to the cloned git repo.
This means the kdfit code installed on your system will automatically be updated
if the repo is updated or edited Reinstallation may still be necessary if there
are new dependencies.



Copyright 2021 by Benjamin J. Land (a.k.a. BenLand100).
Released under the GPLv3 license.
