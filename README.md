# Kernel density fit framework

This is a general purpose statistical analysis framework that uses an adaptive
bandwidth kernel density estimation technique to construct smooth probability
distribution functions with finite data, which are then used to statistically
extract parameters from a dataset using either binned or unbinned maximum 
likelihood fits.

Where possible, and if available, calculations are accelerated on a GPU.

## Dependencies

[CuPy](https://cupy.dev/)
[NumPy](https://numpy.org/)
[SciPy](https://www.scipy.org/)

## Work in progress

API is subject to change at any time.



Copyright 2021 by Benjamin J. Land (a.k.a. BenLand100)
