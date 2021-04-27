import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kdfit-benland100",
    version="0.0.1",
    author="Benjamin J. Land",
    author_email="benland100@gmail.com",
    description="A package for performing kernel density estimation with CUDA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GNU General Public License v3 (GPLv3)",
    platforms="any",
    url="https://github.com/BenLand100/kdfit",
    project_urls={
        "Bug Tracker": "https://github.com/BenLand100/kdfit/issues"
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.9",
        "Environment :: GPU :: NVIDIA CUDA",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    packages=['kdfit'],
    python_requires=">=3.9",
    install_requires=[
        "scipy",
        "numpy",
        "cupy",
        "h5py",
        "uproot4"
    ]
)
