"""
Runb
python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# avoid __init__.py bug
extensions = [Extension("lorenz96_cython", ["lorenz96_cython.pyx"])]

setup(
    name="lorenz96_cython",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],  # include numpy headers
    packages=["lorenz96_cython"],
)
