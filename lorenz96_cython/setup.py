"""
Run
$ python setup.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# setup(
#     ext_modules=cythonize(
#         "lorenz96_cython.pyx",  # Cythonファイル名
#         compiler_directives={"language_level": "3"}  # Python 3対応
#     ),
#     include_dirs=[np.get_include()],
# )

setup(
    name="lorenz96_cython",
    ext_modules=cythonize("lorenz96_cython.pyx"),
    include_dirs=[np.get_include()],
    package_dir={"lorenz96_cython": "lorenz96_cython"},
    packages=["lorenz96_cython"],
)
