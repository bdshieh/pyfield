## interaction / setup.py
'''
Setup script to compile cython files. To compile, use:
'python setup.py build_ext --inplace'

Author: Bernard Shieh (bshieh@gatech.edu)
'''
# from distutils.core import setup, Extension
from setuptools import setup, find_packages
from setuptools.extension import Extension

import numpy as np
import os

setup(
    name='pyfield',
    version='0.1',
    packages=find_packages(),
    package_data={
        'pyfield.core': ['*.m', '*.mat', '*.pdf', '*.mexw64', '*.mexa64']
    },
    #   entry_points={'console_scripts': ['pyfield = pyfield.cli:main']},
    install_requires=['numpy', 'scipy', 'pytest'])
