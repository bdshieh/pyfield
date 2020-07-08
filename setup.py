'''Setup script

Author
-------

Bernard Shieh (bernard.shieh@eng.ox.ac.uk)
'''
# from distutils.core import setup, Extension
from setuptools import setup, find_packages
# from setuptools.extension import Extension

# import numpy as np
# import os

setup(
    name='pyfield',
    version='0.1',
    packages=find_packages(),
    package_data={
        'pyfield.core': ['*.m', '*.mat', '*.pdf', '*.mexw64', '*.mexa64']
    },
    #   entry_points={'console_scripts': ['pyfield = pyfield.cli:main']},
    install_requires=['numpy', 'scipy', 'pytest'],
    extras_require={
        'interactive': ['matplotlib', 'jupyter', 'tqdm', 'ipywidgets']
    })
