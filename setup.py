'''Setup script'''

from setuptools import setup, find_packages

setup(
    name='pyfield',
    version='1.0',
    packages=find_packages(),
    package_data={
        'pyfield.core': ['*.m', '*.mat', '*.pdf', '*.mexw64', '*.mexa64']
    },
    #   entry_points={'console_scripts': ['pyfield = pyfield.cli:main']},
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'pytest', 'jupyter', 'tqdm',
        'ipywidgets'
    ],
    # extras_require={'interactive': []}
)
