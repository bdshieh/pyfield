# PyField

PyField is a Python wrapper for the ultrasound simulator [Field II](http://field-ii.dk/) [1] using the MATLAB Engine API for Python. As such, it requires a licensed MATLAB installation which supports the API. 

PyField also provides a simulation framework for absolute backscattering coefficient as described in [2, 3]. This framework can be used to create more realistic simulated phantoms with backscatter intensity tied to a physical tissue model.

## Installation

Installation is supported using [pip](https://pip.pypa.io/en/stable/).

**Python environment**
The python version must be supported by the version of the MATLAB installation (R2020a supports both Python 3.6 and 3.7, but older versions may not). The easiest way to ensure the right python version is used is to create a virtualenv with the python version explicitly declared, such as in [conda](https://docs.conda.io/en/latest/):
``` sh
conda create -n myenv python=3.6
conda activate myenv
```

**Install PyField from a local copy**

``` sh
git clone https://github.com/bdshieh/pyfield.git
cd pyfield
pip install .
```

**Install PyField from the remote repository**
``` sh
pip install git+https://github.com/bdshieh/pyfield.git
```

**Install the MATLAB Engine API for Python**
Navigate to the MATLAB installation (located at $matlabroot) and run setup.

``` sh
cd $matlabroot/extern/engines/python/
python setup.py install
```

Please refer to the following for help installing on different operating systems or without root priviledges:
[Install MATLAB Engine API for Python](https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
[Install MATLAB Engine API for Python in Nondefault Locations](https://uk.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html)


## Tests
To test the integrity of the installation, use pytest:
``` sh
pytest --pyargs pyfield
```

## Usage
Usage and syntax are nearly identical to the MATLAB version of Field II. For example, to calculate the spatial impulse response of a 32-element linear array:
``` python
from pyfield import PyField

field = PyField()
field.field_init()

# set simulation parameters
field.set_field('c', 1500)
field.set_field('fs', 100e6)
field.set_field('att', 0)
field.set_field('freq_att', 0)
field.set_field('att_f0', 7e6)
field.set_field('use_att', 1)

# define aperture and set focus to infinity
th = field.xdc_linear_array(32, 100e-6, 0.01, 10e-6, 1, 4, [0, 0, 300])
        
# calculate the spatial impulse response
sir, sir_t0 = field.calc_h(th, [0, 0, 0.02])

field.field_end()
```
For more in depth usage, please check out the interactive [examples](examples/) which can be run in a [jupyter](https://jupyter.org/) session.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References

[1] [J.A. Jensen: Field: A Program for Simulating Ultrasound Systems, Paper presented at the 10th Nordic-Baltic Conference on Biomedical Imaging Published in Medical & Biological Engineering & Computing, pp. 351-353, Volume 34, Supplement 1, Part 1, 1996.](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.50.4778)

[2] [Shieh, Bernard, F. Levent Degertekin, and Karim Sabra. "Simulation of absolute backscattering coefficient in Field II." 2014 IEEE International Ultrasonics Symposium. IEEE, 2014.](https://doi.org/10.1109/ULTSYM.2014.0604)

[3] [Shieh, Bernard D. Quantitative simulation of backscatter from tissue and blood flow for ultrasonic transducers. Diss. Georgia Institute of Technology, 2015.](http://hdl.handle.net/1853/53843)

