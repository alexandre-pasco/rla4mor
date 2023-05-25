# README.md

Implementing randomized linear algebra methods for model order reduction.

## Requirements :
This work is based on some packages that can be installed by running the following commands in a conda virtual environment:

```
pip install numpy scipy pymor numba pybind11 matplotlib
pip install spams scikit-learn scikit-umfpack
pip install ffht-unofficial
conda install -c conda-forge scikit-sparse
```
Note that `ffht-unofficial` is optional. Also, if `scikit-umfpack` fails to install, you may need to install libsuitesparse-dev by running:

```
sudo apt-get install libsuitesparse-dev
```
