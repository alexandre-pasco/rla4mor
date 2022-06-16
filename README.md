# README.md

Implementing randomized linear algebra methods for model order reduction.

## Requirements :
This work is based on some packages listed below. For some reasons, conda can take forever to solve environment when installing packages from conda-forge. A solution is to use pip instead (when possible).


- matplotlib (`conda install matplotlib`);
- scikit-sparse (`conda install -c conda-forge scikit-sparse`);
- pybind11 (`conda install -c conda-forge pybind11`);
- spams (`pip install spams`);
- pymor (`pip install pymor[all]`);
- sklearn (`pip install sklearn`);
- FFHT (find it here : https://github.com/dnbaker/FFHT.git, which is an updated version allowing compatibility with recent versions of numpy). If the installation fails (or if the fht function are not found when using python), install the FFHT-matser available in this repo (in external_libs). It is a slightly modified version, only the multi-dim function has been removed.
