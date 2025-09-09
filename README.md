# SPARK-LRS
SPectroscopic Automated Reduction Kit for the Low Resolution Spectrograph on the 2.4 m Thai National Telescope - a Python pipeline for automated spectroscopic data reduction of TNT/LRS observations.

## About
SPARK-LRS automates the reduction of data from the 2.4 m Thai National Telescope’s Low Resolution Spectrograph (LRS).
While it was originally designed for exoplanetary transits, it was expanded to provide a general framework for long-slit spectroscopy for both, single- and dual-target slit observations.

### Features include: 
- automated file recognition and sorting
- creation of master frames
- bias subtraction
- dark subtraction
- flat fielding
- wavelength calibration using HgAr, Ne or custom arcs
- sky subtraction
- cosmic ray removal
- spectrum extraction
- photometric light curve generation, normalisation and response correction

SPARK-LRS is modular and user-friendly, allowing any level of user intervention - from fully automated to step-by-step control.

For explanation behind commands and data reduction please see: [reduction with SPARK-LRS  step by step](docs/reduction_steps.md)

## Installation
To install the pipeline directly from github input:
```
pip install git+https://github.com/idajaniak/SPARK-LRS.git
```
into your terminal.

### Dependencies (automatically installed with pip)
- numpy
- astropy
- matplotlib
- ccdproc
- scipy
- scikit-image
- astroscrappy
- reproject
- scikit-learn
- pandas


### Features to add:
- a worked through example of data reduction in Jupyter Notebook
- a PyPI installation
- an ASCL link
- license
