# Noise Latent space

This repo contains the files for the research project on forecasting in noise latent spaces in course 732A76.

Example code for inverting and sample is in `playground.ipynb`.

To generate more training data run `sqg_nature_run.py` and then `generate_inverted_timeseries.py`

To sample you need the following packages
* pytorch
* numpy
* matplotlib

To generate new training data you also need
* pyfftw
* netcdf4