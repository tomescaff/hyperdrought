import xarray as xr
import numpy as np
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/ENSO/'

pr = xr.open_dataset(join(currentdir, relpath, 'pr_80ens.nc'))
nino34 = xr.open_dataset(join(currentdir, relpath, 'sst_nino34_80ens.nc'))