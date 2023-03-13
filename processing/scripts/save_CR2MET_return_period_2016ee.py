import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.utils import resample as bootstrap
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir, '../../processing'))

import processing.gmst as gmst
import processing.series as se
import processing.math as pmath
import processing.utils as ut

da = ut.get_return_period_cr2met_2016ee()
ds = xr.Dataset({'tau':da})
filepath = '../../../hyperdrought_data/output/CR2METv25_2016_return_period.nc'
ds.to_netcdf(join(currentdir,filepath))