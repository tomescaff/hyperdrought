import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm as gev
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/forest/'

filename = 'BE_pixel_2021.csv'
filepath = join(currentdir, relpath, filename)
df = pd.read_csv(filepath, sep=';')
df = df.set_index('ID')
df = df.transpose()
df = df.reset_index()
df['time'] = df['index'].apply(lambda x: datetime.strptime(x, '%Y_%m_%d_NDVI'))
df = df.set_index('time')
df = df.drop('index', axis=1)

da = xr.DataArray(df.values, coords=[df.index, df.columns.astype('float')], dims=['time', 'pixel'])
ds = xr.Dataset({'ndvi':da})
ds.to_netcdf(join(currentdir, relpath, 'ndvi.nc'))
