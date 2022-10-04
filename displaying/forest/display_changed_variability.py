import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm as dist
from os.path import join, abspath, dirname
from eofs.xarray import Eof

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/forest/'

da = xr.open_dataset(join(currentdir, relpath, 'ndvi.nc'))['ndvi']
da = da.where(da.time.dt.month.isin([1, 2, 3, 4]), drop=True)
da = da.resample(time='1YS').mean('time')
da = da/10000

ndvi_2019 = da.sel(time='2019')
ndvi_2020 = da.sel(time='2020')

diff = ndvi_2020-ndvi_2019.values

changed = diff.where(diff <= -0.2, drop=True)
da_sel = da.sel(pixel=changed.pixel)
da_sel = da_sel-da_sel.mean('time')

solver = Eof(da_sel)
pc1 = solver.pcs(npcs=1, pcscaling=1)*-1*da_sel.mean('pixel').std('time').values

fig = plt.figure(figsize=(10,7))
plt.plot(da_sel.time.dt.year, da_sel.values, color='grey', alpha=0.5)
plt.plot(da_sel.time.dt.year, da_sel.mean('pixel').values, color='red', label='mean pixel')
plt.plot(pc1.time.dt.year, pc1.values, color='blue', ls='--', label='PC1')
plt.legend()
plt.savefig(join(currentdir, relpath, 'changed_variability.png'), dpi=300)
plt.show()