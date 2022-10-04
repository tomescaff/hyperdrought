import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm as dist
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/forest/'

da = xr.open_dataset(join(currentdir, relpath, 'ndvi.nc'))['ndvi']
da = da.where(da.time.dt.month.isin([1, 2, 3, 4]), drop=True)
da = da.resample(time='1YS').mean('time')

ndvi_2019 = da.sel(time='2019')
ndvi_2020 = da.sel(time='2020')

diff = ndvi_2020.values-ndvi_2019.values

ma1 = diff
ma1 = ma1/10000

xmin = -.5
xmax = +.5
bins = np.linspace(xmin, xmax, 100)
hist1, bins1 = np.histogram(ma1, bins=bins, density='True')

width1 = 0.9 * (bins1[1] - bins1[0])
center1 = (bins1[:-1] + bins1[1:]) / 2

x = np.linspace(xmin, xmax, 1000)

fig = plt.figure()
plt.bar(center1, hist1, align='center', width=width1, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75)
plt.savefig(join(currentdir, relpath, 'change_2019_2020.png'), dpi=300)
plt.show()