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
da = da - da.mean('time')

ndvi_2001 = da.sel(time='2001')
ndvi_2010 = da.sel(time='2010')
ndvi_2020 = da.sel(time='2020')

ma1 = ndvi_2001.values
ma2 = ndvi_2020.values
ma3 = ndvi_2010.values

ma1 = ma1/10000
ma2 = ma2/10000
ma3 = ma3/10000

xmin = -.5
xmax = +.5
bins = np.linspace(xmin, xmax, 100)
hist1, bins1 = np.histogram(ma1, bins=bins, density='True')
hist2, bins2 = np.histogram(ma2, bins=bins, density='True')
hist3, bins3 = np.histogram(ma3, bins=bins, density='True')

width1 = 0.9 * (bins1[1] - bins1[0])
center1 = (bins1[:-1] + bins1[1:]) / 2
width2 = 0.9 * (bins2[1] - bins2[0])
center2 = (bins2[:-1] + bins2[1:]) / 2
width3 = 0.9 * (bins3[1] - bins3[0])
center3 = (bins3[:-1] + bins3[1:]) / 2

x = np.linspace(xmin, xmax, 1000)

fig = plt.figure()
plt.bar(center1, hist1, align='center', width=width1, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75)
plt.bar(center2, hist2, align='center', width=width2, edgecolor='red', facecolor='lightcoral', color='red', alpha = 0.75)

plt.plot(x, dist.pdf(x, *dist.fit(ma1)), color='blue')
plt.plot(x, dist.pdf(x, *dist.fit(ma2)), color='red')
plt.plot(x, dist.pdf(x, *dist.fit(ma3)), color='grey')

#plt.axvline(0.45, color='k')
plt.savefig(join(currentdir, relpath, 'summer_differences_anom.png'), dpi=300)
plt.show()