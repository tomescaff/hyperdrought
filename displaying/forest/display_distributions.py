import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as gev
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/forest/'

filename = 'SerieF.csv'
filepath = join(currentdir, relpath, filename)
df = pd.read_csv(filepath, parse_dates={'time':['dat1[, 1]']})
df = df.set_index('time')
ma = df.iloc[:,1:].values
ma = np.mean(ma, axis = 0)

filename = 'BE_pixel_2021.csv'
filepath = join(currentdir, relpath, filename)
df = pd.read_csv(filepath, sep=';')
ma1 = df.iloc[:, 0:100].values
ma1 = np.nanmean(ma1, axis=1)

ma2 = df.iloc[:, -100:].values
ma2 = np.nanmean(ma2, axis=1)

ma1 = ma1/10000
ma2 = ma2/10000

xmin = .2500
xmax = .8500
bins = np.linspace(xmin, xmax, 100)
hist1, bins1 = np.histogram(ma1, bins=bins, density='True')
hist2, bins2 = np.histogram(ma2, bins=bins, density='True')

width1 = 0.9 * (bins1[1] - bins1[0])
center1 = (bins1[:-1] + bins1[1:]) / 2
width2 = 0.9 * (bins2[1] - bins2[0])
center2 = (bins2[:-1] + bins2[1:]) / 2

x = np.linspace(xmin, xmax, 1000)

fig = plt.figure()
plt.bar(center1, hist1, align='center', width=width1, edgecolor='blue', facecolor='lightskyblue', color='blue', alpha = 0.75)
plt.bar(center2, hist2, align='center', width=width2, edgecolor='red', facecolor='lightcoral', color='red', alpha = 0.75)
plt.plot(x, gev.pdf(x, *gev.fit(ma1)), color='blue')
plt.plot(x, gev.pdf(x, *gev.fit(ma2)), color='red')
plt.axvline(0.45, color='k')

plt.show()