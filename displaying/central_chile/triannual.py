import sys
import numpy as np
from scipy.stats import linregress
from scipy.stats import t
from scipy.stats import norm
from scipy.stats import gamma
from sklearn.utils import resample as bootstrap
import matplotlib.pyplot as plt
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir,'../../processing'))

import processing.series as se

# get Quinta Normal time series
pm = se.get_QN_annual_precip_long_record()
data = pm.rolling(time=3).mean().sel(time=slice('1931', '2021'))
data = data.dropna('time')

# get linear trend
data_trend = data.sel(time=slice('1931', '2021'))
x = data_trend.time.dt.year.values
y = data_trend.values
slope, intercept, rvalue, pvalue, stderr  = linregress(x,y)
trend = x*slope + intercept

# compute confidence intervals
xmean = np.mean(x)
SXX = np.sum((x-xmean)**2)
SSE = np.sum((y - trend)**2)
n = x.size
sig_hat_2_E = SSE/(n-2)
sig_hat_E = np.sqrt(sig_hat_2_E)
rad = np.sqrt(1/n + (x-xmean)**2/SXX) 
p = 0.95
q = (1+p)/2
dof = n-2
tval = t.ppf(q, dof)
delta = tval*sig_hat_E*rad
y_sup = trend+delta
y_inf = trend-delta

# plot the data
fig = plt.figure(figsize=(15,7))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 9
plt.plot(data.time.dt.year.values, data.values, lw = 1, marker='o', markersize=7, markeredgecolor='blue', markerfacecolor='white', color='blue')
plt.plot(x, trend, lw = 1.5, alpha=0.7, color='r') 
plt.fill_between(x, y_sup, y_inf, facecolor='grey', linewidth=2, alpha=0.4)
#plt.plot(qn.sel(time='2016').time.dt.year, qn.sel(time='2016').values, lw=1, marker='o', markersize=7, markeredgecolor='blue', markerfacecolor='blue', color='blue', alpha=0.4)
# set grid
plt.grid(lw=0.4, ls='--', color='grey')

# set title and labels
plt.xlabel('Time (year)')
plt.ylabel('3yr precipitation (mm/year)')
plt.xlim([1928, 2023])
plt.ylim([0, 600])

# Hide the right and top spines
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.savefig(join(currentdir, '../../../hyperdrought_data/png/QN_triannual.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)


plt.show()