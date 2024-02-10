import sys
import numpy as np
from scipy.stats import linregress
from scipy.stats import t
from scipy.stats import norm
from sklearn.utils import resample as bootstrap
import matplotlib.pyplot as plt

from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir, '../../processing'))

import processing.series as se
import processing.utils as ut
import processing.cr2met_v25 as cr2met

# get Quinta Normal time series
qn = se.get_PM_JFM_precip()

def linear_trend(se):

    # get linear trend
    x = se.time.dt.year.values
    y = se.values
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
    return x, y, slope, trend, y_inf, y_sup

# create figure
fig, axs = plt.subplots(1,1, figsize=(9,3.3))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
ax_qn = axs

# qn
se = qn
x = se.time.dt.year.values
y = se.values

# plot the mean value
plt.sca(ax_qn)
plt.plot(x, x*0+se.mean().values, lw=1.2, color='b', ls='--', label='mean value')
plt.plot(se.sel(time=slice('1991','2020')).time.dt.year, se.sel(time=slice('1991','2020')).time.dt.year*0+se.sel(time=slice('1991','2020')).mean().values, lw=1.2, color='green', ls='--', label='1991-2020 clim.')

# plot the data
plt.plot(x, y, lw = 0.9, marker='o', markersize=7, markeredgecolor='blue', markerfacecolor='white', color='blue')

# trend
se = se.sel(time=slice('1980', '2021'))
x, y, slope, trend, y_inf, y_sup = linear_trend(se)
se_norm = se/se.sel(time=slice('1980','2021')).mean('time')*100-100
x_norm, y_norm, slope_norm, trend_norm, y_inf_norm, y_sup_norm = linear_trend(se_norm)
plt.plot(x, trend, lw = 1.2, alpha=0.7, color='r', label='Linear trend ({:.1f} mm/dec, {:.1f} %/dec)'.format(slope*10, slope_norm*10))
plt.fill_between(x, y_sup, y_inf, facecolor='grey', linewidth=2, alpha=0.2)
plt.plot(se.sel(time='2016').time.dt.year, se.sel(time='2016').values, lw=1, marker='o', markersize=7, markeredgecolor='blue', markerfacecolor='blue', color='blue', alpha=0.4)

# set grid
plt.grid(lw=0.4, ls='--', color='grey')
plt.xlim(1948, 2022)
plt.ylim(50, 550)

# set title and labels
plt.xlabel('Time (year)')
plt.ylabel('Annual Precipitation (mm)')
plt.legend()
 
# Hide the right and top spines
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")

plt.tight_layout()
plt.savefig(join(currentdir, '../../../hyperdrought_data/png/PM_series_with_trend_full_from_1980.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()