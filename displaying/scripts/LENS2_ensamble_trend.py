import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import t 

sys.path.append('../../processing')

import processing.series as se
import processing.lens as lens

# get Quinta Normal time series
qn = se.get_QN_annual_precip()

# get LENS2 QNEW time series
lens2 = lens.get_LENS2_annual_precip_QNEW()

init_year = '2010'
end_year = '2019'
qn = qn.sel(time=slice(init_year, end_year))
lens2 = lens2.sel(time=slice(init_year, end_year))

qn_slope, qn_intercept, r, p, stderr = linregress(qn.time.dt.year.values, qn.values)

r, t_ = lens2.shape
slopes = np.zeros((r,))

########################
# QN trend CI estimation 
########################

x = qn.time.dt.year.values
x_bar = np.mean(x)
n = qn.size
y = qn.values
y_hat = x*qn_slope + qn_intercept
SE = np.sqrt(np.sum((y-y_hat)**2)/(n-2))/np.sqrt(np.sum((x-x_bar)**2))
alpha = 0.05
dof = n-2
p_star = 1-alpha/2
ME = t.ppf(p_star, dof)*SE

for k in range(r):
    series = lens2[k,:]
    slope, intercept, r, p, stderr = linregress(series.time.dt.year.values, series.values)
    slopes[k] = slope

fig = plt.figure(figsize=(3,6))
plt.boxplot(slopes*10, notch=False, meanline=True,showmeans=True, positions=[0.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.errorbar(x=0.2, y=qn_slope*10, yerr=ME*10, lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.grid(ls='--', lw=0.4, color='grey', axis='y')
plt.yticks(np.arange(-500, 550, 50))
plt.ylim([-500,500])
plt.xlim([-0.1,1.1])
plt.xticks([0.2,0.8], ["",""], rotation=0)
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8) 
    tick.label.set_weight('light') 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8) 
    tick.label.set_weight('light')
plt.savefig('../../../hyperdrought_data/png/LENS2_QNWE_ensamble_trend_confinterv_'+init_year+'_'+end_year+'.png', dpi=300)
plt.show()