import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import sys
from sklearn.utils import resample as bootstrap
from scipy.stats import chi2
from scipy.stats import t
from scipy import stats

sys.path.append('../../processing')

import processing.series as se
import processing.lens as lens

# get Quinta Normal time series
qn = se.get_QN_annual_precip_long_record()
lens2 = lens.get_LENS2_annual_precip_NOAA_QNEW()

qn_1866_2021 = qn.sel(time=slice('1866', '2021'))
qn_1866_2021 = qn_1866_2021/qn_1866_2021.sel(time=slice('1981','2010')).mean('time')

lens2_1866_2021 = lens2.sel(time=slice('1866', '2021'))
lens2_1866_2021 = lens2_1866_2021/lens2_1866_2021.sel(time=slice('1981','2010')).mean('time')

def get_trends(series):
    nruns, ntime = series.shape
    series_trends = np.zeros((nruns,))
    for k in range(nruns):
        y = series[k,:].values
        x = series[k,:].time.dt.year.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        series_trends[k] = slope
    return series_trends

lens2_trends = get_trends(lens2_1866_2021)

########################
# QN trend CI estimation 
########################

slope, intercept, r_value, p_value, std_err = stats.linregress(qn_1866_2021.time.dt.year.values, qn_1866_2021.values)

x = qn_1866_2021.time.dt.year.values
x_bar = np.mean(x)
n = qn_1866_2021.size
y = qn_1866_2021.values
y_hat = x*slope + intercept
SE = np.sqrt(np.sum((y-y_hat)**2)/(n-2))/np.sqrt(np.sum((x-x_bar)**2))
alpha = 0.05
dof = n-2
p_star = 1-alpha/2
ME = t.ppf(p_star, dof)*SE

fig = plt.figure(figsize=(5,6))
# plt.rcParams["font.family"] = 'Arial'
plt.boxplot(lens2_trends*10*100, notch=False, meanline=True,showmeans=True, positions=[0.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.errorbar(x=0.2, y=slope*10*100, yerr=ME*10*100, lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.grid(ls='--', lw=0.4, color='grey', axis='y')
# plt.yticks(np.arange(-1, 4+1, 1.0))
# plt.ylim([-1,4])
plt.xlim([-0.1,1.1])
plt.xticks([0.2,0.8], ["",""], rotation=0)
plt.ylabel('Norm precip annual trend 1866-2021 (%/dec)')
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
plt.savefig('../../../hyperdrought_data/png/LENS2_QNWE_bias_norm_trend_confint.png', dpi=300)
plt.show()