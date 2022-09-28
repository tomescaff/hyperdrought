import sys
import xarray as xr
import numpy as np
from scipy.stats import chi2
from scipy.stats import t
import matplotlib.pyplot as plt

sys.path.append('../../processing')

import processing.series as se
import processing.lens as lens

# get El Tepual time series
pm = se.get_PM_JFM_precip()
lens2 = lens.get_LENS2_JFM_precip_NOAA()

# lon: 286.2, 287.5
# lat: -41.937173, -40.994764

nw = lens2.sel(lat= -40.994764, lon = 286.2, method='nearest').drop(['lat','lon'])
ne = lens2.sel(lat= -40.994764, lon = 287.5, method='nearest').drop(['lat','lon'])
sw = lens2.sel(lat= -41.937173, lon = 286.2, method='nearest').drop(['lat','lon'])
se = lens2.sel(lat= -41.937173, lon = 287.5, method='nearest').drop(['lat','lon'])

pm_1950_2021 = pm.sel(time=slice('1950', '2021'))

nw_1950_2021 = nw.sel(time=slice('1950', '2021'))
ne_1950_2021 = ne.sel(time=slice('1950', '2021'))
sw_1950_2021 = sw.sel(time=slice('1950', '2021'))
se_1950_2021 = se.sel(time=slice('1950', '2021'))

nw_std_1950_2021 = nw_1950_2021.std('time')
ne_std_1950_2021 = ne_1950_2021.std('time')
sw_std_1950_2021 = sw_1950_2021.std('time')
se_std_1950_2021 = se_1950_2021.std('time')

########################
# QN std estimation 
########################

std_dev = pm_1950_2021.std(ddof=1)
n = pm_1950_2021.size
alpha = 0.05
q_l = alpha/2
q_r = 1-alpha/2
chi2_l = chi2.ppf(q_l, n-1)
chi2_r = chi2.ppf(q_r, n-1)
ci_l = np.sqrt((n-1)*std_dev**2/chi2_r)
ci_r = np.sqrt((n-1)*std_dev**2/chi2_l)
ci = np.zeros((2,1))
ci[0] = std_dev-ci_l
ci[1] = ci_r-std_dev


fig = plt.figure(figsize=(5,6))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

plt.boxplot(nw_std_1950_2021, notch=False, meanline=True,showmeans=True, positions=[0.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.boxplot(ne_std_1950_2021, notch=False, meanline=True,showmeans=True, positions=[0.8+0.6], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.boxplot(sw_std_1950_2021, notch=False, meanline=True,showmeans=True, positions=[0.8+1.2], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.boxplot(se_std_1950_2021, notch=False, meanline=True,showmeans=True, positions=[0.8+1.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))

plt.errorbar(x=0.2, y=std_dev, yerr=ci, lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.grid(ls='--', lw=0.4, color='grey', axis='y')

plt.xlim([-0.1,3.1])
plt.xticks([0.2,0.8, 0.8+0.6, 0.8+1.2, 0.8+1.8], ["PM","NW", "NE", "SW", "SE"], rotation=0)
plt.ylabel('JFM acc precip std 1950-2021 (mm)')
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
plt.savefig('../../../hyperdrought_data/png/LENS2_PM_bias_std_confint.png', dpi=300)
plt.show()