import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

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

nw_mean_1950_2021 = nw_1950_2021.mean('time')
ne_mean_1950_2021 = ne_1950_2021.mean('time')
sw_mean_1950_2021 = sw_1950_2021.mean('time')
se_mean_1950_2021 = se_1950_2021.mean('time')

########################
# pm mean estimation 
########################

mean = pm_1950_2021.mean()
std_dev = pm_1950_2021.std(ddof=1)
n = pm_1950_2021.size
dof = n-1
alpha = 0.05
p_star = 1-alpha/2
t_star = t.ppf(p_star, dof)
ME = t_star*std_dev/np.sqrt(n)

fig = plt.figure(figsize=(5,6))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

plt.boxplot(nw_mean_1950_2021, notch=False, meanline=True,showmeans=True, positions=[0.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.boxplot(ne_mean_1950_2021, notch=False, meanline=True,showmeans=True, positions=[0.8+0.6], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.boxplot(sw_mean_1950_2021, notch=False, meanline=True,showmeans=True, positions=[0.8+1.2], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.boxplot(se_mean_1950_2021, notch=False, meanline=True,showmeans=True, positions=[0.8+1.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))

plt.errorbar(x=0.2, y=mean, yerr=ME, lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.grid(ls='--', lw=0.4, color='grey', axis='y')
plt.xlim([-0.1,3.1])
plt.xticks([0.2,0.8, 0.8+0.6, 0.8+1.2, 0.8+1.8], ["PM","NW", "NE", "SW", "SE"], rotation=0)
plt.ylabel('JFM acc precip mean 1950-2021 (mm)')
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
plt.savefig('../../../hyperdrought_data/png/LENS2_PM_bias_mean_confint.png', dpi=300)
plt.show()