import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

sys.path.append('../../processing')

import processing.series as se
import processing.lens as lens

# get Quinta Normal time series
qn = se.get_QN_annual_precip_long_record()
lens2 = lens.get_LENS2_annual_precip_NOAA_QNEW()

qn_1866_2021 = qn.sel(time=slice('1866', '2021'))

lens2_1866_2021 = lens2.sel(time=slice('1866', '2021'))
lens2_mean_1866_2021 = lens2_1866_2021.mean('time')

########################
# QN mean estimation 
########################

mean = qn_1866_2021.mean()
std_dev = qn_1866_2021.std(ddof=1)
n = qn_1866_2021.size
dof = n-1
alpha = 0.05
p_star = 1-alpha/2
t_star = t.ppf(p_star, dof)
ME = t_star*std_dev/np.sqrt(n)

fig = plt.figure(figsize=(5,6))
# plt.rcParams["font.family"] = 'Arial'
plt.boxplot(lens2_mean_1866_2021, notch=False, meanline=True,showmeans=True, positions=[0.8], patch_artist=True, showfliers=True, boxprops={'facecolor':'grey', 'alpha':0.6}, medianprops=dict(color='blue'), meanprops=dict(color='red'))
plt.errorbar(x=0.2, y=mean, yerr=ME, lw=1.2, color='r', capsize=10, fmt = 'o', capthick=1.5)
plt.grid(ls='--', lw=0.4, color='grey', axis='y')
# plt.yticks(np.arange(26, 31+1.0, 1.0))
# plt.ylim([25.5, 30.2])
plt.xlim([-0.1,1.1])
plt.xticks([0.2,0.8], ["",""], rotation=0)
plt.ylabel('Annual precip mean 1866-2021 (mm)')
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
plt.savefig('../../../hyperdrought_data/png/LENS2_QNWE_bias_mean_confint.png', dpi=300)
plt.show()