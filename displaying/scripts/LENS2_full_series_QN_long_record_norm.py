import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import sys


sys.path.append('../../processing')

import processing.lens as lens
import processing.series as se
from scipy.stats import gamma

# get Quinta Normal time series
qn = se.get_QN_annual_precip_long_record()

# get LENS2 QNEW time series
lens2 = lens.get_LENS2_annual_precip_NOAA_QNEW()
lens2_ensmean = lens2.mean('run')

# get LENS1 QNEW time series
lens1 = lens.get_LENS1_annual_precip_NOAA_QNEW()
lens1_ensmean = lens1.mean('run')

qn_norm = qn/qn.sel(time=slice('1980', '2010')).mean('time')
lens2_norm = lens2/lens2.sel(time=slice('1980', '2010')).mean('time')
lens1_norm = lens1/lens1.sel(time=slice('1980', '2010')).mean('time')

lens2_norm_ensmean = lens2_norm.mean('run')
lens1_norm_ensmean = lens1_norm.mean('run')

lens1_ensmean_norm = lens1_ensmean/lens1_ensmean.sel(time=slice('1980', '2010')).mean('time')
lens2_ensmean_norm = lens2_ensmean/lens2_ensmean.sel(time=slice('1980', '2010')).mean('time')


# create figure
fig = plt.figure(figsize=(16,5))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

xq = qn.time.dt.year.values
x1 = lens1.time.dt.year.values
x2 = lens2.time.dt.year.values
# plt.plot(x, lens2.values.T, color='grey', alpha = 0.5, lw=0.8)
plt.plot(x2, lens2_norm_ensmean.values, color='blue', lw=1.3, label='LENS2-QN norm - ensmean')
plt.plot(x2, lens2_ensmean_norm.values, color='green', lw=1.3, label='LENS2-QN ensmean - norm')
plt.plot(x1, lens1_norm_ensmean.values, color='grey', lw=1.3, label='LENS1-QN norm - ensmean')
plt.plot(x1, lens1_ensmean_norm.values, color='brown', lw=1.3, label='LENS1-QN ensmean - norm')
# plt.plot(xq, qn_norm.values, color='r', lw=1.3, label='Quinta Normal norm')
plt.xticks(np.arange(1850, 2125, 25), rotation = 0)
plt.ylabel('Acc anual precip (mm)')
plt.xlabel('Time (yr)')
plt.grid(ls='--', lw=0.4, color='grey')
plt.xlim([1850, 2100])
plt.xticks(rotation=0)
plt.legend(loc='upper right',prop={'size': 10})
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.savefig('../../../hyperdrought_data/png/LENS2_LENS1_full_series_norm.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()
