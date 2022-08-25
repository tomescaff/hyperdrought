import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import sys


sys.path.append('../../processing')

import processing.lens as lens
import processing.series as se
from scipy.stats import gamma

# get Quinta Normal time series
qn = se.get_QN_annual_precip()

# get LENS2 QNEW time series
lens1 = lens.get_LENS1_annual_precip_QNEW()
lens1_ensmean = lens1.mean('run')

# create figure
fig = plt.figure(figsize=(16,5))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

x = lens1_ensmean.time.dt.year.values
y = lens1_ensmean.values
plt.plot(x, lens1.values.T, color='grey', alpha = 0.5, lw=0.8)
plt.plot(x, y, color='b', lw=1.3, label='LENS1-QN ensemble mean')
plt.plot(qn.time.dt.year.values, qn.values, color='r', lw=1.3, label='Quinta Normal')
plt.xticks(np.arange(1920, 2125, 25), rotation = 0)
plt.ylabel('Acc anual precip (mm)')
plt.xlabel('Time (yr)')
plt.grid(ls='--', lw=0.4, color='grey')
plt.xlim([1850, 2100])
plt.ylim([0,1200])
plt.xticks(rotation=0)
plt.legend(loc='upper right',prop={'size': 10})
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.savefig('../../../hyperdrought_data/png/LENS1_full_series_PRECT_NOAA.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()
