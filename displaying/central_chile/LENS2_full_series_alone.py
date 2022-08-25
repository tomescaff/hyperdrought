import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import sys


sys.path.append('../../processing')

import processing.lens as lens
import processing.series as se
from scipy.stats import gamma

# get LENS2 QNEW time series
lens2 = lens.get_LENS2_annual_precip_NOAA_QNEW()
lens2_ensmean = lens2.mean('run')

# create figure
fig = plt.figure(figsize=(16,5))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

x = lens2_ensmean.time.dt.year.values
y = lens2_ensmean.values
plt.plot(x, y, color='b', lw=1.3, label='LENS2-QN ensemble mean')
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
plt.savefig('../../../hyperdrought_data/png/LENS2_full_series_alone.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()
