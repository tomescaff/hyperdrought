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

lens2 = lens.get_LENS2_annual_precip_QNEW()#.sel(run=slice(30,31))
lens2_ensmean = lens2.mean('run')

fig = plt.figure(figsize=(16,5))
# plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

x = lens2_ensmean.time.dt.year.values
y = lens2_ensmean.values
plt.plot(x, lens2.values.T, color='grey', alpha = 0.5, lw=0.8)
plt.plot(x, y, color='b', lw=1.3, label='LENS2-QN ensemble mean')
plt.plot(qn.time.dt.year.values, qn.values, color='r', lw=1.3, label='Quinta Normal')
plt.xticks(np.arange(1850, 2125, 25), rotation = 0)
plt.ylabel('Acc anual precip (mm)')
plt.xlabel('Time (yr)')
plt.grid(ls='--', lw=0.4, color='grey')
plt.xlim([1850, 2100])
plt.xticks(rotation=0)
#plt.ylim([24.5, 38.5])
plt.legend(loc='upper right',prop={'size': 10})
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.show()

# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(10) 
#     tick.label.set_weight('light') 
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(10) 
#     tick.label.set_weight('light')

# plt.sca(axs[1])
# normfit_past = norm.fit(np.ravel(lens2_fixed_1928_2021.sel(time=slice('1851','1880')).values))
# normfit_present = norm.fit(np.ravel(lens2_fixed_1928_2021.sel(time=slice('1991','2020')).values))
# normfit_future = norm.fit(np.ravel(lens2_fixed_1928_2021.sel(time=slice('2071','2100')).values))
# xmin, xmax = 24.5, 38.5
# plt.ylim([0, 0.4])
# plt.xlim([xmin, xmax])
# xval = np.linspace(xmin, xmax, 100)
# plt.plot(xval, norm.pdf(xval, *normfit_past), color='b', lw=1.5)
# plt.plot(xval, norm.pdf(xval, *normfit_present), color='k', lw=1.5)
# plt.plot(xval, norm.pdf(xval, *normfit_future), color='r', lw=1.5)
# plt.axvline(np.mean(np.ravel(lens2_fixed_1928_2021.sel(time=slice('1991','2020')))) + qn_anom, lw=1.8, c='grey')
# plt.grid(ls='--', lw=0.4, color='grey')
# plt.ylabel('PDF')
# plt.xlabel('January Tmax (ºC)')
# ax = plt.gca()
# ax.spines.right.set_visible(False)
# ax.spines.top.set_visible(False)
# ax.tick_params(direction="in")
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(10) 
#     tick.label.set_weight('light') 
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(10) 
#     tick.label.set_weight('light')
# plt.tight_layout()
# plt.savefig('../../../megafires_data/png/LENS2_timeseries_QNWE.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
