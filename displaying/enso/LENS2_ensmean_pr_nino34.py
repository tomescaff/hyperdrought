import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/ENSO/'
sys.path.append(join(currentdir, '../../processing'))

import processing.series as se
import processing.utils as ut

# get LENS2 data
pr = xr.open_dataset(join(currentdir, relpath, 'pr_80ens.nc'))['pr']*3600*1000*24
nino34 = xr.open_dataset(join(currentdir, relpath, 'sst_nino34_80ens.nc'))['sst']-273.15
nino34 = nino34 - nino34.sel(time=slice('1981', '2010')).mean()

pr = pr.where(pr.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='1Y').mean('time')
nino34 = nino34.where(nino34.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='1Y').mean('time')

# get Quinta Normal time series
qn = se.get_QN_MJJAS_precip_long_record()
sst = se.get_MJJAS_Nino34_long_record()

x = pr.mean('run').time.dt.year
y1 = pr.mean('run')
y2 = nino34.mean('run')

s1 = pr.std('run')
s2 = nino34.std('run')

fig, axs = plt.subplots(2,1, figsize=(10,7))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10

ax1 = axs[0]
ax2 = axs[1]

ax1.fill_between(x, y1-s1, y1+s1, color='b', alpha=0.2)
ax2.fill_between(x, y2-s2, y2+s2, color='r', alpha=0.2)

ax1.plot(x, y1, 'b', lw=0.8, label='LENS2')
ax1.plot(qn.time.dt.year, qn/(31+30+31+31+30), 'grey', lw=0.8, label='OBS')
ax2.plot(x, y2, 'r', lw=0.8, label='LENS2')
ax2.plot(sst.time.dt.year, sst, color='grey', lw=0.8, label='OBS')

ax1.set_xlabel('Time')
ax2.set_xlabel('Time')
ax1.set_ylabel('MJJAS precip at Quinta Normal (mm/day)', color='b')
ax2.set_ylabel('MJJAS SST at Nino3.4 anomaly (ºC)', color='r')

ax1.tick_params(direction="in")
ax2.tick_params(direction="in")

ax1.set_xlim([1850,2100])
ax2.set_xlim([1850,2100])

ax1.legend()
ax2.legend()

plt.tight_layout()
plt.savefig(join(currentdir,'../../../hyperdrought_data/png/LENS2_ensmean_pr_nino34.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()
