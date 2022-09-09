import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import norm
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/ENSO/'
sys.path.append(join(currentdir,'../../processing'))

import processing.series as se
import processing.utils as ut

# get LENS2 data
pr = xr.open_dataset(join(currentdir, relpath, 'pr_80ens.nc'))['pr']*3600*1000*24
nino34 = xr.open_dataset(join(currentdir, relpath, 'sst_nino34_80ens.nc'))['sst']-273.15

nino34a = nino34.where(nino34.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='1Y').mean('time')
precip = pr.where(pr.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='1Y').mean('time')

ini_year = '1850'
end_year = '1880'
precip = precip.sel(time=slice(ini_year, end_year))
nino34a = nino34a.sel(time=slice(ini_year, end_year))

nino34a = nino34a - nino34a.sel(time=slice(ini_year, end_year)).mean()
precip = precip/precip.sel(time=slice(ini_year, end_year)).mean('time')*100

precip = precip.sortby('run').sortby('time')
nino34a = nino34a.sortby('run').sortby('time')

print( sum([x==y for x,y in zip(precip.run.values, nino34a.run.values)]) )

precip_ = np.ravel(precip.values)
nino34a_ = np.ravel(nino34a.values)

xx, yy, dd = ut.get_nearest_90p_contour(nino34a_, precip_, -4, 7, 0, 220)
r, p = pearsonr(nino34a_, precip_)

# scatter full period
fig = plt.figure(figsize=(6,8))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
plt.contourf(xx,yy,dd,cmap='Greys',levels=[0,4.61],alpha=0.2)
plt.contour(xx,yy,dd,colors=['red'],levels=[9.21], linewidths=[0.5], linestyles='--')
plt.scatter(nino34a.values, precip.values, color='b', alpha=0.2, label=f'r = {r:0.2f}*')
plt.scatter(np.mean(nino34a_), np.mean(precip_), s=50, color='k', alpha=0.8, marker='+', label = 'mean')
plt.scatter(1.5, 60, s=50, color='r', alpha=0.8, marker='o')
plt.axhline(100, color='grey', ls='--', lw=1.0)
plt.axvline(0, color='grey', ls='--', lw=1.0)
plt.axhline(80, color='blue', ls='--', lw=1.2)
plt.xlim([-3.8, 3.8])
plt.ylim([10, 220])
plt.yticks(np.arange(20, 240, 20))
plt.legend(frameon=False)
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.xlabel('MJJAS Niño3.4 anomaly [ºC]')
plt.ylabel('Quinta Normal norm precipitation (ref. period) [%]')
plt.savefig(join(currentdir,f'../../../hyperdrought_data/png/LENS2_scatter_MJJAS_precip_nino34_{ini_year}_{end_year}_rene.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()