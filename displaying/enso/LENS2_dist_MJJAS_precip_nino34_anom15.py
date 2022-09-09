import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import gamma
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
relpath = '../../../hyperdrought_data/ENSO/'
sys.path.append(join(currentdir,'../../processing'))

import processing.series as se
import processing.utils as ut

# get LENS2 data
pr = xr.open_dataset(join(currentdir, relpath, 'pr_80ens.nc'))['pr']*3600*1000*24
nino34 = xr.open_dataset(join(currentdir, relpath, 'sst_nino34_80ens.nc'))['sst']-273.15

nino34 = nino34.where(nino34.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='1Y').mean('time')
precip = pr.where(pr.time.dt.month.isin([5,6,7,8,9]), drop=True).resample(time='1Y').mean('time')

def get_precip_nino_anom_1p5(precip, nino34, ini_year, end_year):
    precip = precip.sel(time=slice(ini_year, end_year))
    nino34a = nino34.sel(time=slice(ini_year, end_year))

    nino34a = nino34a - nino34a.sel(time=slice(ini_year, end_year)).mean()
    precip = precip/precip.sel(time=slice(ini_year, end_year)).mean('time')*100

    precip = precip.sortby('run').sortby('time')
    nino34a = nino34a.sortby('run').sortby('time')

    print( sum([x==y for x,y in zip(precip.run.values, nino34a.run.values)]) )

    precip_ = np.ravel(precip.values)
    nino34a_ = np.ravel(nino34a.values)

    ans = precip_[nino34a_ >= 1.5]
    return ans

precip_past = get_precip_nino_anom_1p5(precip, nino34, '1850', '1880')
precip_pres = get_precip_nino_anom_1p5(precip, nino34, '1990', '2020')
precip_futu = get_precip_nino_anom_1p5(precip, nino34, '2070', '2100')

x_min = 0
x_max = 450
x = np.linspace(x_min, x_max, 1000)
f = lambda x, z: gamma.pdf(x, *gamma.fit(z, floc=0, scale=1))

# scatter full period
fig = plt.figure(figsize=(8,6))
plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 10
plt.plot(x, f(x, precip_past), 'b')
plt.plot(x, f(x, precip_pres), 'k')
plt.plot(x, f(x, precip_futu), 'r')
plt.axvline(80, color='fuchsia', ls='--', lw=1.2)
plt.xlim([0, 450])
plt.ylim([0,0.01])
plt.xticks([0, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450])
plt.legend(frameon=False)
ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(direction="in")
plt.xlabel('Quinta Normal norm precipitation (ref. period) [%]')
plt.ylabel('PDF(precip | nino34a >= 1.5)')
plt.legend(['1850-1880 "past" distribution', '1990-2020 "present" distribution', '2070-2100 "future" distribution'])
plt.savefig(join(currentdir,f'../../../hyperdrought_data/png/LENS2_dist_MJJAS_precip_nino34_anom15.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
plt.show()