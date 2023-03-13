import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.utils import resample as bootstrap
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir, '../../processing'))

import processing.gmst as gmst
import processing.series as se
import processing.math as pmath

smf = gmst.get_gmst_annual_5year_smooth()
pmf = se.get_PM_JFM_precip()

sm = smf.sel(time=slice('1950','2021'))
pm = pmf.sel(time=slice('1950','2021'))

sm = sm.where(sm.time.dt.year != 2016, drop=True)
pm = pm.where(pm.time.dt.year != 2016, drop=True)

# # bootstrap
nboot = 10
bspreds_sigma0 = np.zeros((nboot,))
bspreds_eta = np.zeros((nboot,))
bspreds_alpha = np.zeros((nboot,))

for i in range(nboot):
    pm_i, sm_i = bootstrap(pm.values, sm.values)
    xopt_i = pmath.mle_gamma_2d_fast(pm_i, sm_i, [70, 4, -0.5])
    bspreds_sigma0[i] = xopt_i[0]
    bspreds_eta[i] = xopt_i[1]
    bspreds_alpha[i] = xopt_i[2]

iter = np.arange(nboot)
ds = xr.Dataset({
    'sigma0': xr.DataArray(bspreds_sigma0, coords=[iter], dims=['iter']),
    'eta':    xr.DataArray(bspreds_eta,    coords=[iter], dims=['iter']),
    'alpha':  xr.DataArray(bspreds_alpha,  coords=[iter], dims=['iter']), 
})
filepath = '../../../hyperdrought_data/output/PM_MLE_precip_PM_GMST_'+str(nboot)+'_evaluation.nc'
ds.to_netcdf(join(currentdir,filepath))